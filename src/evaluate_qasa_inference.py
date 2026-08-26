from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from cache.features import (
    get_features_by_ids,
    get_text_features_by_sample_ids,
    load_features,
    load_text_features,
)
from datasets.common import collate_cir_samples
from datasets.fashioniq import FashionIQDataset, load_correction_dict
from models.taper import TAPER
from teachers.csmcir_compose import CSMCIRComposeTeacher


CATEGORIES = ("dress", "shirt", "toptee")
SLOT_VALUE_ASSIGNMENTS = tuple(sorted(TAPER.SLOT_VALUE_ASSIGNMENTS))


def parse_args():
    p = argparse.ArgumentParser(
        description="QASA-faithful hard-partition evaluation for TAPER."
    )
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    p.add_argument("--dataset-root", type=Path, default=Path("data/FashionIQ"))
    p.add_argument("--cache-root", type=Path, default=Path("features"))
    p.add_argument(
        "--config",
        type=Path,
        default=Path("conf/experiment/taper_e2e.yaml"),
    )
    p.add_argument(
        "--slot-value-assignment",
        choices=SLOT_VALUE_ASSIGNMENTS,
        default=None,
        help="Override model.slot_value_assignment; must match checkpoint provenance.",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--max-queries-per-category",
        type=int,
        default=0,
        help="0 = full validation; e.g. 512 for quick check.",
    )
    p.add_argument("--num-examples", type=int, default=20)
    p.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/qasa_faithful_inference_eval.json"),
    )
    return p.parse_args()


def newest_checkpoint(outputs_root: Path) -> Path:
    candidates = list(outputs_root.rglob("best.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No best.pt found under {outputs_root}; pass --checkpoint explicitly."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_correction_dicts(annotation_root: Path):
    result = {}
    for category in CATEGORIES:
        path = annotation_root / f"correction_dict_{category}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        result[category] = load_correction_dict(path)
    return result


def build_val_loaders(
    *,
    annotation_root: Path,
    batch_size: int,
    num_workers: int,
    caption_policy: str,
    correction_dicts,
):
    loaders = {}
    for category in CATEGORIES:
        ds = FashionIQDataset(
            annotation_root=annotation_root,
            split="val",
            categories=[category],
            caption_policy=caption_policy,
            correction_dicts=correction_dicts,
        )
        loaders[category] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_cir_samples,
            pin_memory=True,
        )
    return loaders


def build_model(cfg, device: torch.device) -> TAPER:
    m = cfg.model
    teacher = CSMCIRComposeTeacher(
        csmcir_root=cfg.teacher.csmcir_root,
        checkpoint_path=cfg.teacher.checkpoint_path,
    ).to(device).eval()

    return TAPER(
        teacher,
        text_dim=m.text_dim,
        reference_dim=m.reference_dim,
        teacher_text_dim=m.teacher_text_dim,
        teacher_query_dim=m.teacher_query_dim,
        query_dim=m.query_dim,
        slot_dim=m.slot_dim,
        state_dim=m.state_dim,
        num_slots=m.num_slots,
        num_primitives=m.num_primitives,
        mask_temperature=m.mask_temperature,
        router_temperature=m.router_temperature,
        retrieval_temperature=m.retrieval_temperature,
        neutral_mode=m.neutral_mode,
        qasa_tau=m.qasa_tau,
        qasa_rho=m.qasa_rho,
        qasa_mu=m.qasa_mu,
        qasa_eps=m.qasa_eps,
        qasa_apply_at_eval=m.qasa_apply_at_eval,
        alpha_max=m.alpha_max,
        counterfactual_chunk_size=m.counterfactual_chunk_size,
        slot_value_source=m.slot_value_source,
        slot_effect_in_value=m.slot_effect_in_value,
        slot_value_assignment=m.slot_value_assignment,
    ).to(device)


def load_checkpoint(model: TAPER, path: Path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")

    checkpoint_provenance = None
    if isinstance(state, dict) and "model_state_dict" in state:
        checkpoint_provenance = state.get("experiment_provenance")
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    expected_provenance = model.experiment_provenance()
    if checkpoint_provenance is not None and checkpoint_provenance != expected_provenance:
        raise RuntimeError(
            "Checkpoint experiment provenance mismatch: "
            f"expected={expected_provenance}, cached={checkpoint_provenance}"
        )
    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
    except RuntimeError as error:
        raise RuntimeError(
            "Incompatible TAPER checkpoint. This experiment changes slot_mlp "
            "from contextual+effect input to teacher-raw-only input and must be "
            "trained from scratch; do not load an A3.1 TAPER checkpoint."
        ) from error
    bad_missing = [k for k in missing if not k.startswith("teacher.")]
    if bad_missing:
        raise RuntimeError("Missing non-teacher keys:\n" + "\n".join(bad_missing))
    if unexpected:
        raise RuntimeError("Unexpected checkpoint keys:\n" + "\n".join(unexpected))

    print(f"Loaded checkpoint: {path}")


class Accumulator:
    def __init__(self, num_slots: int):
        self.num_slots = num_slots
        self.samples = 0
        self.valid_tokens = 0
        self.sum_k = 0.0
        self.k_hist = defaultdict(int)
        self.sum_dominant = 0.0
        self.sum_hard_entropy = 0.0
        self.sum_top1 = 0.0
        self.sum_margin = 0.0
        self.sum_near_tie = 0.0
        self.slot_winner_tokens = torch.zeros(num_slots, dtype=torch.float64)
        self.slot_nonempty = torch.zeros(num_slots, dtype=torch.float64)
        self.slot_share = torch.zeros(num_slots, dtype=torch.float64)
        self.category = {
            c: {"samples": 0, "sum_k": 0.0, "sum_dominant": 0.0}
            for c in CATEGORIES
        }

    def update(
        self,
        category: str,
        effective_k: torch.Tensor,
        winner_counts: torch.Tensor,
        nonempty_slots: torch.Tensor,
        valid_mask: torch.Tensor,
        qasa_attention: torch.Tensor,
    ):
        b, l, _ = qasa_attention.shape
        if l != self.num_slots:
            raise RuntimeError("slot count mismatch")

        valid_count = valid_mask.sum(dim=1).clamp_min(1)
        shares = winner_counts.float() / valid_count[:, None].float()
        dominant = shares.max(dim=1).values

        p = shares.clamp_min(1e-12)
        hard_entropy = -(p * p.log()).sum(dim=1)
        if self.num_slots > 1:
            hard_entropy = hard_entropy / math.log(self.num_slots)

        top = qasa_attention.topk(k=min(2, self.num_slots), dim=1).values
        top1 = top[:, 0]
        margin = top[:, 0] - top[:, 1] if self.num_slots > 1 else top[:, 0]
        vf = valid_mask.to(qasa_attention.dtype)
        denom = vf.sum(dim=1).clamp_min(1.0)
        mean_top1 = (top1 * vf).sum(dim=1) / denom
        mean_margin = (margin * vf).sum(dim=1) / denom
        near_tie = (((margin <= 0.01) & valid_mask).float().sum(dim=1) / denom)

        self.samples += b
        self.valid_tokens += int(valid_mask.sum().item())
        self.sum_k += float(effective_k.float().sum().item())
        self.sum_dominant += float(dominant.sum().item())
        self.sum_hard_entropy += float(hard_entropy.sum().item())
        self.sum_top1 += float(mean_top1.sum().item())
        self.sum_margin += float(mean_margin.sum().item())
        self.sum_near_tie += float(near_tie.sum().item())

        for k in effective_k.tolist():
            self.k_hist[int(k)] += 1

        self.slot_winner_tokens += winner_counts.double().sum(dim=0).cpu()
        self.slot_nonempty += nonempty_slots.double().sum(dim=0).cpu()
        self.slot_share += shares.double().sum(dim=0).cpu()

        c = self.category[category]
        c["samples"] += b
        c["sum_k"] += float(effective_k.float().sum().item())
        c["sum_dominant"] += float(dominant.sum().item())

    def finalize(self):
        n = max(self.samples, 1)
        return {
            "samples": self.samples,
            "valid_tokens": self.valid_tokens,
            "mean_effective_k": self.sum_k / n,
            "k_distribution": {
                str(k): self.k_hist[k] / n
                for k in range(self.num_slots + 1)
            },
            "k1_fraction": self.k_hist[1] / n,
            "mean_dominant_hard_share": self.sum_dominant / n,
            "mean_hard_winner_entropy": self.sum_hard_entropy / n,
            "mean_soft_top1_probability": self.sum_top1 / n,
            "mean_soft_top1_top2_margin": self.sum_margin / n,
            "mean_near_tie_fraction_margin_le_0_01": self.sum_near_tie / n,
            "per_slot": {
                str(s): {
                    "mean_winner_tokens_per_sample": float(self.slot_winner_tokens[s] / n),
                    "nonempty_rate": float(self.slot_nonempty[s] / n),
                    "mean_hard_token_share": float(self.slot_share[s] / n),
                }
                for s in range(self.num_slots)
            },
            "per_category": {
                c: {
                    "samples": v["samples"],
                    "mean_effective_k": v["sum_k"] / max(v["samples"], 1),
                    "mean_dominant_hard_share": v["sum_dominant"] / max(v["samples"], 1),
                }
                for c, v in self.category.items()
                if v["samples"] > 0
            },
        }


def make_examples(batch, winner_ids, winner_counts, effective_k, qasa_attention, valid_mask):
    rows = []
    top = qasa_attention.topk(k=min(2, qasa_attention.shape[1]), dim=1).values
    top1 = top[:, 0]
    margin = top[:, 0] - top[:, 1] if qasa_attention.shape[1] > 1 else top[:, 0]

    for i in range(winner_ids.shape[0]):
        valid = valid_mask[i]
        rows.append({
            "sample_id": str(batch.sample_ids[i]),
            "modification_text": str(batch.modification_texts[i]),
            "effective_k": int(effective_k[i].item()),
            "winner_counts": [int(x) for x in winner_counts[i].tolist()],
            "winner_ids_valid_tokens": [int(x) for x in winner_ids[i][valid].tolist()],
            "mean_top1_probability": float(top1[i, valid].mean().item()) if valid.any() else float("nan"),
            "mean_top1_top2_margin": float(margin[i, valid].mean().item()) if valid.any() else float("nan"),
        })
    return rows


@torch.no_grad()
def run(args):
    if args.max_queries_per_category < 0:
        raise ValueError("--max-queries-per-category must be >= 0")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    checkpoint = args.checkpoint or newest_checkpoint(args.outputs_root)
    device = torch.device(args.device)
    cfg = OmegaConf.load(args.config)
    if args.slot_value_assignment is not None:
        cfg.model.slot_value_assignment = args.slot_value_assignment

    annotation_root = args.dataset_root / "captions"
    correction_dicts = load_correction_dicts(annotation_root)
    loaders = build_val_loaders(
        annotation_root=annotation_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        caption_policy=cfg.val_caption_policy,
        correction_dicts=correction_dicts,
    )

    feature_root = args.cache_root / "fashioniq" / "csmcir" / "val"
    native_features, native_idx = load_features(feature_root / "native")
    text_cache = load_text_features(feature_root / "text")

    model = build_model(cfg, device)
    load_checkpoint(model, checkpoint)
    model.eval()

    acc = Accumulator(model.num_slots)
    examples = []

    for category in CATEGORIES:
        processed = 0
        progress = tqdm(loaders[category], desc=f"QASA eval [{category}]", dynamic_ncols=True)

        for batch in progress:
            if args.max_queries_per_category and processed >= args.max_queries_per_category:
                break

            reference_native = get_features_by_ids(
                batch.reference_ids,
                native_features,
                native_idx,
            ).to(device=device, dtype=torch.float32)
            reference_features = reference_native[:, 0, :]

            text_states, teacher_text_states, attention_mask, content_mask = (
                get_text_features_by_sample_ids(
                    batch.sample_ids,
                    batch.modification_texts,
                    text_cache,
                )
            )
            text_states = text_states.to(device=device, dtype=torch.float32)
            teacher_text_states = teacher_text_states.to(device=device, dtype=torch.float32)
            attention_mask = attention_mask.to(device=device, dtype=torch.bool)
            content_mask = content_mask.to(device=device, dtype=torch.bool)

            # QASA-faithful evaluation only:
            # no model.forward(), no Executor, no retrieval, no qasa_selected_mask.
            out = model.build_edit_slots(
                reference_features,
                text_states,
                attention_mask,
                text_content_mask=content_mask,
                teacher_reference_features=reference_native,
                teacher_text_states=teacher_text_states,
            )

            winner_ids = out["qasa_inference_winner_ids"]
            hard_regions = out["qasa_inference_hard_regions"]
            nonempty_slots = out["qasa_inference_nonempty_slots"]
            effective_k = out["qasa_inference_effective_k"]
            winner_counts = out["qasa_inference_winner_counts"]
            qasa_attention = out["qasa_attention"]
            valid_mask = out["qasa_valid_mask"]

            # Independent evaluator smoke tests.
            hard_sum = hard_regions.sum(dim=1)
            if valid_mask.any() and not torch.equal(
                hard_sum[valid_mask], torch.ones_like(hard_sum[valid_mask])
            ):
                raise RuntimeError("valid token must belong to exactly one slot")
            if (~valid_mask).any() and hard_sum[~valid_mask].any():
                raise RuntimeError("invalid token must belong to no slot")

            acc.update(
                category,
                effective_k,
                winner_counts,
                nonempty_slots,
                valid_mask,
                qasa_attention,
            )

            if len(examples) < args.num_examples:
                remaining = args.num_examples - len(examples)
                examples.extend(
                    make_examples(
                        batch,
                        winner_ids,
                        winner_counts,
                        effective_k,
                        qasa_attention,
                        valid_mask,
                    )[:remaining]
                )

            processed += len(batch.sample_ids)
            progress.set_postfix(
                n=processed,
                meanK=f"{acc.sum_k / max(acc.samples, 1):.3f}",
            )

    summary = acc.finalize()
    report = {
        "checkpoint": str(checkpoint),
        "experiment_provenance": model.experiment_provenance(),
        "protocol": (
            "QASA-faithful inference: token-wise argmax over slot attention; "
            "no QASA selection mask; no Executor; no retrieval."
        ),
        "num_slots": model.num_slots,
        "summary": summary,
        "examples": examples,
    }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 68)
    print("QASA-FAITHFUL HARD-PARTITION EVALUATION")
    print("=" * 68)
    print(f"Samples:                    {summary['samples']}")
    print(f"Valid content tokens:       {summary['valid_tokens']}")
    print()
    print(f"Mean effective K:           {summary['mean_effective_k']:.4f}")
    print(f"K=1 fraction:               {100.0 * summary['k1_fraction']:.2f}%")
    print()
    print("K distribution:")
    for k in range(model.num_slots + 1):
        frac = summary["k_distribution"][str(k)]
        print(f"  K={k}: {100.0 * frac:6.2f}%")
    print()
    print(f"Mean dominant hard share:   {summary['mean_dominant_hard_share']:.4f}")
    print(f"Mean hard winner entropy:   {summary['mean_hard_winner_entropy']:.4f}")
    print(f"Mean soft top1 probability: {summary['mean_soft_top1_probability']:.4f}")
    print(f"Mean top1-top2 margin:       {summary['mean_soft_top1_top2_margin']:.4f}")
    print(
        "Near-tie tokens (<=0.01):   "
        f"{100.0 * summary['mean_near_tie_fraction_margin_le_0_01']:.2f}%"
    )
    print()
    print("Per-slot:")
    for s in range(model.num_slots):
        x = summary["per_slot"][str(s)]
        print(
            f"  S{s}: winner_tokens/sample={x['mean_winner_tokens_per_sample']:.3f} | "
            f"nonempty={100.0 * x['nonempty_rate']:.2f}% | "
            f"hard_share={x['mean_hard_token_share']:.4f}"
        )
    print()
    print("Per-category:")
    for category, x in summary["per_category"].items():
        print(
            f"  {category:7s} n={x['samples']:4d} | "
            f"meanK={x['mean_effective_k']:.4f} | "
            f"dominant={x['mean_dominant_hard_share']:.4f}"
        )
    print()
    print("JSON:", args.json_output)


if __name__ == "__main__":
    run(parse_args())
