from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbones.fgclip2 import FGCLIP2_LARGE_MODEL_ID, FGCLIP2_LARGE_REVISION
from cache.features import (
    get_features_by_ids,
    get_text_features_by_sample_ids,
    load_feature_manifest,
    load_features,
    load_text_features,
    validate_feature_manifest,
    validate_text_cache_subdir,
)
from datasets.common import collate_cir_samples
from datasets.fashioniq import FashionIQDataset, load_correction_dict
from evaluation.fashioniq import (
    build_fashioniq_gallery,
    evaluate_fashioniq_category,
    macro_average_fashioniq,
)
from models.taper import R1_ROUTING_SUPPORT_EPS, TAPER


CATEGORIES = ("dress", "shirt", "toptee")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forensic diagnosis for TAPER R1 FG-CLIP2 + token-axis Entmax-1.5."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/fashionIQ_dataset"))
    parser.add_argument("--cache-root", type=Path, default=Path("features"))
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("conf/experiment/taper_e2e.yaml"),
    )
    parser.add_argument(
        "--protocol",
        choices=("fashioniq_original", "fashioniq_val"),
        default="fashioniq_original",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--max-queries-per-category",
        type=int,
        default=0,
        help="0 = full validation set; positive value = quick forensic subset.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/taper_r1_entmax_diagnosis.json"),
    )
    return parser.parse_args()


def load_correction_dicts(annotation_root: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for category in CATEGORIES:
        path = annotation_root / f"correction_dict_{category}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing FashionIQ correction dictionary: {path}")
        result[category] = load_correction_dict(path)
    return result


def build_val_loaders(
    *,
    annotation_root: Path,
    batch_size: int,
    num_workers: int,
    caption_policy: str,
    correction_dicts: dict[str, dict[str, str]] | None,
) -> tuple[dict[str, DataLoader], dict[str, list]]:
    loaders: dict[str, DataLoader] = {}
    annotations: dict[str, list] = {}

    for category in CATEGORIES:
        dataset = FashionIQDataset(
            annotation_root=annotation_root,
            split="val",
            categories=[category],
            caption_policy=caption_policy,
            correction_dicts=correction_dicts,
        )
        loaders[category] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_cir_samples,
            pin_memory=True,
        )
        annotations[category] = dataset.annotations

    return loaders, annotations


def build_model(cfg, device: torch.device) -> TAPER:
    m = cfg.model
    model = TAPER(
        text_dim=m.text_dim,
        reference_dim=m.reference_dim,
        query_dim=m.query_dim,
        slot_dim=m.slot_dim,
        state_dim=m.state_dim,
        num_slots=m.num_slots,
        num_primitives=m.num_primitives,
        mask_temperature=m.mask_temperature,
        router_temperature=m.router_temperature,
        retrieval_temperature=m.retrieval_temperature,
        qasa_tau=m.qasa_tau,
        qasa_rho=m.qasa_rho,
        qasa_mu=m.qasa_mu,
        qasa_eps=m.qasa_eps,
        qasa_apply_at_eval=m.qasa_apply_at_eval,
        alpha_max=m.alpha_max,
    )
    return model.to(device)


def load_checkpoint(model: TAPER, checkpoint: Path) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=True)


class ScalarAccumulator:
    def __init__(self) -> None:
        self.sum = defaultdict(float)
        self.count = defaultdict(int)

    def add(self, name: str, values: torch.Tensor) -> None:
        x = values.detach().float().reshape(-1)
        x = x[torch.isfinite(x)]
        if x.numel() == 0:
            return
        self.sum[name] += x.sum().item()
        self.count[name] += x.numel()

    def mean(self, name: str) -> float:
        if self.count[name] == 0:
            return float("nan")
        return self.sum[name] / self.count[name]

    def to_dict(self) -> dict[str, float]:
        return {name: self.mean(name) for name in sorted(self.sum)}


def binary_jaccard_per_sample(
    support: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    b, l, _ = support.shape
    if l < 2:
        return torch.zeros(b, device=support.device)

    s = support.to(torch.float32)
    intersection = s @ s.transpose(1, 2)
    size = s.sum(dim=-1)
    union = size[:, :, None] + size[:, None, :] - intersection

    upper = torch.triu(
        torch.ones(l, l, dtype=torch.bool, device=support.device),
        diagonal=1,
    )
    pair_active = active[:, :, None] & active[:, None, :] & upper[None, :, :]

    jaccard = intersection / union.clamp_min(1.0)
    numer = (jaccard * pair_active.to(jaccard.dtype)).sum(dim=(1, 2))
    denom = pair_active.sum(dim=(1, 2))

    result = torch.full(
        (b,),
        float("nan"),
        dtype=torch.float32,
        device=support.device,
    )
    valid = denom > 0
    result[valid] = numer[valid] / denom[valid]
    return result


@torch.no_grad()
def run(args: argparse.Namespace) -> dict:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.max_queries_per_category < 0:
        raise ValueError("--max-queries-per-category must be >= 0")

    device = torch.device(args.device)
    cfg = OmegaConf.load(args.experiment_config)

    if str(cfg.backbone.model_id) != FGCLIP2_LARGE_MODEL_ID:
        raise ValueError(f"Expected {FGCLIP2_LARGE_MODEL_ID}, got {cfg.backbone.model_id}")
    if str(cfg.backbone.revision) != FGCLIP2_LARGE_REVISION:
        raise ValueError(
            f"Expected FG-CLIP2 revision {FGCLIP2_LARGE_REVISION}, got {cfg.backbone.revision}"
        )

    correction_policy = str(cfg.correction_policy)
    text_cache_subdir = validate_text_cache_subdir(
        str(cfg.text_cache_subdir),
        correction_policy,
    )

    annotation_root = args.dataset_root / "captions"
    split_root = args.dataset_root / "image_splits"
    correction_dicts = (
        load_correction_dicts(annotation_root)
        if correction_policy == "fashioniq"
        else None
    )

    val_loaders, val_annotations = build_val_loaders(
        annotation_root=annotation_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        caption_policy=str(cfg.val_caption_policy),
        correction_dicts=correction_dicts,
    )

    feature_root = args.cache_root / "fashioniq" / "fgclip2-large"

    image_manifest = load_feature_manifest(feature_root / "val" / "images")
    validate_feature_manifest(
        image_manifest,
        model_id=FGCLIP2_LARGE_MODEL_ID,
        revision=FGCLIP2_LARGE_REVISION,
        cache_name="val/images",
    )
    text_manifest = load_feature_manifest(feature_root / "val" / text_cache_subdir)
    validate_feature_manifest(
        text_manifest,
        model_id=FGCLIP2_LARGE_MODEL_ID,
        revision=FGCLIP2_LARGE_REVISION,
        cache_name=f"val/{text_cache_subdir}",
        correction_policy=correction_policy,
    )

    val_images, val_image_idx = load_features(feature_root / "val" / "images")
    val_text = load_text_features(feature_root / "val" / text_cache_subdir)

    model = build_model(cfg, device)
    load_checkpoint(model, args.checkpoint)
    model.eval()

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Val images: {tuple(val_images.shape)}")
    print(f"Val text: {tuple(val_text.states.shape)}")
    print(f"Correction policy: {correction_policy}")
    print(f"Text cache: {text_cache_subdir}")
    print()

    scalar = ScalarAccumulator()
    dominant_slot_counter = Counter()
    qasa_selected_slot_counter = Counter()
    routing_active_slot_counter = Counter()
    category_retrieval: dict[str, dict[str, float]] = {}
    category_reference_only: dict[str, dict[str, float]] = {}
    category_slot_drop: dict[str, dict[int, dict[str, float]]] = {}

    total_queries = 0

    for category in CATEGORIES:
        loader = val_loaders[category]
        annotations = val_annotations[category]

        gallery_ids = build_fashioniq_gallery(
            protocol=args.protocol,
            split_root=split_root,
            category=category,
            annotations=annotations,
            split="val",
        )
        gallery_features = get_features_by_ids(
            gallery_ids,
            val_images,
            val_image_idx,
        ).to(device=device, dtype=torch.float32)

        full_score_batches: list[torch.Tensor] = []
        ref_score_batches: list[torch.Tensor] = []
        drop_score_batches: dict[int, list[torch.Tensor]] = {
            k: [] for k in range(model.num_slots)
        }
        target_ids_all: list[str] = []

        seen = 0
        progress = tqdm(loader, desc=f"Diagnose [{category}]")

        for batch in progress:
            if args.max_queries_per_category and seen >= args.max_queries_per_category:
                break

            cached_reference = get_features_by_ids(
                batch.reference_ids,
                val_images,
                val_image_idx,
            ).to(device=device, dtype=torch.float32)
            reference = cached_reference[:, 0, :]

            text_states, attention_mask, content_mask = get_text_features_by_sample_ids(
                batch.sample_ids,
                batch.modification_texts,
                val_text,
            )
            text_states = text_states.to(device=device, dtype=torch.float32)
            attention_mask = attention_mask.to(device=device, dtype=torch.bool)
            content_mask = content_mask.to(device=device, dtype=torch.bool)

            if args.max_queries_per_category:
                remaining = args.max_queries_per_category - seen
                if remaining <= 0:
                    break
                if reference.shape[0] > remaining:
                    reference = reference[:remaining]
                    text_states = text_states[:remaining]
                    attention_mask = attention_mask[:remaining]
                    content_mask = content_mask[:remaining]
                    target_ids_batch = list(batch.target_ids[:remaining])
                else:
                    target_ids_batch = list(batch.target_ids)
            else:
                target_ids_batch = list(batch.target_ids)

            if any(x is None for x in target_ids_batch):
                raise RuntimeError("Validation batch contains target_id=None")
            target_ids_batch = [str(x) for x in target_ids_batch]

            out = model(
                reference,
                text_states,
                attention_mask,
                text_content_mask=content_mask,
            )

            valid = attention_mask & content_mask
            valid_count = valid.sum(dim=-1).float().clamp_min(1.0)

            # -------- pre-sparse soft ownership --------
            slot_masks = out["slot_masks"]
            slot_mass = out["slot_mass"]
            winner = slot_masks.argmax(dim=1)
            winner_oh = F.one_hot(
                winner,
                num_classes=model.num_slots,
            ).permute(0, 2, 1).to(torch.bool)
            winner_oh &= valid[:, None, :]
            winner_count = winner_oh.sum(dim=-1)
            active_soft = (winner_count > 0).sum(dim=-1).float()

            dominant_share = (
                slot_mass.max(dim=1).values
                / slot_mass.sum(dim=1).clamp_min(1e-12)
            )
            monopoly = (dominant_share >= 0.90).float()
            dominant_ids = slot_mass.argmax(dim=1)

            scalar.add("valid_content_tokens", valid_count)
            scalar.add("soft_active_slots", active_soft)
            scalar.add("soft_dominant_share", dominant_share)
            scalar.add("soft_near_monopoly", monopoly)

            for k in range(model.num_slots):
                scalar.add(f"soft_slot_{k}_mass", slot_mass[:, k])
                scalar.add(
                    f"soft_slot_{k}_winner_count",
                    winner_count[:, k].float(),
                )

            dominant_slot_counter.update(dominant_ids.detach().cpu().tolist())

            # -------- QASA --------
            selected = out["qasa_selected_mask"]
            qasa_count = selected.sum(dim=-1).float()
            qasa_quality = out["qasa_quality"]
            selected_q = torch.where(
                selected,
                qasa_quality,
                torch.zeros_like(qasa_quality),
            ).sum(dim=-1) / selected.sum(dim=-1).clamp_min(1)

            scalar.add("qasa_selected_slots", qasa_count)
            scalar.add("qasa_quality_all_slots", qasa_quality.mean(dim=-1))
            scalar.add("qasa_quality_selected_slots", selected_q)
            scalar.add("qasa_final_coverage", out["qasa_final_coverage"])

            for row in selected.detach().cpu():
                for k in torch.nonzero(row, as_tuple=False).flatten().tolist():
                    qasa_selected_slot_counter.update([k])

            # -------- actual token-axis Entmax routing --------
            routing = out["routing_masks"]
            routing_mass = out["routing_slot_mass"]
            support = (routing > R1_ROUTING_SUPPORT_EPS) & valid[:, None, :]
            support_count = support.sum(dim=-1)
            routing_active = routing_mass > R1_ROUTING_SUPPORT_EPS
            routing_active_count = routing_active.sum(dim=-1).float()

            active_f = routing_active.float()
            denom_active = active_f.sum(dim=-1).clamp_min(1.0)
            support_mean_per_sample = (
                support_count.float() * active_f
            ).sum(dim=-1) / denom_active
            support_fraction_per_slot = support_count.float() / valid_count[:, None]
            support_fraction_per_sample = (
                support_fraction_per_slot * active_f
            ).sum(dim=-1) / denom_active
            zero_fraction_per_sample = 1.0 - support_fraction_per_sample
            overlap = binary_jaccard_per_sample(support, routing_active)

            scalar.add("routing_active_slots", routing_active_count)
            scalar.add("routing_support_mean", support_mean_per_sample)
            scalar.add("routing_support_fraction_mean", support_fraction_per_sample)
            scalar.add("routing_zero_fraction", zero_fraction_per_sample)
            scalar.add("routing_support_jaccard", overlap)

            for k in range(model.num_slots):
                active_k = routing_active[:, k]
                scalar.add(
                    f"routing_slot_{k}_support_when_active",
                    support_count[active_k, k].float(),
                )
                scalar.add(
                    f"routing_slot_{k}_support_fraction_when_active",
                    support_fraction_per_slot[active_k, k],
                )

            for row in routing_active.detach().cpu():
                for k in torch.nonzero(row, as_tuple=False).flatten().tolist():
                    routing_active_slot_counter.update([k])

            # -------- retrieval --------
            full_query = out["q0"]
            ref_query = out["q_reference_only"]
            full_scores = model._retrieval_scores(full_query, gallery_features)
            ref_scores = model._retrieval_scores(ref_query, gallery_features)
            full_score_batches.append(full_scores.cpu())
            ref_score_batches.append(ref_scores.cpu())

            # -------- functional slot-drop --------
            drop = model.slot_drop_queries(
                reference_features=reference,
                text_states=text_states,
                text_attention_mask=attention_mask,
                text_content_mask=content_mask,
            )
            full_q = drop["full_query"]
            dropped_q = drop["dropped_queries"]

            for k in range(model.num_slots):
                cosine_change = 1.0 - F.cosine_similarity(
                    full_q,
                    dropped_q[:, k],
                    dim=-1,
                )
                scalar.add(f"slot_drop_{k}_query_cosine_change", cosine_change)
                scores_k = model._retrieval_scores(dropped_q[:, k], gallery_features)
                drop_score_batches[k].append(scores_k.cpu())

            target_ids_all.extend(target_ids_batch)
            batch_n = len(target_ids_batch)
            seen += batch_n
            total_queries += batch_n

        full_scores_cat = torch.cat(full_score_batches, dim=0)
        ref_scores_cat = torch.cat(ref_score_batches, dim=0)

        category_retrieval[category] = evaluate_fashioniq_category(
            scores=full_scores_cat,
            target_ids=target_ids_all,
            gallery_ids=gallery_ids,
        )
        category_reference_only[category] = evaluate_fashioniq_category(
            scores=ref_scores_cat,
            target_ids=target_ids_all,
            gallery_ids=gallery_ids,
        )

        category_slot_drop[category] = {}
        for k in range(model.num_slots):
            scores_k = torch.cat(drop_score_batches[k], dim=0)
            category_slot_drop[category][k] = evaluate_fashioniq_category(
                scores=scores_k,
                target_ids=target_ids_all,
                gallery_ids=gallery_ids,
            )

    full_macro = macro_average_fashioniq(category_retrieval)
    ref_macro = macro_average_fashioniq(category_reference_only)

    slot_drop_macro: dict[str, dict[str, float]] = {}
    for k in range(model.num_slots):
        per_category = {
            category: category_slot_drop[category][k]
            for category in CATEGORIES
        }
        slot_drop_macro[str(k)] = macro_average_fashioniq(per_category)

    report = {
        "checkpoint": str(args.checkpoint),
        "protocol": args.protocol,
        "num_queries": total_queries,
        "r1_contract": {
            "backbone": FGCLIP2_LARGE_MODEL_ID,
            "revision": FGCLIP2_LARGE_REVISION,
            "routing": "token-axis entmax-1.5 after QASA selection",
            "support_eps": R1_ROUTING_SUPPORT_EPS,
            "correction_policy": correction_policy,
            "text_cache_subdir": text_cache_subdir,
        },
        "retrieval": {
            "full": full_macro,
            "reference_only": ref_macro,
            "per_category_full": category_retrieval,
            "per_category_reference_only": category_reference_only,
        },
        "structure": scalar.to_dict(),
        "slot_frequency": {
            "soft_dominant_slot_counts": {
                str(k): int(dominant_slot_counter[k])
                for k in range(model.num_slots)
            },
            "qasa_selected_slot_counts": {
                str(k): int(qasa_selected_slot_counter[k])
                for k in range(model.num_slots)
            },
            "routing_active_slot_counts": {
                str(k): int(routing_active_slot_counter[k])
                for k in range(model.num_slots)
            },
        },
        "functional_slot_drop": {
            "macro_retrieval_after_drop": slot_drop_macro,
            "per_category": {
                category: {
                    str(k): category_slot_drop[category][k]
                    for k in range(model.num_slots)
                }
                for category in CATEGORIES
            },
        },
    }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("\n========== R1 ENTMAX FORENSIC SUMMARY ==========")
    print(
        "Full retrieval: "
        f"R@10={full_macro['recall_at_10']:.2f} "
        f"R@50={full_macro['recall_at_50']:.2f} "
        f"mean={full_macro['mean_recall']:.2f}"
    )
    print(
        "Reference only: "
        f"R@10={ref_macro['recall_at_10']:.2f} "
        f"R@50={ref_macro['recall_at_50']:.2f} "
        f"mean={ref_macro['mean_recall']:.2f}"
    )

    s = report["structure"]
    keys = (
        "valid_content_tokens",
        "soft_active_slots",
        "soft_dominant_share",
        "soft_near_monopoly",
        "qasa_selected_slots",
        "qasa_quality_all_slots",
        "qasa_quality_selected_slots",
        "qasa_final_coverage",
        "routing_active_slots",
        "routing_support_mean",
        "routing_support_fraction_mean",
        "routing_zero_fraction",
        "routing_support_jaccard",
    )
    for key in keys:
        print(f"{key}: {s.get(key, float('nan')):.4f}")

    print("\nPer-slot sparse support when active:")
    for k in range(model.num_slots):
        print(
            f"  S{k}: support="
            f"{s.get(f'routing_slot_{k}_support_when_active', float('nan')):.3f} "
            f"fraction="
            f"{s.get(f'routing_slot_{k}_support_fraction_when_active', float('nan')):.3f} "
            f"drop_cos_change="
            f"{s.get(f'slot_drop_{k}_query_cosine_change', float('nan')):.6f}"
        )

    print("\nSlot frequencies:")
    print("  dominant:", report["slot_frequency"]["soft_dominant_slot_counts"])
    print("  qasa selected:", report["slot_frequency"]["qasa_selected_slot_counts"])
    print("  routing active:", report["slot_frequency"]["routing_active_slot_counts"])

    print("\nFunctional slot-drop macro mean recall:")
    for k in range(model.num_slots):
        value = slot_drop_macro[str(k)]["mean_recall"]
        delta = value - full_macro["mean_recall"]
        print(f"  Drop S{k}: mean={value:.3f} delta={delta:+.3f}")

    print(f"\nSaved JSON: {args.json_output}")
    return report


if __name__ == "__main__":
    run(parse_args())   