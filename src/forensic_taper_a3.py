from __future__ import annotations

import argparse
import json
import math
import types
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from cache.features import (
    get_features_by_ids,
    get_text_features_by_sample_ids,
    load_features,
    load_text_features,
)
from diagnose_taper_checkpoint import (
    CATEGORIES,
    build_model,
    build_val_loaders,
    load_correction_dicts,
    load_taper_checkpoint,
    macro_retrieval_metrics,
    target_ranks,
    update_retrieval_statistics,
)
from evaluation.fashioniq import build_fashioniq_gallery


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "A3 frozen-checkpoint forensic suite: FULL capped at k execution steps, keep-one-slot, "
            "mean-slot x1/x2/x3/x4, duplicate-each-slot x4, hardize-current, ownership margins, "
            "and slot-query gradient cosine."
        )
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, default=Path("data/FashionIQ"))
    p.add_argument("--cache-root", type=Path, default=Path("features"))
    p.add_argument("--config", type=Path, default=Path("conf/experiment/taper_e2e.yaml"))
    p.add_argument(
        "--protocol",
        type=str,
        default="fashioniq_original",
        choices=("fashioniq_original", "fashioniq_val"),
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--max-queries-per-category",
        type=int,
        default=256,
        help="0 = full validation; 256 is a good first forensic pass.",
    )
    p.add_argument(
        "--gradient-batches",
        type=int,
        default=8,
        help="Number of real validation batches used for slot-query gradient-cosine audit.",
    )
    p.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/taper_a3_forensic_depth.json"),
    )
    return p.parse_args()


class VectorCollector:
    def __init__(self):
        self.values: dict[str, list[torch.Tensor]] = defaultdict(list)

    def add(self, name: str, x: torch.Tensor):
        x = x.detach().float().reshape(-1).cpu()
        x = x[torch.isfinite(x)]
        if x.numel():
            self.values[name].append(x)

    def summary(self, name: str):
        chunks = self.values.get(name, [])
        if not chunks:
            return None
        x = torch.cat(chunks)
        qs = torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90])
        q = torch.quantile(x, qs)
        return {
            "n": int(x.numel()),
            "mean": float(x.mean()),
            "std": float(x.std(unbiased=False)),
            "min": float(x.min()),
            "p10": float(q[0]),
            "p25": float(q[1]),
            "median": float(q[2]),
            "p75": float(q[3]),
            "p90": float(q[4]),
            "max": float(x.max()),
            "frac_abs_le_1e-3": float((x.abs() <= 1e-3).float().mean()),
            "frac_abs_le_1e-2": float((x.abs() <= 1e-2).float().mean()),
            "frac_abs_le_1e-1": float((x.abs() <= 1e-1).float().mean()),
        }


def pairwise_row_cosine(x: torch.Tensor) -> tuple[float, torch.Tensor]:
    """x: [L,D]. Return mean off-diagonal cosine + full matrix."""
    if x.ndim != 2:
        raise ValueError("Expected [L,D]")
    z = F.normalize(x.float(), dim=-1, eps=1e-12)
    sim = z @ z.T
    if x.shape[0] < 2:
        return float("nan"), sim
    mask = torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)
    return float(sim[mask].mean()), sim


@contextmanager
def hard_ownership_forward(model):
    """
    Temporarily replace A3 soft ownership by deterministic hard categorical ownership.

    IMPORTANT:
    - no retraining;
    - same learned logits;
    - argmax includes NULL;
    - downstream semantic pooling + teacher counterfactuals are recomputed from the hard masks.
    """
    original = model._competitive_ownership

    def hard_method(self, text_states, slot_valid):
        ownership_logits, _, _ = original(text_states, slot_valid)
        winner = ownership_logits.argmax(dim=1)  # [B,N], NULL is class 0
        hard = F.one_hot(
            winner,
            num_classes=self.num_slots + 1,
        ).permute(0, 2, 1).to(ownership_logits.dtype)
        return ownership_logits, hard[:, 0, :], hard[:, 1:, :]

    model._competitive_ownership = types.MethodType(hard_method, model)
    try:
        yield
    finally:
        model._competitive_ownership = original


def query_from_execution(model, edit_slots, slot_gates, output, disabled_slots=None):
    execution = model.execute(
        edit_slots,
        slot_gates,
        output["z0"],
        output["reference_state"],
        disabled_slots=disabled_slots,
    )
    return model.make_query(execution["final_state"]), execution


def make_execution_variants(model, output):
    """
    Return inference-only queries for:
      - full / reference_only
      - keep_i
      - mean_x1 / mean_x2 / mean_x3 / mean_x4
      - slot{i}_x4 for every original slot i

    The mean controls destroy slot identity by replacing every executable slot
    with the same per-sample mean latent/gate. xK means exactly K execution
    tickets are enabled.

    slot{i}_x4 copies one ORIGINAL learned slot latent/gate into all L slot
    positions, so any recovery to FULL cannot come from averaging across slots.
    Router / Primitive Bank / Executor are left untouched and recomputed each step.
    """
    b, l, _ = output["edit_slots"].shape
    device = output["edit_slots"].device

    queries = {
        "full": output["q0"],
        "reference_only": output["q_reference_only"],
    }
    executions = {}

    # Preserve the real FULL route/slots exactly, but read out the query after
    # only the first k execution steps. execute() already returns checkpoints
    # [z0, z1, ..., zL], so this does not rerun or alter Router/Executor.
    checkpoints = output["checkpoints"]
    if checkpoints.ndim != 3 or checkpoints.shape[1] != l + 1:
        raise ValueError(
            f"Expected checkpoints [B,{l + 1},D], got {tuple(checkpoints.shape)}"
        )
    for k in range(1, l + 1):
        queries[f"full_cap{k}"] = model.make_query(checkpoints[:, k, :])


    # Keep exactly one ORIGINAL learned slot executable.
    for slot_id in range(l):
        disabled = torch.ones(b, l, dtype=torch.bool, device=device)
        disabled[:, slot_id] = False
        q, execution = query_from_execution(
            model,
            output["edit_slots"],
            output["slot_gates"],
            output,
            disabled_slots=disabled,
        )
        queries[f"keep_{slot_id}"] = q
        executions[f"keep_{slot_id}"] = execution

    # Destroy slot identity: every executable slot becomes the same mean latent/gate.
    mean_slot = (
        output["edit_slots"]
        .mean(dim=1, keepdim=True)
        .expand_as(output["edit_slots"])
    )
    mean_gate = (
        output["slot_gates"]
        .mean(dim=1, keepdim=True)
        .expand_as(output["slot_gates"])
    )

    # Exactly k recurrent compute tickets, k=1..L.
    for k in range(1, l + 1):
        disabled = torch.ones(b, l, dtype=torch.bool, device=device)
        disabled[:, :k] = False
        q, execution = query_from_execution(
            model,
            mean_slot,
            mean_gate,
            output,
            disabled_slots=disabled,
        )
        queries[f"mean_x{k}"] = q
        executions[f"mean_x{k}"] = execution

    # Copy each ORIGINAL slot into all L positions and grant all L execution tickets.
    for slot_id in range(l):
        repeated_slot = (
            output["edit_slots"][:, slot_id : slot_id + 1, :]
            .expand_as(output["edit_slots"])
        )
        repeated_gate = (
            output["slot_gates"][:, slot_id : slot_id + 1]
            .expand_as(output["slot_gates"])
        )
        q, execution = query_from_execution(
            model,
            repeated_slot,
            repeated_gate,
            output,
            disabled_slots=None,
        )
        queries[f"slot{slot_id}_x{l}"] = q
        executions[f"slot{slot_id}_x{l}"] = execution

    return queries, executions


def load_runtime(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.max_queries_per_category < 0:
        raise ValueError("--max-queries-per-category must be >= 0")
    if args.gradient_batches < 0:
        raise ValueError("--gradient-batches must be >= 0")

    device = torch.device(args.device)
    cfg = OmegaConf.load(args.config)
    annotation_root = args.dataset_root / "captions"
    split_root = args.dataset_root / "image_splits"

    correction_dicts = load_correction_dicts(annotation_root)
    val_loaders, val_annotations = build_val_loaders(
        annotation_root=annotation_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        caption_policy=cfg.val_caption_policy,
        correction_dicts=correction_dicts,
    )

    feature_root = args.cache_root / "fashioniq" / "csmcir" / "val"
    val_retrieval, val_retrieval_idx = load_features(feature_root / "retrieval")
    val_native, val_native_idx = load_features(feature_root / "native")
    val_text = load_text_features(feature_root / "text")

    model = build_model(cfg=cfg, device=device)
    load_taper_checkpoint(model, args.checkpoint)
    model.eval()

    return {
        "device": device,
        "cfg": cfg,
        "split_root": split_root,
        "val_loaders": val_loaders,
        "val_annotations": val_annotations,
        "val_retrieval": val_retrieval,
        "val_retrieval_idx": val_retrieval_idx,
        "val_native": val_native,
        "val_native_idx": val_native_idx,
        "val_text": val_text,
        "model": model,
    }


def prepare_forward_batch(runtime, batch):
    device = runtime["device"]

    reference_native = get_features_by_ids(
        batch.reference_ids,
        runtime["val_native"],
        runtime["val_native_idx"],
    ).to(device=device, dtype=torch.float32)

    reference_features = reference_native[:, 0, :]

    text_states, teacher_text_states, attention_mask, content_mask = (
        get_text_features_by_sample_ids(
            batch.sample_ids,
            batch.modification_texts,
            runtime["val_text"],
        )
    )

    return {
        "reference_native": reference_native,
        "reference_features": reference_features,
        "text_states": text_states.to(device=device, dtype=torch.float32),
        "teacher_text_states": teacher_text_states.to(device=device, dtype=torch.float32),
        "attention_mask": attention_mask.to(device=device, dtype=torch.bool),
        "content_mask": content_mask.to(device=device, dtype=torch.bool),
    }


def model_forward(model, x):
    return model.forward(
        x["reference_features"],
        x["text_states"],
        x["attention_mask"],
        text_content_mask=x["content_mask"],
        teacher_reference_features=x["reference_native"],
        teacher_text_states=x["teacher_text_states"],
    )


def build_target_indices(batch, gallery_index, device):
    target_ids = list(batch.target_ids)
    if any(x is None for x in target_ids):
        raise ValueError("Validation sample missing target_id")
    missing = [x for x in target_ids if x not in gallery_index]
    if missing:
        raise KeyError(f"Targets not found in gallery, first few: {missing[:5]}")
    return torch.tensor(
        [gallery_index[x] for x in target_ids],
        dtype=torch.long,
        device=device,
    )


def run_inference_forensics(args, runtime):
    model = runtime["model"]
    device = runtime["device"]
    num_slots = model.num_slots

    variant_names = (
        ["full", "reference_only"]
        + [f"full_cap{k}" for k in range(1, num_slots + 1)]
        + [f"keep_{i}" for i in range(num_slots)]
        + [f"mean_x{k}" for k in range(1, num_slots + 1)]
        + [f"slot{i}_x{num_slots}" for i in range(num_slots)]
        + ["hardize_current"]
    )

    category_stats = {
        variant: {
            category: {"n": 0, "hit10": 0, "hit50": 0}
            for category in CATEGORIES
        }
        for variant in variant_names
    }

    margins = VectorCollector()
    execution_stats = VectorCollector()

    with torch.no_grad():
        for category in CATEGORIES:
            gallery_ids = build_fashioniq_gallery(
                protocol=args.protocol,
                split_root=runtime["split_root"],
                split="val",
                category=category,
                annotations=runtime["val_annotations"][category],
            )

            gallery_features = get_features_by_ids(
                gallery_ids,
                runtime["val_retrieval"],
                runtime["val_retrieval_idx"],
            ).to(device=device, dtype=torch.float32)

            gallery_index = {image_id: i for i, image_id in enumerate(gallery_ids)}
            processed = 0

            for batch in tqdm(
                runtime["val_loaders"][category],
                desc=f"A3 forensic [{category}]",
                dynamic_ncols=True,
            ):
                if args.max_queries_per_category and processed >= args.max_queries_per_category:
                    break

                x = prepare_forward_batch(runtime, batch)
                output = model_forward(model, x)
                b = x["reference_features"].shape[0]

                take = b
                if args.max_queries_per_category:
                    take = min(take, args.max_queries_per_category - processed)
                if take <= 0:
                    break

                queries, variant_executions = make_execution_variants(model, output)

                # Hardize CURRENT checkpoint without retraining; recompute all downstream slot effects.
                with hard_ownership_forward(model):
                    hard_output = model_forward(model, x)
                queries["hardize_current"] = hard_output["q0"]

                target_indices = build_target_indices(batch, gallery_index, device)

                for variant, query in queries.items():
                    scores = model._retrieval_scores(query, gallery_features)
                    ranks = target_ranks(scores, target_indices)[:take]
                    update_retrieval_statistics(
                        category_stats=category_stats,
                        variant=variant,
                        category=category,
                        ranks=ranks,
                    )

                # ------------------------------------------------------
                # Top1-top2 margins from CURRENT A3 logits.
                # Two forms:
                # 1) all destinations: NULL + 4 Edit Slots
                # 2) edit-only: 4 Edit Slots (more direct symmetry probe)
                # ------------------------------------------------------
                logits = output["ownership_logits"][:take]
                valid = x["content_mask"][:take]

                all_top2 = logits.topk(k=2, dim=1).values
                all_margin = (all_top2[:, 0] - all_top2[:, 1])[valid]
                margins.add("all_destination_logit_margin", all_margin)

                edit_logits = logits[:, 1:, :]
                if num_slots >= 2:
                    edit_top2 = edit_logits.topk(k=2, dim=1).values
                    edit_margin = (edit_top2[:, 0] - edit_top2[:, 1])[valid]
                    margins.add("edit_only_logit_margin", edit_margin)

                # Verify how many real FULL steps have actually executed by each cap.
                for k in range(1, num_slots + 1):
                    execution_stats.add(
                        f"full_cap{k}_valid_steps",
                        output["trace_valid_mask"][:take, :k].sum(dim=1),
                    )

                # Verify how many recurrent execution tickets each synthetic control actually used.
                for name, execution in variant_executions.items():
                    if name.startswith("mean_x") or (name.startswith("slot") and "_x" in name):
                        execution_stats.add(
                            f"{name}_valid_steps",
                            execution["trace_valid_mask"][:take].sum(dim=1),
                        )

                processed += take

    retrieval = {
        variant: macro_retrieval_metrics(category_stats, variant)
        for variant in variant_names
    }

    return {
        "retrieval": retrieval,
        "ownership_margins": {
            "all_destination": margins.summary("all_destination_logit_margin"),
            "edit_only": margins.summary("edit_only_logit_margin"),
        },
        "execution_controls": {
            name: execution_stats.summary(name)
            for name in sorted(execution_stats.values)
        },
    }


def run_gradient_forensics(args, runtime):
    """
    Measure cosine between per-slot query gradients from real retrieval loss.

    We intentionally use model.eval() so hard execution eligibility matches diagnosis,
    while gradients remain enabled. No optimizer step is performed.
    """
    if args.gradient_batches == 0:
        return {"skipped": True}

    model = runtime["model"]
    device = runtime["device"]
    model.eval()

    pairwise_means: list[float] = []
    matrices: list[torch.Tensor] = []
    grad_norms: list[torch.Tensor] = []
    loss_values: list[float] = []

    used = 0
    for category in CATEGORIES:
        if used >= args.gradient_batches:
            break

        for batch in runtime["val_loaders"][category]:
            if used >= args.gradient_batches:
                break

            x = prepare_forward_batch(runtime, batch)
            target_ids = list(batch.target_ids)
            if any(t is None for t in target_ids):
                raise ValueError("Validation sample missing target_id")

            target_features = get_features_by_ids(
                target_ids,
                runtime["val_retrieval"],
                runtime["val_retrieval_idx"],
            ).to(device=device, dtype=torch.float32)

            loss_batch = {
                "reference_features": x["reference_features"],
                "teacher_reference_features": x["reference_native"],
                "text_states": x["text_states"],
                "teacher_text_states": x["teacher_text_states"],
                "text_attention_mask": x["attention_mask"],
                "text_content_mask": x["content_mask"],
                "target_features": target_features,
                "target_ids": target_ids,
            }

            model.zero_grad(set_to_none=True)
            losses = model.compute_loss(loss_batch)
            loss = losses["retrieval_loss"]
            loss.backward()

            g = model.slot_queries.grad
            if g is None:
                raise RuntimeError("slot_queries.grad is None")
            if not torch.isfinite(g).all():
                raise FloatingPointError("slot_queries gradient contains NaN/Inf")

            mean_cos, matrix = pairwise_row_cosine(g.detach())
            pairwise_means.append(mean_cos)
            matrices.append(matrix.detach().cpu())
            grad_norms.append(g.detach().norm(dim=1).cpu())
            loss_values.append(float(loss.detach()))

            used += 1

    model.zero_grad(set_to_none=True)

    if not matrices:
        return {"skipped": True, "reason": "no batches"}

    matrix_mean = torch.stack(matrices).mean(dim=0)
    norms = torch.stack(grad_norms)

    return {
        "skipped": False,
        "num_batches": used,
        "retrieval_loss_mean": sum(loss_values) / len(loss_values),
        "pairwise_gradient_cosine_mean": sum(pairwise_means) / len(pairwise_means),
        "pairwise_gradient_cosine_per_batch": pairwise_means,
        "mean_cosine_matrix": matrix_mean.tolist(),
        "slot_gradient_norm_mean": norms.mean(dim=0).tolist(),
        "slot_gradient_norm_min": norms.min(dim=0).values.tolist(),
        "slot_gradient_norm_max": norms.max(dim=0).values.tolist(),
    }


def print_retrieval_table(retrieval):
    print("\n" + "=" * 88)
    print("RETRIEVAL FORENSICS")
    print("=" * 88)
    print(f"{'variant':<22} {'R@10':>10} {'R@50':>10} {'mean':>10}")
    print("-" * 58)
    for name, m in retrieval.items():
        print(
            f"{name:<22} "
            f"{m['recall_at_10']:>10.3f} "
            f"{m['recall_at_50']:>10.3f} "
            f"{m['mean_recall']:>10.3f}"
        )


def print_depth_summary(retrieval, num_slots):
    ref = retrieval["reference_only"]["mean_recall"]
    full = retrieval["full"]["mean_recall"]
    denom = full - ref

    print("\n" + "=" * 88)
    print("RECURRENT-DEPTH SHORTCUT SUMMARY")
    print("=" * 88)
    print(f"reference_only mean = {ref:.3f}")
    print(f"full mean           = {full:.3f}")
    print(f"full modification gain = {denom:.3f}")
    print()
    print(f"{'variant':<22} {'mean':>10} {'recovered_mod_gain':>20}")
    print("-" * 56)

    names = (
        [f"mean_x{k}" for k in range(1, num_slots + 1)]
        + [f"slot{i}_x{num_slots}" for i in range(num_slots)]
    )
    for name in names:
        if name not in retrieval:
            continue
        mean = retrieval[name]["mean_recall"]
        frac = float("nan") if abs(denom) < 1e-12 else (mean - ref) / denom
        print(f"{name:<22} {mean:>10.3f} {100.0 * frac:>19.1f}%")


def print_full_vs_mean_cap_summary(retrieval, num_slots):
    ref = retrieval["reference_only"]["mean_recall"]
    full = retrieval["full"]["mean_recall"]
    denom = full - ref

    print("\n" + "=" * 88)
    print("FULL-SLOT INFORMATION vs PURE DEPTH")
    print("=" * 88)
    print(
        f"{'k':>3} {'FULL_capK':>12} {'mean_xK':>12} "
        f"{'delta(full-mean)':>18} {'FULL gain%':>12} {'mean gain%':>12}"
    )
    print("-" * 76)

    for k in range(1, num_slots + 1):
        fk = retrieval[f"full_cap{k}"]["mean_recall"]
        mk = retrieval[f"mean_x{k}"]["mean_recall"]
        if abs(denom) < 1e-12:
            fg = mg = float("nan")
        else:
            fg = 100.0 * (fk - ref) / denom
            mg = 100.0 * (mk - ref) / denom
        print(
            f"{k:>3d} {fk:>12.3f} {mk:>12.3f} "
            f"{fk - mk:>18.3f} {fg:>11.1f}% {mg:>11.1f}%"
        )

    print()
    print(
        "Interpretation: if FULL_capK ~= mean_xK for the same K, then the "
        "slot-specific identities/information add little beyond the number of "
        "Executor refinement steps. A positive FULL_capK - mean_xK gap measures "
        "the residual benefit of using the real distinct slots at that depth."
    )


def main():
    args = parse_args()
    runtime = load_runtime(args)

    print("Device:", runtime["device"])
    print("Checkpoint:", args.checkpoint)
    print("Queries/category:", args.max_queries_per_category or "FULL")
    print("Gradient batches:", args.gradient_batches)

    inference = run_inference_forensics(args, runtime)
    print_retrieval_table(inference["retrieval"])
    print_depth_summary(inference["retrieval"], runtime["model"].num_slots)
    print_full_vs_mean_cap_summary(inference["retrieval"], runtime["model"].num_slots)

    print("\nSynthetic execution-step sanity checks:")
    print(json.dumps(inference["execution_controls"], indent=2))

    print("\nOwnership logit margin summaries:")
    print(json.dumps(inference["ownership_margins"], indent=2))

    print("\nRunning slot-query gradient cosine audit...")
    gradient = run_gradient_forensics(args, runtime)
    print(json.dumps(gradient, indent=2))

    report = {
        "checkpoint": str(args.checkpoint),
        "protocol": args.protocol,
        "max_queries_per_category": args.max_queries_per_category,
        "gradient_batches": args.gradient_batches,
        "inference": inference,
        "gradient": gradient,
        "interpretation_guide": {
            "keep_one": (
                "If every keep_i remains close to FULL, each slot likely carries most of the global modification."
            ),
            "full_cap_vs_mean": (
                "Compare FULL_capK against mean_xK at the same K. If they are close, real slot-specific information contributes little beyond recurrent execution depth; the gap FULL_capK-mean_xK estimates residual complementary slot information."
            ),
            "mean_depth_curve": (
                "If mean_x1 < mean_x2 < mean_x3 < mean_x4 and mean_x4 is near FULL, retrieval benefit scales with repeated execution depth even when slot identity is destroyed."
            ),
            "duplicate_single_slot_x4": (
                "If slot0_x4..slot3_x4 are each near FULL, any single original slot already contains a near-global edit program and four copies mainly provide recurrent compute tickets."
            ),
            "hardize_current": (
                "If hardize_current collapses retrieval, the trained A3 checkpoint materially relies on fractional soft ownership tails."
            ),
            "edit_only_margin": (
                "If edit-only top1-top2 logit margins are near zero, deterministic A4 argmax will be highly tie-sensitive/arbitrary."
            ),
            "gradient_cosine": (
                "If slot-query gradient cosine is consistently near +1, retrieval loss is actively pushing slot queries in similar directions."
            ),
        },
    }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report: {args.json_output}")


if __name__ == "__main__":
    main()