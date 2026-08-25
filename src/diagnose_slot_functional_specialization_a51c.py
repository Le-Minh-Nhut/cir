
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor

from cache.features import (
    get_features_by_ids,
    get_text_features_by_sample_ids,
    load_features,
    load_text_features,
)
from evaluation.fashioniq import build_fashioniq_gallery, macro_average_fashioniq
from models.taper import TAPER
from runtime import configure_torch_runtime, resolve_device, seed_everything
from teachers.csmcir_compose import CSMCIRComposeTeacher
from train import build_val_loaders, load_fashioniq_correction_dicts


"""
A5.1c FUNCTIONAL SPECIALIZATION TEST
====================================

Question:
    Did S0..S3 learn genuinely different retrieval functions,
    or are they mostly interchangeable copies of one global edit?

This is a frozen causal diagnosis. No training.

The test uses FOUR independent kinds of evidence:

A. Necessity / sufficiency
   - DROP_Si: does FULL need slot i?
   - KEEP_Si: can slot i alone recover modification gain?
   - all 2^4 subsets -> exact Shapley contribution.

B. Replication
   - REPEAT_Si_X4: can one slot replace all four execution tickets?

C. Substitution
   - REPLACE_Sdst_WITH_Ssrc while keeping destination gate.
   - If Sj can replace Si with little loss, their payload functions are
     interchangeable.

D. FUNCTIONAL FINGERPRINTS (most directly answers "same function?")
   For every validation query, each intervention produces a score vector
   across the *same gallery*.

   Single-slot fingerprint:
       f_i = score(KEEP_Si) - score(REFERENCE_ONLY)

   Contextual unique fingerprint:
       u_i = score(FULL) - score(DROP_Si)

   We compare slots by cosine similarity of these gallery re-ranking vectors:
       cos(f_i, f_j)
       cos(u_i, u_j)

   Why this is stronger than slot-vector cosine:
       edit_slots can be geometrically different while causing the same
       candidate ranking behavior. Here we compare what they actually DO.

Interpretation:
    High fingerprint similarity + successful substitution + successful
    single-slot replication = strong evidence of shared/redundant function.

    Low fingerprint similarity alone is NOT enough. Functional specialization
    requires unique causal value: drop/substitution/replication should also
    show that one slot cannot simply replace another.
"""


NUM_SLOTS = 4
EPS = 1e-8


class RecallAccumulator:
    def __init__(self, gallery_ids: Sequence[str]) -> None:
        if len(gallery_ids) != len(set(gallery_ids)):
            raise ValueError("gallery_ids must be unique")
        self.gallery_index = {image_id: i for i, image_id in enumerate(gallery_ids)}
        self.n = 0
        self.h10 = 0
        self.h50 = 0

    def update(self, scores: Tensor, target_ids: Sequence[str | None]) -> None:
        if scores.ndim != 2:
            raise ValueError("scores must be [B,G]")
        if scores.shape[0] != len(target_ids):
            raise ValueError("scores/target batch mismatch")

        target_idx = []
        for target_id in target_ids:
            if target_id is None:
                raise ValueError("Missing validation target id")
            target_idx.append(self.gallery_index[target_id])

        k = min(50, scores.shape[1])
        top = scores.topk(k=k, dim=1).indices
        target = torch.tensor(target_idx, device=top.device)[:, None]

        self.h10 += int(top[:, : min(10, k)].eq(target).any(dim=1).sum().item())
        self.h50 += int(top[:, : min(50, k)].eq(target).any(dim=1).sum().item())
        self.n += len(target_ids)

    def result(self) -> dict[str, float]:
        if self.n == 0:
            raise RuntimeError("No samples accumulated")
        return {
            "recall_at_10": 100.0 * self.h10 / self.n,
            "recall_at_50": 100.0 * self.h50 / self.n,
        }


class MeanAccumulator:
    def __init__(self) -> None:
        self.total = 0.0
        self.weight = 0

    def add(self, values: Tensor) -> None:
        values = values.detach().float()
        finite = torch.isfinite(values)
        if finite.any():
            self.total += float(values[finite].sum().item())
            self.weight += int(finite.sum().item())

    def result(self) -> float:
        return self.total / max(self.weight, 1)


def disabled_mask(
    b: int,
    enabled: Sequence[int],
    device: torch.device,
) -> Tensor:
    mask = torch.ones(b, NUM_SLOTS, dtype=torch.bool, device=device)
    for i in enabled:
        mask[:, i] = False
    return mask


def execute_query(
    model: TAPER,
    edit_slots: Tensor,
    slot_gates: Tensor,
    z0: Tensor,
    reference_state: Tensor,
    *,
    enabled: Sequence[int] | None = None,
) -> Tensor:
    b = edit_slots.shape[0]
    disabled = None
    if enabled is not None:
        disabled = disabled_mask(b, enabled, edit_slots.device)

    execution = model.execute(
        edit_slots,
        slot_gates,
        z0,
        reference_state,
        disabled_slots=disabled,
    )
    return model.make_query(execution["final_state"])


def repeat_slot(
    edit_slots: Tensor,
    slot_gates: Tensor,
    slot_id: int,
) -> tuple[Tensor, Tensor]:
    slots = edit_slots[:, slot_id : slot_id + 1].expand(-1, NUM_SLOTS, -1).contiguous()
    gates = slot_gates[:, slot_id : slot_id + 1].expand(-1, NUM_SLOTS).contiguous()
    return slots, gates


def build_variants(
    model: TAPER,
    output: dict[str, Tensor],
) -> dict[str, Tensor]:
    edit_slots = output["edit_slots"]
    slot_gates = output["slot_gates"]
    z0 = output["z0"]
    reference_state = output["reference_state"]

    if edit_slots.shape[1] != NUM_SLOTS:
        raise ValueError(f"Expected 4 slots, got {edit_slots.shape[1]}")

    variants: dict[str, Tensor] = {
        "FULL": output["q0"],
        "REFERENCE_ONLY": output["q_reference_only"],
    }

    # A. Every subset (16 total).
    for mask in range(1 << NUM_SLOTS):
        enabled = [i for i in range(NUM_SLOTS) if mask & (1 << i)]
        name = "SUBSET_NONE" if not enabled else "SUBSET_" + "_".join(f"S{i}" for i in enabled)
        variants[name] = execute_query(
            model, edit_slots, slot_gates, z0, reference_state, enabled=enabled
        )

    # Convenience aliases.
    for i in range(NUM_SLOTS):
        variants[f"KEEP_S{i}"] = variants[f"SUBSET_S{i}"]
        keep_without = [j for j in range(NUM_SLOTS) if j != i]
        variants[f"DROP_S{i}"] = variants[
            "SUBSET_" + "_".join(f"S{j}" for j in keep_without)
        ]

    # B. One slot cloned into all four execution tickets.
    for i in range(NUM_SLOTS):
        slots, gates = repeat_slot(edit_slots, slot_gates, i)
        variants[f"REPEAT_S{i}_X4"] = execute_query(
            model, slots, gates, z0, reference_state, enabled=[0, 1, 2, 3]
        )

    # Mean-slot x4 control.
    mean_slot = edit_slots.mean(dim=1, keepdim=True).expand(-1, NUM_SLOTS, -1).contiguous()
    mean_gate = slot_gates.mean(dim=1, keepdim=True).expand(-1, NUM_SLOTS).contiguous()
    variants["MEAN_SLOT_X4"] = execute_query(
        model, mean_slot, mean_gate, z0, reference_state, enabled=[0, 1, 2, 3]
    )

    # C. Every ordered payload replacement dst <- src, keeping destination gate.
    for dst in range(NUM_SLOTS):
        for src in range(NUM_SLOTS):
            if dst == src:
                continue
            replaced = edit_slots.clone()
            replaced[:, dst] = edit_slots[:, src]
            variants[f"REPLACE_S{dst}_WITH_S{src}"] = execute_query(
                model, replaced, slot_gates, z0, reference_state
            )

    return variants


def normalized_retrieval_scores(query: Tensor, gallery_norm: Tensor) -> Tensor:
    token_scores = torch.einsum("bd,nkd->bnk", query, gallery_norm)
    return token_scores.amax(dim=-1)


def cosine_rows(a: Tensor, b: Tensor) -> Tensor:
    """
    Per-sample cosine over the gallery dimension.
    a,b: [B,G]
    """
    an = a.norm(dim=1)
    bn = b.norm(dim=1)
    denom = an * bn

    cos = (a * b).sum(dim=1) / denom.clamp_min(EPS)

    # If BOTH effects are numerically zero, they are functionally identical
    # "do nothing" effects, so report cosine=1 for that sample.
    both_zero = (an < EPS) & (bn < EPS)
    one_zero = (an < EPS) ^ (bn < EPS)

    cos = torch.where(both_zero, torch.ones_like(cos), cos)
    cos = torch.where(one_zero, torch.zeros_like(cos), cos)

    return cos


def exact_shapley(subset_values: dict[int, float]) -> dict[str, float]:
    n = NUM_SLOTS
    n_fact = math.factorial(n)
    result = {}

    for i in range(n):
        phi = 0.0
        for mask in range(1 << n):
            if mask & (1 << i):
                continue
            s = mask.bit_count()
            weight = math.factorial(s) * math.factorial(n - s - 1) / n_fact
            phi += weight * (
                subset_values[mask | (1 << i)] - subset_values[mask]
            )
        result[f"S{i}"] = phi

    return result


def subset_variant_name(mask: int) -> str:
    enabled = [i for i in range(NUM_SLOTS) if mask & (1 << i)]
    if not enabled:
        return "SUBSET_NONE"
    return "SUBSET_" + "_".join(f"S{i}" for i in enabled)


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    if str(cfg.experiment.get("name", "")) != "taper_e2e":
        raise ValueError("Run with experiment=taper_e2e")
    if "checkpoint" not in cfg:
        raise ValueError("Provide +checkpoint=/path/to/best.pt")

    checkpoint_path = Path(str(cfg.checkpoint))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    report_path = Path(
        str(cfg.get("report", "reports/a5_1c_functional_specialization.json"))
    )

    seed_everything(
        seed=cfg.seed,
        deterministic=cfg.runtime.deterministic,
    )
    configure_torch_runtime(
        deterministic=cfg.runtime.deterministic,
        benchmark=cfg.runtime.benchmark,
    )
    device = resolve_device(
        device_name=cfg.runtime.device,
        accelerator_index=cfg.runtime.accelerator_index,
    )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    dataset_root = Path(cfg.dataset.root)
    annotation_root = dataset_root / "captions"
    split_root = dataset_root / "image_splits"

    correction_dicts = load_fashioniq_correction_dicts(annotation_root)
    val_loaders, val_annotations = build_val_loaders(
        annotation_root=annotation_root,
        batch_size=cfg.experiment.eval_batch_size,
        num_workers=cfg.experiment.num_workers,
        caption_policy=cfg.experiment.val_caption_policy,
        correction_dicts=correction_dicts,
    )

    cache_root = Path(cfg.paths.cache_root)
    val_retrieval, val_retrieval_idx = load_features(
        cache_root / "fashioniq" / "csmcir" / "val" / "retrieval"
    )
    val_native, val_native_idx = load_features(
        cache_root / "fashioniq" / "csmcir" / "val" / "native"
    )
    val_text = load_text_features(
        cache_root / "fashioniq" / "csmcir" / "val" / "text"
    )

    # ------------------------------------------------------------------
    # Exact A5.1c reconstruction
    # ------------------------------------------------------------------
    teacher = CSMCIRComposeTeacher(
        csmcir_root=cfg.experiment.teacher.csmcir_root,
        checkpoint_path=cfg.experiment.teacher.checkpoint_path,
    ).to(device).eval()

    m = cfg.experiment.model
    if int(m.num_slots) != NUM_SLOTS:
        raise ValueError(f"Expected num_slots=4, got {int(m.num_slots)}")

    model = TAPER(
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
        slot_gate_threshold=m.slot_gate_threshold,
        hard_slot_gating_during_training=m.hard_slot_gating_during_training,
        gate_mode=m.gate_mode,
        st_gate_recovery=m.st_gate_recovery,
        alpha_max=m.alpha_max,
        counterfactual_chunk_size=m.counterfactual_chunk_size,
        num_refine_iters=m.num_refine_iters,
        residual_bias_strength=m.residual_bias_strength,
        residual_depletion_power=m.residual_depletion_power,
        residual_eps=m.residual_eps,
        randomize_slot_order_during_training=m.randomize_slot_order_during_training,
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    incompatible = model.load_state_dict(state_dict, strict=False)

    bad_missing = [k for k in incompatible.missing_keys if not k.startswith("teacher.")]
    if bad_missing:
        raise RuntimeError(f"Unexpected missing keys: {bad_missing}")
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {incompatible.unexpected_keys}")

    model.eval()

    category_results: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)

    # D. Functional fingerprint accumulators.
    single_fp_cos = {
        (i, j): MeanAccumulator()
        for i in range(NUM_SLOTS)
        for j in range(i + 1, NUM_SLOTS)
    }
    unique_fp_cos = {
        (i, j): MeanAccumulator()
        for i in range(NUM_SLOTS)
        for j in range(i + 1, NUM_SLOTS)
    }
    single_fp_norm = {i: MeanAccumulator() for i in range(NUM_SLOTS)}
    unique_fp_norm = {i: MeanAccumulator() for i in range(NUM_SLOTS)}

    with torch.no_grad():
        for category, loader in val_loaders.items():
            print(f"\nFunctional specialization diagnosis: {category}")

            gallery_ids = build_fashioniq_gallery(
                protocol=cfg.protocol.name,
                split_root=split_root,
                split="val",
                category=category,
                annotations=val_annotations[category],
            )

            gallery = get_features_by_ids(
                gallery_ids, val_retrieval, val_retrieval_idx
            ).to(device=device, dtype=torch.float32)
            gallery_norm = F.normalize(gallery, dim=-1)

            recalls: dict[str, RecallAccumulator] = {}

            for batch_id, batch in enumerate(loader):
                reference_native = get_features_by_ids(
                    batch.reference_ids, val_native, val_native_idx
                ).to(device=device, dtype=torch.float32)
                reference_features = reference_native[:, 0]

                (
                    text_states,
                    teacher_text_states,
                    attention_mask,
                    content_mask,
                ) = get_text_features_by_sample_ids(
                    batch.sample_ids,
                    batch.modification_texts,
                    val_text,
                )

                text_states = text_states.to(device=device, dtype=torch.float32)
                teacher_text_states = teacher_text_states.to(
                    device=device, dtype=torch.float32
                )
                attention_mask = attention_mask.to(device=device, dtype=torch.bool)
                content_mask = content_mask.to(device=device, dtype=torch.bool)

                # ONE normal forward: slot creation is frozen.
                output = model.forward(
                    reference_features,
                    text_states,
                    attention_mask,
                    text_content_mask=content_mask,
                    teacher_reference_features=reference_native,
                    teacher_text_states=teacher_text_states,
                )

                variants = build_variants(model, output)

                # Score all causal variants.
                scores: dict[str, Tensor] = {}
                for name, query in variants.items():
                    s = normalized_retrieval_scores(query, gallery_norm)
                    scores[name] = s

                    if name not in recalls:
                        recalls[name] = RecallAccumulator(gallery_ids)
                    recalls[name].update(s, batch.target_ids)

                # ------------------------------------------------------
                # Functional fingerprint = actual gallery re-ranking.
                # ------------------------------------------------------
                ref_scores = scores["REFERENCE_ONLY"]
                full_scores = scores["FULL"]

                single_effect = {}
                unique_effect = {}

                for i in range(NUM_SLOTS):
                    # What slot i does by itself.
                    single_effect[i] = scores[f"KEEP_S{i}"] - ref_scores

                    # What is lost specifically when slot i is removed
                    # from the full system.
                    unique_effect[i] = full_scores - scores[f"DROP_S{i}"]

                    single_fp_norm[i].add(single_effect[i].norm(dim=1))
                    unique_fp_norm[i].add(unique_effect[i].norm(dim=1))

                for i in range(NUM_SLOTS):
                    for j in range(i + 1, NUM_SLOTS):
                        single_fp_cos[(i, j)].add(
                            cosine_rows(single_effect[i], single_effect[j])
                        )
                        unique_fp_cos[(i, j)].add(
                            cosine_rows(unique_effect[i], unique_effect[j])
                        )

                if batch_id == 0 or (batch_id + 1) % 10 == 0:
                    print(f"  batch={batch_id + 1:03d}")

            for name, acc in recalls.items():
                category_results[name][category] = acc.result()

    retrieval = {}
    for name, per_category in category_results.items():
        retrieval[name] = {
            **macro_average_fashioniq(per_category),
            "categories": per_category,
        }

    full = float(retrieval["FULL"]["mean_recall"])
    ref = float(retrieval["REFERENCE_ONLY"]["mean_recall"])
    mod_gain = full - ref

    # Exact subset Shapley.
    subset_values = {
        mask: float(retrieval[subset_variant_name(mask)]["mean_recall"])
        for mask in range(1 << NUM_SLOTS)
    }
    shapley = exact_shapley(subset_values)

    necessity = {}
    sufficiency = {}
    full_mask = (1 << NUM_SLOTS) - 1

    for i in range(NUM_SLOTS):
        necessity[f"S{i}"] = (
            subset_values[full_mask]
            - subset_values[full_mask & ~(1 << i)]
        )
        sufficiency[f"S{i}"] = (
            subset_values[1 << i]
            - subset_values[0]
        )

    fingerprint = {
        "single_slot_rerank_cosine": {
            f"S{i}_S{j}": single_fp_cos[(i, j)].result()
            for i in range(NUM_SLOTS)
            for j in range(i + 1, NUM_SLOTS)
        },
        "contextual_unique_rerank_cosine": {
            f"S{i}_S{j}": unique_fp_cos[(i, j)].result()
            for i in range(NUM_SLOTS)
            for j in range(i + 1, NUM_SLOTS)
        },
        "single_slot_effect_norm": {
            f"S{i}": single_fp_norm[i].result()
            for i in range(NUM_SLOTS)
        },
        "contextual_unique_effect_norm": {
            f"S{i}": unique_fp_norm[i].result()
            for i in range(NUM_SLOTS)
        },
    }

    replacement_matrix = {}
    for dst in range(NUM_SLOTS):
        row = {}
        for src in range(NUM_SLOTS):
            if dst == src:
                row[f"S{src}"] = full
            else:
                row[f"S{src}"] = float(
                    retrieval[f"REPLACE_S{dst}_WITH_S{src}"]["mean_recall"]
                )
        replacement_matrix[f"replace_destination_S{dst}_with"] = row

    repeat = {
        f"S{i}": {
            "mean_recall": float(retrieval[f"REPEAT_S{i}_X4"]["mean_recall"]),
            "delta_vs_full": float(
                retrieval[f"REPEAT_S{i}_X4"]["mean_recall"]
            ) - full,
            "gain_fraction": (
                (
                    float(retrieval[f"REPEAT_S{i}_X4"]["mean_recall"])
                    - ref
                )
                / mod_gain
                if abs(mod_gain) > 1e-12
                else float("nan")
            ),
        }
        for i in range(NUM_SLOTS)
    }

    report = {
        "experiment": "A5.1c functional specialization causal diagnosis",
        "checkpoint": str(checkpoint_path),
        "question": (
            "Did S0..S3 learn different retrieval functions, "
            "or mostly one shared/interchangeable function?"
        ),
        "baseline": {
            "full_mean_recall": full,
            "reference_only_mean_recall": ref,
            "modification_gain": mod_gain,
            "mean_slot_x4_mean_recall": float(
                retrieval["MEAN_SLOT_X4"]["mean_recall"]
            ),
        },
        "slot_necessity_drop_from_full_mr_points": necessity,
        "slot_sufficiency_gain_from_reference_mr_points": sufficiency,
        "exact_subset_shapley_mr_points": shapley,
        "repeat_one_slot_x4": repeat,
        "replacement_matrix_mean_recall": replacement_matrix,
        "functional_fingerprints": fingerprint,
        "retrieval": retrieval,
        "interpretation": {
            "strong_redundancy_pattern": (
                "High single-slot rerank cosine + repeat-one-slot x4 "
                "recovers FULL + replacements are cheap + drop necessity "
                "is small => slots are functionally interchangeable/redundant."
            ),
            "strong_specialization_pattern": (
                "Different rerank fingerprints AND asymmetric replacement "
                "costs AND nontrivial unique drop effects => evidence that "
                "slots perform different functions."
            ),
            "warning": (
                "Low slot-vector cosine or different attention maps alone "
                "do not establish functional specialization."
            ),
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("A5.1c FUNCTIONAL SPECIALIZATION SUMMARY")
    print("=" * 88)
    print(f"FULL             = {full:.4f}")
    print(f"REFERENCE_ONLY   = {ref:.4f}")
    print(f"MODIFICATION_GAIN= {mod_gain:.4f}")

    print("\n[1] Necessity / sufficiency / Shapley")
    for i in range(NUM_SLOTS):
        print(
            f"S{i}: necessity={necessity[f'S{i}']:+.4f} | "
            f"sufficiency={sufficiency[f'S{i}']:+.4f} | "
            f"Shapley={shapley[f'S{i}']:+.4f}"
        )

    print("\n[2] Repeat one slot x4")
    for i in range(NUM_SLOTS):
        x = repeat[f"S{i}"]
        print(
            f"S{i}x4: MR={x['mean_recall']:.4f} | "
            f"delta_full={x['delta_vs_full']:+.4f} | "
            f"gain_fraction={x['gain_fraction']:.4f}"
        )

    print("\n[3] Functional fingerprint cosine: KEEP_i - REFERENCE_ONLY")
    for i in range(NUM_SLOTS):
        for j in range(i + 1, NUM_SLOTS):
            print(
                f"S{i} vs S{j}: "
                f"{fingerprint['single_slot_rerank_cosine'][f'S{i}_S{j}']:.6f}"
            )

    print("\n[4] Contextual UNIQUE fingerprint cosine: FULL - DROP_i")
    for i in range(NUM_SLOTS):
        for j in range(i + 1, NUM_SLOTS):
            print(
                f"S{i} vs S{j}: "
                f"{fingerprint['contextual_unique_rerank_cosine'][f'S{i}_S{j}']:.6f}"
            )

    print("\n[5] Replacement matrix: row=destination, col=source, value=MR")
    header = "dst\\src | " + " | ".join(f"S{i:>7}" for i in range(NUM_SLOTS))
    print(header)
    print("-" * len(header))
    for dst in range(NUM_SLOTS):
        row = replacement_matrix[f"replace_destination_S{dst}_with"]
        values = " | ".join(f"{row[f'S{src}']:8.4f}" for src in range(NUM_SLOTS))
        print(f"S{dst:>6} | {values}")

    print(f"\nSaved -> {report_path}")


if __name__ == "__main__":
    main()