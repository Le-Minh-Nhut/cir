from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from cache.features import (
    get_features_by_ids,
    get_text_features_by_sample_ids,
    load_features,
    load_text_features,
)
from evaluation.fashioniq import (
    build_fashioniq_gallery,
    evaluate_fashioniq_category,
    macro_average_fashioniq,
)
from models.taper import TAPER
from runtime import (
    configure_torch_runtime,
    resolve_device,
    seed_everything,
)
from teachers.csmcir_compose import CSMCIRComposeTeacher
from train import (
    build_val_loaders,
    load_fashioniq_correction_dicts,
)


# ---------------------------------------------------------------------
# Small metric accumulator
# ---------------------------------------------------------------------


class Meter:
    def __init__(self):
        self.sums = defaultdict(float)
        self.weights = defaultdict(float)

    def add(self, name: str, value, weight: float = 1.0):
        if isinstance(value, torch.Tensor):
            value = value.detach().float().item()

        value = float(value)

        if not torch.isfinite(torch.tensor(value)):
            return

        self.sums[name] += value * weight
        self.weights[name] += weight

    def result(self):
        out = {}

        for name in self.sums:
            if self.weights[name] > 0:
                out[name] = self.sums[name] / self.weights[name]

        return out


# ---------------------------------------------------------------------
# Pairwise cosine
# ---------------------------------------------------------------------


def pairwise_cosine_mean(
    x: torch.Tensor,
    active: torch.Tensor | None = None,
):
    """
    x: [B,L,D]

    active:
        optional [B,L] bool mask.

    Returns:
        mean cosine,
        number of valid pairs
    """

    b, l, _ = x.shape

    if l < 2:
        return float("nan"), 0

    x = F.normalize(
        x.float(),
        dim=-1,
        eps=1e-6,
    )

    similarity = x @ x.transpose(1, 2)

    upper = torch.triu(
        torch.ones(
            l,
            l,
            dtype=torch.bool,
            device=x.device,
        ),
        diagonal=1,
    )

    values = similarity[:, upper]

    if active is None:
        return values.mean().item(), values.numel()

    pair_active = (
        active[:, :, None]
        & active[:, None, :]
    )[:, upper]

    if not pair_active.any():
        return float("nan"), 0

    selected = values[pair_active]

    return selected.mean().item(), selected.numel()


# ---------------------------------------------------------------------
# One execution -> retrieval query
# ---------------------------------------------------------------------


def execute_query(
    model,
    edit_slots,
    slot_gates,
    z0,
    reference_state,
    disabled_slots=None,
):
    execution = model.execute(
        edit_slots,
        slot_gates,
        z0,
        reference_state,
        disabled_slots=disabled_slots,
    )

    return model.make_query(
        execution["final_state"]
    )


# ---------------------------------------------------------------------
# Build all forensic retrieval variants
# ---------------------------------------------------------------------


def build_query_variants(
    model,
    output,
):
    edit_slots = output["edit_slots"]
    slot_gates = output["slot_gates"]
    z0 = output["z0"]
    reference_state = output["reference_state"]

    b, l, _ = edit_slots.shape
    device = edit_slots.device

    variants = {
        "FULL": output["q0"],
        "REFERENCE_ONLY": output["q_reference_only"],
    }

    # ---------------------------------------------------------------
    # DROP each slot
    # ---------------------------------------------------------------

    for slot_id in range(l):
        disabled = torch.zeros(
            b,
            l,
            dtype=torch.bool,
            device=device,
        )

        disabled[:, slot_id] = True

        variants[f"DROP_S{slot_id}"] = execute_query(
            model,
            edit_slots,
            slot_gates,
            z0,
            reference_state,
            disabled_slots=disabled,
        )

    # ---------------------------------------------------------------
    # KEEP exactly one slot
    # ---------------------------------------------------------------

    for slot_id in range(l):
        disabled = torch.ones(
            b,
            l,
            dtype=torch.bool,
            device=device,
        )

        disabled[:, slot_id] = False

        variants[f"KEEP_S{slot_id}"] = execute_query(
            model,
            edit_slots,
            slot_gates,
            z0,
            reference_state,
            disabled_slots=disabled,
        )

    # ---------------------------------------------------------------
    # Repeat each original slot x1 ... xL
    #
    # This directly tests the A3 "slot-as-compute-ticket" disease.
    # ---------------------------------------------------------------

    for slot_id in range(l):

        repeated_slot = (
            edit_slots[:, slot_id : slot_id + 1, :]
            .expand(-1, l, -1)
            .contiguous()
        )

        repeated_gate = (
            slot_gates[:, slot_id : slot_id + 1]
            .expand(-1, l)
            .contiguous()
        )

        for repeat_count in range(1, l + 1):

            disabled = torch.ones(
                b,
                l,
                dtype=torch.bool,
                device=device,
            )

            disabled[:, :repeat_count] = False

            variants[
                f"REPEAT_S{slot_id}_X{repeat_count}"
            ] = execute_query(
                model,
                repeated_slot,
                repeated_gate,
                z0,
                reference_state,
                disabled_slots=disabled,
            )

    # ---------------------------------------------------------------
    # Mean-slot repeated x1 ... xL
    # ---------------------------------------------------------------

    mean_slot = edit_slots.mean(
        dim=1,
        keepdim=True,
    )

    mean_gate = slot_gates.mean(
        dim=1,
        keepdim=True,
    )

    repeated_mean_slot = (
        mean_slot
        .expand(-1, l, -1)
        .contiguous()
    )

    repeated_mean_gate = (
        mean_gate
        .expand(-1, l)
        .contiguous()
    )

    for repeat_count in range(1, l + 1):

        disabled = torch.ones(
            b,
            l,
            dtype=torch.bool,
            device=device,
        )

        disabled[:, :repeat_count] = False

        variants[
            f"MEAN_SLOT_X{repeat_count}"
        ] = execute_query(
            model,
            repeated_mean_slot,
            repeated_mean_gate,
            z0,
            reference_state,
            disabled_slots=disabled,
        )

    return variants


# ---------------------------------------------------------------------
# A5 structural diagnosis
# ---------------------------------------------------------------------


def collect_a5_diagnostics(
    meter: Meter,
    output: dict[str, torch.Tensor],
    content_mask: torch.Tensor,
):
    states = output["refine_slot_states"]
    masks = output["refine_slot_masks"]
    nulls = output["refine_null_probs"]
    logits = output["refine_ownership_logits"]
    masses = output["refine_slot_masses"]
    update_norms = output["refine_update_norms"]

    b, t, l, _ = states.shape

    valid = content_mask.to(torch.bool)
    valid_count = valid.sum().item()

    # ===============================================================
    # Per refinement round
    # ===============================================================

    for round_id in range(t):

        round_states = states[:, round_id]
        round_masks = masks[:, round_id]
        round_null = nulls[:, round_id]
        round_logits = logits[:, round_id]
        round_mass = masses[:, round_id]

        # -----------------------------------------------------------
        # Slot-state geometry
        # -----------------------------------------------------------

        state_cos, state_pairs = pairwise_cosine_mean(
            round_states
        )

        meter.add(
            f"round_{round_id}/slot_state_pair_cos",
            state_cos,
            max(state_pairs, 1),
        )

        meter.add(
            f"round_{round_id}/slot_state_norm",
            round_states.norm(dim=-1).mean(),
            b,
        )

        # -----------------------------------------------------------
        # Ownership support geometry
        # -----------------------------------------------------------

        valid_masks = (
            round_masks
            * valid[:, None, :].to(round_masks.dtype)
        )

        mask_cos, mask_pairs = pairwise_cosine_mean(
            valid_masks
        )

        meter.add(
            f"round_{round_id}/slot_mask_pair_cos",
            mask_cos,
            max(mask_pairs, 1),
        )

        # -----------------------------------------------------------
        # Per-slot mass
        # -----------------------------------------------------------

        for slot_id in range(l):
            meter.add(
                f"round_{round_id}/slot_{slot_id}_mass",
                round_mass[:, slot_id].mean(),
                b,
            )

        # -----------------------------------------------------------
        # Joint NULL + Edit ownership
        # -----------------------------------------------------------

        all_probs = torch.cat(
            [
                round_null[:, None, :],
                round_masks,
            ],
            dim=1,
        )

        valid_f = valid.to(all_probs.dtype)

        entropy = -(
            all_probs.clamp_min(1e-12)
            * all_probs.clamp_min(1e-12).log()
        ).sum(dim=1)

        if valid_count > 0:
            meter.add(
                f"round_{round_id}/assignment_entropy",
                (entropy * valid_f).sum() / valid_f.sum(),
                valid_count,
            )

        top2 = all_probs.topk(
            k=2,
            dim=1,
        ).values

        winner_confidence = top2[:, 0]
        margin = top2[:, 0] - top2[:, 1]

        if valid_count > 0:
            meter.add(
                f"round_{round_id}/winner_confidence",
                (
                    winner_confidence * valid_f
                ).sum()
                / valid_f.sum(),
                valid_count,
            )

            meter.add(
                f"round_{round_id}/top1_top2_margin",
                (
                    margin * valid_f
                ).sum()
                / valid_f.sum(),
                valid_count,
            )

        # -----------------------------------------------------------
        # NULL rate
        # -----------------------------------------------------------

        if valid_count > 0:
            meter.add(
                f"round_{round_id}/null_mass",
                (
                    round_null * valid_f
                ).sum()
                / valid_f.sum(),
                valid_count,
            )

        # -----------------------------------------------------------
        # Logit scale
        # -----------------------------------------------------------

        valid_logits = (
            valid[:, None, :]
            .expand_as(round_logits)
        )

        selected_logits = round_logits[
            valid_logits
        ]

        if selected_logits.numel() > 1:
            meter.add(
                f"round_{round_id}/ownership_logit_std",
                selected_logits.float().std(),
                selected_logits.numel(),
            )

    # ===============================================================
    # GRU update magnitude
    # ===============================================================

    for transition_id in range(update_norms.shape[1]):

        transition = update_norms[
            :,
            transition_id,
        ]

        meter.add(
            f"update_{transition_id}_to_{transition_id + 1}/mean_norm",
            transition.mean(),
            b,
        )

        for slot_id in range(l):
            meter.add(
                f"update_{transition_id}_to_{transition_id + 1}/slot_{slot_id}_norm",
                transition[:, slot_id].mean(),
                b,
            )

    # ===============================================================
    # Final factor geometry
    # ===============================================================

    slot_mass = output["slot_mass"]

    # Match the threshold already used in current TAPER diagnostics.
    active = slot_mass >= 0.10

    semantic_cos_all, count = pairwise_cosine_mean(
        output["slot_semantics"]
    )

    meter.add(
        "final/semantic_pair_cos_all",
        semantic_cos_all,
        max(count, 1),
    )

    semantic_cos_active, count = pairwise_cosine_mean(
        output["slot_semantics"],
        active=active,
    )

    meter.add(
        "final/semantic_pair_cos_active",
        semantic_cos_active,
        max(count, 1),
    )

    effect_cos_all, count = pairwise_cosine_mean(
        output["slot_effects"]
    )

    meter.add(
        "final/effect_pair_cos_all",
        effect_cos_all,
        max(count, 1),
    )

    effect_cos_active, count = pairwise_cosine_mean(
        output["slot_effects"],
        active=active,
    )

    meter.add(
        "final/effect_pair_cos_active",
        effect_cos_active,
        max(count, 1),
    )

    edit_cos_all, count = pairwise_cosine_mean(
        output["edit_slots"]
    )

    meter.add(
        "final/edit_slot_pair_cos_all",
        edit_cos_all,
        max(count, 1),
    )

    edit_cos_active, count = pairwise_cosine_mean(
        output["edit_slots"],
        active=active,
    )

    meter.add(
        "final/edit_slot_pair_cos_active",
        edit_cos_active,
        max(count, 1),
    )

    # ---------------------------------------------------------------
    # Balanced/global packing indicators
    # ---------------------------------------------------------------

    total_edit_mass = slot_mass.sum(
        dim=1
    ).clamp_min(1e-12)

    dominant_share = (
        slot_mass.max(dim=1).values
        / total_edit_mass
    )

    meter.add(
        "final/dominant_slot_share",
        dominant_share.mean(),
        b,
    )

    meter.add(
        "final/active_slot_count",
        active.float().sum(dim=1).mean(),
        b,
    )

    # ---------------------------------------------------------------
    # Per-slot gate / hard-active / execution
    # ---------------------------------------------------------------

    hard_active = output["hard_active_slot_mask"]

    for slot_id in range(l):

        meter.add(
            f"final/slot_{slot_id}_gate",
            output["slot_gates"][:, slot_id].mean(),
            b,
        )

        meter.add(
            f"final/slot_{slot_id}_hard_active_rate",
            hard_active[:, slot_id].float().mean(),
            b,
        )

    # ===============================================================
    # Executor diagnosis
    # ===============================================================

    valid_steps = output["trace_valid_mask"]

    meter.add(
        "execution/valid_steps_per_sample",
        valid_steps.float().sum(dim=1).mean(),
        b,
    )

    trace_slots = output["trace_slot_ids"]
    trace_primitives = output["trace_primitive_ids"]

    for slot_id in range(l):

        executed = (
            trace_slots == slot_id
        ).any(dim=1)

        meter.add(
            f"execution/slot_{slot_id}_execution_rate",
            executed.float().mean(),
            b,
        )

    total_valid_steps = valid_steps.sum().item()

    if total_valid_steps > 0:

        confidence = output["route_confidences"][
            valid_steps
        ]

        meter.add(
            "execution/route_confidence",
            confidence.mean(),
            total_valid_steps,
        )

        strengths = output["transition_strengths"][
            valid_steps
        ]

        meter.add(
            "execution/transition_strength",
            strengths.mean(),
            total_valid_steps,
        )

        changes = output[
            "actual_state_changes"
        ].norm(dim=-1)[valid_steps]

        meter.add(
            "execution/actual_state_change_norm",
            changes.mean(),
            total_valid_steps,
        )

        for primitive_id in range(
            model_num_primitives(output, trace_primitives)
        ):
            fraction = (
                trace_primitives[valid_steps]
                == primitive_id
            ).float().mean()

            meter.add(
                f"execution/primitive_{primitive_id}_fraction",
                fraction,
                total_valid_steps,
            )


def model_num_primitives(
    output,
    trace_primitives,
):
    """
    Infer K safely from selected IDs.

    Main caller later overwrites this behavior with model.num_primitives
    through report completion if some primitive is never selected.
    """
    valid = trace_primitives >= 0

    if not valid.any():
        return 0

    return int(
        trace_primitives[valid].max().item()
    ) + 1


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="config",
)
def main(cfg: DictConfig):

    if str(
        cfg.experiment.get("name", "")
    ) != "taper_e2e":
        raise ValueError(
            "Run with experiment=taper_e2e"
        )

    if "checkpoint" not in cfg:
        raise ValueError(
            "Provide +checkpoint=/path/to/best.pt"
        )

    checkpoint_path = Path(
        str(cfg.checkpoint)
    )

    report_path = Path(
        str(
            cfg.get(
                "report",
                "reports/a5_0_forensic.json",
            )
        )
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

    print("Device:", device)
    print("Checkpoint:", checkpoint_path)

    # ===============================================================
    # Data / cached features
    # ===============================================================

    dataset_root = Path(cfg.dataset.root)
    annotation_root = dataset_root / "captions"
    split_root = dataset_root / "image_splits"

    correction_dicts = (
        load_fashioniq_correction_dicts(
            annotation_root
        )
    )

    val_loaders, val_annotations = (
        build_val_loaders(
            annotation_root=annotation_root,
            batch_size=cfg.experiment.eval_batch_size,
            num_workers=cfg.experiment.num_workers,
            caption_policy=cfg.experiment.val_caption_policy,
            correction_dicts=correction_dicts,
        )
    )

    cache_root = Path(
        cfg.paths.cache_root
    )

    val_retrieval, val_retrieval_idx = (
        load_features(
            cache_root
            / "fashioniq"
            / "csmcir"
            / "val"
            / "retrieval"
        )
    )

    val_native, val_native_idx = (
        load_features(
            cache_root
            / "fashioniq"
            / "csmcir"
            / "val"
            / "native"
        )
    )

    val_text = load_text_features(
        cache_root
        / "fashioniq"
        / "csmcir"
        / "val"
        / "text"
    )

    # ===============================================================
    # Model
    # ===============================================================

    teacher = CSMCIRComposeTeacher(
        csmcir_root=cfg.experiment.teacher.csmcir_root,
        checkpoint_path=cfg.experiment.teacher.checkpoint_path,
    ).to(device).eval()

    m = cfg.experiment.model

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
    ).to(device)

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    incompatible = model.load_state_dict(
        state_dict,
        strict=False,
    )

    bad_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("teacher.")
    ]

    if bad_missing:
        raise RuntimeError(
            f"Unexpected missing keys: {bad_missing}"
        )

    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected checkpoint keys: "
            f"{incompatible.unexpected_keys}"
        )

    model.eval()

    meter = Meter()

    category_results = defaultdict(dict)

    # ===============================================================
    # Validation pass
    # ===============================================================

    with torch.no_grad():

        for category, val_loader in val_loaders.items():

            print(
                f"\nDiagnosing category: {category}"
            )

            annotations = val_annotations[
                category
            ]

            gallery_ids = build_fashioniq_gallery(
                protocol=cfg.protocol.name,
                split_root=split_root,
                split="val",
                category=category,
                annotations=annotations,
            )

            gallery_features = get_features_by_ids(
                gallery_ids,
                val_retrieval,
                val_retrieval_idx,
            ).to(
                device=device,
                dtype=torch.float32,
            )

            variant_scores = defaultdict(list)
            target_ids = []

            for batch in val_loader:

                reference_native = get_features_by_ids(
                    batch.reference_ids,
                    val_native,
                    val_native_idx,
                ).to(
                    device=device,
                    dtype=torch.float32,
                )

                reference_features = (
                    reference_native[:, 0, :]
                )

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

                text_states = text_states.to(
                    device=device,
                    dtype=torch.float32,
                )

                teacher_text_states = (
                    teacher_text_states.to(
                        device=device,
                        dtype=torch.float32,
                    )
                )

                attention_mask = attention_mask.to(
                    device=device,
                    dtype=torch.bool,
                )

                content_mask = content_mask.to(
                    device=device,
                    dtype=torch.bool,
                )

                # ---------------------------------------------------
                # ONE expensive slot/teacher forward.
                # Every forensic execution variant reuses these slots.
                # ---------------------------------------------------

                output = model.forward(
                    reference_features,
                    text_states,
                    attention_mask,
                    text_content_mask=content_mask,
                    teacher_reference_features=reference_native,
                    teacher_text_states=teacher_text_states,
                )

                collect_a5_diagnostics(
                    meter,
                    output,
                    content_mask,
                )

                variants = build_query_variants(
                    model,
                    output,
                )

                for name, query in variants.items():

                    scores = model._retrieval_scores(
                        query,
                        gallery_features,
                    )

                    variant_scores[name].append(
                        scores.cpu()
                    )

                for target_id in batch.target_ids:

                    if target_id is None:
                        raise ValueError(
                            "Validation target_id missing"
                        )

                    target_ids.append(
                        target_id
                    )

            # -------------------------------------------------------
            # Category recall for every forensic variant
            # -------------------------------------------------------

            for name, score_batches in (
                variant_scores.items()
            ):

                scores = torch.cat(
                    score_batches,
                    dim=0,
                )

                category_results[name][
                    category
                ] = evaluate_fashioniq_category(
                    scores=scores,
                    target_ids=target_ids,
                    gallery_ids=gallery_ids,
                )

    # ===============================================================
    # Retrieval report
    # ===============================================================

    retrieval_report = {}

    for name, results in category_results.items():

        macro = macro_average_fashioniq(
            results
        )

        retrieval_report[name] = {
            **macro,
            "categories": results,
        }

    # ===============================================================
    # Derived forensic ratios
    # ===============================================================

    derived = {}

    full = retrieval_report[
        "FULL"
    ]["mean_recall"]

    reference_only = retrieval_report[
        "REFERENCE_ONLY"
    ]["mean_recall"]

    modification_gain = (
        full - reference_only
    )

    derived[
        "full_mean_recall"
    ] = full

    derived[
        "reference_only_mean_recall"
    ] = reference_only

    derived[
        "modification_gain"
    ] = modification_gain

    if abs(modification_gain) > 1e-8:

        for name, result in (
            retrieval_report.items()
        ):

            if (
                name.startswith("KEEP_")
                or name.startswith("REPEAT_")
                or name.startswith("MEAN_SLOT_")
            ):

                derived[
                    f"{name}/gain_fraction"
                ] = (
                    result["mean_recall"]
                    - reference_only
                ) / modification_gain

    # ===============================================================
    # Final report
    # ===============================================================

    report = {
        "checkpoint": str(checkpoint_path),
        "config": {
            "num_refine_iters": int(
                m.num_refine_iters
            ),
            "num_slots": int(
                m.num_slots
            ),
            "num_primitives": int(
                m.num_primitives
            ),
            "gate_mode": str(
                m.gate_mode
            ),
            "st_gate_recovery": bool(
                m.st_gate_recovery
            ),
        },
        "structural": meter.result(),
        "retrieval": retrieval_report,
        "derived": derived,
    }

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ===============================================================
    # Important console summary
    # ===============================================================

    structural = report["structural"]

    print("\n" + "=" * 80)
    print("A5.0 FORENSIC SUMMARY")
    print("=" * 80)

    print(
        f"FULL               = {full:.4f}"
    )
    print(
        f"REFERENCE_ONLY     = {reference_only:.4f}"
    )
    print(
        f"MODIFICATION_GAIN  = {modification_gain:.4f}"
    )

    print("\n--- Refinement state cosine ---")

    for round_id in range(
        int(m.num_refine_iters)
    ):
        key = (
            f"round_{round_id}/"
            "slot_state_pair_cos"
        )

        print(
            f"round {round_id}: "
            f"{structural.get(key, float('nan')):.6f}"
        )

    print("\n--- Refinement mask cosine ---")

    for round_id in range(
        int(m.num_refine_iters)
    ):
        key = (
            f"round_{round_id}/"
            "slot_mask_pair_cos"
        )

        print(
            f"round {round_id}: "
            f"{structural.get(key, float('nan')):.6f}"
        )

    print("\n--- Final factor cosine ---")

    for key in [
        "final/semantic_pair_cos_active",
        "final/effect_pair_cos_active",
        "final/edit_slot_pair_cos_active",
    ]:
        print(
            f"{key}: "
            f"{structural.get(key, float('nan')):.6f}"
        )

    print("\n--- Execution ---")

    print(
        "valid steps/sample:",
        structural.get(
            "execution/valid_steps_per_sample",
            float("nan"),
        ),
    )

    print("\n--- KEEP one slot ---")

    for slot_id in range(
        int(m.num_slots)
    ):

        name = f"KEEP_S{slot_id}"

        mr = retrieval_report[
            name
        ]["mean_recall"]

        fraction = derived.get(
            f"{name}/gain_fraction",
            float("nan"),
        )

        print(
            f"{name}: "
            f"MR={mr:.4f} "
            f"gain_fraction={fraction:.4f}"
        )

    print("\n--- Repeat each slot x4 ---")

    for slot_id in range(
        int(m.num_slots)
    ):

        name = (
            f"REPEAT_S{slot_id}_"
            f"X{int(m.num_slots)}"
        )

        mr = retrieval_report[
            name
        ]["mean_recall"]

        fraction = derived.get(
            f"{name}/gain_fraction",
            float("nan"),
        )

        print(
            f"{name}: "
            f"MR={mr:.4f} "
            f"gain_fraction={fraction:.4f}"
        )

    print("\n--- Mean slot depth curve ---")

    for repeat_count in range(
        1,
        int(m.num_slots) + 1,
    ):

        name = (
            f"MEAN_SLOT_X"
            f"{repeat_count}"
        )

        mr = retrieval_report[
            name
        ]["mean_recall"]

        fraction = derived.get(
            f"{name}/gain_fraction",
            float("nan"),
        )

        print(
            f"{name}: "
            f"MR={mr:.4f} "
            f"gain_fraction={fraction:.4f}"
        )

    print(
        f"\nSaved forensic report → {report_path}"
    )


if __name__ == "__main__":
    main()