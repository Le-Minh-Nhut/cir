from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from cache.features import (
    get_features_by_ids,
    get_text_features_by_sample_ids,
    load_features,
    load_text_features,
)
from datasets.common import collate_cir_samples
from datasets.fashioniq import (
    FashionIQDataset,
    load_correction_dict,
)
from models.taper import TAPER
from teachers.csmcir_compose import CSMCIRComposeTeacher


CATEGORIES = ("dress", "shirt", "toptee")
CAPTION_POLICY = "normalized_ordered_and"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/FashionIQ"),
    )

    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("features/fashioniq/csmcir"),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("conf/experiment/taper_e2e.yaml"),
    )

    parser.add_argument(
        "--csmcir-root",
        type=Path,
        default=Path("teacher/repos/CSMCIR"),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "teacher/checkpoints/csmcir/"
            "fashioniq_tuned_clip_best.pt"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    return parser.parse_args()


def load_corrections(annotation_root: Path):
    return {
        category: load_correction_dict(
            annotation_root
            / f"correction_dict_{category}.json"
        )
        for category in CATEGORIES
    }


def prepare_real_batch(
    *,
    dataset,
    batch_size,
    native_features,
    native_idx,
    retrieval_features,
    retrieval_idx,
    text_cache,
    device,
):
    samples = [
        dataset[i]
        for i in range(batch_size)
    ]

    batch = collate_cir_samples(samples)

    target_ids = list(batch.target_ids)

    if any(
        target_id is None
        for target_id in target_ids
    ):
        raise ValueError(
            "Parity batch contains missing target_id"
        )

    reference_native = get_features_by_ids(
        batch.reference_ids,
        native_features,
        native_idx,
    ).to(
        device=device,
        dtype=torch.float32,
    )

    target_features = get_features_by_ids(
        target_ids,
        retrieval_features,
        retrieval_idx,
    ).to(
        device=device,
        dtype=torch.float32,
    )

    (
        text_states,
        teacher_text_states,
        attention_mask,
        content_mask,
    ) = get_text_features_by_sample_ids(
        batch.sample_ids,
        batch.modification_texts,
        text_cache,
    )

    return {
        "reference_features":
            reference_native[:, 0, :],

        "teacher_reference_features":
            reference_native,

        "target_features":
            target_features,

        "text_states":
            text_states.to(
                device=device,
                dtype=torch.float32,
            ),

        "teacher_text_states":
            teacher_text_states.to(
                device=device,
                dtype=torch.float32,
            ),

        "text_attention_mask":
            attention_mask.to(
                device=device,
                dtype=torch.bool,
            ),

        "text_content_mask":
            content_mask.to(
                device=device,
                dtype=torch.bool,
            ),

        "target_ids":
            target_ids,
    }


def build_model(
    *,
    teacher,
    config,
    device,
):
    m = config.model

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
        hard_slot_gating_during_training=
            m.hard_slot_gating_during_training,
        gate_mode=m.gate_mode,
        st_gate_recovery=m.st_gate_recovery,
        alpha_max=m.alpha_max,
        counterfactual_chunk_size=
            m.counterfactual_chunk_size,
    )

    return model.to(device)


def snapshot_grads(model):
    names = (
        "slot_queries",
        "null_query",
        "slot_query_projection.weight",
        "text_key_projection.weight",
        "slot_mlp.0.weight",
    )

    params = dict(
        model.named_parameters()
    )

    result = {}

    for name in names:
        parameter = params[name]

        if parameter.grad is None:
            raise AssertionError(
                f"No gradient for {name}"
            )

        result[name] = (
            parameter.grad
            .detach()
            .float()
            .cpu()
            .clone()
        )

    return result


def run_once(
    *,
    model,
    batch,
    chunk_size,
):
    model.zero_grad(set_to_none=True)

    model.counterfactual_chunk_size = (
        chunk_size
    )

    output = model.forward(
        batch["reference_features"],
        batch["text_states"],
        batch["text_attention_mask"],
        text_content_mask=
            batch["text_content_mask"],
        teacher_reference_features=
            batch["teacher_reference_features"],
        teacher_text_states=
            batch["teacher_text_states"],
    )

    loss = model._retrieval_loss(
        output["q0"],
        batch["target_features"],
        batch["target_ids"],
    )

    if not torch.isfinite(loss):
        raise AssertionError(
            f"Non-finite loss at chunk={chunk_size}"
        )

    loss.backward()

    grads = snapshot_grads(model)

    return {
        "q_minus":
            output["q_teacher_minus"]
            .detach()
            .float()
            .cpu(),

        "slot_effects":
            output["slot_effects"]
            .detach()
            .float()
            .cpu(),

        "q0":
            output["q0"]
            .detach()
            .float()
            .cpu(),

        "loss":
            float(loss.detach().cpu()),

        "grads":
            grads,
    }


def max_abs_diff(a, b):
    return (
        a - b
    ).abs().max().item()


def relative_l2(a, b):
    numerator = (
        a - b
    ).norm()

    denominator = (
        a.norm()
        .clamp_min(1e-12)
    )

    return (
        numerator / denominator
    ).item()


def assert_tensor_close(
    *,
    name,
    candidate,
    reference,
    atol,
    rtol,
):
    abs_diff = max_abs_diff(
        candidate,
        reference,
    )

    rel_diff = relative_l2(
        candidate,
        reference,
    )

    print(
        f"{name:<40} "
        f"max_abs={abs_diff:.8e} "
        f"rel_l2={rel_diff:.8e}"
    )

    if not torch.allclose(
        candidate,
        reference,
        atol=atol,
        rtol=rtol,
    ):
        raise AssertionError(
            f"Parity failed for {name}"
        )


def main():
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError(
            "batch-size must be >= 1"
        )

    device = torch.device(args.device)

    torch.manual_seed(12345)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(12345)
        torch.backends.cudnn.benchmark = False

    config = OmegaConf.load(args.config)

    annotation_root = (
        args.dataset_root / "captions"
    )

    corrections = load_corrections(
        annotation_root
    )

    dataset = FashionIQDataset(
        annotation_root=annotation_root,
        split="train",
        categories=CATEGORIES,
        caption_policy=CAPTION_POLICY,
        correction_dicts=corrections,
        seed=42,
    )

    if len(dataset) < args.batch_size:
        raise ValueError(
            "Dataset smaller than requested batch"
        )

    native_features, native_idx = (
        load_features(
            args.cache_root
            / "train"
            / "native"
        )
    )

    retrieval_features, retrieval_idx = (
        load_features(
            args.cache_root
            / "train"
            / "retrieval"
        )
    )

    text_cache = load_text_features(
        args.cache_root
        / "train"
        / "text"
    )

    batch = prepare_real_batch(
        dataset=dataset,
        batch_size=args.batch_size,
        native_features=native_features,
        native_idx=native_idx,
        retrieval_features=retrieval_features,
        retrieval_idx=retrieval_idx,
        text_cache=text_cache,
        device=device,
    )

    print("Loading compose-only teacher...")

    teacher = CSMCIRComposeTeacher(
        csmcir_root=args.csmcir_root,
        checkpoint_path=args.checkpoint,
    ).to(device).eval()

    model = build_model(
        teacher=teacher,
        config=config,
        device=device,
    )

    # Important:
    # TAPER is in training mode, matching the real
    # training execution semantics.
    model.train()

    full_chunk = (
        args.batch_size
        * model.num_slots
    )

    chunk_sizes = []

    for chunk in (
        full_chunk,
        16,
        1,
    ):
        chunk = min(
            chunk,
            full_chunk,
        )

        if chunk not in chunk_sizes:
            chunk_sizes.append(chunk)

    print()
    print(
        "Counterfactual total:",
        full_chunk,
    )

    print(
        "Testing chunk sizes:",
        chunk_sizes,
    )

    print()

    results = {}

    for chunk_size in chunk_sizes:
        print(
            f"Running chunk_size={chunk_size}..."
        )

        results[chunk_size] = run_once(
            model=model,
            batch=batch,
            chunk_size=chunk_size,
        )

        result = results[chunk_size]

        print(
            "  loss:",
            f"{result['loss']:.10f}",
        )

        for name, grad in (
            result["grads"].items()
        ):
            print(
                f"  grad {name:<30} "
                f"norm={grad.norm().item():.8e}"
            )

        print()

    reference_chunk = full_chunk
    reference = results[
        reference_chunk
    ]

    print("=" * 80)
    print(
        f"REFERENCE: chunk_size={reference_chunk}"
    )
    print("=" * 80)

    # CUDA GEMMs/QFormer can differ slightly when
    # effective mini-batch size changes.
    output_atol = 2e-5
    output_rtol = 2e-5

    grad_atol = 5e-5
    grad_rtol = 5e-5

    for chunk_size in chunk_sizes:
        if chunk_size == reference_chunk:
            continue

        candidate = results[
            chunk_size
        ]

        print()
        print(
            f"Compare chunk={chunk_size} "
            f"vs full={reference_chunk}"
        )

        assert_tensor_close(
            name="q_minus",
            candidate=candidate["q_minus"],
            reference=reference["q_minus"],
            atol=output_atol,
            rtol=output_rtol,
        )

        assert_tensor_close(
            name="slot_effects",
            candidate=candidate[
                "slot_effects"
            ],
            reference=reference[
                "slot_effects"
            ],
            atol=output_atol,
            rtol=output_rtol,
        )

        assert_tensor_close(
            name="q0",
            candidate=candidate["q0"],
            reference=reference["q0"],
            atol=output_atol,
            rtol=output_rtol,
        )

        loss_diff = abs(
            candidate["loss"]
            - reference["loss"]
        )

        print(
            f"{'retrieval_loss':<40} "
            f"abs_diff={loss_diff:.8e}"
        )

        if loss_diff > 2e-5:
            raise AssertionError(
                "Retrieval-loss parity failed"
            )

        for name in reference["grads"]:
            assert_tensor_close(
                name=f"grad/{name}",
                candidate=
                    candidate["grads"][name],
                reference=
                    reference["grads"][name],
                atol=grad_atol,
                rtol=grad_rtol,
            )

    print()
    print("=" * 80)
    print("TAPER COUNTERFACTUAL CHUNK PARITY: PASS")
    print("=" * 80)

    print(
        "Forward, retrieval loss, and "
        "Competitive-NULL gradients agree "
        "across chunk sizes."
    )


if __name__ == "__main__":
    main()