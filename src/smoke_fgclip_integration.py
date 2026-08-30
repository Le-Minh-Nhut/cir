from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from PIL import Image
from torch.optim import SGD

from losses.objective import IAGSRMEObjective, ObjectiveConfig
from models.iag_srme import FGCLIPBackbone, FGCLIPRegime, IAGSRME, IAGSRMEConfig, IAGSRMECore
from training.engine import assert_training_setup, trainable_parameters


def _parameter_change(parameter: torch.nn.Parameter, before: torch.Tensor) -> float:
    return float((parameter.detach() - before).abs().max())


def _error_statistics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = (actual.detach().float() - expected.detach().float()).abs()
    relative = difference.norm() / expected.detach().float().norm().clamp_min(1e-12)
    return {
        "max_absolute": float(difference.max()),
        "mean_absolute": float(difference.mean()),
        "relative_l2": float(relative),
    }


def _gradient_statistics(
    actual: torch.Tensor, expected: torch.Tensor
) -> dict[str, float]:
    result = _error_statistics(actual, expected)
    result["official_norm"] = float(expected.norm())
    result["manual_norm"] = float(actual.norm())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Real pinned FG-CLIP/IAG-SRME smoke update")
    parser.add_argument("--checkpoint", default="qihoo360/fg-clip-base")
    parser.add_argument(
        "--revision", default="454d76372c2cf5eb48fa0d871fd0534481484d97"
    )
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--r1b", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(20260829)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    regime = FGCLIPRegime(
        checkpoint=args.checkpoint,
        revision=args.revision,
        train_vision=True,
        train_text=True,
        train_text_projection=False,
    )
    backbone = FGCLIPBackbone.from_pretrained(regime, internal_width=256)
    tokenizer, processor = FGCLIPBackbone.load_processor(
        regime.checkpoint, regime.revision, regime.trust_remote_code
    )
    core = IAGSRMECore(
        IAGSRMEConfig(
            width=256,
            num_candidates=4,
            max_steps=args.max_steps,
            num_heads=8,
            retrieval_dim=backbone.retrieval_dim,
            selector_gumbel_noise=False,
            query_cap=1000.0 if args.r1b else 0.5,
            enable_dynamic_applicability=args.r1b,
            initial_applicability=0.98,
        )
    )
    model = IAGSRME(backbone, core).to(device).eval()
    objective = IAGSRMEObjective(ObjectiveConfig(), width=256).to(device).train()
    optimizer = SGD(trainable_parameters(model, objective), lr=1e-3)
    assert_training_setup(model, objective, optimizer, device)

    arrays = [
        np.full((224, 224, 3), (48, 96, 160), dtype=np.uint8),
        np.full((224, 224, 3), (176, 72, 40), dtype=np.uint8),
        np.full((224, 224, 3), (32, 152, 88), dtype=np.uint8),
        np.full((224, 224, 3), (144, 128, 56), dtype=np.uint8),
    ]
    images = [Image.fromarray(array) for array in arrays]
    reference_pixels = processor(images=images[:2], return_tensors="pt")["pixel_values"].to(
        device
    )
    target_pixels = processor(images=images[2:], return_tensors="pt")["pixel_values"].to(
        device
    )
    tokenized = tokenizer(
        ["make it red and add long sleeves", "make it green and remove the collar"],
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    content_mask = attention_mask.bool()

    parity_parameters = {
        "vision_model": next(model.backbone.model.vision_model.parameters()),
        "visual_projection": model.backbone.model.visual_projection.weight,
        "anchor_projection": model.backbone.anchor_projection[0].weight,
    }
    official_dense = model.backbone.model.get_image_dense_features(
        pixel_values=reference_pixels[:1]
    )
    official_global = model.backbone.model.get_image_features(
        pixel_values=reference_pixels[:1]
    )
    official_anchor = model.backbone.anchor_projection(official_dense)
    official_parity_objective = (
        official_dense.square().mean()
        + official_global.square().mean()
        + official_anchor.square().mean()
    )
    official_gradients = torch.autograd.grad(
        official_parity_objective, tuple(parity_parameters.values())
    )
    vision_outputs = model.backbone.model.vision_model(
        pixel_values=reference_pixels[:1], output_hidden_states=True, return_dict=True
    )
    manual_dense, manual_global = model.backbone.reference_features_from_vision_outputs(
        vision_outputs
    )
    manual_anchor = model.backbone.anchor_projection(manual_dense)
    manual_parity_objective = (
        manual_dense.square().mean()
        + manual_global.square().mean()
        + manual_anchor.square().mean()
    )
    manual_gradients = torch.autograd.grad(
        manual_parity_objective, tuple(parity_parameters.values())
    )
    if not torch.allclose(manual_dense, official_dense, atol=1e-6, rtol=1e-5):
        raise AssertionError("manual dense features differ from official FG-CLIP helper")
    if not torch.allclose(manual_global, official_global, atol=1e-6, rtol=1e-5):
        raise AssertionError("manual global features differ from official FG-CLIP helper")
    gradient_parity = {}
    for (name, _), manual_gradient, official_gradient in zip(
        parity_parameters.items(), manual_gradients, official_gradients, strict=True
    ):
        gradient_parity[name] = _gradient_statistics(manual_gradient, official_gradient)
        if not torch.allclose(manual_gradient, official_gradient, atol=5e-5, rtol=1e-4):
            raise AssertionError(f"manual {name} gradient differs from official helpers")
        if manual_gradient.abs().sum() == 0:
            raise AssertionError(f"manual {name} gradient is zero")

    with torch.no_grad():
        official_text = model.backbone.model.text_model(
            input_ids=input_ids[:1],
            attention_mask=attention_mask[:1],
            return_dict=True,
            walk_short_pos=True,
        )
        official_text_projected = model.backbone.model.text_projection(
            official_text.pooler_output
        )

    model.train()
    tracked = {
        "intent_query_bank": model.core.intent_encoder.query_bank,
        "grounding_projection": model.core.grounder.intent_projection.weight,
        "editor_direction": model.core.editor.direction.weight,
        "readout_projection": model.core.readout.output_projection.weight,
        "score_head": model.core.scorer.score_head[-1].weight,
        "text_encoder": next(model.backbone.model.text_model.parameters()),
        "iag_text_projection": model.backbone.text_projection[0].weight,
        "vision_encoder": next(model.backbone.model.vision_model.parameters()),
        "visual_projection": model.backbone.model.visual_projection.weight,
        "anchor_projection": model.backbone.anchor_projection[0].weight,
    }
    if args.r1b:
        tracked["applicability_weight"] = (
            model.core.applicability_head.projection.weight
        )
        tracked["applicability_bias"] = model.core.applicability_head.projection.bias
    before = {name: parameter.detach().clone() for name, parameter in tracked.items()}
    real_calls = {
        "vision_model": 0,
        "anchor_projection": 0,
        "intent_encoder": 0,
        "grounder": 0,
        "applicability": 0,
    }

    def count_real_call(name):
        def hook(_module, _inputs):
            real_calls[name] += 1

        return hook

    vision_handle = model.backbone.model.vision_model.register_forward_pre_hook(
        count_real_call("vision_model")
    )
    anchor_handle = model.backbone.anchor_projection.register_forward_pre_hook(
        count_real_call("anchor_projection")
    )
    intent_handle = model.core.intent_encoder.register_forward_pre_hook(
        count_real_call("intent_encoder")
    )
    grounder_handle = model.core.grounder.register_forward_pre_hook(
        count_real_call("grounder")
    )
    applicability_handle = (
        model.core.applicability_head.register_forward_pre_hook(
            count_real_call("applicability")
        )
        if model.core.applicability_head is not None
        else None
    )
    optimizer.zero_grad(set_to_none=True)
    encoded = model.backbone(reference_pixels, input_ids, attention_mask, content_mask)
    output = model.core(encoded)
    with torch.no_grad():
        projected_intents = model.core.grounder.intent_projection(output.intents)
        real_grounding_logits = (
            torch.einsum(
                "bkd,bnd->bkn",
                projected_intents,
                model.core.grounder.anchor_projection(output.anchor),
            )
            / model.core.grounder.scale
        )
    target_embeddings = model.encode_global_images(target_pixels)
    training_call_counts = dict(real_calls)
    if training_call_counts != {
        "vision_model": 2,
        "anchor_projection": 1,
        "intent_encoder": 1,
        "grounder": 1,
        "applicability": args.max_steps if args.r1b else 0,
    }:
        raise AssertionError(f"unexpected real training call counts: {training_call_counts}")
    positives = torch.eye(2, dtype=torch.bool, device=device)
    losses = objective(output, target_embeddings, positives)
    losses["total"].backward()
    gradient_norms = {
        name: 0.0 if parameter.grad is None else float(parameter.grad.norm())
        for name, parameter in tracked.items()
    }
    optimizer.step()
    real_calls = {
        "vision_model": 0,
        "anchor_projection": 0,
        "intent_encoder": 0,
        "grounder": 0,
        "applicability": 0,
    }
    with torch.no_grad():
        model.encode_global_images(target_pixels[:1])
    gallery_call_counts = dict(real_calls)
    vision_handle.remove()
    anchor_handle.remove()
    intent_handle.remove()
    grounder_handle.remove()
    if applicability_handle is not None:
        applicability_handle.remove()
    if gallery_call_counts != {
        "vision_model": 1,
        "anchor_projection": 0,
        "intent_encoder": 0,
        "grounder": 0,
        "applicability": 0,
    }:
        raise AssertionError(f"unexpected real gallery call counts: {gallery_call_counts}")
    parameter_deltas = {
        name: _parameter_change(parameter, before[name]) for name, parameter in tracked.items()
    }
    finite_tensors = all(
        torch.isfinite(tensor).all()
        for tensor in (
            encoded.anchor,
            encoded.reference_global,
            encoded.text_tokens,
            output.final_query,
            target_embeddings,
            losses["terminal"],
            losses["marginal"],
        )
    )
    if not finite_tensors:
        raise AssertionError("real FG-CLIP smoke produced non-finite values")
    if not all(value > 0 for value in gradient_norms.values()):
        raise AssertionError(f"expected nonzero real gradients: {gradient_norms}")
    required_parameter_deltas = {
        name: value
        for name, value in parameter_deltas.items()
        if name != "applicability_bias"
    }
    if not all(value > 0 for value in required_parameter_deltas.values()):
        raise AssertionError(f"expected real parameter changes: {parameter_deltas}")
    if args.r1b and max(
        parameter_deltas["applicability_weight"],
        parameter_deltas["applicability_bias"],
    ) == 0:
        raise AssertionError("dynamic applicability head did not update")
    model.eval()
    with torch.no_grad():
        dynamic_output = model.core(encoded, control="repeat_candidate_1")
    dynamic_changes: dict[str, list[float]] = {
        "current_evidence": [],
        "accumulated_local_change": [],
        "contexts": [],
        "delta_z": [],
        "scores": [],
    }
    for previous, current in zip(
        dynamic_output.trace[:-1], dynamic_output.trace[1:], strict=True
    ):
        pairs = {
            "current_evidence": (previous.current_evidence, current.current_evidence),
            "accumulated_local_change": (
                previous.accumulated_local_change,
                current.accumulated_local_change,
            ),
            "contexts": (previous.contexts, current.contexts),
            "delta_z": (previous.delta_z, current.delta_z),
            "scores": (previous.scores, current.scores),
        }
        for name, (left, right) in pairs.items():
            dynamic_changes[name].append(float((right - left).abs().max()))
    if args.max_steps > 1 and not all(
        all(value > 0 for value in values) for values in dynamic_changes.values()
    ):
        raise AssertionError(f"expected state-conditioned recurrence dynamics: {dynamic_changes}")
    print(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "revision": args.revision,
                "device": str(device),
                "official_shapes": {
                    "get_image_dense_features": list(official_dense.shape),
                    "get_image_features": list(official_global.shape),
                    "text_model_last_hidden_state": list(official_text.last_hidden_state.shape),
                    "text_projection": list(official_text_projected.shape),
                },
                "reference_value_parity": {
                    "dense": _error_statistics(manual_dense, official_dense),
                    "global": _error_statistics(manual_global, official_global),
                    "normalized_global": _error_statistics(
                        torch.nn.functional.normalize(manual_global.float(), dim=-1),
                        torch.nn.functional.normalize(official_global.float(), dim=-1),
                    ),
                },
                "reference_gradient_parity": gradient_parity,
                "real_vision_call_counts": {
                    "training_reference_plus_target": training_call_counts,
                    "gallery_global_only": gallery_call_counts,
                },
                "iag_srme_shapes": {
                    "anchor": list(encoded.anchor.shape),
                    "text_tokens": list(encoded.text_tokens.shape),
                    "intents": list(output.intents.shape),
                    "supports": list(output.supports.shape),
                    "candidate_states": list(output.trace[0].candidate_states.shape),
                    "candidate_queries": list(output.trace[0].candidate_queries.shape),
                    "scores": list(output.trace[0].scores.shape),
                    "final_query": list(output.final_query.shape),
                    "target_global": list(target_embeddings.shape),
                    "per_step": [
                        {
                            "delta_z": list(step.delta_z.shape),
                            "candidate_queries": list(step.candidate_queries.shape),
                            "scores": list(step.scores.shape),
                        }
                        for step in output.trace
                    ],
                },
                "r1b_dynamic_applicability": {
                    "enabled": model.core.config.enable_dynamic_applicability,
                    "query_cap": model.core.config.query_cap,
                    "mean_p_null": (
                        None
                        if output.visual_null_probabilities is None
                        else float(
                            output.visual_null_probabilities.detach().float().mean()
                        )
                    ),
                    "p_null_standard_deviation": (
                        None
                        if output.visual_null_probabilities is None
                        else float(
                            output.visual_null_probabilities.detach()
                            .float()
                            .std(unbiased=False)
                        )
                    ),
                    "p_null_minimum": (
                        None
                        if output.visual_null_probabilities is None
                        else float(output.visual_null_probabilities.detach().float().min())
                    ),
                    "p_null_maximum": (
                        None
                        if output.visual_null_probabilities is None
                        else float(output.visual_null_probabilities.detach().float().max())
                    ),
                    "mean_confidence_by_timestep": (
                        None
                        if output.visual_confidence is None
                        else output.visual_confidence.detach()
                        .float()
                        .mean(dim=(0, 2))
                        .tolist()
                    ),
                    "spatial_mass_invariant_max_abs_error": float(
                        (output.supports.detach().float().sum(dim=-1) - 1.0)
                        .abs()
                        .max()
                    ),
                    "dtype_contract": {
                        "context": str(output.trace[0].contexts.dtype),
                        "applicability_logits": (
                            None
                            if output.trace[0].applicability_logits is None
                            else str(output.trace[0].applicability_logits.dtype)
                        ),
                        "confidence": (
                            None
                            if output.trace[0].visual_confidence is None
                            else str(output.trace[0].visual_confidence.dtype)
                        ),
                        "p_null": (
                            None
                            if output.trace[0].visual_null_probability is None
                            else str(output.trace[0].visual_null_probability.dtype)
                        ),
                        "delta_z": str(output.trace[0].delta_z.dtype),
                        "candidate_state": str(
                            output.trace[0].candidate_states.dtype
                        ),
                    },
                    "initial_real_visual_logit_statistics": {
                        "mean": float(real_grounding_logits.mean()),
                        "standard_deviation": float(real_grounding_logits.std()),
                        "minimum": float(real_grounding_logits.min()),
                        "maximum": float(real_grounding_logits.max()),
                        "initial_applicability": (
                            model.core.config.initial_applicability
                        ),
                    },
                },
                "three_step_contract": {
                    "intent_encoder_calls": training_call_counts["intent_encoder"],
                    "grounder_calls": training_call_counts["grounder"],
                    "applicability_calls": training_call_counts["applicability"],
                    "dynamic_max_abs_changes": dynamic_changes,
                },
                "losses": {
                    "terminal": float(losses["terminal"].detach()),
                    "marginal": float(losses["marginal"].detach()),
                    "total": float(losses["total"].detach()),
                },
                "gradient_norms": gradient_norms,
                "text_trainability_core": {
                    "fgclip_text_model": {
                        "requires_grad": next(
                            model.backbone.model.text_model.parameters()
                        ).requires_grad,
                        "gradient_norm": gradient_norms["text_encoder"],
                    },
                    "fgclip_text_projection": {
                        "requires_grad": model.backbone.model.text_projection.weight.requires_grad,
                        "gradient_norm": 0.0,
                    },
                    "iag_text_projection": {
                        "requires_grad": model.backbone.text_projection[0].weight.requires_grad,
                        "gradient_norm": gradient_norms["iag_text_projection"],
                    },
                },
                "parameter_max_abs_deltas": parameter_deltas,
                "finite": finite_tensors,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
