from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from diagnostics.iag_srme import summarize_trajectory
from losses.objective import IAGSRMEObjective, ObjectiveConfig
from models.iag_srme import BackboneOutput, IAGSRMEConfig, IAGSRMECore


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic IAG-SRME end-to-end smoke test")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--r1b", action="store_true")
    parser.add_argument("--r1c1", action="store_true")
    args = parser.parse_args()
    if args.r1b and args.r1c1:
        raise ValueError("R1b applicability and R1c1 dynamic grounding are separate experiments")
    torch.manual_seed(20260829)
    batch, tokens, length, width, retrieval_dim = 4, 17, 9, 32, 24
    core = IAGSRMECore(
        IAGSRMEConfig(
            width=width,
            num_heads=4,
            retrieval_dim=retrieval_dim,
            max_steps=3,
            selector_gumbel_noise=False,
            query_cap=1000.0 if (args.r1b or args.r1c1) else 0.5,
            enable_dynamic_applicability=args.r1b,
            initial_applicability=0.98,
            enable_dynamic_regrounding=args.r1c1,
        )
    )
    with torch.no_grad():
        core.scorer.score_head[-1].weight.zero_()
        core.scorer.score_head[-1].bias.fill_(0.5)
        if args.r1b:
            # Exercise continuous FP32 actuator resolution rather than only the exact
            # zero-weight initialization point.
            core.applicability_head.projection.weight.normal_(std=1e-3)
    encoded = BackboneOutput(
        anchor=torch.randn(batch, tokens, width, requires_grad=True),
        reference_global=F.normalize(torch.randn(batch, retrieval_dim), dim=-1).requires_grad_(),
        text_tokens=torch.randn(batch, length, width, requires_grad=True),
        text_global=torch.randn(batch, width, requires_grad=True),
        text_semantic_global=F.normalize(
            torch.randn(batch, retrieval_dim), dim=-1
        ).requires_grad_(),
        text_content_mask=torch.ones(batch, length, dtype=torch.bool),
    )
    target_embeddings = F.normalize(torch.randn(batch, retrieval_dim), dim=-1).requires_grad_()
    positive_mask = torch.eye(batch, dtype=torch.bool)

    core.eval()
    before = core(encoded)
    permuted_target = target_embeddings[torch.tensor([2, 0, 3, 1])]
    after = core(encoded)
    firewall = all(
        torch.equal(left, right)
        for left, right in (
            (before.intents, after.intents),
            (before.supports, after.supports),
            (before.trace[0].contexts, after.trace[0].contexts),
            (before.trace[0].delta_z, after.trace[0].delta_z),
            (before.trace[0].candidate_queries, after.trace[0].candidate_queries),
            (before.trace[0].scores, after.trace[0].scores),
        )
    )
    assert permuted_target.shape == target_embeddings.shape and firewall

    core.train()
    output = core(encoded)
    objective = IAGSRMEObjective(ObjectiveConfig(), width=width)
    losses = objective(output, target_embeddings, positive_mask)
    losses["total"].backward()
    gradient_checks = {
        "intent_query_bank": core.intent_encoder.query_bank.grad,
        "grounding_projection": core.grounder.intent_projection.weight.grad,
        "editor_direction": core.editor.direction.weight.grad,
        "readout_projection": core.readout.output_projection.weight.grad,
        "score_head": core.scorer.score_head[-1].weight.grad,
        "anchor_input": encoded.anchor.grad,
        "text_input": encoded.text_tokens.grad,
    }
    if args.r1b:
        gradient_checks["applicability_weight"] = (
            core.applicability_head.projection.weight.grad
        )
        gradient_checks["applicability_bias"] = (
            core.applicability_head.projection.bias.grad
        )
    gradient_norms = {
        name: 0.0 if gradient is None else float(gradient.norm())
        for name, gradient in gradient_checks.items()
    }
    if not all(
        value > 0 and torch.isfinite(torch.tensor(value)) for value in gradient_norms.values()
    ):
        raise AssertionError(f"expected nonzero finite gradients: {gradient_norms}")
    report: dict[str, object] = {
        "shapes": {
            "E": list(output.intents.shape),
            "P": list(output.supports.shape),
            "DeltaZ": list(output.trace[0].delta_z.shape),
            "Z_hat": list(output.trace[0].candidate_states.shape),
            "q_hat": list(output.trace[0].candidate_queries.shape),
            "scores": list(output.trace[0].scores.shape),
            "final_query": list(output.final_query.shape),
        },
        "losses": {name: float(value.detach()) for name, value in losses.items()},
        "gradient_norms": gradient_norms,
        "target_firewall": firewall,
        "finite": bool(torch.isfinite(losses["total"]).item()),
        "selected_actions": [step.selected_index.tolist() for step in output.trace],
        "model_config": {
            "query_cap": core.config.query_cap,
            "enable_dynamic_applicability": core.config.enable_dynamic_applicability,
            "initial_applicability": core.config.initial_applicability,
            "grounding_normalization": core.config.grounding_normalization,
            "enable_dynamic_regrounding": core.config.enable_dynamic_regrounding,
        },
        "visual_null": (
            None
            if output.visual_null_probabilities is None
            else {
                "mean": float(output.visual_null_probabilities.detach().mean()),
                "standard_deviation": float(
                    output.visual_null_probabilities.detach().std(unbiased=False)
                ),
                "maximum": float(output.visual_null_probabilities.detach().max()),
                "minimum": float(output.visual_null_probabilities.detach().min()),
                "per_timestep_mean": output.visual_null_probabilities.detach()
                .mean(dim=(0, 2))
                .tolist(),
                "spatial_mass_error": float(
                    (output.supports.detach().sum(dim=-1) - 1.0).abs().max()
                ),
                "dtype_contract": {
                    "context": str(output.trace[0].contexts.dtype),
                    "applicability_logits": str(
                        output.trace[0].applicability_logits.dtype
                    ),
                    "confidence": str(output.trace[0].visual_confidence.dtype),
                    "p_null": str(output.trace[0].visual_null_probability.dtype),
                    "delta_z": str(output.trace[0].delta_z.dtype),
                    "candidate_state": str(output.trace[0].candidate_states.dtype),
                },
            }
        ),
    }
    temporal_supports = output.temporal_supports.detach().float()
    temporal_cosine = []
    temporal_l1 = []
    for previous, current in zip(
        temporal_supports[:, :-1].unbind(dim=1),
        temporal_supports[:, 1:].unbind(dim=1),
        strict=True,
    ):
        temporal_cosine.append(
            float(F.cosine_similarity(previous, current, dim=-1).mean())
        )
        temporal_l1.append(float((current - previous).abs().sum(dim=-1).mean()))
    static_t0 = core.grounder(output.intents, output.anchor)
    report["dynamic_grounding"] = {
        "enabled": output.dynamic_regrounding,
        "temporal_support_shape": list(temporal_supports.shape),
        "support_mass_max_abs_error_by_timestep": [
            float((temporal_supports[:, timestep].sum(dim=-1) - 1.0).abs().max())
            for timestep in range(temporal_supports.shape[1])
        ],
        "temporal_support_cosine": temporal_cosine,
        "temporal_support_l1_change": temporal_l1,
        "t0_static_anchor_parity_max_abs_error": float(
            (temporal_supports[:, 0] - static_t0.detach().float()).abs().max()
        ),
        "support_dtype": str(output.temporal_supports.dtype),
        "state_dtype": str(output.final_state.dtype),
        "same_parent_exact": all(
            torch.equal(
                step.candidate_states,
                step.current_state[:, None] + step.delta_z,
            )
            for step in output.trace
        ),
        "applicability_disabled": core.applicability_head is None,
    }
    if args.diagnostics:
        diagnostics = summarize_trajectory(output)
        report["diagnostics"] = {
            name: float(value.detach().float().mean()) for name, value in diagnostics.items()
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
