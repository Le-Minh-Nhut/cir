from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F
from torch.optim import SGD

from diagnostics.iag_srme import summarize_trajectory
from losses.objective import IAGSRMEObjective, ObjectiveConfig
from models.iag_srme import BackboneOutput, IAGSRMEConfig, IAGSRMECore


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic IAG-SRME end-to-end smoke test")
    parser.add_argument("--diagnostics", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--r1b", action="store_true")
    modes.add_argument("--r1c1", action="store_true")
    modes.add_argument("--r1c2", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(20260829)
    batch, tokens, length, width, retrieval_dim = 4, 17, 9, 32, 24
    core = IAGSRMECore(
        IAGSRMEConfig(
            width=width,
            num_heads=4,
            retrieval_dim=retrieval_dim,
            max_steps=3,
            selector_gumbel_noise=False,
            query_cap=(
                1000.0 if (args.r1b or args.r1c1 or args.r1c2) else 0.5
            ),
            enable_dynamic_applicability=args.r1b,
            initial_applicability=0.98,
            enable_dynamic_regrounding=args.r1c1 or args.r1c2,
            enable_dynamic_reproposal=args.r1c2,
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

    calls = {"intent": 0, "reproposal": 0, "grounder": 0, "applicability": 0}

    def count(name):
        def hook(_module, _inputs) -> None:
            calls[name] += 1

        return hook

    handles = [
        core.intent_encoder.register_forward_pre_hook(count("intent")),
        core.grounder.register_forward_pre_hook(count("grounder")),
    ]
    if core.reproposal is not None:
        handles.append(core.reproposal.register_forward_pre_hook(count("reproposal")))
    if core.applicability_head is not None:
        handles.append(
            core.applicability_head.register_forward_pre_hook(count("applicability"))
        )
    core.eval()
    before = core(encoded)
    for handle in handles:
        handle.remove()
    static_t0_before = core.grounder(before.intents, before.anchor)
    permuted_target = target_embeddings[torch.tensor([2, 0, 3, 1])]
    after = core(encoded)
    firewall = all(
        torch.equal(left, right)
        for left, right in (
            (before.intents, after.intents),
            (before.supports, after.supports),
            (before.temporal_intents, after.temporal_intents),
            (before.temporal_supports, after.temporal_supports),
            (before.trace[0].contexts, after.trace[0].contexts),
            (before.trace[0].delta_z, after.trace[0].delta_z),
            (before.trace[0].candidate_queries, after.trace[0].candidate_queries),
            (before.trace[0].scores, after.trace[0].scores),
        )
    )
    assert permuted_target.shape == target_embeddings.shape and firewall

    reproposal_before: dict[str, torch.Tensor] = {}
    if args.r1c2:
        reproposal_before = {
            "output": core.reproposal.output_projection.weight.detach().clone(),
            "upstream": core.reproposal.state_query_projection.weight.detach().clone(),
        }
        with torch.no_grad():
            core.reproposal.output_projection.weight.normal_(std=1e-3)
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
    if args.r1c2:
        gradient_checks["reproposal_output"] = (
            core.reproposal.output_projection.weight.grad
        )
        gradient_checks["reproposal_upstream"] = (
            core.reproposal.state_query_projection.weight.grad
        )
    gradient_norms = {
        name: 0.0 if gradient is None else float(gradient.norm())
        for name, gradient in gradient_checks.items()
    }
    if not all(
        value > 0 and torch.isfinite(torch.tensor(value)) for value in gradient_norms.values()
    ):
        raise AssertionError(f"expected nonzero finite gradients: {gradient_norms}")
    reproposal_parameter_movement = None
    if args.r1c2:
        optimizer = SGD(core.parameters(), lr=1e-2)
        activated_output = core.reproposal.output_projection.weight.detach().clone()
        activated_upstream = core.reproposal.state_query_projection.weight.detach().clone()
        optimizer.step()
        reproposal_parameter_movement = {
            "output_after_optimizer": float(
                (
                    core.reproposal.output_projection.weight.detach()
                    - activated_output
                )
                .abs()
                .max()
            ),
            "upstream_after_optimizer": float(
                (
                    core.reproposal.state_query_projection.weight.detach()
                    - activated_upstream
                )
                .abs()
                .max()
            ),
        }
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
            "enable_dynamic_reproposal": core.config.enable_dynamic_reproposal,
        },
        "forward_call_counts": calls,
        "reproposal_parameter_movement": reproposal_parameter_movement,
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
            (
                before.temporal_supports[:, 0].detach().float()
                - static_t0_before.detach().float()
            )
            .abs()
            .max()
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
    report["dynamic_reproposal"] = {
        "enabled": output.dynamic_reproposal,
        "initial_intents_shape": list(output.initial_intents.shape),
        "temporal_intents_shape": list(output.temporal_intents.shape),
        "t0_intent_parity_max_abs_error": float(
            (
                output.temporal_intents[:, 0].detach()
                - output.initial_intents.detach()
            )
            .abs()
            .max()
        ),
        "zero_init_residual_max_abs": float(
            (
                before.temporal_intents.detach()
                - before.initial_intents.detach()[:, None]
            )
            .abs()
            .max()
        ),
        "controlled_intent_t1_change": float(
            (
                output.temporal_intents[:, 1].detach()
                - output.temporal_intents[:, 0].detach()
            )
            .norm(dim=-1)
            .mean()
        ),
        "controlled_intent_t2_change": float(
            (
                output.temporal_intents[:, 2].detach()
                - output.temporal_intents[:, 1].detach()
            )
            .norm(dim=-1)
            .mean()
        ),
        "controlled_support_response": float(
            (
                output.temporal_supports[:, 1].detach()
                - before.temporal_supports[:, 1].detach()
            )
            .abs()
            .sum(dim=-1)
            .mean()
        ),
        "pre_activation_output_was_zero": bool(
            torch.count_nonzero(reproposal_before.get("output", torch.ones(1)))
            == 0
        ),
    }
    if args.diagnostics:
        diagnostics = summarize_trajectory(output)
        report["diagnostics"] = {
            name: float(value.detach().float().mean()) for name, value in diagnostics.items()
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
