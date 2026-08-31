from __future__ import annotations

import inspect
import json

import torch
import torch.nn.functional as F
from torch.optim import SGD

from losses.objective import IAGSRMEObjective, ObjectiveConfig
from models.iag_srme import BackboneOutput, IAGSRMEConfig, IAGSRMECore


def _config(*, enabled: bool) -> IAGSRMEConfig:
    return IAGSRMEConfig(
        width=32,
        num_candidates=4,
        max_steps=3,
        num_heads=4,
        retrieval_dim=24,
        lambda_z=0.1,
        query_cap=1000.0,
        selector_gumbel_noise=False,
        enable_dynamic_regrounding=True,
        enable_semantic_residual=enabled,
    )


def _force_repeat(core: IAGSRMECore) -> None:
    with torch.no_grad():
        core.scorer.score_head[-1].weight.zero_()
        core.scorer.score_head[-1].bias.fill_(1.0)


def _grad_norm(parameters) -> float:
    values = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def main() -> None:
    torch.manual_seed(20260831)
    batch, visual_tokens, text_length, width, retrieval_dim = 4, 17, 9, 32, 24
    encoded = BackboneOutput(
        anchor=torch.randn(batch, visual_tokens, width),
        reference_global=F.normalize(torch.randn(batch, retrieval_dim), dim=-1),
        text_tokens=torch.randn(batch, text_length, width),
        text_global=torch.randn(batch, width),
        text_semantic_global=F.normalize(torch.randn(batch, retrieval_dim), dim=-1),
        text_content_mask=torch.ones(batch, text_length, dtype=torch.bool),
    )
    target = F.normalize(torch.randn(batch, retrieval_dim), dim=-1)
    positives = torch.eye(batch, dtype=torch.bool)
    parent = IAGSRMECore(_config(enabled=False)).eval()
    r2 = IAGSRMECore(_config(enabled=True)).eval()
    missing, unexpected = r2.load_state_dict(parent.state_dict(), strict=False)
    if unexpected or not all(name.startswith("semantic_claim.") for name in missing):
        raise AssertionError((missing, unexpected))
    _force_repeat(parent)
    _force_repeat(r2)
    parent_output = parent(encoded, control="repeat_candidate_1")
    initial_output = r2(encoded, control="repeat_candidate_1")
    t0_errors = {
        "intent": float(
            (initial_output.intents - parent_output.intents).detach().abs().max()
        ),
        "support": float(
            (
                initial_output.trace[0].raw_spatial_supports
                - parent_output.trace[0].raw_spatial_supports
            ).detach().abs().max()
        ),
        "delta_z": float(
            (initial_output.trace[0].delta_z - parent_output.trace[0].delta_z)
            .detach()
            .abs()
            .max()
        ),
    }
    snapshot = r2(encoded, control="repeat_candidate_1")
    rerun = r2(encoded, control="repeat_candidate_1")
    firewall = "target" not in inspect.signature(r2.forward).parameters and all(
        torch.equal(left, right)
        for left, right in (
            (snapshot.temporal_semantic_residuals, rerun.temporal_semantic_residuals),
            (snapshot.temporal_semantic_claims, rerun.temporal_semantic_claims),
            (snapshot.temporal_intents, rerun.temporal_intents),
            (snapshot.temporal_supports, rerun.temporal_supports),
            (snapshot.final_state, rerun.final_state),
            (snapshot.final_query, rerun.final_query),
        )
    )

    r2.train()
    # A deliberately large synthetic-only step makes the expected two-stage
    # zero-init learning path observable above FP32 parameter resolution.
    optimizer = SGD(r2.parameters(), lr=1.0)
    objective = IAGSRMEObjective(ObjectiveConfig(), width=width)
    tracked = {
        "claim_output": r2.semantic_claim.claim_projection.weight,
        "consumption_output": r2.semantic_claim.consumption_projection.weight,
        "claim_query": r2.semantic_claim.query_projection.weight,
        "claim_token": r2.semantic_claim.token_projection.weight,
        "claim_state": r2.semantic_claim.state_projection.weight,
        "grounder": r2.grounder.intent_projection.weight,
        "editor": r2.editor.direction.weight,
        "readout": r2.readout.output_projection.weight,
    }
    initial = {name: parameter.detach().clone() for name, parameter in tracked.items()}
    pass_gradients = []
    losses = None
    output = None
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = r2(encoded, control="repeat_candidate_1")
        losses = objective(output, target, positives)
        losses["total"].backward()
        pass_gradients.append(
            {
                "claim_output": _grad_norm(r2.semantic_claim.claim_projection.parameters()),
                "consumption_output": _grad_norm(
                    r2.semantic_claim.consumption_projection.parameters()
                ),
                "claim_upstream": _grad_norm(
                    list(r2.semantic_claim.query_projection.parameters())
                    + list(r2.semantic_claim.token_projection.parameters())
                    + list(r2.semantic_claim.state_projection.parameters())
                ),
                "grounder": _grad_norm(r2.grounder.parameters()),
            }
        )
        optimizer.step()
    assert output is not None and losses is not None
    same_parent_error = max(
        float(
            (
                step.candidate_states
                - (step.current_state[:, None] + step.delta_z)
            ).detach().abs().max()
        )
        for step in output.trace
    )
    report = {
        "architecture_generation": "r2_semantic_residual_claim_firewall_v2",
        "shapes": {
            "rho": list(output.temporal_semantic_residuals.shape),
            "claims": list(output.temporal_semantic_claims.shape),
            "intents": list(output.temporal_intents.shape),
            "supports": list(output.temporal_supports.shape),
            "delta_z": list(output.trace[0].delta_z.shape),
            "final_query": list(output.final_query.shape),
        },
        "t0_parent_parity_max_abs_error": t0_errors,
        "initial_rho_mean_by_state": initial_output.temporal_semantic_residuals.mean(
            dim=(0, 2)
        ).tolist(),
        "rho_mean_by_state_after_updates": output.temporal_semantic_residuals.mean(
            dim=(0, 2)
        ).tolist(),
        "claim_mean_by_timestep": initial_output.temporal_semantic_claims.mean(
            dim=(0, 2, 3)
        ).tolist(),
        "consumption_mean_by_timestep": initial_output.temporal_semantic_consumption.mean(
            dim=(0, 2, 3)
        ).tolist(),
        "effective_consumption_mean_by_timestep": (
            initial_output.temporal_effective_semantic_consumption.mean(
                dim=(0, 2, 3)
            ).tolist()
        ),
        "semantic_mass_mean_by_timestep": [
            float(step.claimed_semantic_mass.detach().mean())
            for step in initial_output.trace
        ],
        "semantic_content_norm_by_timestep": [
            float(
                step.claimed_text_content.detach().float().norm(dim=-1).mean()
            )
            for step in initial_output.trace
        ],
        "claim_dtype": str(output.temporal_semantic_claims.dtype),
        "rho_dtype": str(output.temporal_semantic_residuals.dtype),
        "gradient_norms_by_update": pass_gradients,
        "parameter_max_abs_delta": {
            name: float((parameter.detach() - initial[name]).abs().max())
            for name, parameter in tracked.items()
        },
        "same_parent_max_abs_error": same_parent_error,
        "target_firewall": firewall,
        "losses": {name: float(value.detach()) for name, value in losses.items()},
        "finite": bool(torch.isfinite(losses["total"])),
    }
    initial_rho = report["initial_rho_mean_by_state"]
    if not 0.94 <= initial_rho[1] / initial_rho[0] <= 0.96:
        raise AssertionError(
            f"conservative initial consumption contract failed: {initial_rho}"
        )
    if not report["finite"] or not firewall or same_parent_error != 0.0:
        raise AssertionError(report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
