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
        )
    )
    with torch.no_grad():
        core.scorer.score_head[-1].weight.zero_()
        core.scorer.score_head[-1].bias.fill_(0.5)
    encoded = BackboneOutput(
        anchor=torch.randn(batch, tokens, width, requires_grad=True),
        reference_global=F.normalize(torch.randn(batch, retrieval_dim), dim=-1).requires_grad_(),
        text_tokens=torch.randn(batch, length, width, requires_grad=True),
        text_global=torch.randn(batch, width, requires_grad=True),
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
    gradient_norms = {
        name: 0.0 if gradient is None else float(gradient.norm())
        for name, gradient in gradient_checks.items()
    }
    if not all(value > 0 and torch.isfinite(torch.tensor(value)) for value in gradient_norms.values()):
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
    }
    if args.diagnostics:
        diagnostics = summarize_trajectory(output)
        report["diagnostics"] = {
            name: float(value.detach().float().mean()) for name, value in diagnostics.items()
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
