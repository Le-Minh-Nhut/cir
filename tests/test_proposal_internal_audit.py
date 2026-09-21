from __future__ import annotations

from dataclasses import replace
import copy
import json

import pytest
import torch

from analyze_proposal_internal import _heuristic, _records, _summarize
from diagnostics.functional_collapse import generic_diversity, proposal_internal_audit
from models.iag_srme import IAGSRMEConfig, ProposalNet


STAGES = (
    "base_query",
    "expanded_query",
    "conditioned_residual",
    "query_pre_norm",
    "query_post_norm",
    "raw_attention_output",
    "proposal_output",
)


def _step(timestep: int, batch: int, *, collapsed_pre: bool, collapsed_output: bool) -> dict[str, object]:
    base = torch.eye(3).unsqueeze(0).expand(batch, -1, -1).requires_grad_()
    pre = torch.ones_like(base) if collapsed_pre else base
    output = torch.ones_like(base) if collapsed_output else pre
    return {
        "timestep": timestep,
        "proposal_internal": {
            "base_query": base[:1],
            "expanded_query": base,
            "conditioned_residual": pre - base,
            "query_pre_norm": pre,
            "query_post_norm": pre,
            "raw_attention_output": output,
            "proposal_output": output,
        },
    }


def test_heuristic_labels_pre_attention_and_attention_output() -> None:
    pre = proposal_internal_audit({"steps": [_step(0, 2, collapsed_pre=True, collapsed_output=True)]})
    output = proposal_internal_audit({"steps": [_step(0, 2, collapsed_pre=False, collapsed_output=True)]})

    assert _heuristic(pre["overall"]) == "pre_attention"
    assert _heuristic(output["overall"]) == "attention_output"


def test_base_query_metrics_match_generic_diversity_and_identical_rank() -> None:
    base = torch.ones(1, 5, 4)
    step = _step(0, 2, collapsed_pre=False, collapsed_output=False)
    step["proposal_internal"]["base_query"] = base
    audit = proposal_internal_audit({"steps": [step]})

    assert audit["overall"]["base_query"] == generic_diversity(base)
    assert audit["overall"]["base_query"]["pairwise_cosine"] == pytest.approx(1.0)
    assert audit["overall"]["base_query"]["effective_rank"] == pytest.approx(1.0, abs=1e-3)
    assert len([key for key in audit["overall"]["base_query"] if key.startswith("mean_norm_c")]) == 5


def test_attention_diversity_retention_and_recurrent_aggregation_are_weighted() -> None:
    first = _step(0, 2, collapsed_pre=False, collapsed_output=True)
    second = _step(1, 1, collapsed_pre=True, collapsed_output=True)
    audit = proposal_internal_audit({"steps": [first, second]})

    expected = (
        audit["by_step"]["t0"]["attention_diversity_retention"] * 2
        + audit["by_step"]["t1"]["attention_diversity_retention"]
    ) / 3
    assert audit["overall"]["live_samples"] == 3
    assert audit["overall"]["attention_diversity_retention"] == pytest.approx(expected)

def test_audit_emits_detached_stages_without_changing_model_behavior(model, features) -> None:
    state, tokens, text, mask = features
    model.eval()
    enabled = copy.deepcopy(model).eval()
    enabled.config = replace(enabled.config, proposal_internal_audit_enabled=True)
    enabled.proposal.internal_audit_enabled = True

    baseline = model.forward_from_features(state, tokens, text, mask)
    output = enabled.forward_from_features(state, tokens, text, mask)
    audit = proposal_internal_audit(output)

    torch.testing.assert_close(output["state"], baseline["state"])
    torch.testing.assert_close(output["query"], baseline["query"])
    assert "proposal_internal" not in baseline["steps"][0]
    internal = output["steps"][0]["proposal_internal"]
    assert set(internal) == set(STAGES)
    assert all(not value.requires_grad for value in internal.values())
    assert set(audit["by_step"]) == {"t0", "t1", "t2"}
    assert set(audit["overall"]) >= {*STAGES, "attention_diversity_retention", "attention_rank_retention"}


def test_disabled_audit_preserves_proposal_forward_and_supports_dynamic_k(features) -> None:
    _, tokens, text, mask = features
    current = torch.randn(tokens.shape[0], 16)
    proposal = ProposalNet(16, 8, 5, 1).eval()
    enabled = copy.deepcopy(proposal)
    enabled.internal_audit_enabled = True

    torch.testing.assert_close(proposal(tokens, text, current, mask), enabled(tokens, text, current, mask))
    assert proposal.last_internal_audit is None
    assert enabled.last_internal_audit is not None
    assert enabled.last_internal_audit["proposal_output"].shape[1] == 5


def test_audit_query_snapshots_survive_parameter_update(features) -> None:
    _, tokens, text, mask = features
    proposal = ProposalNet(16, 8, 3, 1).eval()
    proposal.internal_audit_enabled = True
    output = proposal(tokens, text, torch.randn(tokens.shape[0], 16), mask)
    assert proposal.last_internal_audit is not None
    base = proposal.last_internal_audit["base_query"].clone()
    expanded = proposal.last_internal_audit["expanded_query"].clone()
    output_before = output.clone()

    with torch.no_grad():
        proposal.queries.add_(1.0)

    torch.testing.assert_close(proposal.last_internal_audit["base_query"], base)
    torch.testing.assert_close(proposal.last_internal_audit["expanded_query"], expanded)
    torch.testing.assert_close(output, output_before)


@pytest.mark.parametrize("mode", ("attention", "attention_ln", "residual", "residual_ln"))
def test_proposal_modes_are_finite_and_expose_attention_stages(features, mode) -> None:
    _, tokens, text, mask = features
    proposal = ProposalNet(16, 8, 3, 1, proposal_mode=mode).eval()
    proposal.internal_audit_enabled = True
    output = proposal(tokens, text, torch.randn(tokens.shape[0], 16), mask)

    assert output.shape == (tokens.shape[0], 3, 16)
    assert torch.isfinite(output).all()
    assert proposal.last_internal_audit is not None
    assert {"query_post_norm", "raw_attention_output", "proposal_output"} <= set(proposal.last_internal_audit)
    torch.testing.assert_close(proposal.last_internal_audit["proposal_output"], output)


def test_attention_mode_matches_the_pre_ablation_attention_path(features) -> None:
    _, tokens, text, mask = features
    proposal = ProposalNet(16, 8, 3, 1, proposal_mode="attention").eval()
    current = torch.randn(tokens.shape[0], 16)
    text_context = proposal.text_context(text)
    visual_context = proposal.visual_context(current)
    context = proposal.context(torch.cat([text_context, visual_context, text_context * visual_context, text_context - visual_context], dim=-1))
    query = proposal.queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
    query = proposal.query_norm(proposal.query_conditioner(torch.cat([query, context[:, None].expand_as(query), query * context[:, None]], dim=-1)) + query)
    expected, _ = proposal.token_attention(query, tokens, tokens, key_padding_mask=~mask, need_weights=False)

    torch.testing.assert_close(proposal(tokens, text, current, mask), expected)


def test_residual_mode_preserves_query_difference_when_attention_is_zero(features) -> None:
    _, tokens, text, mask = features
    attention = ProposalNet(16, 8, 3, 1, proposal_mode="attention").eval()
    residual = ProposalNet(16, 8, 3, 1, proposal_mode="residual").eval()
    residual.load_state_dict(attention.state_dict())
    with torch.no_grad():
        for parameter in attention.token_attention.parameters():
            parameter.zero_()
        for parameter in residual.token_attention.parameters():
            parameter.zero_()
    residual.internal_audit_enabled = True
    current = torch.randn(tokens.shape[0], 16)

    assert torch.equal(attention(tokens, text, current, mask), torch.zeros(tokens.shape[0], 3, 16))
    output = residual(tokens, text, current, mask)
    assert not torch.allclose(output[:, 0], output[:, 1])
    assert residual.last_internal_audit is not None
    torch.testing.assert_close(output, residual.last_internal_audit["query_post_norm"])


def test_unknown_proposal_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="proposal_mode"):
        IAGSRMEConfig(proposal_mode="unknown")


def test_analyzer_reads_and_summarizes_audit_records(tmp_path) -> None:
    audit = proposal_internal_audit({"steps": [_step(0, 2, collapsed_pre=True, collapsed_output=True)]})
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps({"proposal_internal_audit": audit}) + "\n")

    summary = _summarize(_records(path))
    assert _heuristic(summary) == "pre_attention"
