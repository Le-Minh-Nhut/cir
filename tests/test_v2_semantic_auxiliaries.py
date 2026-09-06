from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
from omegaconf import OmegaConf

from losses.objective import (
    ConceptSetAuxiliary,
    IAGSRMEObjective,
    ObjectiveConfig,
    RelationAuxiliary,
)
from models.iag_srme.utils.semantic import ConceptVocabulary, parse_instruction_concepts


def _grad_sum(module: torch.nn.Module) -> torch.Tensor:
    values = [
        parameter.grad.abs().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    return sum(values, torch.tensor(0.0))


def _concept_objective(state_dim: int = 16) -> IAGSRMEObjective:
    vocabulary = ConceptVocabulary(("red", "long sleeve", "striped"))
    prototypes = torch.randn(3, 8, requires_grad=True)
    return IAGSRMEObjective(
        ObjectiveConfig(
            terminal_weight=0.0,
            lambda_pair=0.0,
            lambda_gain=0.0,
            concept_enabled=True,
            lambda_c=1.0,
        ),
        state_dim=state_dim,
        concept_vocabulary=vocabulary,
        concept_prototypes=prototypes,
    )


def test_concept_parser_and_vocabulary_use_instruction_text_only() -> None:
    instructions = [
        "Make it red and long-sleeved",
        "The shirt is more colorful and has no sleeves",
    ]
    first = ConceptVocabulary.build(instructions, min_frequency=1)
    second = ConceptVocabulary.build(instructions, min_frequency=1)

    assert first == second
    assert first.fingerprint == second.fingerprint
    assert "red" in parse_instruction_concepts(instructions[0])
    assert "long sleeved" in parse_instruction_concepts(instructions[0])
    assert "more colorful" in parse_instruction_concepts(instructions[1])
    assert "no sleeves" in parse_instruction_concepts(instructions[1])
    assert "is" not in first.concepts
    assert "more" not in first.concepts
    labels = first.labels(instructions, torch.device("cpu"))
    # No target object is accepted anywhere in the parser/label API.
    assert labels.shape == (2, len(first.concepts))


def test_concept_set_pools_candidates_before_loss_and_is_permutation_invariant() -> None:
    torch.manual_seed(1)
    vocabulary = ConceptVocabulary(("red", "striped"))
    module = ConceptSetAuxiliary(
        16,
        vocabulary,
        torch.randn(2, 8),
        tau_concept=0.2,
        tau_mil=0.5,
        beta_pos=1.0,
        beta_neg=4.0,
        threshold=0.5,
    )
    proposals = torch.randn(2, 4, 16)
    texts = ["red", "striped"]
    normal = module(proposals, texts)
    permuted = module(proposals[:, [2, 0, 3, 1]], texts)

    assert normal["candidate_logits"].shape == (2, 4, 2)
    assert normal["set_probability"].shape == (2, 2)
    assert torch.allclose(normal["loss"], permuted["loss"])
    assert torch.allclose(normal["set_probability"], permuted["set_probability"])


def test_concept_loss_is_t0_only_and_routes_only_to_proposal(model, features) -> None:
    initial, tokens, text, mask = features
    output = model.forward_from_features(initial, tokens, text, mask)
    objective = _concept_objective()
    targets = torch.randn(3, 12)
    texts = ["red", "long sleeve", "striped"]

    changed_later_step = {**output, "steps": [dict(step) for step in output["steps"]]}
    changed_later_step["steps"][1]["proposals"] = torch.randn_like(
        changed_later_step["steps"][1]["proposals"]
    )
    base = objective(output, targets, ["a", "b", "c"], texts)["concept_loss"]
    changed = objective(changed_later_step, targets.flip(0), ["c", "b", "a"], texts)[
        "concept_loss"
    ]
    assert torch.allclose(base, changed)

    objective(output, targets, ["a", "b", "c"], texts)["total"].backward()
    assert _grad_sum(model.proposal) > 0
    for module in (model.grounder, model.action_fusion, model.executor, model.score_net):
        assert _grad_sum(module) == 0
    assert objective.concept is not None
    assert not objective.concept.concept_prototypes.requires_grad


def test_bind_uses_separate_relation_bank_and_routes_expected_gradients(model, features) -> None:
    initial, tokens, text, mask = features
    model.config = replace(model.config, max_steps=1)
    output = model.forward_from_features(initial, tokens, text, mask)
    objective = IAGSRMEObjective(
        ObjectiveConfig(
            terminal_weight=0.0,
            lambda_pair=0.0,
            lambda_gain=0.0,
            bind_enabled=True,
            lambda_bind=1.0,
        ),
        state_dim=16,
    )
    assert objective.relation is not None
    assert objective.relation.relation_prototypes.data_ptr() != model.proposal.queries.data_ptr()

    targets = torch.randn(3, 12)
    objective(output, targets, ["a", "b", "c"])["total"].backward()
    assert _grad_sum(model.proposal) > 0
    assert _grad_sum(model.grounder) > 0
    assert objective.relation.relation_prototypes.grad is not None
    assert objective.relation.relation_prototypes.grad.abs().sum() > 0
    for module in (model.action_fusion, model.executor, model.score_net):
        assert _grad_sum(module) == 0


def test_bind_kl_is_candidate_permutation_invariant() -> None:
    torch.manual_seed(3)
    relation = RelationAuxiliary(16, 5, 8, tau_rel=0.2)
    edits = torch.randn(2, 4, 16)
    entities = torch.randn(2, 4, 16)
    normal, b_edit, b_entity = relation.bind_loss(edits, entities)
    order = [3, 1, 0, 2]
    permuted, _, _ = relation.bind_loss(edits[:, order], entities[:, order])

    expected = (b_edit * (b_edit.log() - b_entity.log())).sum(dim=-1).mean()
    assert torch.allclose(normal, expected)
    assert torch.allclose(normal, permuted)


def test_relation_orthogonality_regularizes_only_relation_prototypes() -> None:
    relation = RelationAuxiliary(8, 4, 8, tau_rel=0.1)
    with torch.no_grad():
        relation.relation_prototypes.fill_(1.0)
    collapsed = relation.orthogonality_loss()
    with torch.no_grad():
        relation.relation_prototypes.copy_(torch.eye(4, 8))
    orthogonal = relation.orthogonality_loss()

    assert collapsed > orthogonal
    assert torch.allclose(orthogonal, torch.zeros_like(orthogonal), atol=1e-7)
    orthogonal.backward()
    assert relation.relation_prototypes.grad is not None
    assert all(parameter.grad is None for parameter in relation.edit_projection.parameters())
    assert all(parameter.grad is None for parameter in relation.entity_projection.parameters())


def test_correspondence_is_disabled_and_contributes_no_loss(model, features) -> None:
    config = ObjectiveConfig()
    objective = IAGSRMEObjective(config)
    assert not config.correspondence_enabled
    checked_in = OmegaConf.load(Path("conf/objective/core.yaml"))
    assert not checked_in.correspondence_enabled
    assert not any("corr" in name or "cycle" in name for name, _ in objective.named_modules())
    output = model.forward_from_features(*features)
    components = objective(output, torch.randn(3, 12), ["a", "b", "c"])
    assert not any("corr" in name or "cycle" in name or "gcons" in name for name in components)
