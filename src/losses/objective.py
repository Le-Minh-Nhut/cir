from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.iag_srme.utils.retrieval import (
    build_teacher_masks,
    marginal_teacher_utilities,
)
from models.iag_srme.utils.semantic import ConceptVocabulary, PARSER_VERSION

from .retrieval import TerminalRetrievalLoss


@dataclass(frozen=True, slots=True)
class ObjectiveConfig:
    terminal_weight: float = 1.0
    lambda_pair: float = 0.5
    lambda_gain: float = 0.5
    retrieval_temperature: float = 0.07
    pair_temperature: float = 1.0
    pair_weight_temperature: float = 1.0
    epsilon_pair: float = 0.01
    huber_delta: float = 1.0

    concept_enabled: bool = False
    bind_enabled: bool = False
    rel_ortho_enabled: bool = False
    dpp_enabled: bool = False
    correspondence_enabled: bool = False
    lambda_c: float = 0.01
    lambda_bind: float = 0.01
    lambda_rel: float = 0.001
    lambda_dpp: float = 0.01

    tau_concept: float = 0.1
    tau_mil: float = 1.0
    beta_pos: float = 1.0
    beta_neg: float = 4.0
    concept_threshold: float = 0.5
    concept_min_frequency: int = 2
    concept_max_size: int = 2048
    concept_parser_version: str = PARSER_VERSION

    num_relation_prototypes: int = 8
    tau_rel: float = 0.1

    kappa_dpp: float = 1.0
    sigma_dpp: float = 1.0
    tau_dpp: float = 0.1
    useful_threshold: float = 0.55
    dpp_jitter: float = 1e-4


def pairwise_ranking_loss(
    predicted: Tensor,
    teacher: Tensor,
    valid_rows: Tensor,
    *,
    epsilon: float,
    temperature: float,
    weight_temperature: float,
) -> tuple[Tensor, Tensor]:
    """Confidence-weighted ranking over every unordered sibling pair."""

    candidates = predicted.shape[-1]
    left, right = torch.triu_indices(candidates, candidates, offset=1, device=predicted.device)
    teacher_delta = teacher[:, left] - teacher[:, right]
    confident = teacher_delta.abs() >= epsilon
    pair_mask = valid_rows[:, None] & confident
    weight = 1.0 - torch.exp(-teacher_delta.abs() / weight_temperature)
    signed_margin = teacher_delta.sign() * (predicted[:, left] - predicted[:, right])
    values = -weight * F.logsigmoid(signed_margin / temperature)
    normalizer = (weight * pair_mask).sum()
    loss = (values * pair_mask).sum() / normalizer.clamp_min(1e-8)
    return loss, normalizer


def absolute_gain_loss(
    predicted: Tensor, teacher: Tensor, valid_rows: Tensor, delta: float
) -> tuple[Tensor, Tensor]:
    values = F.huber_loss(predicted, teacher, reduction="none", delta=delta)
    mask = valid_rows[:, None].expand_as(values)
    count = mask.sum()
    return (values * mask).sum() / count.clamp_min(1), count


class ConceptSetAuxiliary(nn.Module):
    """Instruction-level concept coverage after pooling the proposal set."""

    def __init__(
        self,
        state_dim: int,
        vocabulary: ConceptVocabulary,
        prototypes: Tensor,
        *,
        tau_concept: float,
        tau_mil: float,
        beta_pos: float,
        beta_neg: float,
        threshold: float,
    ) -> None:
        super().__init__()
        if prototypes.shape[0] != len(vocabulary.concepts):
            raise ValueError("one frozen prototype is required for every concept")
        self.vocabulary = vocabulary
        self.tau_concept = tau_concept
        self.tau_mil = tau_mil
        self.beta_pos = beta_pos
        self.beta_neg = beta_neg
        self.threshold = threshold
        self.projection = nn.Linear(state_dim, prototypes.shape[-1], bias=False)
        self.register_buffer(
            "concept_prototypes", F.normalize(prototypes.detach().float(), dim=-1)
        )

    def forward(
        self, proposals_t0: Tensor, instructions: Sequence[str]
    ) -> Mapping[str, Tensor]:
        # proposals_t0: [B,K,D]. Pool K before comparing to instruction labels.
        z = F.normalize(self.projection(proposals_t0).float(), dim=-1)
        candidate_logits = (
            torch.einsum("bkd,md->bkm", z, self.concept_prototypes)
            / self.tau_concept
        )
        set_logits = self.tau_mil * torch.logsumexp(
            candidate_logits / self.tau_mil, dim=1
        )
        probability = torch.sigmoid(set_logits)
        positive = self.vocabulary.labels(instructions, proposals_t0.device)
        negative = ~positive
        eps = torch.finfo(probability.dtype).eps
        positive_term = (
            (1.0 - probability).pow(self.beta_pos)
            * torch.log(probability.clamp_min(eps))
            * positive
        )
        negative_term = (
            probability.pow(self.beta_neg)
            * torch.log((1.0 - probability).clamp_min(eps))
            * negative
        )
        loss = -(positive_term + negative_term).sum(dim=-1).mean() / probability.shape[-1]

        predicted = probability >= self.threshold
        positive_count = positive.sum().clamp_min(1)
        negative_count = negative.sum().clamp_min(1)
        recall = (predicted & positive).sum().float() / positive_count
        false_positive_rate = (predicted & negative).sum().float() / negative_count
        rows_with_concepts = positive.any(dim=-1)
        covered = ((~positive) | predicted).all(dim=-1)
        coverage = (covered & rows_with_concepts).sum().float() / rows_with_concepts.sum().clamp_min(1)
        return {
            "loss": loss,
            "candidate_logits": candidate_logits,
            "set_probability": probability,
            "labels": positive,
            "concept_positive_recall": recall.detach(),
            "concept_negative_false_positive_rate": false_positive_rate.detach(),
            "instruction_concept_coverage": coverage.detach(),
        }


class RelationAuxiliary(nn.Module):
    """Edit/entity binding against a stable, separate relation prototype bank."""

    def __init__(
        self, state_dim: int, num_prototypes: int, relation_dim: int, tau_rel: float
    ) -> None:
        super().__init__()
        self.tau_rel = tau_rel
        self.edit_projection = nn.Linear(state_dim, relation_dim, bias=False)
        self.entity_projection = nn.Linear(state_dim, relation_dim, bias=False)
        self.relation_prototypes = nn.Parameter(torch.empty(num_prototypes, relation_dim))
        nn.init.normal_(self.relation_prototypes, std=0.02)

    def distributions(self, edits: Tensor, entities: Tensor) -> tuple[Tensor, Tensor]:
        prototypes = F.normalize(self.relation_prototypes, dim=-1)
        edit = F.normalize(self.edit_projection(edits), dim=-1)
        entity = F.normalize(self.entity_projection(entities), dim=-1)
        b_edit = torch.softmax(torch.einsum("bkd,md->bkm", edit, prototypes) / self.tau_rel, dim=-1)
        b_entity = torch.softmax(
            torch.einsum("bkd,md->bkm", entity, prototypes) / self.tau_rel, dim=-1
        )
        return b_edit, b_entity

    def bind_loss(self, edits: Tensor, entities: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        b_edit, b_entity = self.distributions(edits, entities)
        # Canonical direction: KL(edit relation distribution || entity distribution).
        loss = (
            b_edit
            * (b_edit.clamp_min(1e-8).log() - b_entity.clamp_min(1e-8).log())
        ).sum(dim=-1).mean()
        return loss, b_edit, b_entity

    def orthogonality_loss(self) -> Tensor:
        prototypes = F.normalize(self.relation_prototypes, dim=-1)
        count = prototypes.shape[0]
        gram = prototypes @ prototypes.T
        identity = torch.eye(count, device=gram.device, dtype=gram.dtype)
        return (gram - identity).square().sum() / max(count * (count - 1), 1)

    def diagnostics(self, distributions: Tensor) -> Mapping[str, Tensor]:
        occupancy = distributions.mean(dim=(0, 1))
        entropy = -(occupancy * occupancy.clamp_min(1e-8).log()).sum()
        prototype = F.normalize(self.relation_prototypes.detach(), dim=-1)
        cosine = prototype @ prototype.T
        count = prototype.shape[0]
        off_diagonal = ~torch.eye(count, device=cosine.device, dtype=torch.bool)
        pairwise = cosine[off_diagonal].mean() if count > 1 else cosine.new_zeros(())
        active = (occupancy > 1.0 / (10.0 * count)).float().mean()
        return {
            "prototype_occupancy": active,
            "prototype_entropy": entropy.detach(),
            "prototype_pairwise_cosine": pairwise.detach(),
        }


def _rbf_similarity(values: Tensor, sigma: float) -> Tensor:
    squared_distance = (values[:, None] - values[None, :]).square().sum(dim=-1)
    return torch.exp(-squared_distance / (2.0 * sigma**2))


def functional_dpp_loss(
    effects: Tensor,
    history: Tensor | None,
    teacher_utility: Tensor,
    *,
    kappa: float,
    sigma: float,
    tau: float,
    useful_threshold: float,
    jitter: float,
) -> Mapping[str, Tensor]:
    """History-conditioned DPP over current retrieval consequences [K,D]."""

    effect_values = effects.float()
    effects_normalized = effect_values / (
        effect_values.norm(dim=-1, keepdim=True) + 1e-8
    )
    quality = torch.sigmoid(teacher_utility.detach().float() / tau)
    active = quality > useful_threshold
    useful_count = active.sum()
    if useful_count < 2:
        return {
            "loss": effects.sum() * 0.0,
            "normalized_effects": effects_normalized,
            "quality": quality,
            "useful_count": useful_count.float(),
            "valid": torch.zeros((), dtype=torch.bool, device=effects.device),
        }

    similarity_cc = _rbf_similarity(effects_normalized.float(), sigma)
    conditional = similarity_cc
    if history is not None and history.numel() > 0:
        history_normalized = F.normalize(history.detach().float(), dim=-1)
        all_values = torch.cat([effects_normalized.float(), history_normalized], dim=0)
        similarity = _rbf_similarity(all_values, sigma)
        candidates = effects.shape[0]
        similarity_ch = similarity[:candidates, candidates:]
        similarity_hh = similarity[candidates:, candidates:]
        regularized_history = similarity_hh + jitter * torch.eye(
            similarity_hh.shape[0], device=effects.device, dtype=similarity_hh.dtype
        )
        conditional = similarity_cc - similarity_ch @ torch.linalg.solve(
            regularized_history, similarity_ch.T
        )
    conditional = 0.5 * (conditional + conditional.T)
    kernel = quality[:, None] * conditional * quality[None, :]
    matrix = torch.eye(effects.shape[0], device=effects.device, dtype=kernel.dtype)
    matrix = matrix + kappa * kernel
    sign, logabsdet = torch.linalg.slogdet(matrix)
    loss = torch.where(sign > 0, -logabsdet, effects.sum() * 0.0)
    return {
        "loss": loss,
        "normalized_effects": effects_normalized,
        "quality": quality,
        "useful_count": useful_count.float(),
        "valid": torch.ones((), dtype=torch.bool, device=effects.device),
    }


def update_executed_history(
    histories: list[list[Tensor]],
    live_indices: Tensor,
    selected_indices: Tensor,
    delta_q: Tensor,
    num_candidates: int,
) -> None:
    """Append only detached effects that were actually committed; STOP appends nothing."""

    for local_row, sample_index in enumerate(live_indices.tolist()):
        selected = int(selected_indices[local_row])
        if selected < num_candidates:
            effect = delta_q[local_row, selected]
            histories[sample_index].append(
                F.normalize(effect.float(), dim=-1).detach()
            )


def _functional_diagnostics(effects: list[Tensor], zero: Tensor) -> tuple[Tensor, Tensor]:
    if not effects:
        return zero.detach(), zero.detach()
    cosine_values = []
    rank_values = []
    for value in effects:
        # value: [B_live,K,D]. Diagnostics stay within sibling sets.
        values = value.detach().float()
        normalized = values / (values.norm(dim=-1, keepdim=True) + 1e-8)
        cosine = normalized @ normalized.transpose(-1, -2)
        candidates = value.shape[1]
        off_diagonal = ~torch.eye(candidates, dtype=torch.bool, device=value.device)
        cosine_values.append(cosine[:, off_diagonal].mean())
        singular_values = torch.linalg.svdvals(value.detach().float())
        effective_rank = singular_values.sum(dim=-1).square() / singular_values.square().sum(
            dim=-1
        ).clamp_min(1e-8)
        rank_values.append(effective_rank.mean())
    return torch.stack(cosine_values).mean(), torch.stack(rank_values).mean().to(zero.dtype)


class IAGSRMEObjective(nn.Module):
    """Canonical V2 objective; all target-derived candidate judgments are detached."""

    def __init__(
        self,
        config: ObjectiveConfig,
        width: int = 256,
        *,
        state_dim: int | None = None,
        concept_vocabulary: ConceptVocabulary | None = None,
        concept_prototypes: Tensor | None = None,
    ) -> None:
        super().__init__()
        if config.correspondence_enabled:
            raise ValueError("correspondence is intentionally disabled in V2 R0")
        self.config = config
        self.terminal = TerminalRetrievalLoss(config.retrieval_temperature)
        state_dim = state_dim or width
        self.concept: ConceptSetAuxiliary | None = None
        if config.concept_enabled:
            if concept_vocabulary is None or concept_prototypes is None:
                raise ValueError("enabled concept loss requires training-split vocabulary and prototypes")
            if concept_vocabulary.parser_version != config.concept_parser_version:
                raise ValueError("configured concept parser version does not match the vocabulary")
            self.concept = ConceptSetAuxiliary(
                state_dim,
                concept_vocabulary,
                concept_prototypes,
                tau_concept=config.tau_concept,
                tau_mil=config.tau_mil,
                beta_pos=config.beta_pos,
                beta_neg=config.beta_neg,
                threshold=config.concept_threshold,
            )
        self.relation: RelationAuxiliary | None = None
        if config.bind_enabled or config.rel_ortho_enabled:
            self.relation = RelationAuxiliary(
                state_dim,
                config.num_relation_prototypes,
                state_dim,
                config.tau_rel,
            )

    def forward(
        self,
        output: Mapping[str, object],
        target_embeddings: Tensor,
        target_ids: Sequence[str | None],
        modification_texts: Sequence[str] | None = None,
    ) -> Mapping[str, Tensor]:
        positive, negative, _ = build_teacher_masks(target_ids, target_embeddings.device)
        query = output["query"]
        assert isinstance(query, Tensor)
        terminal = self.terminal(query, target_embeddings, positive)

        steps = output["steps"]
        assert isinstance(steps, list)
        zero = query.sum() * 0.0
        pair_numerator = zero
        gain_numerator = zero
        pair_count = query.new_zeros(())
        gain_count = query.new_zeros(())
        invalid_rows = query.new_zeros(())
        bind_numerator = zero
        bind_count = query.new_zeros(())
        binding_distributions: list[Tensor] = []
        dpp_numerator = zero
        dpp_count = query.new_zeros(())
        useful_total = query.new_zeros(())
        useful_rows = query.new_zeros(())
        functional_effects: list[Tensor] = []
        histories: list[list[Tensor]] = [[] for _ in range(query.shape[0])]

        for step in steps:
            live_indices = step["live_indices"]
            current_query = step["current_query"]
            candidate_queries = step["candidate_queries"]
            predicted = step["scores"]
            delta_q = step["delta_q"]
            pos_live = positive.index_select(0, live_indices)
            neg_live = negative.index_select(0, live_indices)
            teacher, valid_rows = marginal_teacher_utilities(
                current_query,
                candidate_queries,
                target_embeddings.detach(),
                pos_live,
                neg_live,
                self.config.retrieval_temperature,
            )
            pair, pair_weight = pairwise_ranking_loss(
                predicted,
                teacher,
                valid_rows,
                epsilon=self.config.epsilon_pair,
                temperature=self.config.pair_temperature,
                weight_temperature=self.config.pair_weight_temperature,
            )
            gain, gains = absolute_gain_loss(
                predicted, teacher, valid_rows, self.config.huber_delta
            )
            pair_numerator = pair_numerator + pair * pair_weight
            gain_numerator = gain_numerator + gain * gains.clamp_min(1)
            pair_count = pair_count + pair_weight
            gain_count = gain_count + gains
            invalid_rows = invalid_rows + (~valid_rows).sum()

            if self.config.bind_enabled:
                assert self.relation is not None
                bind, edit_distribution, _ = self.relation.bind_loss(
                    step["proposals"], step["entities"]
                )
                candidates = step["proposals"].shape[0] * step["proposals"].shape[1]
                bind_numerator = bind_numerator + bind * candidates
                bind_count = bind_count + candidates
                binding_distributions.append(edit_distribution.detach())

            if self.config.dpp_enabled:
                functional_effects.append(delta_q)
                for local_row, sample_index in enumerate(live_indices.tolist()):
                    if not bool(valid_rows[local_row]):
                        continue
                    history_values = histories[sample_index]
                    history = torch.stack(history_values) if history_values else None
                    dpp = functional_dpp_loss(
                        delta_q[local_row],
                        history,
                        teacher[local_row],
                        kappa=self.config.kappa_dpp,
                        sigma=self.config.sigma_dpp,
                        tau=self.config.tau_dpp,
                        useful_threshold=self.config.useful_threshold,
                        jitter=self.config.dpp_jitter,
                    )
                    useful_total = useful_total + dpp["useful_count"]
                    useful_rows = useful_rows + 1
                    if bool(dpp["valid"]):
                        dpp_numerator = dpp_numerator + dpp["loss"]
                        dpp_count = dpp_count + 1

            update_executed_history(
                histories,
                live_indices,
                step["selected_idx"],
                delta_q,
                delta_q.shape[1],
            )

        pair = pair_numerator / pair_count.clamp_min(1e-8)
        gain = gain_numerator / gain_count.clamp_min(1)
        bind = bind_numerator / bind_count.clamp_min(1)
        dpp_raw = dpp_numerator / dpp_count.clamp_min(1)
        rel_ortho = self.relation.orthogonality_loss() if self.config.rel_ortho_enabled else zero

        concept_loss = zero
        concept_metrics = {
            "concept_positive_recall": zero.detach(),
            "concept_negative_false_positive_rate": zero.detach(),
            "instruction_concept_coverage": zero.detach(),
        }
        if self.config.concept_enabled:
            if not steps or modification_texts is None:
                raise ValueError("concept loss needs t=0 proposals and modification_texts")
            assert self.concept is not None
            concept = self.concept(steps[0]["proposals"], modification_texts)
            concept_loss = concept["loss"]
            concept_metrics = {name: concept[name] for name in concept_metrics}

        relation_metrics = {
            "prototype_occupancy": zero.detach(),
            "prototype_entropy": zero.detach(),
            "prototype_pairwise_cosine": zero.detach(),
        }
        if binding_distributions:
            assert self.relation is not None
            relation_metrics = self.relation.diagnostics(torch.cat(binding_distributions, dim=0))
        functional_cosine, functional_rank = _functional_diagnostics(functional_effects, zero)
        useful_count = useful_total / useful_rows.clamp_min(1)

        dpp_weighted = self.config.lambda_dpp * dpp_raw
        total = (
            self.config.terminal_weight * terminal
            + self.config.lambda_pair * pair
            + self.config.lambda_gain * gain
            + self.config.lambda_c * concept_loss
            + self.config.lambda_bind * bind
            + self.config.lambda_rel * rel_ortho
            + dpp_weighted
        )
        return {
            "terminal": terminal,
            "pair": pair,
            "gain": gain,
            "concept_loss": concept_loss,
            "bind_loss": bind,
            "rel_ortho_loss": rel_ortho,
            "dpp_raw": dpp_raw,
            "dpp_weighted": dpp_weighted,
            **concept_metrics,
            **relation_metrics,
            "functional_pairwise_cosine": functional_cosine,
            "functional_rank": functional_rank,
            "useful_candidate_count": useful_count.detach(),
            "dpp_valid_timestep_count": dpp_count.detach(),
            "kappa_dpp": query.new_tensor(self.config.kappa_dpp),
            "lambda_dpp": query.new_tensor(self.config.lambda_dpp),
            "sigma_dpp": query.new_tensor(self.config.sigma_dpp),
            "tau_dpp": query.new_tensor(self.config.tau_dpp),
            "teacher_invalid_rows": invalid_rows.detach(),
            "total": total,
        }
