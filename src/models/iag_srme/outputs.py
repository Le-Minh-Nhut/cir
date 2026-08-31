from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(slots=True)
class BackboneOutput:
    """Target-free reference/text encoding contract."""

    anchor: Tensor  # [B,N,d]
    reference_global: Tensor  # [B,D]
    text_tokens: Tensor  # [B,L,d]
    text_global: Tensor  # [B,d]
    text_semantic_global: Tensor  # [B,D], official FG-CLIP retrieval projection
    text_content_mask: Tensor  # [B,L]


@dataclass(slots=True)
class RecurrentStepOutput:
    timestep: int
    live_before: Tensor  # [B] bool
    current_state: Tensor  # [B,N,d]
    current_query: Tensor  # [B,D]
    original_evidence: Tensor  # [B,K,d]
    current_evidence: Tensor  # [B,K,d]
    accumulated_local_change: Tensor  # [B,K,d]
    contexts: Tensor  # [B,K,d]
    delta_z: Tensor  # [B,K,N,d]
    candidate_states: Tensor  # [B,K,N,d]
    candidate_queries: Tensor  # [B,K,D]
    delta_q: Tensor  # [B,K,D]
    scores: Tensor  # [B,K]
    logits_with_stop: Tensor  # [B,K+1]
    action_st: Tensor  # [B,K+1]
    action_hard: Tensor  # [B,K+1]
    selected_index: Tensor  # [B]
    stopped_now: Tensor  # [B] bool
    next_state: Tensor  # [B,N,d]
    next_query: Tensor  # [B,D]
    applicability_logits: Tensor | None = None  # [B,K]
    visual_confidence: Tensor | None = None  # [B,K]
    visual_null_probability: Tensor | None = None  # [B,K] = 1-confidence
    ungated_delta_z: Tensor | None = None  # [B,K,N,d], before applicability
    # Backward-compatible alias of effective_spatial_supports. Canonical FULL and
    # REPEAT equal raw Ground(I,Z_t); CLONE/MEAN may expose a controlled view here.
    spatial_supports: Tensor | None = None  # [B,K,N]
    raw_spatial_supports: Tensor | None = None  # [B,K,N], Ground(I, Z_t)
    effective_spatial_supports: Tensor | None = None  # [B,K,N], scorer/control view
    base_intents: Tensor | None = None  # [B,K,d], immutable text-only WHAT
    current_intents: Tensor | None = None  # [B,K,d], raw WHAT used at this step
    intent_residual: Tensor | None = None  # [B,K,d], current_intents-base_intents
    parent_semantic_residual: Tensor | None = None  # [B,L], shared rho_t
    raw_semantic_claims: Tensor | None = None  # [B,K,L], before diagnostic controls
    effective_semantic_claims: Tensor | None = None  # [B,K,L], executable claims
    semantic_consumption: Tensor | None = None  # [B,K,L], raw gamma
    effective_semantic_consumption: Tensor | None = None  # [B,K,L], alpha*gamma
    claimed_semantic_mass: Tensor | None = None  # [B,K]
    claimed_semantic_direction: Tensor | None = None  # [B,K,d]
    claimed_text_content: Tensor | None = None  # [B,K,d]
    candidate_semantic_residuals: Tensor | None = None  # [B,K,L]
    next_semantic_residual: Tensor | None = None  # [B,L]
    selected_semantic_consumption: Tensor | None = None  # [B,L]


@dataclass(slots=True)
class IAGSRMEOutput:
    final_query: Tensor  # [B,D]
    final_state: Tensor  # [B,N,d]
    anchor: Tensor  # [B,N,d]
    intents: Tensor  # [B,K,d]
    supports: Tensor  # [B,K,N], backward-compatible t0/initial WHERE
    text_tokens: Tensor  # [B,L,d]
    text_content_mask: Tensor  # [B,L]
    reference_global: Tensor  # [B,D]
    trace: tuple[RecurrentStepOutput, ...]
    claim_logits: Tensor | None = None  # [B,K,L]
    claims: Tensor | None = None  # [B,K,L]
    factors: Tensor | None = None  # [B,K,D_f]
    auxiliary_anchor: Tensor | None = None  # [B,D_f]
    conditional_supports: Tensor | None = None  # legacy diagnostic alias of supports
    visual_null_probabilities: Tensor | None = None  # [B,T,K], dynamic p_null
    visual_confidence: Tensor | None = None  # [B,T,K], dynamic confidence
    initial_supports: Tensor | None = None  # [B,K,N], explicit t0 WHERE
    temporal_supports: Tensor | None = None  # [B,T,K,N]
    dynamic_regrounding: bool = False
    initial_intents: Tensor | None = None  # [B,K,d], explicit immutable base WHAT
    temporal_intents: Tensor | None = None  # [B,T,K,d], raw per-step WHAT
    dynamic_reproposal: bool = False
    semantic_residual_enabled: bool = False
    initial_semantic_residual: Tensor | None = None  # [B,L]
    temporal_semantic_residuals: Tensor | None = None  # [B,T+1,L]
    temporal_semantic_claims: Tensor | None = None  # [B,T,K,L], raw claims
    temporal_semantic_consumption: Tensor | None = None  # [B,T,K,L], raw gamma
    temporal_effective_semantic_consumption: Tensor | None = None  # [B,T,K,L]
    final_semantic_residual: Tensor | None = None  # [B,L]
