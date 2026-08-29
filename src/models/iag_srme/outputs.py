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


@dataclass(slots=True)
class IAGSRMEOutput:
    final_query: Tensor  # [B,D]
    final_state: Tensor  # [B,N,d]
    anchor: Tensor  # [B,N,d]
    intents: Tensor  # [B,K,d]
    supports: Tensor  # [B,K,N]
    text_tokens: Tensor  # [B,L,d]
    text_content_mask: Tensor  # [B,L]
    reference_global: Tensor  # [B,D]
    trace: tuple[RecurrentStepOutput, ...]
    claim_logits: Tensor | None = None  # [B,K,L]
    claims: Tensor | None = None  # [B,K,L]
    factors: Tensor | None = None  # [B,K,D_f]
    auxiliary_anchor: Tensor | None = None  # [B,D_f]
