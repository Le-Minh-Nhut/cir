from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .backbone import FGCLIPBackbone
from .context import GroundedEditContext
from .editor import SharedTokenEditor
from .factorization import SemanticFullQueryAnchor, StableFactorFuser
from .grounded_reader import GroundedStateReader
from .grounding import AnchorGrounder
from .intent import SemanticClaimHead, TextIntentEncoder
from .outputs import BackboneOutput, IAGSRMEOutput, RecurrentStepOutput
from .readout import TokenStateReadout
from .scorer import ConsequenceScorer
from .selector import HardStopSelector, select_next_state


@dataclass(frozen=True, slots=True)
class IAGSRMEConfig:
    width: int = 256
    num_candidates: int = 4
    max_steps: int = 3
    num_heads: int = 8
    retrieval_dim: int = 512
    lambda_z: float = 0.1
    query_cap: float = 0.5
    selector_temperature: float = 1.0
    selector_gumbel_noise: bool = True
    enable_claim_head: bool = False
    enable_factor_head: bool = False
    factor_dim: int | None = None
    enable_visual_null: bool = False
    visual_null_initial_logit: float = 0.0
    grounding_normalization: str = "entmax15"


class IAGSRMECore(nn.Module):
    """Target-free IAG-SRME recurrence over already encoded reference/text tensors."""

    def __init__(self, config: IAGSRMEConfig) -> None:
        super().__init__()
        if config.num_candidates != 4:
            raise ValueError("canonical IAG-SRME requires exactly four candidate identities")
        if config.max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.config = config
        self.intent_encoder = TextIntentEncoder(
            config.width, config.num_candidates, config.num_heads
        )
        self.grounder = AnchorGrounder(
            config.width,
            enable_visual_null=config.enable_visual_null,
            visual_null_initial_logit=config.visual_null_initial_logit,
            normalization=config.grounding_normalization,
        )
        self.grounded_reader = GroundedStateReader(config.width)
        self.context_fuser = GroundedEditContext(config.width)
        self.editor = SharedTokenEditor(config.width, config.lambda_z)
        self.readout = TokenStateReadout(config.width, config.retrieval_dim, config.query_cap)
        self.scorer = ConsequenceScorer(config.width, config.retrieval_dim)
        self.selector = HardStopSelector(config.selector_temperature, config.selector_gumbel_noise)
        self.claim_head = SemanticClaimHead(config.width) if config.enable_claim_head else None
        factor_dim = config.retrieval_dim if config.factor_dim is None else config.factor_dim
        if config.enable_factor_head and factor_dim != config.retrieval_dim:
            raise ValueError(
                "factor_dim must equal FG-CLIP retrieval_dim: no untrained projection is "
                "permitted in the detached semantic-anchor path"
            )
        self.factor_fuser = (
            StableFactorFuser(config.width, factor_dim)
            if config.enable_factor_head
            else None
        )
        self.auxiliary_anchor = (
            SemanticFullQueryAnchor()
            if config.enable_factor_head
            else None
        )

    def forward(self, encoded: BackboneOutput, control: str = "full") -> IAGSRMEOutput:
        valid_controls = {
            "full",
            "zero_edit",
            "single_candidate",
            "repeat_candidate_1",
            "repeat_candidate_2",
            "repeat_candidate_3",
            "repeat_candidate_4",
            "repeat_best",
            "clone_candidate_1",
            "mean_candidate",
            "random_candidate",
            "frozen_t0_order",
        }
        if control not in valid_controls:
            raise ValueError(f"unsupported rollout control: {control}")
        anchor = encoded.anchor
        if anchor.ndim != 3 or anchor.shape[-1] != self.config.width:
            raise ValueError("anchor must be [B,N,d]")
        batch_size, tokens, width = anchor.shape
        if encoded.reference_global.shape != (batch_size, self.config.retrieval_dim):
            raise ValueError("reference_global must be [B,D]")
        intents = self.intent_encoder(encoded.text_tokens, encoded.text_content_mask)
        grounding = self.grounder(intents, anchor)
        visual_supports = grounding.visual_supports
        spatial_supports = grounding.conditional_supports
        execution_confidence = (
            grounding.visual_confidence if self.config.enable_visual_null else None
        )
        # A is immutable; assignment never changes after this point. Z starts at A.
        state = anchor
        current_query = self.readout(state, anchor, encoded.text_global, encoded.reference_global)
        live = torch.ones(batch_size, dtype=torch.bool, device=anchor.device)
        trace: list[RecurrentStepOutput] = []
        fixed_best: Tensor | None = None
        frozen_order: Tensor | None = None

        original_static, _, _ = self.grounded_reader(spatial_supports, anchor, anchor)
        claims = None
        claim_logits = None
        if self.claim_head is not None:
            claim_logits = self.claim_head(intents, encoded.text_tokens, encoded.text_content_mask)
            claims = torch.sigmoid(claim_logits) * encoded.text_content_mask[:, None].to(
                claim_logits.dtype
            )
        factors = None
        auxiliary_anchor = None
        if self.factor_fuser is not None and self.auxiliary_anchor is not None:
            factors = self.factor_fuser(intents, original_static)
            auxiliary_anchor = self.auxiliary_anchor(
                encoded.reference_global, encoded.text_semantic_global
            )

        for timestep in range(self.config.max_steps):
            current_state = state
            query_before = current_query
            live_before = live
            original, current, change = self.grounded_reader(
                spatial_supports, anchor, current_state
            )
            contexts = self.context_fuser(intents, original, current, change)
            delta_z, candidate_states = self.editor(
                contexts,
                spatial_supports,
                anchor,
                current_state,
                execution_confidence=execution_confidence,
            )
            expected_shape = (
                batch_size,
                self.config.num_candidates,
                tokens,
                width,
            )
            if candidate_states.shape != expected_shape:
                raise AssertionError("same-parent candidate shape invariant failed")
            # This explicit expression is the scientific same-parent contract.
            if not torch.equal(candidate_states, current_state.unsqueeze(1) + delta_z):
                raise AssertionError(
                    "all counterfactual candidates must branch from the same parent"
                )
            candidate_queries = self.readout(
                candidate_states, anchor, encoded.text_global, encoded.reference_global
            )
            delta_q = candidate_queries - query_before[:, None, :]
            scorer_change = change
            scorer_supports = spatial_supports
            if control == "clone_candidate_1":
                original = original[:, :1].expand_as(original)
                current = current[:, :1].expand_as(current)
                change = change[:, :1].expand_as(change)
                contexts = contexts[:, :1].expand_as(contexts)
                delta_z = delta_z[:, :1].expand_as(delta_z)
                candidate_states = candidate_states[:, :1].expand_as(candidate_states)
                candidate_queries = candidate_queries[:, :1].expand_as(candidate_queries)
                delta_q = delta_q[:, :1].expand_as(delta_q)
                scorer_change = change
                scorer_supports = spatial_supports[:, :1].expand_as(spatial_supports)
            elif control == "mean_candidate":
                original = original.mean(dim=1, keepdim=True).expand_as(original)
                current = current.mean(dim=1, keepdim=True).expand_as(current)
                change = change.mean(dim=1, keepdim=True).expand_as(change)
                contexts = contexts.mean(dim=1, keepdim=True).expand_as(contexts)
                delta_z = delta_z.mean(dim=1, keepdim=True).expand_as(delta_z)
                candidate_states = current_state[:, None] + delta_z
                candidate_queries = self.readout(
                    candidate_states, anchor, encoded.text_global, encoded.reference_global
                )
                delta_q = candidate_queries - query_before[:, None, :]
                scorer_change = change
                scorer_supports = spatial_supports.mean(dim=1, keepdim=True).expand_as(
                    spatial_supports
                )
            scores = self.scorer(
                contexts, delta_z, delta_q, scorer_change, scorer_supports
            )
            stop_score = torch.zeros(batch_size, 1, dtype=scores.dtype, device=scores.device)
            logits = torch.cat([scores, stop_score], dim=-1)
            if control == "full":
                action_st, action_hard = self.selector(logits, live_before)
            else:
                if timestep == 0:
                    fixed_best = scores.argmax(dim=-1)
                    frozen_order = scores.argsort(dim=-1, descending=True)
                if control == "zero_edit" or (control == "single_candidate" and timestep > 0):
                    forced = torch.full_like(fixed_best, self.config.num_candidates)
                elif control == "single_candidate":
                    forced = fixed_best
                elif control.startswith("repeat_candidate_"):
                    forced = torch.full_like(fixed_best, int(control[-1]) - 1)
                elif control == "repeat_best":
                    forced = fixed_best
                elif control == "frozen_t0_order":
                    if frozen_order is None:
                        raise AssertionError("frozen order was not initialized")
                    forced = frozen_order[:, min(timestep, self.config.num_candidates - 1)]
                elif control == "random_candidate":
                    forced = torch.randint(
                        self.config.num_candidates, (batch_size,), device=anchor.device
                    )
                else:  # clone/mean: preserve the normal target-free selection policy.
                    forced = logits.argmax(dim=-1)
                forced = torch.where(
                    live_before,
                    forced,
                    torch.full_like(forced, self.config.num_candidates),
                )
                action_hard = torch.nn.functional.one_hot(
                    forced, self.config.num_candidates + 1
                ).to(logits.dtype)
                action_st = action_hard
            next_state, next_query = select_next_state(
                candidate_states, current_state, candidate_queries, query_before, action_st
            )
            selected_index = action_hard.argmax(dim=-1)
            stopped_now = live_before & selected_index.eq(self.config.num_candidates)
            live = live_before & ~stopped_now
            trace.append(
                RecurrentStepOutput(
                    timestep=timestep,
                    live_before=live_before,
                    current_state=current_state,
                    current_query=query_before,
                    original_evidence=original,
                    current_evidence=current,
                    accumulated_local_change=change,
                    contexts=contexts,
                    delta_z=delta_z,
                    candidate_states=candidate_states,
                    candidate_queries=candidate_queries,
                    delta_q=delta_q,
                    scores=scores,
                    logits_with_stop=logits,
                    action_st=action_st,
                    action_hard=action_hard,
                    selected_index=selected_index,
                    stopped_now=stopped_now,
                    next_state=next_state,
                    next_query=next_query,
                )
            )
            state = next_state
            current_query = next_query

        return IAGSRMEOutput(
            final_query=current_query,
            final_state=state,
            anchor=anchor,
            intents=intents,
            supports=visual_supports,
            text_tokens=encoded.text_tokens,
            text_content_mask=encoded.text_content_mask,
            reference_global=encoded.reference_global,
            trace=tuple(trace),
            claim_logits=claim_logits,
            claims=claims,
            factors=factors,
            auxiliary_anchor=auxiliary_anchor,
            conditional_supports=spatial_supports,
            visual_null_probabilities=(
                grounding.null_probabilities if self.config.enable_visual_null else None
            ),
            visual_confidence=(
                grounding.visual_confidence if self.config.enable_visual_null else None
            ),
        )


class IAGSRME(nn.Module):
    """Raw reference+text public model. Target tensors are intentionally absent."""

    def __init__(self, backbone: FGCLIPBackbone, core: IAGSRMECore) -> None:
        super().__init__()
        if backbone.internal_width != core.config.width:
            raise ValueError("backbone and core internal widths differ")
        if backbone.retrieval_dim != core.config.retrieval_dim:
            raise ValueError("backbone and core retrieval dimensions differ")
        self.backbone = backbone
        self.core = core

    def forward(
        self,
        reference_pixels: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        content_mask: Tensor,
        control: str = "full",
    ) -> IAGSRMEOutput:
        encoded = self.backbone(reference_pixels, input_ids, attention_mask, content_mask)
        return self.core(encoded, control=control)

    def encode_global_images(self, pixel_values: Tensor) -> Tensor:
        return self.backbone.encode_global_images(pixel_values)

    def encode_gallery(self, pixel_values: Tensor) -> Tensor:
        """Backward-compatible generic gallery name; always global-only."""

        return self.encode_global_images(pixel_values)
