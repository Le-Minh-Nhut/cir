from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .applicability import DynamicApplicabilityGate
from .backbone import FGCLIPBackbone
from .context import GroundedEditContext
from .editor import SharedTokenEditor
from .factorization import SemanticFullQueryAnchor, StableFactorFuser
from .grounded_reader import GroundedStateReader
from .grounding import AnchorGrounder
from .intent import SemanticClaimHead, TextIntentEncoder
from .outputs import BackboneOutput, IAGSRMEOutput, RecurrentStepOutput
from .readout import TokenStateReadout
from .reproposal import DynamicIntentReproposal
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
    # Deprecated N+1 Entmax R1b-v1 fields are retained only for explicit replay rejection.
    enable_visual_null: bool = False
    visual_null_initial_logit: float = 0.0
    enable_dynamic_applicability: bool = False
    initial_applicability: float = 0.98
    grounding_normalization: str = "entmax15"
    enable_dynamic_regrounding: bool = False
    enable_dynamic_reproposal: bool = False


def architecture_generation(config: IAGSRMEConfig) -> str:
    if config.enable_dynamic_reproposal:
        return "r1c2_dynamic_current_state_reproposal_v1"
    if config.enable_dynamic_regrounding:
        return "r1c1_dynamic_current_state_reground_v1"
    if config.enable_dynamic_applicability:
        return "r1b_dynamic_applicability_gate_v2"
    if config.enable_visual_null:
        return "r1b_visual_null_entmax_v1"
    return "legacy_iag_srme"


class IAGSRMECore(nn.Module):
    """Target-free IAG-SRME recurrence over already encoded reference/text tensors."""

    def __init__(self, config: IAGSRMEConfig) -> None:
        super().__init__()
        if config.num_candidates != 4:
            raise ValueError("canonical IAG-SRME requires exactly four candidate identities")
        if config.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if config.enable_visual_null:
            raise ValueError(
                "N+1 Entmax Visual NULL is superseded and cannot be replayed as the "
                "corrected dynamic-applicability architecture"
            )
        if config.enable_dynamic_reproposal and not config.enable_dynamic_regrounding:
            raise ValueError("dynamic reproposal requires dynamic regrounding")
        if config.enable_dynamic_reproposal and config.enable_dynamic_applicability:
            raise ValueError(
                "R1c2 isolates dynamic WHAT/WHERE and cannot enable R1b applicability"
            )
        if config.enable_dynamic_regrounding and config.enable_dynamic_applicability:
            raise ValueError(
                "R1c1 isolates dynamic WHERE and cannot enable R1b applicability"
            )
        self.config = config
        self.intent_encoder = TextIntentEncoder(
            config.width, config.num_candidates, config.num_heads
        )
        self.grounder = AnchorGrounder(
            config.width,
            normalization=config.grounding_normalization,
        )
        self.reproposal = (
            DynamicIntentReproposal(config.width, config.num_heads)
            if config.enable_dynamic_reproposal
            else None
        )
        self.grounded_reader = GroundedStateReader(config.width)
        self.context_fuser = GroundedEditContext(config.width)
        self.applicability_head = (
            DynamicApplicabilityGate(config.width, config.initial_applicability)
            if config.enable_dynamic_applicability
            else None
        )
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
            "frozen_t0_what",
        }
        if control not in valid_controls:
            raise ValueError(f"unsupported rollout control: {control}")
        anchor = encoded.anchor
        if anchor.ndim != 3 or anchor.shape[-1] != self.config.width:
            raise ValueError("anchor must be [B,N,d]")
        batch_size, tokens, width = anchor.shape
        if encoded.reference_global.shape != (batch_size, self.config.retrieval_dim):
            raise ValueError("reference_global must be [B,D]")
        base_intents = self.intent_encoder(
            encoded.text_tokens, encoded.text_content_mask
        )
        # Static R0/R1a/R1b checkpoints retain one immutable-anchor grounding call.
        # R1c1 instead resolves current_supports from Z_t inside each recurrence step.
        static_supports = (
            None
            if self.config.enable_dynamic_regrounding
            else self.grounder(base_intents, anchor)
        )
        # A is immutable; assignment never changes after this point. Z starts at A.
        state = anchor
        current_query = self.readout(state, anchor, encoded.text_global, encoded.reference_global)
        live = torch.ones(batch_size, dtype=torch.bool, device=anchor.device)
        trace: list[RecurrentStepOutput] = []
        fixed_best: Tensor | None = None
        frozen_order: Tensor | None = None

        claims = None
        claim_logits = None
        if self.claim_head is not None:
            claim_logits = self.claim_head(
                base_intents, encoded.text_tokens, encoded.text_content_mask
            )
            claims = torch.sigmoid(claim_logits) * encoded.text_content_mask[:, None].to(
                claim_logits.dtype
            )
        factors = None
        auxiliary_anchor = None
        initial_supports: Tensor | None = None
        temporal_intents: list[Tensor] = []

        for timestep in range(self.config.max_steps):
            current_state = state
            query_before = current_query
            live_before = live
            current_intents = base_intents
            intent_residual = torch.zeros_like(base_intents)
            if (
                timestep > 0
                and self.reproposal is not None
                and control != "frozen_t0_what"
            ):
                current_intents, intent_residual = self.reproposal(
                    base_intents,
                    current_state,
                    anchor,
                    encoded.text_tokens,
                    encoded.text_content_mask,
                )
            temporal_intents.append(current_intents)
            current_supports = (
                self.grounder(current_intents, current_state)
                if self.config.enable_dynamic_regrounding
                else static_supports
            )
            if current_supports is None:
                raise AssertionError("grounding support was not constructed")
            if initial_supports is None:
                initial_supports = current_supports
            original, current, change = self.grounded_reader(
                current_supports, anchor, current_state
            )
            if timestep == 0 and self.factor_fuser is not None and self.auxiliary_anchor is not None:
                factors = self.factor_fuser(current_intents, original)
                auxiliary_anchor = self.auxiliary_anchor(
                    encoded.reference_global, encoded.text_semantic_global
                )
            contexts = self.context_fuser(
                current_intents, original, current, change
            )
            applicability_logits = None
            execution_confidence = None
            null_probability = None
            if self.applicability_head is not None:
                (
                    applicability_logits,
                    execution_confidence,
                    null_probability,
                ) = self.applicability_head(contexts)
            ungated_delta_z, delta_z, candidate_states = self.editor.forward_with_ungated(
                contexts,
                current_supports,
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
            scorer_supports = current_supports
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
                scorer_supports = current_supports[:, :1].expand_as(current_supports)
                if applicability_logits is not None:
                    applicability_logits = applicability_logits[:, :1].expand_as(
                        applicability_logits
                    )
                    execution_confidence = execution_confidence[:, :1].expand_as(
                        execution_confidence
                    )
                    null_probability = null_probability[:, :1].expand_as(null_probability)
                ungated_delta_z = ungated_delta_z[:, :1].expand_as(ungated_delta_z)
            elif control == "mean_candidate":
                original = original.mean(dim=1, keepdim=True).expand_as(original)
                current = current.mean(dim=1, keepdim=True).expand_as(current)
                change = change.mean(dim=1, keepdim=True).expand_as(change)
                contexts = contexts.mean(dim=1, keepdim=True).expand_as(contexts)
                delta_z = delta_z.mean(dim=1, keepdim=True).expand_as(delta_z)
                ungated_delta_z = ungated_delta_z.mean(
                    dim=1, keepdim=True
                ).expand_as(ungated_delta_z)
                candidate_states = current_state[:, None] + delta_z
                candidate_queries = self.readout(
                    candidate_states, anchor, encoded.text_global, encoded.reference_global
                )
                delta_q = candidate_queries - query_before[:, None, :]
                scorer_change = change
                scorer_supports = current_supports.mean(dim=1, keepdim=True).expand_as(
                    current_supports
                )
                if applicability_logits is not None:
                    execution_confidence = execution_confidence.mean(
                        dim=1, keepdim=True
                    ).expand_as(execution_confidence)
                    null_probability = 1.0 - execution_confidence
                    applicability_logits = torch.logit(
                        execution_confidence.float().clamp(1e-7, 1.0 - 1e-7)
                    ).to(execution_confidence.dtype)
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
                    applicability_logits=applicability_logits,
                    visual_confidence=execution_confidence,
                    visual_null_probability=null_probability,
                    ungated_delta_z=ungated_delta_z,
                    # Keep raw dynamic WHERE distinct from a diagnostic control's
                    # effective scorer support. Canonical FULL/REPEAT have equality.
                    spatial_supports=scorer_supports,
                    raw_spatial_supports=current_supports,
                    effective_spatial_supports=scorer_supports,
                    base_intents=base_intents,
                    current_intents=current_intents,
                    intent_residual=intent_residual,
                )
            )
            state = next_state
            current_query = next_query

        if initial_supports is None:
            raise AssertionError("rollout did not construct initial grounding support")
        temporal_supports = torch.stack(
            [step.raw_spatial_supports for step in trace], dim=1
        )
        return IAGSRMEOutput(
            final_query=current_query,
            final_state=state,
            anchor=anchor,
            intents=base_intents,
            # Backward-compatible alias with explicit t0 semantics.
            supports=initial_supports,
            text_tokens=encoded.text_tokens,
            text_content_mask=encoded.text_content_mask,
            reference_global=encoded.reference_global,
            trace=tuple(trace),
            claim_logits=claim_logits,
            claims=claims,
            factors=factors,
            auxiliary_anchor=auxiliary_anchor,
            conditional_supports=initial_supports,
            initial_supports=initial_supports,
            temporal_supports=temporal_supports,
            dynamic_regrounding=self.config.enable_dynamic_regrounding,
            initial_intents=base_intents,
            temporal_intents=torch.stack(temporal_intents, dim=1),
            dynamic_reproposal=self.config.enable_dynamic_reproposal,
            visual_null_probabilities=(
                torch.stack(
                    [step.visual_null_probability for step in trace], dim=1
                )
                if self.config.enable_dynamic_applicability
                else None
            ),
            visual_confidence=(
                torch.stack([step.visual_confidence for step in trace], dim=1)
                if self.config.enable_dynamic_applicability
                else None
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
