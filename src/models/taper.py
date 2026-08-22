from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TAPER(nn.Module):
    def __init__(
        self,
        teacher: nn.Module,
        *,
        text_dim: int,
        reference_dim: int,
        teacher_text_dim: int | None = None,
        teacher_query_dim: int,
        query_dim: int,
        slot_dim: int = 512,
        state_dim: int = 512,
        num_slots: int = 4,
        num_primitives: int = 8,
        mask_temperature: float = 1.0,
        router_temperature: float = 1.0,
        retrieval_temperature: float = 0.07,
        neutral_mode: str = "zero",
        slot_gate_threshold: float = 0.5,
        hard_slot_gating_during_training: bool = False,
        gate_mode: str = "legacy_soft_train_hard_eval",
        st_gate_recovery: bool = False,
        overlap_margin: float = 0.60,
        effect_diversity_margin: float = 0.50,
        alpha_max: float = 1.0,
        counterfactual_chunk_size: int = 8,
        num_refine_iters: int = 3
    ) -> None:
        super().__init__()

        if num_refine_iters < 1:
            raise ValueError("num_refine_iters must be >= 1")
        if num_slots < 1 or num_primitives < 1:
            raise ValueError("num_slots and num_primitives must be >= 1")
        if min(mask_temperature, router_temperature, retrieval_temperature) <= 0:
            raise ValueError("all temperatures must be > 0")
        if neutral_mode not in {"zero", "mean", "learned"}:
            raise ValueError("neutral_mode must be: zero, mean, or learned")
        if not 0.0 <= slot_gate_threshold <= 1.0:
            raise ValueError("slot_gate_threshold must be in [0, 1]")
        if alpha_max <= 0:
            raise ValueError("alpha_max must be > 0")
        if gate_mode not in {"legacy_soft_train_hard_eval", "straight_through_hard"}:
            raise ValueError("gate_mode must be 'legacy_soft_train_hard_eval' or " "'straight_through_hard'")
        if counterfactual_chunk_size < 1:
            raise ValueError("counterfactual_chunk_size must be >= 1")

        self.teacher = teacher
        self.text_dim = text_dim
        self.teacher_text_dim = text_dim if teacher_text_dim is None else teacher_text_dim
        if self.teacher_text_dim < 1:
            raise ValueError("teacher_text_dim must be >= 1")
        self.reference_dim = reference_dim
        self.teacher_query_dim = teacher_query_dim
        self.query_dim = query_dim
        self.slot_dim = slot_dim
        self.state_dim = state_dim
        self.num_slots = num_slots
        self.num_primitives = num_primitives

        self.mask_temperature = mask_temperature
        self.router_temperature = router_temperature
        self.retrieval_temperature = retrieval_temperature
        self.neutral_mode = neutral_mode
        self.slot_gate_threshold = slot_gate_threshold
        self.hard_slot_gating_during_training = hard_slot_gating_during_training
        self.gate_mode = gate_mode
        self.st_gate_recovery = bool(st_gate_recovery)
        self.overlap_margin = overlap_margin
        self.effect_diversity_margin = effect_diversity_margin
        self.alpha_max = alpha_max
        self.counterfactual_chunk_size = counterfactual_chunk_size

        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

        # Ephemeral Edit Slot queries + one non-executable NULL competitor.
        self.slot_queries = nn.Parameter(torch.randn(num_slots, slot_dim) * 0.02)
        self.null_query = nn.Parameter(torch.randn(1, slot_dim) * 0.02)
        self.slot_query_projection = nn.Linear(slot_dim, slot_dim, bias=False)
        self.text_key_projection = nn.Linear(text_dim, slot_dim, bias=False)

        if neutral_mode == "learned":
            self.neutral_embedding = nn.Parameter(torch.zeros(self.teacher_text_dim))
        else:
            self.register_buffer("neutral_embedding", torch.zeros(self.teacher_text_dim))

        self.slot_mlp = nn.Sequential(nn.Linear(text_dim + teacher_query_dim, slot_dim), nn.GELU(), nn.Linear(slot_dim, slot_dim), nn.LayerNorm(slot_dim))

        self.slot_gate = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim), nn.GELU(), nn.Linear(slot_dim, 1))
        nn.init.constant_(self.slot_gate[-1].bias, 1.0)

        self.reference_to_state = nn.Sequential(nn.Linear(reference_dim, state_dim), nn.GELU(), nn.Linear(state_dim, state_dim), nn.LayerNorm(state_dim))

        self.primitive_bank = nn.Parameter(torch.randn(num_primitives, state_dim) * 0.02)
        router_dim = state_dim + slot_dim + state_dim + state_dim
        self.router = nn.Sequential(nn.Linear(router_dim, state_dim), nn.GELU(), nn.Linear(state_dim, 1))

        transition_dim = state_dim + slot_dim + state_dim + state_dim
        self.transition_delta = nn.Sequential(nn.Linear(transition_dim, state_dim), nn.GELU(), nn.Linear(state_dim, state_dim))
        self.transition_strength = nn.Sequential(nn.Linear(transition_dim, state_dim), nn.GELU(), nn.Linear(state_dim, 1), nn.Sigmoid())
        self.state_norm = nn.LayerNorm(state_dim)
        # TAPER V3 contract: valid controlled residual transitions are followed by LN.
        # Invalid steps bypass this module and preserve the previous state exactly.
        # self.state_update_norm = nn.LayerNorm(state_dim)

        self.num_refine_iters = num_refine_iters
        self.text_value_projection = nn.Linear(text_dim, slot_dim, bias=False,)
        self.slot_update = nn.GRUCell(input_size=slot_dim, hidden_size=slot_dim)

        self.query_head = nn.Sequential(nn.Linear(state_dim, query_dim), nn.GELU(), nn.Linear(query_dim, query_dim))

    def train(self, mode: bool = True) -> "TAPER":
        super().train(mode)
        # The teacher is a frozen functional measuring instrument.
        self.teacher.eval()
        return self

    def _pool_text(self, text_states: Tensor, mask: Tensor) -> Tensor:
        mask_f = mask.to(text_states.dtype).unsqueeze(-1)
        return (text_states * mask_f).sum(1) / mask_f.sum(1).clamp_min(1.0)

    def _neutral_text(self, text_states: Tensor, mask: Tensor) -> Tensor:
        if self.neutral_mode in {"zero", "learned"}:
            return self.neutral_embedding.view(1, 1, -1).expand_as(text_states)
        return self._pool_text(text_states, mask).unsqueeze(1).expand_as(text_states)

    @staticmethod
    def _gather_slots(slots: Tensor, slot_ids: Tensor) -> Tensor:
        batch_ids = torch.arange(slots.shape[0], device=slots.device)
        return slots[batch_ids, slot_ids]

    def _validate_text_inputs(
        self,
        text_states: Tensor,
        text_attention_mask: Tensor,
        text_content_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if text_states.ndim != 3 or text_attention_mask.ndim != 2:
            raise ValueError("text_states must be [B,N,D] and mask must be [B,N]")
        if text_states.shape[:2] != text_attention_mask.shape:
            raise ValueError("text_states and text_attention_mask do not match")
        if text_states.shape[1] < 1:
            raise ValueError("text sequence length must be >= 1")
        if text_states.shape[-1] != self.text_dim:
            raise ValueError(f"text dim must be {self.text_dim}")
        if not torch.isfinite(text_states).all():
            raise ValueError("text_states contains NaN or Inf")

        attention_valid = text_attention_mask.to(torch.bool)
        if text_content_mask is None:
            raise ValueError("text_content_mask is required for competitive NULL ownership so " "special/content tokens are excluded explicitly")
        if text_content_mask.shape != text_attention_mask.shape:
            raise ValueError("text_content_mask must match text_attention_mask")
        slot_valid = attention_valid & text_content_mask.to(torch.bool)

        return attention_valid, slot_valid

    def _competitive_ownership(self, text_states: Tensor, slot_valid: Tensor, slot_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = text_states.shape[0]

        if slot_states.shape != (batch_size, self.num_slots, self.slot_dim):
            raise ValueError(f"slot_states must be [B,{self.num_slots},{self.slot_dim}], got {tuple(slot_states.shape)}")

        null_states = self.null_query.unsqueeze(0).expand(batch_size, -1, -1)  # [B, 1, Ds]
        all_states = torch.cat([null_states, slot_states], dim=1)  # [B, L+1, Ds]
        queries = self.slot_query_projection(all_states)  # [B, L+1, Ds]
        keys = self.text_key_projection(text_states)  # [B, N, Ds]
        logits = torch.einsum("bld,bnd->bln", queries, keys)
        logits = logits / math.sqrt(self.slot_dim) / self.mask_temperature
        valid = slot_valid[:, None, :]
        null_logits = torch.where(slot_valid, logits[:, 0, :], torch.zeros_like(logits[:, 0, :]))
        invalid_logit = torch.finfo(logits.dtype).min
        edit_logits = logits[:, 1:, :].masked_fill(~valid, invalid_logit)
        ownership_logits = torch.cat([null_logits[:, None, :], edit_logits], dim=1)
        ownership = F.softmax(ownership_logits, dim=1)
        if not torch.isfinite(ownership).all():
            raise FloatingPointError("non-finite competitive slot ownership")

        null_probs = ownership[:, 0, :]
        slot_masks = ownership[:, 1:, :]

        return ownership_logits, null_probs, slot_masks

    def _refine_slot_states(self, slot_states: Tensor, text_states: Tensor, slot_masks: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = text_states.shape[0]
        if slot_states.shape != (batch_size, self.num_slots, self.slot_dim):
            raise ValueError("slot_states has invalid shape")

        expected_slot_mask_shape = (batch_size, self.num_slots, text_states.shape[1])

        if slot_masks.shape != expected_slot_mask_shape:
            raise ValueError(f"slot_masks must be {expected_slot_mask_shape}, got {tuple(slot_masks.shape)}")

        text_values = self.text_value_projection(text_states)  # [B, N, Ds]
        slot_evidence, slot_mass, slot_activity = self._mass_aware_slot_pool(text_values, slot_masks)
        if not torch.isfinite(slot_evidence).all():
            raise FloatingPointError("non-finite iterative slot evidence")
        b, l, d = slot_states.shape
        candidate_states = self.slot_update(slot_evidence.reshape(b * l, d), slot_states.reshape(b * l, d)).reshape(b, l, d)
        active = slot_mass > 0
        next_slot_states = torch.where(active.unsqueeze(-1), candidate_states, slot_states,)
        if not torch.isfinite(next_slot_states).all():
            raise FloatingPointError("non-finite refined slot states")

        return (next_slot_states, slot_mass, slot_activity)

    @staticmethod
    def _mass_aware_slot_pool(
        text_states: Tensor,
        slot_masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        slot_mass = slot_masks.sum(dim=2)  # [B, L], expected owned-token count
        weighted_sum = torch.einsum("bln,bnd->bld", slot_masks, text_states)
        slot_activity = slot_mass.clamp(max=1.0)
        slot_semantics = weighted_sum / slot_mass.clamp_min(1.0).unsqueeze(-1)
        return slot_semantics, slot_mass, slot_activity

    def _compose_counterfactual_queries(
        self,
        *,
        teacher_reference_features: Tensor,
        teacher_text_states: Tensor,
        text_attention_mask: Tensor,
        slot_masks: Tensor,
        neutral: Tensor,
    ) -> Tensor:
        batch_size, num_tokens, _ = teacher_text_states.shape

        total = batch_size * self.num_slots
        outputs = []

        for start in range(0, total, self.counterfactual_chunk_size,):
            end = min(start + self.counterfactual_chunk_size, total)
            flat_ids = torch.arange(start, end, device=teacher_text_states.device)
            batch_ids = torch.div(flat_ids, self.num_slots, rounding_mode="floor")
            slot_ids = flat_ids % self.num_slots
            chunk_masks = slot_masks[batch_ids, slot_ids, :].unsqueeze(-1)
            chunk_teacher_text = teacher_text_states[batch_ids]
            chunk_neutral = neutral[batch_ids]
            counterfactual_text = ((1.0 - chunk_masks) * chunk_teacher_text + chunk_masks * chunk_neutral)
            counterfactual_reference = teacher_reference_features[batch_ids]
            counterfactual_mask = text_attention_mask[batch_ids]
            q_chunk = self.teacher.compose(
                counterfactual_reference,
                counterfactual_text,
                counterfactual_mask,
                normalize=False,
            )

            expected_chunk_shape = (end - start, self.teacher_query_dim)

            if q_chunk.shape != expected_chunk_shape:
                raise ValueError(
                    "counterfactual teacher query chunk "
                    f"must be {expected_chunk_shape}, "
                    f"got {tuple(q_chunk.shape)}"
                )

            outputs.append(q_chunk)

        q_minus_flat = torch.cat(outputs, dim=0)
        expected_shape = (total, self.teacher_query_dim)

        if q_minus_flat.shape != expected_shape:
            raise ValueError(
                "counterfactual teacher query must be "
                f"{expected_shape}, "
                f"got {tuple(q_minus_flat.shape)}"
            )

        return q_minus_flat.reshape(batch_size, self.num_slots, self.teacher_query_dim)

    def build_edit_slots(
        self,
        reference_features: Tensor,
        text_states: Tensor,
        text_attention_mask: Tensor,
        text_content_mask: Tensor | None = None,
        *,
        teacher_reference_features: Tensor | None = None,
        teacher_text_states: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if reference_features.ndim != 2:
            raise ValueError("reference_features must be [B,D] for TAPER state initialization")
        if reference_features.shape[0] != text_states.shape[0]:
            raise ValueError("reference_features and text_states batch sizes do not match")
        if teacher_reference_features is None:
            raise ValueError("teacher_reference_features is required explicitly; do not silently " "reuse TAPER reference_features for the frozen teacher")
        if teacher_text_states is None:
            raise ValueError("teacher_text_states is required explicitly; do not silently reuse " "contextual TAPER text_states as teacher-native inputs")
        if teacher_reference_features.shape[0] != text_states.shape[0]:
            raise ValueError("teacher_reference_features batch size does not match text_states")
        if teacher_text_states.ndim != 3 or teacher_text_states.shape[:2] != text_states.shape[:2]:
            raise ValueError("teacher_text_states must be [B,N,D_teacher] aligned with text_states")
        if teacher_text_states.shape[-1] != self.teacher_text_dim:
            raise ValueError(f"teacher text dim must be {self.teacher_text_dim}")
        if not torch.isfinite(teacher_text_states).all():
            raise ValueError("teacher_text_states contains NaN or Inf")
        if not torch.isfinite(teacher_reference_features).all():
            raise ValueError("teacher_reference_features contains NaN or Inf")

        _, slot_valid = self._validate_text_inputs(text_states, text_attention_mask, text_content_mask)

        batch_size, num_tokens, _ = text_states.shape
        slot_states = self.slot_queries.unsqueeze(0).expand(batch_size, -1,-1)
        refine_slot_states = []
        refine_slot_masses = []
        refine_update_norms = []
        for refine_idx in range(self.num_refine_iters):
            refine_slot_states.append(slot_states)
            ownership_logits, null_probs, slot_masks = self._competitive_ownership(text_states=text_states, slot_valid=slot_valid, slot_states=slot_states)
            refine_slot_masses.append(slot_masks.sum(dim=2))
            if refine_idx + 1 < self.num_refine_iters:
                old_states = slot_states
                slot_states, _, _ = self._refine_slot_states(
                    slot_states=slot_states,
                    text_states=text_states,
                    slot_masks=slot_masks,
                )

                refine_update_norms.append((slot_states - old_states).norm(dim=-1))
        
        slot_semantics, slot_mass, slot_activity = self._mass_aware_slot_pool(text_states, slot_masks)
        q_full = self.teacher.compose(teacher_reference_features, teacher_text_states, text_attention_mask, normalize=False)
        expected_shape = (batch_size, self.teacher_query_dim)
        if q_full.shape != expected_shape:
            raise ValueError(f"teacher query must be {expected_shape}, got {tuple(q_full.shape)}")

        neutral = self._neutral_text(teacher_text_states, text_attention_mask)
        q_minus = self._compose_counterfactual_queries(
            teacher_reference_features=teacher_reference_features,
            teacher_text_states=teacher_text_states,
            text_attention_mask=text_attention_mask,
            slot_masks=slot_masks,
            neutral=neutral,
        )

        slot_effects = q_full.unsqueeze(1) - q_minus

        raw_edit_slots = self.slot_mlp(torch.cat([slot_semantics, slot_effects], dim=-1))
        edit_slots = raw_edit_slots * slot_activity.unsqueeze(-1)

        slot_gate_logits = self.slot_gate(edit_slots).squeeze(-1)
        raw_slot_gates = torch.sigmoid(slot_gate_logits)
        slot_gates = torch.where(slot_activity > 0, raw_slot_gates, torch.zeros_like(raw_slot_gates))
        if refine_update_norms:
            refine_update_norms_tensor = torch.stack(refine_update_norms, dim=1)
        else:
            refine_update_norms_tensor = slot_states.new_empty(batch_size, 0, self.num_slots)
        return {
            "edit_slots": edit_slots,
            "raw_edit_slots": raw_edit_slots,
            "ownership_logits": ownership_logits,
            "null_probs": null_probs,
            "slot_masks": slot_masks,
            "slot_semantics": slot_semantics,
            "slot_mass": slot_mass,
            "slot_activity": slot_activity,
            "slot_peak_ownership": slot_masks.amax(dim=2),
            "slot_effects": slot_effects,
            "slot_gate_logits": slot_gate_logits,
            "raw_slot_gates": raw_slot_gates,
            "slot_gates": slot_gates,
            "q_teacher_full": q_full,
            "q_teacher_minus": q_minus,
            "refine_slot_states": torch.stack(refine_slot_states, dim=1),  # [B,T,L,D]
            "refine_update_norms": torch.stack(refine_update_norms, dim=1),  # [B,T-1,L]
            "refine_slot_masses": torch.stack(refine_slot_masses, dim=1),
            "refine_update_norms": refine_update_norms_tensor,
        }

    def initialize_state(self, reference_features: Tensor) -> tuple[Tensor, Tensor]:
        if reference_features.ndim != 2:
            raise ValueError("reference_features must be [B,D]")
        if reference_features.shape[-1] != self.reference_dim:
            raise ValueError(f"reference dim must be {self.reference_dim}")

        reference_state = self.reference_to_state(reference_features)
        z0 = reference_state
        return z0, reference_state

    def _joint_router_scores(
        self,
        state: Tensor,
        edit_slots: Tensor,
        reference_state: Tensor,
    ) -> Tensor:
        """Score every candidate active-slot x primitive pair [B,L,K]."""

        b, l, _ = edit_slots.shape
        k = self.num_primitives

        state_x = state[:, None, None, :].expand(b, l, k, -1)
        slot_x = edit_slots[:, :, None, :].expand(b, l, k, -1)
        primitive_x = self.primitive_bank[None, None, :, :].expand(b, l, k, -1)
        reference_x = reference_state[:, None, None, :].expand(b, l, k, -1)

        x = torch.cat([state_x, slot_x, primitive_x, reference_x], dim=-1)
        return self.router(x).squeeze(-1)

    def _transition(
        self,
        state: Tensor,
        slot: Tensor,
        primitive: Tensor,
        reference_state: Tensor,
        selected_slot_gate: Tensor,
        valid_step: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        state_context = self.state_norm(state)
        x = torch.cat([state_context, slot, primitive, reference_state], dim=-1)

        proposed_delta = self.transition_delta(x)
        alpha = self.transition_strength(x).squeeze(-1) * self.alpha_max
        effective_strength = alpha * selected_slot_gate
        state_update = effective_strength[:, None] * proposed_delta

        # TAPER V3: valid transitions are normalized after the controlled residual;
        # invalid steps preserve state exactly and do not reapply LayerNorm.
        # proposed_next = self.state_update_norm(state + state_update)
        # next_state = torch.where(valid_step[:, None], proposed_next, state)
        proposed_next = state + state_update
        next_state = torch.where(valid_step[:, None], proposed_next, state)
        actual_change = next_state - state
        proposed_delta = torch.where(valid_step[:, None], proposed_delta, torch.zeros_like(proposed_delta))
        alpha = torch.where(valid_step, alpha, torch.zeros_like(alpha))

        return next_state, proposed_delta, alpha, actual_change

    def _st_gate_recovery_shadow(
        self,
        state: Tensor,
        edit_slots: Tensor,
        slot_gates: Tensor,
        reference_state: Tensor,
        available_slots: Tensor,
    ) -> Tensor:
        if available_slots.shape != slot_gates.shape:
            raise ValueError("available_slots must match slot_gates")

        available_f = available_slots.to(slot_gates.dtype)
        has_available = available_slots.any(dim=1)
        if not has_available.any():
            return state.detach()

        with torch.no_grad():
            router_scores = self._joint_router_scores(state.detach(), edit_slots.detach(), reference_state.detach())

        gate_eps = 1e-6
        shadow_logits = router_scores + torch.log(slot_gates.clamp_min(gate_eps))[:, :, None]
        shadow_logits = shadow_logits.masked_fill(~available_slots[:, :, None], -1e4)
        conditional_route = F.softmax(shadow_logits.reshape(state.shape[0], -1) / self.router_temperature, dim=-1).reshape(state.shape[0], self.num_slots, self.num_primitives)
        conditional_route = conditional_route * has_available[:, None, None].to(conditional_route.dtype)

        # Smooth probability that at least one currently available gate participates.
        soft_any = 1.0 - torch.prod(1.0 - slot_gates.clamp(0.0, 1.0) * available_f, dim=1)
        shadow_route = conditional_route * soft_any[:, None, None]

        selected_slot = torch.einsum("bl,bld->bd", shadow_route.sum(2), edit_slots.detach())
        selected_primitive = torch.einsum("bk,kd->bd", shadow_route.sum(1), self.primitive_bank.detach())

        state_context = self.state_norm(state.detach()).detach()
        x = torch.cat([state_context, selected_slot, selected_primitive, reference_state.detach()], dim=-1)

        def detached_call(module: nn.Module, arg: Tensor) -> Tensor:
            params_and_buffers = {
                name: value.detach()
                for name, value in module.named_parameters()
            }
            params_and_buffers.update({name: value.detach() for name, value in module.named_buffers()})
            return torch.func.functional_call(module, params_and_buffers, (arg,))

        proposed_delta = detached_call(self.transition_delta, x)
        alpha = (
            detached_call(self.transition_strength, x).squeeze(-1)
            * self.alpha_max
        )
        update = alpha[:, None] * proposed_delta
        # shadow_next = detached_call(self.state_update_norm, state.detach() + update)
        shadow_next = state.detach() + update
        return torch.where(has_available[:, None], shadow_next, state.detach())

    def execute(
        self,
        edit_slots: Tensor,
        slot_gates: Tensor,
        z0: Tensor,
        reference_state: Tensor,
        *,
        disabled_slots: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if edit_slots.ndim != 3 or edit_slots.shape[1] != self.num_slots:
            raise ValueError("edit_slots must be [B,num_slots,slot_dim]")
        if slot_gates.shape != edit_slots.shape[:2]:
            raise ValueError("slot_gates must be [B,num_slots]")

        b = edit_slots.shape[0]
        device = edit_slots.device
        dtype = edit_slots.dtype

        if disabled_slots is None:
            disabled_slots = torch.zeros(b, self.num_slots, dtype=torch.bool, device=device)
        else:
            if disabled_slots.shape != (b, self.num_slots):
                raise ValueError("disabled_slots must be [B,num_slots]")
            disabled_slots = disabled_slots.to(device=device, dtype=torch.bool)

        state = z0
        completed = torch.zeros(b, self.num_slots, dtype=torch.bool, device=device)
        hard_active_slots = (
            slot_gates.detach() >= self.slot_gate_threshold
        ) & ~disabled_slots
        if self.gate_mode == "straight_through_hard":
            hard_gate_values = hard_active_slots.to(dtype)
            gate_values = (
                hard_gate_values + slot_gates - slot_gates.detach()
            )
        else:
            gate_values = slot_gates
        recovery_used = torch.zeros(b, dtype=torch.bool, device=device)

        checkpoints = [state]
        slot_trace: list[Tensor] = []
        primitive_trace: list[Tensor] = []
        valid_trace: list[Tensor] = []
        active_trace: list[Tensor] = []
        selected_gate_trace: list[Tensor] = []
        route_confidences: list[Tensor] = []
        proposed_deltas: list[Tensor] = []
        actual_state_changes: list[Tensor] = []
        transition_strengths: list[Tensor] = []

        slot_to_step = torch.full((b, self.num_slots), -1, dtype=torch.long, device=device)

        for step in range(self.num_slots):
            if self.gate_mode == "straight_through_hard":
                # Forward candidate eligibility is identical in train and eval.
                candidate_mask = ~completed & hard_active_slots
            elif self.training and not self.hard_slot_gating_during_training:
                candidate_mask = (
                    ~completed
                    & ~disabled_slots
                    & (slot_gates.detach() > 0.0)
                )
            else:
                candidate_mask = ~completed & hard_active_slots

            valid_step = candidate_mask.any(dim=1)
            scores = self._joint_router_scores(state, edit_slots, reference_state)

            gate_bias = torch.log(slot_gates.clamp_min(1e-6))[:, :, None]
            scores = scores + gate_bias
            scores = scores.masked_fill(~candidate_mask[:, :, None], -1e4)

            flat_scores = scores.reshape(b, -1)
            soft = F.softmax(flat_scores / self.router_temperature, dim=-1).reshape(b, self.num_slots, self.num_primitives)

            hard_index = flat_scores.argmax(dim=-1)
            raw_slot_ids = torch.div(hard_index, self.num_primitives, rounding_mode="floor")
            raw_primitive_ids = hard_index % self.num_primitives

            hard = F.one_hot(hard_index, num_classes=self.num_slots * self.num_primitives).to(dtype).reshape_as(soft)
            route = hard + soft - soft.detach()
            route = route * valid_step[:, None, None].to(dtype)

            selected_slot = torch.einsum("bl,bld->bd", route.sum(2), edit_slots)
            selected_primitive = torch.einsum("bk,kd->bd", route.sum(1), self.primitive_bank)
            selected_gate = torch.einsum("bl,bl->b", route.sum(2), gate_values)
            next_state, proposed_delta, alpha, actual_change = self._transition(state, selected_slot, selected_primitive, reference_state, selected_gate, valid_step)

            if (
                self.training
                and self.gate_mode == "straight_through_hard"
                and self.st_gate_recovery
            ):
                available_for_recovery = (
                    ~completed
                    & ~disabled_slots
                    & (slot_gates.detach() > 0.0)
                )
                recovery_mask = (
                    ~valid_step
                    & ~recovery_used
                    & available_for_recovery.any(dim=1)
                )
                if recovery_mask.any():
                    shadow_next = self._st_gate_recovery_shadow(state, edit_slots, slot_gates, reference_state, available_for_recovery)
                    # Zero-valued forward, nonzero backward surrogate.
                    next_state = next_state + recovery_mask[:, None].to(dtype) * (
                        shadow_next - shadow_next.detach()
                    )
                    actual_change = next_state - state
                    recovery_used = recovery_used | recovery_mask

            confidence = soft.reshape(b, -1).gather(1, hard_index[:, None]).squeeze(1)
            confidence = torch.where(valid_step, confidence, torch.zeros_like(confidence))
            slot_ids = torch.where(valid_step, raw_slot_ids, torch.full_like(raw_slot_ids, -1))
            primitive_ids = torch.where(valid_step, raw_primitive_ids, torch.full_like(raw_primitive_ids, -1))
            selected_gate = torch.where(valid_step, selected_gate, torch.zeros_like(selected_gate))
            selected_is_hard_active = valid_step & (
                selected_gate.detach() >= self.slot_gate_threshold
            )

            slot_trace.append(slot_ids)
            primitive_trace.append(primitive_ids)
            valid_trace.append(valid_step)
            active_trace.append(selected_is_hard_active)
            selected_gate_trace.append(selected_gate)
            route_confidences.append(confidence)
            proposed_deltas.append(proposed_delta)
            actual_state_changes.append(actual_change)
            transition_strengths.append(alpha)

            valid_batches = valid_step.nonzero(as_tuple=False).squeeze(1)
            if valid_batches.numel() > 0:
                chosen_slots = raw_slot_ids[valid_batches]
                completed = completed.clone()
                completed[valid_batches, chosen_slots] = True
                slot_to_step[valid_batches, chosen_slots] = step

            state = next_state
            checkpoints.append(state)

        return {
            "final_state": state,
            "checkpoints": torch.stack(checkpoints, dim=1),
            "trace_slot_ids": torch.stack(slot_trace, dim=1),
            "trace_primitive_ids": torch.stack(primitive_trace, dim=1),
            "trace_valid_mask": torch.stack(valid_trace, dim=1),
            "trace_active_mask": torch.stack(active_trace, dim=1),
            "trace_selected_slot_gates": torch.stack(selected_gate_trace, dim=1),
            "route_confidences": torch.stack(route_confidences, dim=1),
            "proposed_state_deltas": torch.stack(proposed_deltas, dim=1),
            "actual_state_changes": torch.stack(actual_state_changes, dim=1),
            "transition_strengths": torch.stack(transition_strengths, dim=1),
            "slot_to_step": slot_to_step,
            "hard_active_slot_mask": hard_active_slots,
        }

    def make_query(self, final_state: Tensor) -> Tensor:
        return F.normalize(self.query_head(final_state), dim=-1)

    def forward(
        self,
        reference_features: Tensor,
        text_states: Tensor,
        text_attention_mask: Tensor,
        *,
        text_content_mask: Tensor | None = None,
        teacher_reference_features: Tensor | None = None,
        teacher_text_states: Tensor | None = None,
        disable_execution: bool = False,
        disabled_slots: Tensor | None = None,
    ) -> dict[str, Tensor]:
        slot_output = self.build_edit_slots(
            reference_features,
            text_states,
            text_attention_mask,
            text_content_mask=text_content_mask,
            teacher_reference_features=teacher_reference_features,
            teacher_text_states=teacher_text_states,
        )
        z0, reference_state = self.initialize_state(reference_features)

        if disable_execution:
            disabled_slots = torch.ones(slot_output["edit_slots"].shape[:2], dtype=torch.bool, device=slot_output["edit_slots"].device)

        execution = self.execute(slot_output["edit_slots"], slot_output["slot_gates"], z0, reference_state, disabled_slots=disabled_slots)

        q0 = self.make_query(execution["final_state"])
        q_reference_only = self.make_query(z0)

        return {
            **slot_output,
            **execution,
            "z0": z0,
            "reference_state": reference_state,
            "q_reference_only": q_reference_only,
            "q0": q0,
        }

    @staticmethod
    def _positive_mask(
        target_ids: object,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        if isinstance(target_ids, Tensor):
            ids = target_ids.reshape(-1)
            if ids.numel() != batch_size:
                raise ValueError("target_ids Tensor must have B elements")
            ids = ids.to(device=device)
            return ids[:, None].eq(ids[None, :])

        if isinstance(target_ids, Sequence) and not isinstance(target_ids, (str, bytes)):
            ids = list(target_ids)
            if len(ids) != batch_size:
                raise ValueError("target_ids sequence must have B elements")
            mask = [
                [ids[i] == ids[j] for j in range(batch_size)]
                for i in range(batch_size)
            ]
            return torch.tensor(mask, dtype=torch.bool, device=device)

        raise TypeError("target_ids must be a Tensor or a non-string sequence")

    def _retrieval_loss(self, query: Tensor, targets: Tensor, target_ids: object | None = None) -> Tensor:
        if query.shape[0] != targets.shape[0]:
            raise ValueError("query and target batch sizes must match")

        logits = self._retrieval_scores(query, targets) / self.retrieval_temperature

        if target_ids is None:
            labels = torch.arange(query.shape[0], device=query.device)
            return F.cross_entropy(logits, labels)

        positive = self._positive_mask(target_ids, query.shape[0], query.device)
        positive_logits = logits.masked_fill(~positive, float("-inf"))
        log_numerator = torch.logsumexp(positive_logits, dim=1)
        log_denominator = torch.logsumexp(logits, dim=1)
        return (log_denominator - log_numerator).mean()

    def _retrieval_scores(self, query: Tensor, candidates: Tensor) -> Tensor:
        if query.ndim != 2 or query.shape[-1] != self.query_dim:
            raise ValueError(f"query must be [B,{self.query_dim}]")

        if candidates.ndim != 3 or candidates.shape[-1] != self.query_dim:
            raise ValueError(f"candidates must be [N,K,{self.query_dim}]")

        candidates = F.normalize(candidates, dim=-1)
        token_scores = torch.einsum("bd,nkd->bnk", query, candidates)
        return token_scores.amax(dim=-1)

    def _assignment_diagnostics(
        self,
        *,
        null_probs: Tensor,
        slot_masks: Tensor,
        slot_mass: Tensor,
        slot_effects: Tensor,
        slot_gates: Tensor,
        hard_active_slot_mask: Tensor,
        text_attention_mask: Tensor,
        text_content_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Non-prescriptive diagnostics for the competitive NULL formulation."""

        if text_content_mask is not None:
            valid = text_attention_mask.to(torch.bool) & text_content_mask.to(torch.bool)
        else:
            valid = text_attention_mask.to(torch.bool)
        valid_f = valid.to(slot_masks.dtype)
        denom = valid_f.sum().clamp_min(1.0)

        all_probs = torch.cat([null_probs[:, None, :], slot_masks], dim=1)
        entropy_per_token = -(
            all_probs.clamp_min(1e-12) * all_probs.clamp_min(1e-12).log()
        ).sum(dim=1)
        assignment_entropy = (entropy_per_token * valid_f).sum() / denom
        null_rate = (null_probs * valid_f).sum() / denom
        edit_rate = ((1.0 - null_probs) * valid_f).sum() / denom
        ownership_winner = all_probs.argmax(dim=1)
        null_winner = ownership_winner.eq(0)
        null_argmax_fraction = (null_winner.to(slot_masks.dtype) * valid_f).sum() / denom
        has_valid_content = valid.any(dim=1)
        all_null_argmax = (null_winner | ~valid).all(dim=1) & has_valid_content
        valid_sample_count = has_valid_content.to(slot_masks.dtype).sum().clamp_min(1.0)
        all_null_argmax_sample_fraction = all_null_argmax.to(slot_masks.dtype).sum() / valid_sample_count
        edit_mass_per_sample = slot_mass.sum(dim=1)
        edit_mass_fraction = edit_mass_per_sample.sum() / denom
        ownership_active_slot_count = (slot_mass >= 0.10).to(slot_masks.dtype).sum(dim=1).mean()
        execution_hard_active_slot_count = hard_active_slot_mask.to(slot_masks.dtype).sum(dim=1).mean()
        nontrivial_edit = edit_mass_per_sample >= 0.10
        dominant_slot_share_per_sample = slot_mass.max(dim=1).values/ edit_mass_per_sample.clamp_min(1e-12)
        nontrivial_count = nontrivial_edit.to(slot_masks.dtype).sum().clamp_min(1.0)
        dominant_slot_share = (dominant_slot_share_per_sample * nontrivial_edit.to(slot_masks.dtype)).sum() / nontrivial_count
        near_monopoly_fraction = (
            (dominant_slot_share_per_sample >= 0.90) & nontrivial_edit
        ).to(slot_masks.dtype).sum() / nontrivial_count

        zero = slot_masks.sum() * 0.0
        if self.num_slots == 1:
            overlap = zero
            effect_similarity_mean = zero
        else:
            masked = slot_masks * valid_f[:, None, :]
            mask_vectors = F.normalize(masked, dim=-1, eps=1e-6)
            mask_similarity = mask_vectors @ mask_vectors.transpose(1, 2)

            effect_vectors = F.normalize(slot_effects, dim=-1, eps=1e-6)
            effect_similarity = effect_vectors @ effect_vectors.transpose(1, 2)

            upper = torch.triu(torch.ones(self.num_slots, self.num_slots, dtype=torch.bool, device=slot_masks.device), diagonal=1)
            overlap = mask_similarity[:, upper].mean()
            effect_similarity_mean = effect_similarity[:, upper].mean()

        diagnostics = {
            "assignment_entropy": assignment_entropy,
            "null_ownership_rate": null_rate,
            "edit_ownership_rate": edit_rate,

            "edit_mass_fraction": edit_mass_fraction,
            "null_argmax_fraction": null_argmax_fraction,
            "all_null_argmax_sample_fraction":all_null_argmax_sample_fraction,
            "ownership_active_slot_count": ownership_active_slot_count,
            "execution_hard_active_slot_count": execution_hard_active_slot_count,
            "dominant_slot_share": dominant_slot_share,
            "near_monopoly_fraction": near_monopoly_fraction,
            "slot_overlap_mean": overlap,
            "slot_effect_similarity_mean": effect_similarity_mean,
            "slot_gate_mean": slot_gates.mean(),
        }

        for slot_id in range(self.num_slots):
            diagnostics[f"slot_{slot_id}_mass_mean"] = slot_mass[:, slot_id].mean()

        return diagnostics

    def _slot_regularizers(
        self,
        slot_masks: Tensor,
        slot_effects: Tensor,
        slot_gates: Tensor,
        text_attention_mask: Tensor,
        text_content_mask: Tensor | None = None,
        *,
        null_probs: Tensor | None = None,
    ) -> dict[str, Tensor]:

        del slot_masks, slot_effects, slot_gates, text_attention_mask
        del text_content_mask, null_probs
        raise RuntimeError(
            "Standalone Stage-1 structural losses are incompatible with "
            "competitive NULL ownership. Train this formulation end-to-end with "
            "retrieval supervision; keep assignment statistics as diagnostics."
        )

    def compute_loss(self, batch: Mapping[str, object]) -> dict[str, Tensor]:
        required = {
            "reference_features",
            "teacher_reference_features",
            "text_states",
            "teacher_text_states",
            "text_attention_mask",
            "text_content_mask",
            "target_features",
        }
        missing = required.difference(batch)
        if missing:
            raise KeyError(f"TAPER batch missing keys: {sorted(missing)}")

        reference = batch["reference_features"]
        text = batch["text_states"]
        mask = batch["text_attention_mask"]
        targets = batch["target_features"]

        if not all(isinstance(x, Tensor) for x in (reference, text, mask, targets)):
            raise TypeError("reference_features, text_states, text_attention_mask, and " "target_features must be Tensors")
        assert isinstance(reference, Tensor)
        assert isinstance(text, Tensor)
        assert isinstance(mask, Tensor)
        assert isinstance(targets, Tensor)

        if targets.ndim != 3 or targets.shape[-1] != self.query_dim:
            raise ValueError(f"target_features must be [B,K,{self.query_dim}]")
        if targets.shape[0] != reference.shape[0]:
            raise ValueError("target_features batch size must match reference_features batch size")

        content_mask = batch["text_content_mask"]
        teacher_reference = batch["teacher_reference_features"]
        teacher_text = batch["teacher_text_states"]
        if not isinstance(content_mask, Tensor):
            raise TypeError("text_content_mask must be a Tensor")
        if not isinstance(teacher_reference, Tensor):
            raise TypeError("teacher_reference_features must be a Tensor")
        if not isinstance(teacher_text, Tensor):
            raise TypeError("teacher_text_states must be a Tensor")

        output = self.forward(reference, text, mask, text_content_mask=content_mask, teacher_reference_features=teacher_reference, teacher_text_states=teacher_text)
        losses = {
            "retrieval_loss": self._retrieval_loss(output["q0"], targets, batch.get("target_ids"))
        }
        with torch.no_grad():
            diagnostics = self._assignment_diagnostics(
                null_probs=output["null_probs"],
                slot_masks=output["slot_masks"],
                slot_mass=output["slot_mass"],
                slot_effects=output["slot_effects"],
                slot_gates=output["slot_gates"],
                hard_active_slot_mask=output["hard_active_slot_mask"],
                text_attention_mask=mask,
                text_content_mask=content_mask,
            )
        losses.update({f"diagnostic/{k}": v for k, v in diagnostics.items()})
        return losses

    @torch.no_grad()
    def slot_drop_queries(
        self,
        *,
        reference_features: Tensor,
        text_states: Tensor,
        text_attention_mask: Tensor,
        text_content_mask: Tensor | None = None,
        teacher_reference_features: Tensor | None = None,
        teacher_text_states: Tensor | None = None,
    ) -> dict[str, Tensor]:
        was_training = self.training
        self.eval()
        try:
            slots = self.build_edit_slots(
                reference_features,
                text_states,
                text_attention_mask,
                text_content_mask=text_content_mask,
                teacher_reference_features=teacher_reference_features,
                teacher_text_states=teacher_text_states,
            )
            z0, reference_state = self.initialize_state(reference_features)
            full_execution = self.execute(slots["edit_slots"], slots["slot_gates"], z0, reference_state)
            full_query = self.make_query(full_execution["final_state"])

            dropped_queries = []
            b = reference_features.shape[0]
            for slot_id in range(self.num_slots):
                disabled = torch.zeros(b, self.num_slots, dtype=torch.bool, device=reference_features.device)
                disabled[:, slot_id] = True
                execution = self.execute(slots["edit_slots"], slots["slot_gates"], z0, reference_state, disabled_slots=disabled)
                dropped_queries.append(self.make_query(execution["final_state"]))

            return {
                "full_query": full_query,
                "dropped_queries": torch.stack(dropped_queries, dim=1),
                "slot_gates": slots["slot_gates"],
                "slot_mass": slots["slot_mass"],
                "slot_activity": slots["slot_activity"],
                "null_probs": slots["null_probs"],
                "hard_active_slot_mask": full_execution[
                    "hard_active_slot_mask"
                ],
            }
        finally:
            self.train(was_training)

    def compute_stage1_loss(self, batch: Mapping[str, object]) -> dict[str, Tensor]:
        del batch
        raise RuntimeError(
            "compute_stage1_loss() is intentionally disabled for competitive "
            "NULL ownership. This branch requires end-to-end retrieval "
            "supervision; the previous structural Stage-1 objective is not "
            "mathematically compatible with a learnable NULL sink."
        )

    @torch.no_grad()
    def retrieve(
        self,
        *,
        reference_features: Tensor,
        text_states: Tensor,
        text_attention_mask: Tensor,
        gallery_features: Tensor,
        text_content_mask: Tensor | None = None,
        teacher_reference_features: Tensor | None = None,
        teacher_text_states: Tensor | None = None,
        topk: int | None = None,
    ) -> dict[str, Tensor]:
        if gallery_features.ndim != 3 or gallery_features.shape[-1] != self.query_dim:
            raise ValueError(f"gallery_features must be [G,K,{self.query_dim}]")

        was_training = self.training
        self.eval()
        try:
            output = self.forward(
                reference_features, text_states, text_attention_mask,
                text_content_mask=text_content_mask,
                teacher_reference_features=teacher_reference_features,
                teacher_text_states=teacher_text_states,
            )
            scores = self._retrieval_scores(output["q0"], gallery_features)
            result = {
                **output,
                "scores": scores,
            }
            if topk is not None:
                if topk < 1:
                    raise ValueError("topk must be >= 1 when provided")

                k = min(topk, gallery_features.shape[0])
                top_scores, top_indices = scores.topk(k, dim=1)
                result.update({"top_scores": top_scores, "top_indices": top_indices,})
            return result
        finally:
            self.train(was_training)
