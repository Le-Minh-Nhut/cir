from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TAPER(nn.Module):
    def __init__(
        self,
        *,
        text_dim: int,
        reference_dim: int,
        query_dim: int,
        slot_dim: int = 512,
        state_dim: int = 512,
        num_slots: int = 4,
        num_primitives: int = 8,
        mask_temperature: float = 1.0,
        router_temperature: float = 1.0,
        retrieval_temperature: float = 0.07,
        neutral_mode: str = "zero",
        overlap_margin: float = 0.60,
        effect_diversity_margin: float = 0.50,
        alpha_max: float = 1.0,
        counterfactual_chunk_size: int = 8,
        qasa_tau: float = 0.5,
        qasa_rho: float = 0.8,
        qasa_mu: float = 0.3,
        qasa_eps: float = 1e-8,
        qasa_apply_at_eval: bool = True,
    ) -> None:
        super().__init__()

        if not 0.0 < qasa_tau <= 1.0:
            raise ValueError("qasa_tau must be in (0, 1]")
        if not 0.0 < qasa_rho <= 1.0:
            raise ValueError("qasa_rho must be in (0, 1]")
        if not 0.0 <= qasa_mu < 1.0:
            raise ValueError("qasa_mu must be in [0, 1)")
        if qasa_eps <= 0:
            raise ValueError("qasa_eps must be > 0")
        if num_slots < 1 or num_primitives < 1:
            raise ValueError("num_slots and num_primitives must be >= 1")
        if min(mask_temperature, router_temperature, retrieval_temperature) <= 0:
            raise ValueError("all temperatures must be > 0")
        if neutral_mode not in {"zero", "mean", "learned"}:
            raise ValueError("neutral_mode must be: zero, mean, or learned")
        if alpha_max <= 0:
            raise ValueError("alpha_max must be > 0")
        if counterfactual_chunk_size < 1:
            raise ValueError("counterfactual_chunk_size must be >= 1")

        self.text_dim = text_dim
        self.reference_dim = reference_dim
        self.query_dim = query_dim
        self.slot_dim = slot_dim
        self.state_dim = state_dim
        self.num_slots = num_slots
        self.num_primitives = num_primitives

        self.mask_temperature = mask_temperature
        self.router_temperature = router_temperature
        self.retrieval_temperature = retrieval_temperature
        self.neutral_mode = neutral_mode
        self.overlap_margin = overlap_margin
        self.effect_diversity_margin = effect_diversity_margin
        self.alpha_max = alpha_max
        self.counterfactual_chunk_size = counterfactual_chunk_size

        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

        # Ephemeral competing Edit Slot queries.
        self.slot_queries = nn.Parameter(torch.randn(num_slots, slot_dim) * 0.02)
        self.slot_query_projection = nn.Linear(slot_dim, slot_dim, bias=False)
        self.text_key_projection = nn.Linear(text_dim, slot_dim, bias=False)

        if neutral_mode == "learned":
            self.neutral_embedding = nn.Parameter(torch.zeros(self.teacher_text_dim))
        else:
            self.register_buffer("neutral_embedding", torch.zeros(self.teacher_text_dim))

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

        self.query_head = nn.Sequential(nn.Linear(state_dim, query_dim), nn.GELU(), nn.Linear(query_dim, query_dim))
        self.qasa_tau = qasa_tau
        self.qasa_rho = qasa_rho
        self.qasa_mu = qasa_mu
        self.qasa_eps = qasa_eps
        self.qasa_apply_at_eval = bool(qasa_apply_at_eval)

    def train(self, mode: bool = True) -> "TAPER":
        super().train(mode)
        self.teacher.eval()
        return self

    def _pool_text(self, text_states: Tensor, mask: Tensor) -> Tensor:
        mask_f = mask.to(text_states.dtype).unsqueeze(-1)
        return (text_states * mask_f).sum(1) / mask_f.sum(1).clamp_min(1.0)

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
            raise ValueError("text_content_mask is required so special/padding tokens are excluded from Edit-Slot ownership")
        if text_content_mask.shape != text_attention_mask.shape:
            raise ValueError("text_content_mask must match text_attention_mask")
        slot_valid = attention_valid & text_content_mask.to(torch.bool)

        return attention_valid, slot_valid

    def _competitive_ownership(self, text_states: Tensor, slot_valid: Tensor) -> tuple[Tensor, Tensor]:
        queries = self.slot_query_projection(self.slot_queries)  # [L, Ds]
        keys = self.text_key_projection(text_states)  # [B,N,Ds]
        logits = torch.einsum("ld,bnd->bln", queries, keys)
        logits = logits / math.sqrt(self.slot_dim) / self.mask_temperature
        valid = slot_valid[:, None, :]
        invalid_logit = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~valid, invalid_logit)
        ownership = F.softmax(logits, dim=1)
        ownership = ownership * valid.to(ownership.dtype)
        if not torch.isfinite(ownership).all():
            raise FloatingPointError("non-finite slot ownership")

        return logits, ownership

    @torch.no_grad()
    def _qasa_attention_fp32(self, text_states: Tensor, slot_valid: Tensor) -> Tensor:
        if text_states.ndim != 3:
            raise ValueError("text_states must be [B,N,D]")
        if slot_valid.shape != text_states.shape[:2]:
            raise ValueError("slot_valid must match text_states [B,N]")

        with torch.autocast(device_type=text_states.device.type, enabled=False):
            text_fp32 = text_states.float()
            queries = F.linear(self.slot_queries.float(), self.slot_query_projection.weight.float(), bias=None)  # [L, Ds]
            keys = F.linear(text_fp32, self.text_key_projection.weight.float(), bias=None)  # [B,N,Ds]
            logits = torch.einsum("ld,bnd->bln", queries, keys)
            logits = logits / math.sqrt(self.slot_dim) / self.mask_temperature
            valid = slot_valid[:, None, :]
            invalid_logit = torch.finfo(torch.float32).min
            logits = logits.masked_fill(~valid, invalid_logit)
            attention = F.softmax(logits, dim=1,)
            attention = attention * valid.to(attention.dtype)
        if attention.dtype != torch.float32:
            raise RuntimeError("QASA attention must be FP32")

        if not torch.isfinite(attention).all():
            raise FloatingPointError("non-finite QASA attention")

        return attention

    def _qasa_select_slots(self, attention: Tensor, valid: Tensor) -> dict[str, Tensor]:
        if attention.ndim != 3:
            raise ValueError("attention must be [B,L,N]")
        attention = attention.float()
        valid = valid.to(torch.bool)
        b, l, n = attention.shape

        if l != self.num_slots:
            raise ValueError("attention slot dimension mismatch")
        if valid.shape != (b, n):
            raise ValueError("valid shape mismatch")

        dtype = attention.dtype
        device = attention.device
        valid_f = valid.to(dtype)
        winner = attention.argmax(dim=1)  # [B,N]
        winner_one_hot = F.one_hot(winner, num_classes=l).permute(0, 2, 1).to(dtype)
        winner_one_hot = winner_one_hot * valid_f[:, None, :]
        total_mass = (attention* valid_f[:, None, :]).sum(dim=-1)  # [B,L]
        winning_mass = (attention* winner_one_hot).sum(dim=-1)
        # quality = winning_mass / total_mass.clamp_min(self.qasa_eps)
        quality = winning_mass / (total_mass + self.qasa_eps)
        # quality = torch.where(total_mass > self.qasa_eps, quality, torch.zeros_like(quality))
        quality_for_selection = quality.detach()
        attention_for_selection = attention.detach()
        valid_for_selection = valid.detach()
        selected = torch.zeros(b, l, dtype=torch.bool, device=device)
        final_coverage = torch.zeros(b, dtype=dtype, device=device)
        novelty_skip_count = torch.zeros(b, dtype=dtype, device=device)
        for batch_id in range(b):
            valid_tokens = valid_for_selection[batch_id]
            num_valid = int(valid_tokens.sum().item())
            if num_valid == 0:
                continue

            order = torch.argsort(quality_for_selection[batch_id], descending=True, stable=True)
            covered = torch.zeros(n, dtype=torch.bool, device=device)
            for slot_id_tensor in order:
                slot_id = int(slot_id_tensor.item())
                slot_mass = attention_for_selection[batch_id, slot_id, valid_tokens].sum()
                # if slot_mass <= self.qasa_eps:
                #     continue

                if selected[batch_id].any():
                    already_covered = covered & valid_tokens
                    mass_on_covered = attention_for_selection[batch_id, slot_id, already_covered].sum()
                    # novelty = 1.0 - mass_on_covered / slot_mass.clamp_min(self.qasa_eps)
                    novelty = 1.0 - mass_on_covered / (slot_mass + self.qasa_eps)
                else:
                    novelty = torch.ones((), dtype=dtype, device=device)
                if novelty < self.qasa_mu:
                    novelty_skip_count[batch_id] += 1.0

                    continue

                selected[batch_id, slot_id] = True

                selected_attention = attention_for_selection[batch_id, selected[batch_id], :].sum(dim=0)
                covered = (selected_attention >= self.qasa_tau) & valid_tokens
                coverage = covered.sum().to(dtype) / float(num_valid)
                final_coverage[batch_id] = coverage
                if coverage >= self.qasa_rho:
                    break

        selected_count = selected.sum(dim=1)

        return {
            "qasa_quality": quality,
            "qasa_selected_mask": selected,
            "qasa_selected_count": selected_count,
            "qasa_final_coverage": final_coverage,
            "qasa_novelty_skip_count": novelty_skip_count,
        }

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

    @torch.no_grad()
    def qasa_inference_partition(self, attention: Tensor, valid: Tensor) -> dict[str, Tensor]:
        if attention.ndim != 3:
            raise ValueError("attention must be [B,L,N]")
        b, l, n = attention.shape
        if l != self.num_slots:
            raise ValueError("attention slot dimension mismatch")
        if valid.shape != (b, n):
            raise ValueError("valid must be [B,N]")
        valid = valid.to(torch.bool)
        winner = attention.argmax(dim=1)  # [B,N]
        hard_regions = F.one_hot(winner, num_classes=l).permute(0, 2, 1).to(torch.bool)  # [B,L,N]
        hard_regions = hard_regions & valid[:, None, :]
        winner_ids = torch.where(valid, winner, torch.full_like(winner, -1))
        nonempty_slots = hard_regions.any(dim=2)  # [B,L]
        effective_k = nonempty_slots.sum(dim=1)   # [B]
        winner_counts = hard_regions.sum(dim=2)   # [B,L]
        hard_sum = hard_regions.sum(dim=1)  # [B,N]
        if not torch.equal(hard_sum[valid], torch.ones_like(hard_sum[valid])):
            raise RuntimeError("Every valid token must belong to exactly one slot")

        if (~valid).any():
            if hard_sum[~valid].any():
                raise RuntimeError("Invalid/special tokens must belong to no slot")

        if (effective_k > self.num_slots).any():
            raise RuntimeError("effective_k cannot exceed num_slots")

        return {
            "qasa_inference_winner_ids": winner_ids,
            "qasa_inference_hard_regions": hard_regions,
            "qasa_inference_nonempty_slots": nonempty_slots,
            "qasa_inference_effective_k": effective_k,
            "qasa_inference_winner_counts": winner_counts,
        }

    def build_edit_slots(
        self,
        reference_features: Tensor,
        text_states: Tensor,
        text_attention_mask: Tensor,
        *,
        text_content_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if reference_features.ndim != 2:
            raise ValueError("reference_features must be [B,D] for TAPER state initialization")
        if reference_features.shape[0] != text_states.shape[0]:
            raise ValueError("reference_features and text_states batch sizes do not match")

        _, slot_valid = self._validate_text_inputs(text_states, text_attention_mask, text_content_mask)

        batch_size, num_tokens, _ = text_states.shape
        ownership_logits, slot_masks = self._competitive_ownership(text_states, slot_valid)
        slot_semantics, slot_mass, slot_activity = self._mass_aware_slot_pool(text_states, slot_masks)

        expected_shape = (batch_size, self.teacher_query_dim)

        raw_edit_slots = self.slot_mlp(slot_semantics)
        edit_slots = raw_edit_slots * slot_activity.unsqueeze(-1)
        edit_slots = raw_edit_slots * slot_activity.unsqueeze(-1)

        # qasa_attention = F.softmax(ownership_logits.float(), dim=1)
        # qasa_attention = qasa_attention * slot_valid[:, None, :].to(qasa_attention.dtype)
        # qasa_valid = slot_valid.to(torch.bool)
        # qasa = self._qasa_select_slots(qasa_attention, qasa_valid)
        qasa_attention = self._qasa_attention_fp32(text_states, slot_valid)
        qasa_valid = slot_valid.to(torch.bool)
        qasa = self._qasa_select_slots(qasa_attention, qasa_valid)
        qasa_inference = self.qasa_inference_partition(qasa_attention, qasa_valid)
        if (not self.training and not self.qasa_apply_at_eval):
            selected_mask = torch.ones(
                qasa["qasa_selected_mask"].shape,
                dtype=torch.bool,
                device=qasa["qasa_selected_mask"].device,
            )
        else:
            selected_mask = qasa["qasa_selected_mask"]

        return {
            "edit_slots": edit_slots,
            "raw_edit_slots": raw_edit_slots,
            "ownership_logits": ownership_logits,
            "slot_masks": slot_masks,
            "slot_semantics": slot_semantics,
            "slot_mass": slot_mass,
            "slot_activity": slot_activity,
            "slot_peak_ownership": (slot_masks.amax(dim=2)),
            "qasa_attention": qasa_attention,
            "qasa_valid_mask": qasa_valid,
            "qasa_quality": qasa["qasa_quality"],
            "qasa_selected_mask": selected_mask,
            "qasa_selected_count": (selected_mask.sum(dim=1)),
            "qasa_final_coverage": qasa["qasa_final_coverage"],
            "qasa_novelty_skip_count": qasa["qasa_novelty_skip_count"],
            **qasa_inference,
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
        valid_step: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        state_context = self.state_norm(state)
        x = torch.cat([state_context, slot, primitive, reference_state], dim=-1)
        proposed_delta = self.transition_delta(x)
        alpha = self.transition_strength(x).squeeze(-1) * self.alpha_max
        state_update = alpha[:, None] * proposed_delta
        proposed_next = state + state_update
        next_state = torch.where(valid_step[:, None], proposed_next, state)
        actual_change = next_state - state
        proposed_delta = torch.where(
            valid_step[:, None],
            proposed_delta,
            torch.zeros_like(proposed_delta),
        )
        alpha = torch.where(
            valid_step,
            alpha,
            torch.zeros_like(alpha),
        )

        return (next_state, proposed_delta, alpha, actual_change)

    def execute(
        self,
        edit_slots: Tensor,
        selected_slots: Tensor,
        z0: Tensor,
        reference_state: Tensor,
        *,
        disabled_slots: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if edit_slots.ndim != 3 or edit_slots.shape[1] != self.num_slots:
            raise ValueError("edit_slots must be [B,num_slots,slot_dim]")
        if selected_slots.shape != (edit_slots.shape[0], self.num_slots):
            raise ValueError("selected_slots must be [B,num_slots]")
        if selected_slots.dtype != torch.bool:
            raise TypeError("selected_slots must be bool")

        b = edit_slots.shape[0]
        device = edit_slots.device
        dtype = edit_slots.dtype

        if disabled_slots is None:
            disabled_slots = torch.zeros(b, self.num_slots, dtype=torch.bool, device=device)
        else:
            if disabled_slots.shape != (b, self.num_slots):
                raise ValueError("disabled_slots shape mismatch")

            disabled_slots = disabled_slots.to(device=device, dtype=torch.bool)
        active_slots = (selected_slots & ~disabled_slots)
        state = z0
        completed = torch.zeros(b, self.num_slots, dtype=torch.bool, device=device)
        checkpoints = [state]
        slot_trace = []
        primitive_trace = []
        valid_trace = []
        active_trace = []
        route_confidences = []
        proposed_deltas = []
        actual_state_changes = []
        transition_strengths = []
        slot_to_step = torch.full((b, self.num_slots), -1, dtype=torch.long, device=device)

        for step in range(self.num_slots):
            candidate_mask = ~completed & active_slots
            valid_step = candidate_mask.any(dim=1)
            scores = self._joint_router_scores(state, edit_slots, reference_state)
            scores = scores.masked_fill(~candidate_mask[:, :, None], -1e4)
            flat_scores = scores.reshape(b, -1)
            soft = F.softmax(flat_scores / self.router_temperature, dim=-1).reshape(b, self.num_slots, self.num_primitives)
            hard_index = flat_scores.argmax(dim=-1)
            raw_slot_ids = torch.div(hard_index, self.num_primitives, rounding_mode="floor")
            raw_primitive_ids = hard_index % self.num_primitives
            hard = F.one_hot(
                hard_index,
                num_classes=self.num_slots * self.num_primitives,
            ).to(dtype).reshape_as(soft)
            route = hard + soft - soft.detach()
            route = route * valid_step[:, None, None].to(dtype)
            selected_slot = torch.einsum("bl,bld->bd", route.sum(2), edit_slots)
            selected_primitive = torch.einsum("bk,kd->bd", route.sum(1), self.primitive_bank)
            (next_state, proposed_delta, alpha, actual_change) = self._transition(state, selected_slot, selected_primitive, reference_state, valid_step)
            confidence = soft.reshape(b, -1).gather(1, hard_index[:, None]).squeeze(1)
            confidence = torch.where(
                valid_step,
                confidence,
                torch.zeros_like(confidence),
            )
            slot_ids = torch.where(
                valid_step,
                raw_slot_ids,
                torch.full_like(raw_slot_ids, -1),
            )

            primitive_ids = torch.where(
                valid_step,
                raw_primitive_ids,
                torch.full_like(raw_primitive_ids, -1),
            )

            slot_trace.append(slot_ids)
            primitive_trace.append(primitive_ids)
            valid_trace.append(valid_step)
            active_trace.append(valid_step)
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
            "route_confidences": torch.stack(route_confidences, dim=1),
            "proposed_state_deltas": torch.stack(proposed_deltas, dim=1),
            "actual_state_changes": torch.stack(actual_state_changes, dim=1),
            "transition_strengths": torch.stack(transition_strengths, dim=1),
            "slot_to_step": slot_to_step,
            "hard_active_slot_mask": active_slots,
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

        execution = self.execute(slot_output["edit_slots"], slot_output["qasa_selected_mask"], z0, reference_state, disabled_slots=disabled_slots)
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
        slot_masks: Tensor,
        slot_mass: Tensor,
        slot_effects: Tensor,
        qasa_selected_mask: Tensor,
        qasa_quality: Tensor,
        qasa_final_coverage: Tensor,
        hard_active_slot_mask: Tensor,
        text_attention_mask: Tensor,
        text_content_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if text_content_mask is not None:
            valid = text_attention_mask.to(torch.bool) & text_content_mask.to(torch.bool)
        else:
            valid = text_attention_mask.to(torch.bool)

        valid_f = valid.to(slot_masks.dtype)
        denom = valid_f.sum().clamp_min(1.0)
        probs = slot_masks.clamp_min(1e-12)
        entropy_per_token = -(probs * probs.log()).sum(dim=1)
        assignment_entropy = (entropy_per_token * valid_f).sum() / denom
        ownership_winner = slot_masks.argmax(dim=1)
        winner_one_hot = F.one_hot(ownership_winner, num_classes=self.num_slots).permute(0, 2, 1)
        winner_one_hot = winner_one_hot & valid[:, None, :]
        winner_count = winner_one_hot.sum(dim=2)
        ownership_active_slot_count = (winner_count > 0).to(slot_masks.dtype).sum(dim=1).mean()
        execution_hard_active_slot_count = hard_active_slot_mask.to(slot_masks.dtype).sum(dim=1).mean()
        total_mass_per_sample = slot_mass.sum(dim=1)
        dominant_slot_share_per_sample = slot_mass.max(dim=1).values / total_mass_per_sample.clamp_min(1e-12)
        dominant_slot_share = dominant_slot_share_per_sample.mean()
        near_monopoly_fraction = (dominant_slot_share_per_sample >= 0.90).to(slot_masks.dtype).mean()
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
            "ownership_active_slot_count":ownership_active_slot_count,
            "execution_hard_active_slot_count": execution_hard_active_slot_count,
            "dominant_slot_share": dominant_slot_share,
            "near_monopoly_fraction": near_monopoly_fraction,
            "slot_overlap_mean": overlap,
            "slot_effect_similarity_mean": effect_similarity_mean,
            "qasa_selected_slot_count": qasa_selected_mask.to(slot_masks.dtype).sum(dim=1).mean(),
            "qasa_quality_mean": qasa_quality.mean(),
            "qasa_final_coverage_mean": qasa_final_coverage.mean(),
        }

        for slot_id in range(self.num_slots):
            diagnostics[f"slot_{slot_id}_mass_mean"] = slot_mass[:, slot_id].mean()
            diagnostics[f"slot_{slot_id}_winner_count_mean"] = winner_count[:, slot_id].to(slot_masks.dtype).mean()
            diagnostics[f"slot_{slot_id}_quality_mean"] = qasa_quality[:, slot_id].mean()

        return diagnostics

    def _slot_regularizers(
        self,
        slot_masks: Tensor,
        slot_effects: Tensor,
        text_attention_mask: Tensor,
        *,
        text_content_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:

        del slot_masks, slot_effects, text_attention_mask
        del text_content_mask
        raise RuntimeError(
            "Standalone Stage-1 structural losses are incompatible with "
            "Train this formulation end-to-end with "
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
                slot_masks=output["slot_masks"],
                slot_mass=output["slot_mass"],
                slot_effects=output["slot_effects"],
                hard_active_slot_mask=output["hard_active_slot_mask"],
                text_attention_mask=mask,
                text_content_mask=content_mask,
                qasa_selected_mask=output["qasa_selected_mask"],
                qasa_quality=output["qasa_quality"],
                qasa_final_coverage=output["qasa_final_coverage"],
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
            full_execution = self.execute(slots["edit_slots"], slots["qasa_selected_mask"], z0, reference_state)
            full_query = self.make_query(full_execution["final_state"])

            dropped_queries = []
            b = reference_features.shape[0]
            for slot_id in range(self.num_slots):
                disabled = torch.zeros(b, self.num_slots, dtype=torch.bool, device=reference_features.device)
                disabled[:, slot_id] = True
                execution = self.execute(slots["edit_slots"], slots["qasa_selected_mask"], z0, reference_state, disabled_slots=disabled)
                dropped_queries.append(self.make_query(execution["final_state"]))

            return {
                "full_query": full_query,
                "dropped_queries": torch.stack(dropped_queries, dim=1),
                "slot_mass": slots["slot_mass"],
                "slot_activity": slots["slot_activity"],
                "hard_active_slot_mask": full_execution["hard_active_slot_mask"],
                "qasa_selected_mask": slots["qasa_selected_mask"],
                "qasa_quality": slots["qasa_quality"],
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
