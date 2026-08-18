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
        overlap_margin: float = 0.60,
        effect_diversity_margin: float = 0.50,
        alpha_max: float = 1.0,
    ) -> None:
        super().__init__()

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

        self.teacher = teacher
        self.text_dim = text_dim
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
        self.overlap_margin = overlap_margin
        self.effect_diversity_margin = effect_diversity_margin
        self.alpha_max = alpha_max

        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

        self.slot_queries = nn.Parameter(torch.randn(num_slots, slot_dim) * 0.02)
        self.slot_query_projection = nn.Linear(slot_dim, slot_dim, bias=False)
        self.text_key_projection = nn.Linear(text_dim, slot_dim, bias=False)

        if neutral_mode == "learned":
            self.neutral_embedding = nn.Parameter(torch.zeros(text_dim))
        else:
            self.register_buffer("neutral_embedding", torch.zeros(text_dim))

        self.slot_mlp = nn.Sequential(
            nn.Linear(text_dim + teacher_query_dim, slot_dim),
            nn.GELU(),
            nn.Linear(slot_dim, slot_dim),
            nn.LayerNorm(slot_dim),
        )

        self.slot_gate = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, slot_dim),
            nn.GELU(),
            nn.Linear(slot_dim, 1),
        )
        nn.init.constant_(self.slot_gate[-1].bias, 1.0)

        self.reference_to_state = nn.Sequential(
            nn.Linear(reference_dim, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, state_dim),
            nn.LayerNorm(state_dim),
        )

        self.primitive_bank = nn.Parameter(torch.randn(num_primitives, state_dim) * 0.02)
        router_dim = state_dim + slot_dim + state_dim + state_dim
        self.router = nn.Sequential(
            nn.Linear(router_dim, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, 1),
        )

        transition_dim = state_dim + slot_dim + state_dim + state_dim
        self.transition_delta = nn.Sequential(
            nn.Linear(transition_dim, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, state_dim),
        )
        self.transition_strength = nn.Sequential(
            nn.Linear(transition_dim, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, 1),
            nn.Sigmoid(),
        )
        self.state_norm = nn.LayerNorm(state_dim)

        self.query_head = nn.Sequential(
            nn.Linear(state_dim, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, query_dim),
        )

    def train(self, mode: bool = True) -> "TAPER":
        super().train(mode)
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

    def build_edit_slots(self, reference_features: Tensor, text_states: Tensor, text_attention_mask: Tensor) -> dict[str, Tensor]:
        """M -> A -> s -> counterfactual delta -> u -> slot gate g."""
        if text_states.ndim != 3 or text_attention_mask.ndim != 2:
            raise ValueError("text_states must be [B,N,D] and mask must be [B,N]")
        if text_states.shape[:2] != text_attention_mask.shape:
            raise ValueError("text_states and text_attention_mask do not match")
        if text_states.shape[-1] != self.text_dim:
            raise ValueError(f"text dim must be {self.text_dim}")

        batch_size, num_tokens, _ = text_states.shape
        valid = text_attention_mask.to(torch.bool)

        queries = self.slot_query_projection(self.slot_queries)       # [L,Ds]
        keys = self.text_key_projection(text_states)                  # [B,N,Ds]
        logits = torch.einsum("ld,bnd->bln", queries, keys)
        logits = logits / math.sqrt(self.slot_dim) / self.mask_temperature

        # Independent token membership. Padding never belongs to a slot.
        slot_masks = torch.sigmoid(logits) * valid[:, None, :].to(text_states.dtype)

        mask_mass = slot_masks.sum(2, keepdim=True).clamp_min(1e-6)
        slot_semantics = (torch.einsum("bln,bnd->bld", slot_masks, text_states) / mask_mass)

        # Full frozen-teacher query. No torch.no_grad() here during training.
        q_full = self.teacher.compose(reference_features, text_states, text_attention_mask, normalize=False)
        expected_shape = (batch_size, self.teacher_query_dim)
        if q_full.shape != expected_shape:
            raise ValueError(f"teacher query must be {expected_shape}, got {tuple(q_full.shape)}")

        neutral = self._neutral_text(text_states, text_attention_mask)
        masks = slot_masks.unsqueeze(-1)
        counterfactual_text = ((1.0 - masks) * text_states.unsqueeze(1) + masks * neutral.unsqueeze(1)).reshape(batch_size * self.num_slots, num_tokens, self.text_dim)

        counterfactual_mask = (
            text_attention_mask[:, None, :]
            .expand(batch_size, self.num_slots, num_tokens)
            .reshape(batch_size * self.num_slots, num_tokens)
        )
        counterfactual_reference = reference_features.repeat_interleave(self.num_slots, dim=0)

        q_minus_flat = self.teacher.compose(counterfactual_reference, counterfactual_text, counterfactual_mask, normalize=False,)
        expected_minus_shape = (batch_size * self.num_slots, self.teacher_query_dim)
        if q_minus_flat.shape != expected_minus_shape:
            raise ValueError(f"counterfactual teacher query must be {expected_minus_shape}, got {tuple(q_minus_flat.shape)}")
        q_minus = q_minus_flat.reshape(batch_size, self.num_slots, self.teacher_query_dim)

        slot_effects = q_full.unsqueeze(1) - q_minus
        edit_slots = self.slot_mlp(torch.cat([slot_semantics, slot_effects], dim=-1))
        slot_gate_logits = self.slot_gate(edit_slots).squeeze(-1)
        slot_gates = torch.sigmoid(slot_gate_logits)

        return {
            "edit_slots": edit_slots,
            "slot_masks": slot_masks,
            "slot_semantics": slot_semantics,
            "slot_effects": slot_effects,
            "slot_gate_logits": slot_gate_logits,
            "slot_gates": slot_gates,
            "q_teacher_full": q_full,
            "q_teacher_minus": q_minus,
        }

    def initialize_state(self, reference_features: Tensor) -> tuple[Tensor, Tensor]:
        if reference_features.ndim != 2:
            raise ValueError("reference_features must be [B,D]")

        if reference_features.shape[-1] != self.reference_dim:
            raise ValueError(f"reference dim must be {self.reference_dim}")

        reference_state = self.reference_to_state(reference_features)
        z0 = reference_state
        return z0, reference_state


    def _joint_router_scores(self, state: Tensor, edit_slots: Tensor, reference_state: Tensor) -> Tensor:
        """Score every candidate active-slot x primitive pair [B,L,K]."""
        b, l, _ = edit_slots.shape
        k = self.num_primitives

        state_x = state[:, None, None, :].expand(b, l, k, -1)
        slot_x = edit_slots[:, :, None, :].expand(b, l, k, -1)
        primitive_x = self.primitive_bank[None, None, :, :].expand(b, l, k, -1)
        reference_x = reference_state[:, None, None, :].expand(b, l, k, -1)

        x = torch.cat([state_x, slot_x, primitive_x, reference_x], dim=-1)
        return self.router(x).squeeze(-1)

    def _transition(self, state: Tensor, slot: Tensor, primitive: Tensor, reference_state: Tensor, selected_slot_gate: Tensor, valid_step: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        state_context = self.state_norm(state)
        x = torch.cat([state_context, slot, primitive, reference_state], dim=-1)
        proposed_delta = self.transition_delta(x)
        alpha = (self.transition_strength(x).squeeze(-1) * self.alpha_max)
        effective_strength = alpha * selected_slot_gate
        state_update = (effective_strength[:, None]* proposed_delta)
        proposed_next = state + state_update
        next_state = torch.where(valid_step[:, None], proposed_next, state)
        actual_change = next_state - state
        proposed_delta = torch.where(valid_step[:, None], proposed_delta, torch.zeros_like(proposed_delta))
        alpha = torch.where(valid_step, alpha, torch.zeros_like(alpha),)

        return next_state, proposed_delta,alpha, actual_change

    def execute(self, edit_slots: Tensor, slot_gates: Tensor, z0: Tensor, reference_state: Tensor, *, disabled_slots: Tensor | None = None) -> dict[str, Tensor]:
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
        hard_active_slots = (slot_gates.detach() >= self.slot_gate_threshold) & ~disabled_slots

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
            if self.training and not self.hard_slot_gating_during_training:
                candidate_mask = ~completed & ~disabled_slots
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
            selected_gate = torch.einsum("bl,bl->b", route.sum(2), slot_gates)
            next_state, proposed_delta, alpha, actual_change = self._transition(state,selected_slot, selected_primitive, reference_state, selected_gate, valid_step)
            confidence = soft.reshape(b, -1).gather(1, hard_index[:, None]).squeeze(1)
            confidence = torch.where(valid_step, confidence, torch.zeros_like(confidence))
            slot_ids = torch.where(valid_step, raw_slot_ids, torch.full_like(raw_slot_ids, -1))
            primitive_ids = torch.where(valid_step, raw_primitive_ids, torch.full_like(raw_primitive_ids, -1))
            selected_gate = torch.where(valid_step, selected_gate, torch.zeros_like(selected_gate))
            selected_is_hard_active = valid_step & (selected_gate.detach() >= self.slot_gate_threshold)

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

    def forward(self, reference_features: Tensor, text_states: Tensor, text_attention_mask: Tensor, *, disable_execution: bool = False, disabled_slots: Tensor | None = None) -> dict[str, Tensor]:
        slot_output = self.build_edit_slots(reference_features, text_states, text_attention_mask)
        z0, reference_state = self.initialize_state(reference_features)

        if disable_execution:
            disabled_slots = torch.ones(slot_output["edit_slots"].shape[:2], dtype=torch.bool, device=slot_output["edit_slots"].device,)

        execution = self.execute(
            slot_output["edit_slots"],
            slot_output["slot_gates"],
            z0,
            reference_state,
            disabled_slots=disabled_slots,
        )

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
    def _positive_mask(target_ids: object, batch_size: int, device: torch.device) -> Tensor:
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
        targets = F.normalize(targets, dim=-1)
        logits = (query @ targets.t()) / self.retrieval_temperature

        if target_ids is None:
            labels = torch.arange(query.shape[0], device=query.device)
            return F.cross_entropy(logits, labels)

        positive = self._positive_mask(target_ids, query.shape[0], query.device)
        positive_logits = logits.masked_fill(~positive, float("-inf"))
        log_numerator = torch.logsumexp(positive_logits, dim=1)
        log_denominator = torch.logsumexp(logits, dim=1)
        return (log_denominator - log_numerator).mean()

    def _slot_regularizers(self, slot_masks: Tensor, slot_effects: Tensor, slot_gates: Tensor, text_attention_mask: Tensor, text_content_mask: Tensor | None = None) -> dict[str, Tensor]:
        attention_valid = text_attention_mask[:, None, :].to(slot_masks.dtype)
        attention_count = attention_valid.sum().clamp_min(1.0)
        gated_masks = slot_masks * slot_gates[:, :, None]

        sparse = (gated_masks * attention_valid).sum() / (attention_count * self.num_slots)
        if text_content_mask is None:
            coverage = slot_masks.new_zeros(())
        else:
            if text_content_mask.shape != text_attention_mask.shape:
                raise ValueError("text_content_mask must match text_attention_mask")
            coverage_valid = (text_content_mask.to(torch.bool) & text_attention_mask.to(torch.bool))[:, None, :].to(slot_masks.dtype)
            coverage_count = coverage_valid.sum().clamp_min(1.0)
            max_claim = gated_masks.max(1, keepdim=True).values
            coverage = (((1.0 - max_claim) ** 2) * coverage_valid).sum() / coverage_count

        gate_sparsity = slot_gates.mean()
        if self.num_slots == 1:
            overlap = slot_masks.new_zeros(())
            effect_diversity = slot_masks.new_zeros(())
        else:
            eye = torch.eye(self.num_slots, device=slot_masks.device, dtype=torch.bool)[None]
            offdiag = ~eye
            pair_weight = (slot_gates[:, :, None] * slot_gates[:, None, :])
            mask_vectors = F.normalize(slot_masks * attention_valid, dim=-1, eps=1e-6)
            mask_similarity = mask_vectors @ mask_vectors.transpose(1, 2)
            overlap_penalty = F.relu(mask_similarity - self.overlap_margin)
            overlap_weight = pair_weight * offdiag.to(pair_weight.dtype)
            overlap = (overlap_penalty * overlap_weight).sum() / overlap_weight.sum().clamp_min(1e-6)
            effect_vectors = F.normalize(slot_effects, dim=-1, eps=1e-6)
            effect_similarity = effect_vectors @ effect_vectors.transpose(1, 2)
            effect_penalty = F.relu(effect_similarity - self.effect_diversity_margin)
            effect_weight = pair_weight * offdiag.to(pair_weight.dtype)
            effect_diversity = (effect_penalty * effect_weight).sum() / effect_weight.sum().clamp_min(1e-6)

        return {
            "slot_sparse_loss": sparse,
            "slot_coverage_loss": coverage,
            "slot_overlap_loss": overlap,
            "slot_effect_diversity_loss": effect_diversity,
            "slot_gate_sparsity_loss": gate_sparsity,
        }

    def compute_loss(self, batch: Mapping[str, object]) -> dict[str, Tensor]:
        required = {
            "reference_features",
            "text_states",
            "text_attention_mask",
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
            raise TypeError("reference_features, text_states, text_attention_mask, and target_features must be Tensors")
        assert isinstance(reference, Tensor)
        assert isinstance(text, Tensor)
        assert isinstance(mask, Tensor)
        assert isinstance(targets, Tensor)

        if targets.ndim != 2 or targets.shape[-1] != self.query_dim:
            raise ValueError(f"target_features must be [B,{self.query_dim}]")

        content_mask = batch.get("text_content_mask")
        if content_mask is not None and not isinstance(content_mask, Tensor):
            raise TypeError("text_content_mask must be a Tensor when provided")

        output = self.forward(reference, text, mask)
        losses = {
            "retrieval_loss": self._retrieval_loss(
                output["q0"],
                targets,
                batch.get("target_ids"),
            )
        }
        losses.update(
            self._slot_regularizers(
                output["slot_masks"],
                output["slot_effects"],
                output["slot_gates"],
                mask,
                content_mask,
            )
        )

        return losses

    @torch.no_grad()
    def slot_drop_queries(self, *, reference_features: Tensor, text_states: Tensor, text_attention_mask: Tensor) -> dict[str, Tensor]:
        was_training = self.training
        self.eval()
        try:
            slots = self.build_edit_slots(reference_features, text_states, text_attention_mask,)
            z0, reference_state = self.initialize_state(reference_features)
            full_execution = self.execute(slots["edit_slots"], slots["slot_gates"], z0, reference_state,)
            full_query = self.make_query(full_execution["final_state"])

            dropped_queries = []
            b = reference_features.shape[0]
            for slot_id in range(self.num_slots):
                disabled = torch.zeros(b, self.num_slots, dtype=torch.bool, device=reference_features.device,)
                disabled[:, slot_id] = True
                execution = self.execute(slots["edit_slots"], slots["slot_gates"], z0, reference_state, disabled_slots=disabled,)
                dropped_queries.append(self.make_query(execution["final_state"]))

            return {
                "full_query": full_query,
                "dropped_queries": torch.stack(dropped_queries, dim=1),
                "slot_gates": slots["slot_gates"],
                "hard_active_slot_mask": full_execution["hard_active_slot_mask"],
            }
        finally:
            self.train(was_training)

    def compute_stage1_loss(self, batch: Mapping[str, object]) -> dict[str, Tensor]:
        reference = batch["teacher_reference_features"]
        text = batch["text_states"]
        attention_mask = batch["text_attention_mask"]
        output = self.build_edit_slots(reference, text, attention_mask)
        return self._slot_regularizers(
            output["slot_masks"],
            output["slot_effects"],
            output["slot_gates"],
            attention_mask,
            batch.get("text_content_mask"),
        )

    @torch.no_grad()
    def retrieve(self, *, reference_features: Tensor, text_states: Tensor, text_attention_mask: Tensor, gallery_features: Tensor, topk: int | None = None) -> dict[str, Tensor]:
        if gallery_features.ndim != 2 or gallery_features.shape[-1] != self.query_dim:
            raise ValueError(f"gallery_features must be [G,{self.query_dim}]")

        was_training = self.training
        self.eval()
        try:
            output = self.forward(reference_features, text_states, text_attention_mask,)
            gallery = F.normalize(gallery_features, dim=-1,)
            scores = output["q0"] @ gallery.t()
            result = {
                **output,
                "scores": scores,
            }
            if topk is not None:
                if topk < 1:
                    raise ValueError(
                        "topk must be >= 1 when provided"
                    )

                k = min(topk,gallery.shape[0])
                top_scores, top_indices = scores.topk(k,dim=1)
                result.update(
                    {
                        "top_scores": top_scores,
                        "top_indices": top_indices,
                    }
                )
            return result
        finally:
            self.train(was_training)