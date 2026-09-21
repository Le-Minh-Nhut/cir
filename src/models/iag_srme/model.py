from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class IAGSRMEConfig:
    width: int = 256
    num_candidates: int = 4
    max_steps: int = 3
    num_heads: int = 8
    exec_dim: int = 256
    epsilon_stop: float = 0.0
    stop_enabled: bool = True
    read_scale_init: float = 10.0
    exec_scale_init: float = 10.0
    exec_bias_init: float = 0.0
    scale_min: float = 1.0
    scale_max: float = 100.0
    score_dropout: float = 0.1
    loss_free_balance_enabled: bool = False
    loss_free_bias_update_rate: float = 1.0e-3
    functional_collapse_audit_enabled: bool = False
    proposal_internal_audit_enabled: bool = False
    proposal_mode: str = "attention"

    def __post_init__(self) -> None:
        if self.proposal_mode not in {"attention", "attention_ln", "residual", "residual_ln"}:
            raise ValueError(f"unsupported proposal_mode: {self.proposal_mode}")


class ProposalNet(nn.Module):
    """Fresh, state-conditioned WHAT proposals with token-only semantic values."""

    def __init__(
        self, state_dim: int, text_dim: int, num_candidates: int, num_heads: int, proposal_mode: str = "attention"
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.empty(num_candidates, state_dim))
        nn.init.normal_(self.queries, std=0.02)
        self.text_context = nn.Linear(text_dim, state_dim)
        self.visual_context = nn.Linear(state_dim, state_dim)
        self.context = nn.Sequential(
            nn.Linear(4 * state_dim, 2 * state_dim),
            nn.GELU(),
            nn.Linear(2 * state_dim, state_dim),
        )
        self.query_conditioner = nn.Sequential(
            nn.Linear(3 * state_dim, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, state_dim),
        )
        self.query_norm = nn.LayerNorm(state_dim)
        self.token_attention = nn.MultiheadAttention(
            state_dim,
            num_heads,
            kdim=text_dim,
            vdim=text_dim,
            batch_first=True,
        )
        if proposal_mode not in {"attention", "attention_ln", "residual", "residual_ln"}:
            raise ValueError(f"unsupported proposal_mode: {proposal_mode}")
        self.proposal_mode = proposal_mode
        self.output_norm = nn.LayerNorm(state_dim) if proposal_mode.endswith("_ln") else None
        self.internal_audit_enabled = False
        self.last_internal_audit: dict[str, Tensor] | None = None

    def forward(
        self,
        text_tokens: Tensor,
        text_global: Tensor,
        current_global: Tensor,
        content_mask: Tensor,
    ) -> Tensor:
        # text_tokens: [B,L,Dt], current_global: [B,Dv]
        text = self.text_context(text_global)
        visual = self.visual_context(current_global)
        context = self.context(
            torch.cat([text, visual, text * visual, text - visual], dim=-1)
        )

        # Learned rows are capacity priors, not fixed semantic identities.
        base_query = self.queries.unsqueeze(0)
        expanded_query = base_query.expand(text.shape[0], -1, -1)
        conditioned = self.query_conditioner(
            torch.cat(
                [
                    expanded_query,
                    context[:, None].expand_as(expanded_query),
                    expanded_query * context[:, None],
                ],
                dim=-1,
            )
        )
        query_pre_norm = expanded_query + conditioned
        query_post_norm = self.query_norm(query_pre_norm)
        raw_attention, _ = self.token_attention(
            query_post_norm,
            text_tokens,
            text_tokens,
            key_padding_mask=~content_mask,
            need_weights=False,
        )
        edits = raw_attention + query_post_norm if self.proposal_mode.startswith("residual") else raw_attention
        if self.output_norm is not None:
            edits = self.output_norm(edits)
        # Attention weights are intentionally omitted: need_weights=False preserves this call's kernels.
        if self.internal_audit_enabled:
            self.last_internal_audit = {
                "base_query": base_query.detach().clone(),
                "expanded_query": expanded_query.detach().clone(),
                "conditioned_residual": conditioned.detach(),
                "query_pre_norm": query_pre_norm.detach(),
                "query_post_norm": query_post_norm.detach(),
                "raw_attention_output": raw_attention.detach(),
                "proposal_output": edits.detach(),
            }
        else:
            self.last_internal_audit = None
        return edits


class Grounder(nn.Module):
    """Native-dense cosine WHERE grounding with distinct read and write maps."""

    def __init__(
        self,
        state_dim: int,
        dense_dim: int,
        read_scale: float = 10.0,
        exec_scale: float = 10.0,
        exec_bias: float = 0.0,
        scale_bounds: tuple[float, float] = (1.0, 100.0),
    ) -> None:
        super().__init__()
        self.where = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, dense_dim),
        )
        self.log_read_scale = nn.Parameter(torch.tensor(read_scale).log())
        self.log_exec_scale = nn.Parameter(torch.tensor(exec_scale).log())
        self.exec_bias = nn.Parameter(torch.tensor(exec_bias))
        self.scale_bounds = scale_bounds

    def forward(self, edits: Tensor, dense: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # edits: [B,K,Dv], dense: [B,N,Dd]
        query = F.normalize(self.where(edits).float(), dim=-1)
        keys = F.normalize(dense.float(), dim=-1)
        raw = torch.einsum("bkd,bnd->bkn", query, keys)
        low, high = self.scale_bounds
        read_scale = self.log_read_scale.exp().clamp(low, high)
        exec_scale = self.log_exec_scale.exp().clamp(low, high)
        alpha_read = torch.softmax(read_scale * raw, dim=-1)
        exec_mask = torch.sigmoid(exec_scale * (raw - self.exec_bias))
        return raw.to(edits.dtype), alpha_read.to(edits.dtype), exec_mask.to(edits.dtype)


class ActionFusion(nn.Module):
    """Entity-conditioned bounded independent dual-gate fusion."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gates = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, 2 * dim),
        )

    def forward(self, entity: Tensor, edit: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        gamma, beta = torch.sigmoid(self.gates(torch.cat([entity, edit], dim=-1))).chunk(
            2, dim=-1
        )
        action = gamma * entity + beta * edit
        return action, gamma, beta


class Executor(nn.Module):
    """Action-conditioned, masked local signed-residual state transition."""

    def __init__(self, state_dim: int, action_dim: int, exec_dim: int = 256) -> None:
        super().__init__()
        self.state_down = nn.Linear(state_dim, exec_dim)
        self.state_norm = nn.LayerNorm(exec_dim)
        self.action_norm = nn.LayerNorm(action_dim)
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, 2 * exec_dim),
            nn.SiLU(),
            nn.Linear(2 * exec_dim, 2 * exec_dim),
        )
        self.local_mixer = nn.Conv2d(
            exec_dim, exec_dim, kernel_size=3, padding=1, groups=exec_dim
        )
        self.ffn_norm = nn.LayerNorm(exec_dim)
        self.ffn = nn.Sequential(
            nn.Linear(exec_dim, 4 * exec_dim),
            nn.GELU(),
            nn.Linear(4 * exec_dim, exec_dim),
        )
        self.state_up = nn.Linear(exec_dim, state_dim)
        nn.init.zeros_(self.action_mlp[-1].weight)
        nn.init.zeros_(self.action_mlp[-1].bias)
        nn.init.zeros_(self.state_up.weight)
        nn.init.zeros_(self.state_up.bias)

    def forward(
        self,
        parent: Tensor,
        actions: Tensor,
        exec_mask: Tensor,
        patch_grid: tuple[int, int],
    ) -> tuple[Tensor, Tensor, Tensor]:
        # parent: [B,N,Dv], actions: [B,K,Da], exec_mask: [B,K,N]
        batch, tokens, _ = parent.shape
        candidates = actions.shape[1]
        height, width = patch_grid
        if tokens != height * width:
            raise ValueError(f"state has {tokens} patches but grid is {height}x{width}")

        x = self.state_down(parent)  # [B,N,dx], computed once for the shared parent
        gamma, beta = self.action_mlp(self.action_norm(actions)).chunk(2, dim=-1)
        normalized = self.state_norm(x)[:, None]
        modulated = (1.0 + gamma[:, :, None]) * normalized + beta[:, :, None]

        local = modulated.reshape(batch * candidates, height, width, -1)
        local = local.permute(0, 3, 1, 2)
        local = local + self.local_mixer(local)
        local = local.permute(0, 2, 3, 1).reshape(batch, candidates, tokens, -1)
        residual_features = local + self.ffn(self.ffn_norm(local))
        raw_delta = self.state_up(residual_features)  # signed: deliberately no activation
        delta = exec_mask[..., None] * raw_delta
        candidate_states = parent[:, None] + delta
        return raw_delta, delta, candidate_states


class ScoreNet(nn.Module):
    """Shared independent predictor of marginal utility versus KEEP state."""

    def __init__(
        self, state_dim: int, text_dim: int, action_dim: int, dim: int = 256, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.context_global = nn.Linear(state_dim, dim)
        self.context_text = nn.Linear(text_dim, dim)
        self.context_norm = nn.LayerNorm(dim)
        self.action_projection = nn.Linear(action_dim, dim)
        self.local_projection = nn.Linear(state_dim, dim)
        self.global_projection = nn.Linear(state_dim, dim)
        self.net = nn.Sequential(
            nn.LayerNorm(5 * dim),
            nn.Linear(5 * dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def build_features(
        self,
        current_global: Tensor,
        text_global: Tensor,
        actions: Tensor,
        delta: Tensor,
        exec_mask: Tensor,
        candidate_global: Tensor,
    ) -> Tensor:
        # All candidate groups below are [B,K,d].
        context = self.context_norm(
            self.context_global(current_global) + self.context_text(text_global)
        )
        action = self.action_projection(actions)
        support = exec_mask.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        local_mean = delta.sum(dim=-2) / support  # delta is already support-masked
        local_effect = self.local_projection(local_mean)
        global_effect = self.global_projection(candidate_global - current_global[:, None])
        compatibility = action * global_effect
        context = context[:, None].expand_as(action)
        return torch.cat(
            [context, action, local_effect, global_effect, compatibility], dim=-1
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features).squeeze(-1)


class IAGSRME(nn.Module):
    """Canonical target-free V2 R0 recurrent model."""

    def __init__(self, backbone: nn.Module, config: IAGSRMEConfig = IAGSRMEConfig()) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config
        state_dim = int(backbone.state_dim)
        text_dim = int(backbone.text_dim)
        dense_dim = int(backbone.dense_dim)
        self.proposal = ProposalNet(
            state_dim, text_dim, config.num_candidates, config.num_heads, config.proposal_mode
        )
        self.grounder = Grounder(
            state_dim,
            dense_dim,
            config.read_scale_init,
            config.exec_scale_init,
            config.exec_bias_init,
            (config.scale_min, config.scale_max),
        )
        self.action_fusion = ActionFusion(state_dim)
        self.executor = Executor(state_dim, state_dim, config.exec_dim)
        self.score_net = ScoreNet(
            state_dim, text_dim, state_dim, config.width, config.score_dropout
        )
        self.proposal.internal_audit_enabled = config.proposal_internal_audit_enabled
        self.register_buffer("routing_bias", torch.zeros(config.num_candidates))

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # V2-R0 checkpoints predate this persistent controller state.
        key = prefix + "routing_bias"
        if key not in state_dict:
            state_dict[key] = torch.zeros_like(self.routing_bias)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def route_scores(self, raw_scores: Tensor) -> dict[str, Tensor]:
        """Route with a detached historical slot bias while preserving raw STOP semantics."""

        if raw_scores.shape[-1] != self.config.num_candidates:
            raise ValueError("raw scores have the wrong candidate dimension")
        raw_best_score, raw_best_idx = raw_scores.max(dim=-1)
        selection_scores = raw_scores + (
            self.routing_bias.detach() if self.config.loss_free_balance_enabled else 0.0
        )
        if self.config.stop_enabled:
            eligible = raw_scores > self.config.epsilon_stop
            routed_idx = selection_scores.masked_fill(~eligible, -torch.inf).argmax(dim=-1)
            stop_now = ~eligible.any(dim=-1)
        else:
            routed_idx = selection_scores.argmax(dim=-1)
            stop_now = torch.zeros_like(raw_best_score, dtype=torch.bool)
        selected_idx = torch.where(
            stop_now,
            torch.full_like(routed_idx, self.config.num_candidates),
            routed_idx,
        )
        selected_raw_score = raw_scores.gather(1, routed_idx[:, None]).squeeze(1)
        return {
            "selection_scores": selection_scores,
            "raw_best_score": raw_best_score,
            "raw_best_idx": raw_best_idx,
            "routed_idx": routed_idx,
            "selected_idx": selected_idx,
            "selected_raw_score": selected_raw_score,
            "stop_now": stop_now,
        }

    @torch.no_grad()
    def update_routing_bias(self, output: dict[str, object]) -> dict[str, object]:
        """Apply Algorithm 1 once after a completed training batch and return compact metrics."""

        candidates = self.config.num_candidates
        routing_bias_before = self.routing_bias.detach().clone()
        counts = torch.zeros(candidates, device=self.routing_bias.device, dtype=torch.long)
        raw_counts_all_decisions = torch.zeros_like(counts)
        raw_counts_executed = torch.zeros_like(counts)
        raw_scores: list[Tensor] = []
        stop_count = 0
        decisions = 0
        disagreements = 0
        for step in output["steps"]:
            assert isinstance(step, dict)
            scores = step["scores"]
            selected = step["selected_idx"]
            raw_best = step["raw_best_idx"]
            routed = step["routed_idx"]
            assert isinstance(scores, Tensor)
            assert isinstance(selected, Tensor)
            assert isinstance(raw_best, Tensor)
            assert isinstance(routed, Tensor)
            execute = selected < candidates
            raw_counts_all_decisions += torch.bincount(raw_best, minlength=candidates)
            if execute.any():
                counts += torch.bincount(routed[execute], minlength=candidates)
                raw_counts_executed += torch.bincount(raw_best[execute], minlength=candidates)
                disagreements += int((raw_best[execute] != routed[execute]).sum())
            stop_count += int((~execute).sum())
            decisions += selected.numel()
            raw_scores.append(scores.detach().float().reshape(-1))

        total = int(counts.sum())
        if self.training and self.config.loss_free_balance_enabled and total:
            mean_count = counts.float().mean()
            self.routing_bias.add_(
                self.config.loss_free_bias_update_rate * torch.sign(mean_count - counts)
            )
        executed_denominator = max(total, 1)
        decision_denominator = max(decisions, 1)
        score_values = torch.cat(raw_scores) if raw_scores else self.routing_bias.new_zeros(1)
        mean_count = counts.float().mean()
        max_vio = (
            float((counts.float().max() - mean_count) / mean_count) if total else 0.0
        )
        routing_bias_after = self.routing_bias.detach().clone()
        return {
            "loss_free_balance_enabled": self.config.loss_free_balance_enabled,
            "routing_bias": routing_bias_after.float().cpu().tolist(),
            "routing_bias_before": routing_bias_before.float().cpu().tolist(),
            "routing_bias_after": routing_bias_after.float().cpu().tolist(),
            "routing_bias_delta": (routing_bias_after - routing_bias_before).float().cpu().tolist(),
            "committed_selection_count": counts.cpu().tolist(),
            "committed_selection_fraction": (counts.float() / executed_denominator).cpu().tolist(),
            "raw_argmax_all_decision_count": raw_counts_all_decisions.cpu().tolist(),
            "raw_argmax_all_decision_fraction": (
                raw_counts_all_decisions.float() / decision_denominator
            ).cpu().tolist(),
            "raw_argmax_executed_count": raw_counts_executed.cpu().tolist(),
            "raw_argmax_executed_fraction": (
                raw_counts_executed.float() / executed_denominator
            ).cpu().tolist(),
            "routed_selection_count": counts.cpu().tolist(),
            "routed_selection_fraction": (counts.float() / executed_denominator).cpu().tolist(),
            "raw_routed_disagreement_fraction": disagreements / executed_denominator,
            "routing_max_vio": max_vio,
            "raw_argmax_all_decision_monopoly_fraction": float(
                raw_counts_all_decisions.max() / decision_denominator
            ),
            "raw_argmax_executed_monopoly_fraction": float(
                raw_counts_executed.max() / executed_denominator
            ),
            "raw_monopoly_fraction": float(raw_counts_executed.max() / executed_denominator),
            "routed_monopoly_fraction": float(counts.max() / executed_denominator),
            "raw_score_mean": float(score_values.mean()),
            "raw_score_std": float(score_values.std(unbiased=False)),
            "max_abs_routing_bias": float(self.routing_bias.abs().max()),
            "routing_bias_to_raw_score_std": float(
                self.routing_bias.abs().max() / (score_values.std(unbiased=False) + 1e-8)
            ),
            "stop_count": stop_count,
            "stop_rate": stop_count / decision_denominator,
        }

    @staticmethod
    def _gather_candidate(values: Tensor, indices: Tensor) -> Tensor:
        shape = [values.shape[0], 1] + [1] * (values.ndim - 2)
        index = indices.view(*shape).expand(-1, 1, *values.shape[2:])
        return values.gather(1, index).squeeze(1)

    def forward_from_features(
        self,
        initial_state: Tensor,
        text_tokens: Tensor,
        text_global: Tensor,
        content_mask: Tensor,
        cls_anchor: Tensor | None = None,
    ) -> dict[str, object]:
        # V0 is the immutable reference anchor; V is always updated out of place.
        V0 = initial_state
        V = V0.clone()
        cls_anchor0 = cls_anchor
        batch = V.shape[0]
        alive = torch.ones(batch, dtype=torch.bool, device=V.device)
        steps: list[dict[str, Tensor | int | None]] = []

        for timestep in range(self.config.max_steps):
            if not alive.any():
                break
            live_indices = alive.nonzero(as_tuple=False).squeeze(-1)
            parent = V.index_select(0, live_indices)
            tokens = text_tokens.index_select(0, live_indices)
            text = text_global.index_select(0, live_indices)
            mask = content_mask.index_select(0, live_indices)
            live_cls = (
                cls_anchor0.index_select(0, live_indices)
                if cls_anchor0 is not None
                else None
            )

            current_global = self.backbone.global_readout(parent, live_cls)
            dense = self.backbone.dense_readout(parent)
            edits = self.proposal(tokens, text, current_global, mask)
            proposal_internal = self.proposal.last_internal_audit
            grounding, alpha_read, exec_mask = self.grounder(edits, dense)
            entity = torch.einsum("bkn,bnd->bkd", alpha_read, parent)
            actions, fuse_gamma, fuse_beta = self.action_fusion(entity, edits)
            _, delta, candidate_states = self.executor(
                parent, actions, exec_mask, self.backbone.patch_grid
            )

            # Compute each expensive final-block readout once, then project it.
            current_query = self.backbone.retrieval_from_global(current_global)
            candidate_global = self.backbone.global_readout(candidate_states, live_cls)
            candidate_queries = self.backbone.retrieval_from_global(candidate_global)
            delta_q = candidate_queries - current_query[:, None]
            score_features = self.score_net.build_features(
                current_global.detach(),
                text.detach(),
                actions.detach(),
                delta.detach(),
                exec_mask.detach(),
                candidate_global.detach(),
            )
            # Score losses train all ScoreNet projections, but never upstream modules.
            scores = self.score_net(score_features)
            routing = self.route_scores(scores)
            # `scores` remains the raw ScoreNet prediction for every loss and diagnostic.
            best_score = routing["raw_best_score"]
            best_idx = routing["routed_idx"]
            stop_now = routing["stop_now"]
            selected_idx = routing["selected_idx"]

            execute = ~stop_now
            if execute.any():
                execute_rows = live_indices[execute]
                chosen = self._gather_candidate(candidate_states[execute], best_idx[execute])
                V = V.index_copy(0, execute_rows, chosen)
            if stop_now.any():
                alive = alive.clone()
                alive[live_indices[stop_now]] = False

            steps.append(
                {
                    "timestep": timestep,
                    "live_indices": live_indices,
                    "cls_anchor": live_cls,
                    "parent_state": parent,
                    "current_global": current_global,
                    "current_query": current_query,
                    # Already computed live text; scorer-refit tooling detaches it when caching.
                    "text_global": text,
                    "proposals": edits,
                    "grounding": grounding,
                    "alpha_read": alpha_read,
                    "exec_mask": exec_mask,
                    "entities": entity,
                    "actions": actions,
                    "fuse_gamma": fuse_gamma,
                    "fuse_beta": fuse_beta,
                    "delta": delta,
                    "candidate_states": candidate_states,
                    # Already computed for ScoreNet/readout; exposed only for diagnostics.
                    "candidate_global": candidate_global,
                    "candidate_queries": candidate_queries,
                    "delta_q": delta_q,
                    **(
                        {"proposal_internal": proposal_internal}
                        if self.config.proposal_internal_audit_enabled
                        else {}
                    ),
                    "scores": scores,
                    "stop_score": scores.new_zeros(scores.shape[0]),
                    "best_score": best_score,
                    "selected_idx": selected_idx,
                    "selection_scores": routing["selection_scores"],
                    "routing_bias": self.routing_bias.detach().clone(),
                    "raw_best_idx": routing["raw_best_idx"],
                    "routed_idx": routing["routed_idx"],
                    "selected_raw_score": routing["selected_raw_score"],
                    "stopped_now": stop_now,
                }
            )

        terminal_query = self.backbone.retrieval_readout(V, cls_anchor0)
        return {
            "query": terminal_query,
            "state": V,
            "initial_state": V0,
            "cls_anchor": cls_anchor0,
            "steps": steps,
            "stopped": ~alive,
        }

    def forward(
        self,
        reference_images: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        content_mask: Tensor,
    ) -> dict[str, object]:
        if getattr(self.backbone, "global_readout_mode", "learned_qg") == "native_cls":
            initial_state, cls_anchor = self.backbone.initial_state_with_anchor(
                reference_images
            )
        else:
            initial_state = self.backbone.initial_state(reference_images)
            cls_anchor = None
        text_tokens, text_global = self.backbone.encode_text(
            input_ids, attention_mask, content_mask
        )
        return self.forward_from_features(
            initial_state, text_tokens, text_global, content_mask, cls_anchor
        )

    def encode_global_images(self, pixel_values: Tensor) -> Tensor:
        return self.backbone.encode_global_images(pixel_values)

    def encode_gallery(self, pixel_values: Tensor) -> Tensor:
        return self.encode_global_images(pixel_values)
