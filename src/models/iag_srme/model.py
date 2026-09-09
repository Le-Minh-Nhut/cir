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


class ProposalNet(nn.Module):
    """Fresh, state-conditioned WHAT proposals with token-only semantic values."""

    def __init__(
        self, state_dim: int, text_dim: int, num_candidates: int, num_heads: int
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
        q = self.queries.unsqueeze(0).expand(text.shape[0], -1, -1)
        conditioned = self.query_conditioner(
            torch.cat([q, context[:, None].expand_as(q), q * context[:, None]], dim=-1)
        )
        q = self.query_norm(q + conditioned)
        # No q residual: candidate content comes only from instruction token values.
        edits, _ = self.token_attention(
            q,
            text_tokens,
            text_tokens,
            key_padding_mask=~content_mask,
            need_weights=False,
        )
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
            state_dim, text_dim, config.num_candidates, config.num_heads
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
            best_score, best_idx = scores.max(dim=-1)
            stop_now = (
                best_score <= self.config.epsilon_stop
                if self.config.stop_enabled
                else torch.zeros_like(best_score, dtype=torch.bool)
            )
            selected_idx = torch.where(
                stop_now,
                torch.full_like(best_idx, self.config.num_candidates),
                best_idx,
            )

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
                    "candidate_queries": candidate_queries,
                    "delta_q": delta_q,
                    "scores": scores,
                    "stop_score": scores.new_zeros(scores.shape[0]),
                    "best_score": best_score,
                    "selected_idx": selected_idx,
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
