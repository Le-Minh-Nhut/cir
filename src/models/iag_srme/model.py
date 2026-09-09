from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class IAGSRMEConfig:
    width: int = 256
    num_context_edits: int = 4
    num_heads: int = 8
    exec_dim: int = 256
    read_scale_init: float = 10.0
    exec_scale_init: float = 10.0
    exec_bias_init: float = 0.0
    scale_min: float = 1.0
    scale_max: float = 100.0


class ProposalNet(nn.Module):
    """Initial-state-conditioned context edits with token-only semantic values."""

    def __init__(
        self, state_dim: int, text_dim: int, num_context_edits: int, num_heads: int
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.empty(num_context_edits, state_dim))
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

        # Learned rows are ordered latent context-edit slots, not semantic labels.
        q = self.queries.unsqueeze(0).expand(text.shape[0], -1, -1)
        conditioned = self.query_conditioner(
            torch.cat([q, context[:, None].expand_as(q), q * context[:, None]], dim=-1)
        )
        q = self.query_norm(q + conditioned)
        # No q residual: context-edit content comes only from instruction token values.
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
        # edits: [B,S,Dv], dense: [B,N,Dd]
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
        nn.init.normal_(self.action_mlp[-1].weight, std=1e-3)
        nn.init.zeros_(self.action_mlp[-1].bias)
        nn.init.normal_(self.state_up.weight, std=1e-3)
        nn.init.zeros_(self.state_up.bias)

    def forward(
        self,
        parent: Tensor,
        actions: Tensor,
        exec_mask: Tensor,
        patch_grid: tuple[int, int],
    ) -> tuple[Tensor, Tensor, Tensor]:
        # parent: [B,N,Dv], actions: [B,S,Da], exec_mask: [B,S,N]
        batch, tokens, _ = parent.shape
        slots = actions.shape[1]
        height, width = patch_grid
        if tokens != height * width:
            raise ValueError(f"state has {tokens} patches but grid is {height}x{width}")

        x = self.state_down(parent)  # [B,N,dx], computed once for the shared parent
        gamma, beta = self.action_mlp(self.action_norm(actions)).chunk(2, dim=-1)
        normalized = self.state_norm(x)[:, None]
        modulated = (1.0 + gamma[:, :, None]) * normalized + beta[:, :, None]

        local = modulated.reshape(batch * slots, height, width, -1)
        local = local.permute(0, 3, 1, 2)
        local = local + self.local_mixer(local)
        local = local.permute(0, 2, 3, 1).reshape(batch, slots, tokens, -1)
        residual_features = local + self.ffn(self.ffn_norm(local))
        raw_delta = self.state_up(residual_features)  # signed: deliberately no activation
        delta = exec_mask[..., None] * raw_delta
        next_states = parent[:, None] + delta
        return raw_delta, delta, next_states


class IAGSRME(nn.Module):
    """Target-free sequential execution of one fixed set of context edits."""

    def __init__(self, backbone: nn.Module, config: IAGSRMEConfig = IAGSRMEConfig()) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config
        state_dim = int(backbone.state_dim)
        text_dim = int(backbone.text_dim)
        dense_dim = int(backbone.dense_dim)
        self.proposal = ProposalNet(
            state_dim, text_dim, config.num_context_edits, config.num_heads
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

    def forward_from_features(
        self,
        initial_state: Tensor,
        text_tokens: Tensor,
        text_global: Tensor,
        content_mask: Tensor,
        cls_anchor: Tensor | None = None,
    ) -> dict[str, object]:
        # Generate all ordered context edits once from the immutable initial state.
        V0 = initial_state
        V = V0.clone()
        cls_anchor0 = cls_anchor
        initial_global = self.backbone.global_readout(V, cls_anchor0)
        context_edits = self.proposal(
            text_tokens,
            text_global,
            initial_global,
            content_mask,
        )
        steps: list[dict[str, Tensor | int | None]] = []

        for slot in range(self.config.num_context_edits):
            parent = V
            current_global = (
                initial_global
                if slot == 0
                else self.backbone.global_readout(parent, cls_anchor0)
            )
            current_dense = self.backbone.dense_readout(parent)
            edit = context_edits[:, slot : slot + 1]
            grounding, alpha_read, exec_mask = self.grounder(edit, current_dense)
            entity = torch.einsum("bsn,bnd->bsd", alpha_read, parent)
            action, fuse_gamma, fuse_beta = self.action_fusion(entity, edit)
            raw_delta, delta, next_states = self.executor(
                parent,
                action,
                exec_mask,
                self.backbone.patch_grid,
            )
            V = next_states[:, 0]

            steps.append(
                {
                    "slot": slot,
                    "cls_anchor": cls_anchor0,
                    "parent_state": parent,
                    "current_global": current_global,
                    "prefix_query": self.backbone.retrieval_from_global(current_global),
                    "edit": edit[:, 0],
                    "grounding": grounding[:, 0],
                    "alpha_read": alpha_read[:, 0],
                    "exec_mask": exec_mask[:, 0],
                    "entity": entity[:, 0],
                    "action": action[:, 0],
                    "fuse_gamma": fuse_gamma[:, 0],
                    "fuse_beta": fuse_beta[:, 0],
                    "raw_delta": raw_delta[:, 0],
                    "delta": delta[:, 0],
                    "state": V,
                }
            )

        terminal_query = self.backbone.retrieval_readout(V, cls_anchor0)
        return {
            "query": terminal_query,
            "state": V,
            "initial_state": V0,
            "cls_anchor": cls_anchor0,
            "context_edits": context_edits,
            "steps": steps,
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
