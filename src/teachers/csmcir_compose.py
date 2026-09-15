from __future__ import annotations

import gc
from pathlib import Path

import torch
from torch import nn

from teachers.csmcir import CSMCIRStage1Teacher


class CSMCIRComposeTeacher(nn.Module):
    """
    Minimal frozen CSMCIR module required by TAPER E2E.

    Online training already receives cached:
      - native reference features
      - teacher-native text states
      - contextual TAPER text states
      - text masks

    Therefore the only remaining teacher operation is compose().
    """

    def __init__(
        self,
        *,
        csmcir_root: str | Path,
        checkpoint_path: str | Path,
    ) -> None:
        super().__init__()

        # Important:
        # Load the full teacher on CPU only.
        # We never want the visual encoder to become a persistent CUDA module.
        full_teacher = CSMCIRStage1Teacher(
            csmcir_root=csmcir_root,
            checkpoint_path=checkpoint_path,
            device="cpu",
        ).eval()

        model = full_teacher.model

        # compose() only requires these three pieces.
        self.bert = model.Qformer.bert
        self.text_proj = model.text_proj

        # query_tokens is frozen teacher state.
        # Keep a private copy so this module does not depend on the parent model.
        self.query_tokens = nn.Parameter(
            model.query_tokens.detach().clone(),
            requires_grad=False,
        )

        # Freeze every teacher parameter.
        for parameter in self.parameters():
            parameter.requires_grad_(False)

        self.eval()

        # Drop references to all unused CSMCIR components:
        # visual encoder, tokenizer-side model state, vision projection, etc.
        del model
        del full_teacher

        gc.collect()

    def train(
        self,
        mode: bool = True,
    ) -> "CSMCIRComposeTeacher":
        # Teacher must always stay in eval mode.
        super().train(False)
        return self

    def compose(
        self,
        reference_features: torch.Tensor,
        text_states: torch.Tensor,
        text_attention_mask: torch.Tensor,
        normalize: bool = False,
    ) -> torch.Tensor:
        if reference_features.ndim != 3:
            raise ValueError(
                "CSMCIR reference_features must be [B,K,D], "
                f"got {tuple(reference_features.shape)}"
            )

        if text_states.ndim != 3:
            raise ValueError(
                "text_states must be [B,N,D]"
            )

        if (
            text_attention_mask.shape
            != (
                text_states.shape[0],
                text_states.shape[1],
            )
        ):
            raise ValueError(
                "text_attention_mask shape mismatch"
            )

        if (
            reference_features.shape[0]
            != text_states.shape[0]
        ):
            raise ValueError(
                "reference/text batch size mismatch"
            )

        if not torch.isfinite(reference_features).all():
            raise FloatingPointError(
                "reference_features contain NaN/Inf"
            )

        if not torch.isfinite(text_states).all():
            raise FloatingPointError(
                "text_states contain NaN/Inf"
            )

        batch_size = reference_features.shape[0]
        device = reference_features.device

        query_tokens = self.query_tokens.expand(
            batch_size,
            -1,
            -1,
        )

        query_atts = torch.ones(
            query_tokens.shape[:-1],
            dtype=torch.long,
            device=device,
        )

        image_atts = torch.ones(
            reference_features.shape[:-1],
            dtype=torch.long,
            device=device,
        )

        attention_mask = torch.cat(
            [
                query_atts,
                text_attention_mask.to(
                    device=device,
                    dtype=torch.long,
                ),
            ],
            dim=1,
        )

        output = self.bert(
            inputs_embeds=text_states,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=reference_features,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        num_query_tokens = query_tokens.shape[1]

        q_pre = self.text_proj(
            output.last_hidden_state[
                :,
                num_query_tokens,
                :
            ]
        )

        if not torch.isfinite(q_pre).all():
            raise FloatingPointError(
                "CSMCIR compose produced NaN/Inf"
            )

        if normalize:
            return torch.nn.functional.normalize(
                q_pre,
                dim=-1,
            )

        return q_pre