from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn


@contextmanager
def temporary_cwd(path: Path):
    old_cwd = Path.cwd()

    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old_cwd)


class CSMCIRStage1Teacher(nn.Module):
    def __init__(self, csmcir_root: str | Path, checkpoint_path: str | Path, device: str = "cpu") -> None:
        super().__init__()

        self.csmcir_root = Path(csmcir_root).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.load_device = torch.device(device)

        if not self.csmcir_root.is_dir():
            raise FileNotFoundError(f"CSMCIR repo not found: {self.csmcir_root}")

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"CSMCIR checkpoint not found: {self.checkpoint_path}")

        src_root = self.csmcir_root / "src"

        if not src_root.is_dir():
            raise FileNotFoundError(f"CSMCIR src not found: {src_root}")

        if str(src_root) not in sys.path:
            sys.path.insert(0, str(src_root))

        with temporary_cwd(self.csmcir_root):
            from data_utils_csmcir import targetpad_transform
            from lavis.models import load_model_and_preprocess

            model, _, txt_processors = load_model_and_preprocess(
                name="blip2_cir_align_prompt_csmcir",
                model_type="pretrain",
                is_eval=False,
                device=self.load_device,
            )

            preprocess = targetpad_transform(1.25, 224)

        checkpoint = torch.load(self.checkpoint_path, map_location=self.load_device)

        state_keys = [
            key
            for key in checkpoint
            if key != "epoch"
        ]

        if len(state_keys) != 1:
            raise RuntimeError(f"Ambiguous CSMCIR checkpoint keys: {state_keys}")

        state_dict = checkpoint[state_keys[0]]
        load_result = model.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys:
            raise RuntimeError(f"CSMCIR checkpoint has missing model keys: {load_result.missing_keys}")

        unexpected = set(load_result.unexpected_keys)
        allowed_unexpected = {
            "token_importance",
        }

        invalid_unexpected = unexpected - allowed_unexpected

        if invalid_unexpected:
            raise RuntimeError(f"Unexpected CSMCIR checkpoint keys: {sorted(invalid_unexpected)}")

        self.model = model
        self.txt_processor = txt_processors["eval"]
        self.preprocess = preprocess
        self._freeze_teacher()

    def _freeze_teacher(self) -> None:
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "CSMCIRStage1Teacher":
        super().train(mode)
        self.model.eval()

        return self


    @torch.no_grad()
    def encode_text_tokens(self, texts: list[str]):
        processed = [
            self.txt_processor(text)
            for text in texts
        ]

        tokens = self.model.tokenizer(
            processed,
            padding="max_length",
            truncation=True,
            max_length=self.model.max_txt_len,
            return_tensors="pt",
        )

        device = next(self.model.parameters()).device
        tokens = tokens.to(device)
        text_states = self.model.Qformer.bert.embeddings.word_embeddings(tokens.input_ids)
        attention_mask = tokens.attention_mask.bool()
        special_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        for special_id in self.model.tokenizer.all_special_ids:
            special_mask |= tokens.input_ids == special_id

        content_mask = attention_mask & ~special_mask

        return (
            text_states,
            attention_mask,
            content_mask,
        )

    @torch.no_grad()
    def encode_contextual_text_tokens(self, reference_features, teacher_text_states, text_attention_mask):
        batch_size = reference_features.shape[0]
        query_tokens = self.model.query_tokens.expand(batch_size, -1, -1)

        query_atts = torch.ones(query_tokens.shape[:-1], dtype=torch.long, device=reference_features.device)
        image_atts = torch.ones(reference_features.shape[:-1], dtype=torch.long, device=reference_features.device)
        attention_mask = torch.cat([query_atts, text_attention_mask.long()], dim=1)

        output = self.model.Qformer.bert(
            inputs_embeds=teacher_text_states,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=reference_features,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        num_query_tokens = query_tokens.shape[1]
        text_states = output.last_hidden_state[:, num_query_tokens:, :]

        if text_states.shape != teacher_text_states.shape:
            raise ValueError("Contextual text states must match teacher text states shape")

        return text_states

    @torch.no_grad()
    def encode_image_tokens(self, images):
        with self.model.maybe_autocast():
            native_features = self.model.ln_vision(self.model.visual_encoder(images))

        native_features = native_features.float()
        image_atts = torch.ones(native_features.shape[:-1], dtype=torch.long, device=native_features.device)
        query_tokens = self.model.query_tokens.expand(native_features.shape[0], -1, -1)

        output = self.model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=native_features,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        retrieval_features = self.model.vision_proj(output.last_hidden_state)
        retrieval_features = torch.nn.functional.normalize(retrieval_features, dim=-1)

        return retrieval_features, native_features

    def compose(self, reference_features: torch.Tensor, text_states: torch.Tensor, text_attention_mask: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        if reference_features.ndim != 3:
            raise ValueError(f"CSMCIR reference_features must be [B,K,D], got {tuple(reference_features.shape)}")

        if text_states.ndim != 3:
            raise ValueError("text_states must be [B,N,D]")

        if text_attention_mask.shape != (text_states.shape[0], text_states.shape[1]):
            raise ValueError("text_attention_mask shape mismatch")

        batch_size = reference_features.shape[0]
        device = reference_features.device

        query_tokens = self.model.query_tokens.expand(batch_size, -1, -1)
        query_atts = torch.ones(query_tokens.shape[:-1], dtype=torch.long, device=device)
        image_atts = torch.ones(reference_features.shape[:-1], dtype=torch.long, device=device)
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

        output = self.model.Qformer.bert(
            inputs_embeds=text_states,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=reference_features,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        num_query_tokens = query_tokens.shape[1]
        q_pre = self.model.text_proj(
            output.last_hidden_state[
                :,
                num_query_tokens,
                :
            ]
        )

        if not torch.isfinite(q_pre).all():
            raise FloatingPointError("CSMCIR compose produced NaN/Inf")

        if normalize:
            return torch.nn.functional.normalize(q_pre, dim=-1)

        return q_pre

    @torch.no_grad()
    def encode_reference(self, images: torch.Tensor) -> torch.Tensor:
        with self.model.maybe_autocast():
            reference_features = self.model.ln_vision(self.model.visual_encoder(images))

        reference_features = reference_features.float()
        if reference_features.ndim != 3:
            raise ValueError(f"CSMCIR reference features must be [B,K,D], got {tuple(reference_features.shape)}")

        if not torch.isfinite(reference_features).all():
            raise FloatingPointError("CSMCIR reference features contain NaN/Inf")

        return reference_features

    @torch.no_grad()
    def encode_gallery(self, images: torch.Tensor, captions: list[str]) -> torch.Tensor:
        gallery_features, _ = self.model.extract_target_caption_features(images, captions, mode="mean")

        if gallery_features.ndim != 3:
            raise ValueError(f"CSMCIR gallery features must be [B,K,D], got {tuple(gallery_features.shape)}")

        if not torch.isfinite(gallery_features).all():
            raise FloatingPointError("CSMCIR gallery features contain NaN/Inf")

        return gallery_features.float()