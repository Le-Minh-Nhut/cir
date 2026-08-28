from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn

try:
    import transformers
    from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer
except ImportError as error:  # pragma: no cover - exercised only in minimal deployments
    raise ImportError("FG-CLIP2 requires the 'transformers' package") from error


FGCLIP2_BASE_MODEL_ID = "qihoo360/fg-clip2-base"
FGCLIP2_BASE_REVISION = "430fbc8a912c86fd4de601381b6245a0edab22f0"
FGCLIP2_SHORT_TEXT_LENGTH = 64
FGCLIP2_TEXT_WALK_TYPE = "short"
FGCLIP2_PATCH_SIZE = 16
FGCLIP2_DYNAMIC_PATCH_BUDGETS = (128, 256, 576, 784, 1024)
FGCLIP2_PATCH_POLICY = "official_dynamic_v1"

TextTuningMode = Literal["frozen", "last_n_blocks", "full"]
VisionTuningMode = Literal["frozen", "last_n_blocks", "full"]


def _sha256_json(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _config_dict(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    if isinstance(config, Mapping):
        return dict(config)
    return {key: value for key, value in vars(config).items() if not key.startswith("_")}


def validate_revision(revision: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", revision or "") is None:
        raise ValueError("FG-CLIP2 revision must be an immutable 40-character git SHA")
    return revision


@dataclass(frozen=True, slots=True)
class TextTuningConfig:
    mode: TextTuningMode = "last_n_blocks"
    num_unfrozen_blocks: int = 4
    train_final_norm: bool = True
    train_projection: bool = False

    def validate(self, num_blocks: int) -> None:
        if self.mode not in {"frozen", "last_n_blocks", "full"}:
            raise ValueError(f"Unsupported text tuning mode: {self.mode}")
        if self.mode == "last_n_blocks" and not 1 <= self.num_unfrozen_blocks <= num_blocks:
            raise ValueError(
                f"num_unfrozen_blocks must be in [1,{num_blocks}] for last_n_blocks"
            )
        if self.mode == "frozen" and (self.train_final_norm or self.train_projection):
            raise ValueError("frozen text mode cannot train final norm/projection")


@dataclass(frozen=True, slots=True)
class VisionTuningConfig:
    mode: VisionTuningMode = "frozen"
    num_unfrozen_blocks: int = 0

    def validate(self, num_blocks: int) -> None:
        if self.mode not in {"frozen", "last_n_blocks", "full"}:
            raise ValueError(f"Unsupported vision tuning mode: {self.mode}")
        if self.mode != "frozen":
            raise ValueError(
                "This branch intentionally supports only frozen vision; future modes are typed "
                "but cannot be enabled in the first experiment"
            )
        if self.num_unfrozen_blocks != 0:
            raise ValueError("Frozen vision requires num_unfrozen_blocks=0")


@dataclass(frozen=True, slots=True)
class BackboneContract:
    text_dim: int
    vision_dim: int
    retrieval_dim: int
    text_blocks: int
    vision_blocks: int
    patch_size: int
    max_text_length: int


@dataclass(frozen=True, slots=True)
class BackboneManifest:
    schema_version: int
    model_id: str
    revision: str
    transformers_version: str
    tokenizer_config: dict[str, Any]
    image_processor_config: dict[str, Any]
    preprocessing_policy: str
    dtype: str
    normalization_policy: str
    text_walk_type: str
    max_text_length: int
    vision_patch_policy: str
    contract: dict[str, int]

    @property
    def sha256(self) -> str:
        return _sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class TokenizedTextBatch:
    input_ids: Tensor
    attention_mask: Tensor
    content_mask: Tensor


@dataclass(frozen=True, slots=True)
class DenseVisualBatch:
    tokens: Tensor
    mask: Tensor
    spatial_shapes: Tensor

    def validate(self, vision_dim: int) -> None:
        if self.tokens.ndim != 3 or self.tokens.shape[-1] != vision_dim:
            raise ValueError("Dense visual tokens must be [B,N,vision_dim]")
        if self.mask.shape != self.tokens.shape[:2] or self.mask.dtype != torch.bool:
            raise ValueError("Dense visual mask must be bool [B,N]")
        if self.spatial_shapes.shape != (self.tokens.shape[0], 2):
            raise ValueError("spatial_shapes must be [B,2]")
        if not self.mask.any(dim=1).all():
            raise ValueError("Every image must contain at least one valid local token")
        if not torch.isfinite(self.tokens).all():
            raise FloatingPointError("Dense visual tokens contain NaN/Inf")


def determine_max_num_patches(image: Image.Image) -> int:
    width, height = image.size
    patch_count = (width // FGCLIP2_PATCH_SIZE) * (height // FGCLIP2_PATCH_SIZE)
    if patch_count > 784:
        return 1024
    if patch_count > 576:
        return 784
    if patch_count > 256:
        return 576
    if patch_count > 128:
        return 256
    return 128


class FGCLIP2BaseBackbone(nn.Module):
    """Official pinned FG-CLIP2-Base with online text and frozen vision.

    Model/tokenizer/processor injection exists only to make contract and gradient tests
    independent of network/GPU availability. Production loading always uses the pinned ID/SHA.
    """

    def __init__(
        self,
        *,
        model_id: str = FGCLIP2_BASE_MODEL_ID,
        revision: str = FGCLIP2_BASE_REVISION,
        max_text_length: int = FGCLIP2_SHORT_TEXT_LENGTH,
        dtype: torch.dtype | None = None,
        text_tuning: TextTuningConfig | None = None,
        vision_tuning: VisionTuningConfig | None = None,
        model: nn.Module | None = None,
        tokenizer: Any | None = None,
        image_processor: Any | None = None,
    ) -> None:
        super().__init__()
        if model_id != FGCLIP2_BASE_MODEL_ID:
            raise ValueError(f"This experiment requires {FGCLIP2_BASE_MODEL_ID!r}")
        self.model_id = model_id
        self.revision = validate_revision(revision)
        if max_text_length != FGCLIP2_SHORT_TEXT_LENGTH:
            raise ValueError("Pinned short-walk policy requires max_text_length=64")
        self.max_text_length = max_text_length
        self.text_tuning = text_tuning or TextTuningConfig()
        self.vision_tuning = vision_tuning or VisionTuningConfig()

        if (model is None) != (tokenizer is None) or (model is None) != (image_processor is None):
            raise ValueError("model, tokenizer, and image_processor must be injected together")
        if model is None:
            load_kwargs: dict[str, Any] = {
                "revision": self.revision,
                "trust_remote_code": True,
            }
            if dtype is not None:
                load_kwargs["dtype"] = dtype
            model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, revision=self.revision, trust_remote_code=True
            )
            image_processor = AutoImageProcessor.from_pretrained(
                self.model_id, revision=self.revision, trust_remote_code=True
            )
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.contract = self._inspect_contract()
        self.text_tuning.validate(self.contract.text_blocks)
        self.vision_tuning.validate(self.contract.vision_blocks)
        self._configure_trainable_parameters()

    def _inspect_contract(self) -> BackboneContract:
        config = self.model.config
        text_model = self.model.text_model
        vision_model = self.model.vision_model
        required = (
            (text_model, "encoder"),
            (text_model, "final_layer_norm"),
            (text_model, "head"),
            (vision_model, "encoder"),
        )
        for module, attribute in required:
            if not hasattr(module, attribute):
                raise RuntimeError(f"Pinned FG-CLIP2 runtime lacks required module: {attribute}")
        text_blocks = len(text_model.encoder.layers)
        vision_blocks = len(vision_model.encoder.layers)
        contract = BackboneContract(
            text_dim=int(config.text_config.hidden_size),
            vision_dim=int(config.vision_config.hidden_size),
            retrieval_dim=int(config.text_config.projection_size),
            text_blocks=text_blocks,
            vision_blocks=vision_blocks,
            patch_size=int(config.vision_config.patch_size),
            max_text_length=self.max_text_length,
        )
        return contract

    def _configure_trainable_parameters(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        text_model = self.model.text_model
        if self.text_tuning.mode == "full":
            for parameter in text_model.embeddings.parameters():
                parameter.requires_grad_(True)
            blocks = list(text_model.encoder.layers)
        elif self.text_tuning.mode == "last_n_blocks":
            blocks = list(text_model.encoder.layers)[-self.text_tuning.num_unfrozen_blocks :]
        else:
            blocks = []
        for block in blocks:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        for parameter in text_model.final_layer_norm.parameters():
            parameter.requires_grad_(self.text_tuning.train_final_norm)
        for parameter in text_model.head.parameters():
            parameter.requires_grad_(self.text_tuning.train_projection)

        if any(parameter.requires_grad for parameter in self.model.vision_model.parameters()):
            raise RuntimeError("Vision encoder was accidentally unfrozen")
        dense_head = getattr(self.model, "dense_feature_head", None)
        if dense_head is not None and any(p.requires_grad for p in dense_head.parameters()):
            raise RuntimeError("Frozen dense image projection was accidentally unfrozen")

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def unfrozen_text_block_names(self) -> tuple[str, ...]:
        return tuple(
            f"text_model.encoder.layers.{index}"
            for index, block in enumerate(self.model.text_model.encoder.layers)
            if any(parameter.requires_grad for parameter in block.parameters())
        )

    def parameter_report(self, taper_parameters: int = 0) -> dict[str, Any]:
        parameters = list(self.model.parameters())
        text_parameters = list(self.model.text_model.parameters())
        vision_parameters = list(self.model.vision_model.parameters())
        return {
            "total_fgclip2_params": sum(p.numel() for p in parameters),
            "trainable_fgclip2_params": sum(p.numel() for p in parameters if p.requires_grad),
            "trainable_text_params": sum(p.numel() for p in text_parameters if p.requires_grad),
            "trainable_vision_params": sum(p.numel() for p in vision_parameters if p.requires_grad),
            "trainable_taper_params": taper_parameters,
            "unfrozen_text_blocks": self.unfrozen_text_block_names,
        }

    def manifest(self) -> BackboneManifest:
        tokenizer_config = {
            key: getattr(self.tokenizer, key, None)
            for key in (
                "padding_side",
                "truncation_side",
                "pad_token_id",
                "eos_token_id",
                "bos_token_id",
                "model_max_length",
            )
        }
        processor_config = _config_dict(self.image_processor)
        if not processor_config and hasattr(self.image_processor, "to_dict"):
            processor_config = self.image_processor.to_dict()
        return BackboneManifest(
            schema_version=1,
            model_id=self.model_id,
            revision=self.revision,
            transformers_version=transformers.__version__,
            tokenizer_config=tokenizer_config,
            image_processor_config=processor_config,
            preprocessing_policy="official AutoImageProcessor; RGB; resize/rescale/normalize",
            dtype=str(next(self.model.parameters()).dtype).removeprefix("torch."),
            normalization_policy="official global output then L2 normalize; dense tokens unnormalized",
            text_walk_type=FGCLIP2_TEXT_WALK_TYPE,
            max_text_length=self.max_text_length,
            vision_patch_policy=FGCLIP2_PATCH_POLICY,
            contract=asdict(self.contract),
        )

    def train(self, mode: bool = True) -> FGCLIP2BaseBackbone:
        super().train(mode)
        self.model.vision_model.eval()
        if hasattr(self.model, "dense_feature_head"):
            self.model.dense_feature_head.eval()
        for index, block in enumerate(self.model.text_model.encoder.layers):
            trainable = any(p.requires_grad for p in block.parameters())
            block.train(mode and trainable)
        return self

    def tokenize_texts(self, texts: Sequence[str]) -> TokenizedTextBatch:
        if not texts:
            raise ValueError("texts must not be empty")
        encoded = self.tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_attention_mask=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].long()
        attention_mask = encoded["attention_mask"].bool()
        special_mask = encoded["special_tokens_mask"].bool()
        content_mask = attention_mask & ~special_mask
        if not content_mask.any(dim=1).all():
            raise ValueError("Every modification must contain at least one content token")
        return TokenizedTextBatch(input_ids, attention_mask, content_mask)

    def encode_text_tokens(self, batch: TokenizedTextBatch) -> Tensor:
        input_ids = batch.input_ids.to(self.device)
        attention_mask = batch.attention_mask.to(self.device)
        position_ids = torch.arange(
            input_ids.shape[1], device=self.device, dtype=torch.long
        ).unsqueeze(0)
        outputs = self.model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            walk_type=FGCLIP2_TEXT_WALK_TYPE,
        )
        states = outputs.last_hidden_state
        expected = (*input_ids.shape, self.contract.text_dim)
        if tuple(states.shape) != expected or not torch.isfinite(states).all():
            raise RuntimeError(f"Invalid online text states: expected {expected}, got {states.shape}")
        return states

    def pool_short_text_states(self, states: Tensor) -> Tensor:
        """Project already-computed short-walk states into text retrieval space.

        Pinned FG-CLIP2 short walk applies ``text_model.head`` to the final
        position of its final-layer-normalized ``last_hidden_state``. Keeping
        this contract here avoids a redundant text-transformer forward in
        representation-drift instrumentation.
        """
        if states.ndim != 3 or states.shape[-1] != self.contract.text_dim:
            raise ValueError(
                "Short-walk text states must be [B,L,text_dim]; "
                f"got {tuple(states.shape)}"
            )
        if states.shape[0] == 0 or states.shape[1] != self.max_text_length:
            raise ValueError(
                "Short-walk text states must have a non-empty batch and the pinned "
                f"sequence length {self.max_text_length}; got {tuple(states.shape)}"
            )
        expected = (states.shape[0], self.contract.retrieval_dim)
        head = getattr(self.model.text_model, "head", None)
        if not isinstance(head, nn.Module):
            raise RuntimeError(
                "Pinned FG-CLIP2 runtime lacks text_model.head required for short pooling"
            )
        pooled = head(states[:, -1, :])
        if tuple(pooled.shape) != expected or not torch.isfinite(pooled).all():
            raise RuntimeError(
                f"Invalid short-walk pooled text: expected {expected}, got {pooled.shape}"
            )
        return pooled

    @torch.inference_mode()
    def encode_image_global(self, images: Sequence[Image.Image]) -> Tensor:
        return self._encode_images(images, dense=False)

    @torch.inference_mode()
    def encode_image_dense(self, images: Sequence[Image.Image]) -> DenseVisualBatch:
        return self._encode_images(images, dense=True)

    def _assert_frozen_vision(self) -> None:
        if any(parameter.requires_grad for parameter in self.model.vision_model.parameters()):
            raise RuntimeError("Online/cached image extraction requires frozen vision")

    def _encode_images(
        self, images: Sequence[Image.Image], *, dense: bool
    ) -> Tensor | DenseVisualBatch:
        self._assert_frozen_vision()
        if not images:
            raise ValueError("images must not be empty")
        grouped: dict[int, list[int]] = {}
        for index, image in enumerate(images):
            grouped.setdefault(determine_max_num_patches(image), []).append(index)
        globals_: list[Tensor | None] = [None] * len(images)
        locals_: list[Tensor | None] = [None] * len(images)
        shapes = torch.empty(len(images), 2, device=self.device, dtype=torch.long)
        for budget, indices in grouped.items():
            processor_batch = self.image_processor(
                images=[images[index] for index in indices],
                max_num_patches=budget,
                return_tensors="pt",
            ).to(self.device)
            if dense:
                values = self.model.get_image_dense_feature(**processor_batch)
                spatial_shapes = processor_batch["spatial_shapes"].long()
                mask = processor_batch.get("pixel_attention_mask")
                for local_index, original_index in enumerate(indices):
                    real_count = int(spatial_shapes[local_index].prod().item())
                    if real_count <= 0 or real_count > values.shape[1]:
                        raise RuntimeError("Processor spatial_shapes disagree with dense output")
                    if mask is not None:
                        flat_mask = mask[local_index].reshape(-1).bool()
                        if flat_mask.numel() == values.shape[1] and (
                            not flat_mask[:real_count].all() or flat_mask[real_count:].any()
                        ):
                            raise RuntimeError("Processor mask disagrees with spatial_shapes")
                    locals_[original_index] = values[local_index, :real_count].float()
                    shapes[original_index] = spatial_shapes[local_index]
            else:
                values = F.normalize(self.model.get_image_features(**processor_batch).float(), dim=-1)
                for local_index, original_index in enumerate(indices):
                    globals_[original_index] = values[local_index]
        if dense:
            if any(value is None for value in locals_):
                raise RuntimeError("Dense batching failed to restore input order")
            typed = [value for value in locals_ if value is not None]
            padded = nn.utils.rnn.pad_sequence(typed, batch_first=True)
            mask = torch.arange(padded.shape[1], device=padded.device).unsqueeze(0) < torch.tensor(
                [value.shape[0] for value in typed], device=padded.device
            ).unsqueeze(1)
            result = DenseVisualBatch(padded, mask, shapes)
            result.validate(self.contract.vision_dim)
            return result
        if any(value is None for value in globals_):
            raise RuntimeError("Global batching failed to restore input order")
        result = torch.stack([value for value in globals_ if value is not None])
        if result.shape != (len(images), self.contract.retrieval_dim):
            raise RuntimeError("Invalid official global image output shape")
        if not torch.isfinite(result).all():
            raise FloatingPointError("Global image output contains NaN/Inf")
        return result
