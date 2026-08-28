from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ImageCacheManifest:
    schema_version: int
    model_id: str
    revision: str
    processor_config_hash: str
    extraction_method: str
    normalization: str
    dtype: str
    image_id_mapping_hash: str
    feature_dim: int
    patch_policy: str
    split: str
    spatial_shapes_present: bool
    image_count: int
    complete_split: bool

    @property
    def sha256(self) -> str:
        return stable_json_hash(asdict(self))

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def read(cls, path: str | Path) -> ImageCacheManifest:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def require_exact(self, expected: ImageCacheManifest) -> None:
        actual = asdict(self)
        wanted = asdict(expected)
        differences = {
            key: {"expected": wanted[key], "actual": actual[key]}
            for key in wanted
            if wanted[key] != actual[key]
        }
        if differences:
            raise RuntimeError(
                "Incompatible FG-CLIP2 image cache manifest: "
                + json.dumps(differences, sort_keys=True)
            )


@dataclass(frozen=True, slots=True)
class FeatureSourcePolicy:
    text_encoder_trainable: bool
    vision_encoder_trainable: bool
    gallery_projection_trainable: bool
    use_cached_text_states: bool
    use_cached_reference_dense: bool
    use_cached_reference_global: bool
    use_cached_gallery_global: bool

    def validate(self) -> None:
        if self.text_encoder_trainable and self.use_cached_text_states:
            raise ValueError("Trainable text encoder forbids cached text states during training")
        if self.vision_encoder_trainable and (
            self.use_cached_reference_dense or self.use_cached_reference_global
        ):
            raise ValueError("Trainable vision encoder forbids cached reference image features")
        if self.gallery_projection_trainable and self.use_cached_gallery_global:
            raise ValueError("Changing gallery/image projection invalidates cached gallery embeddings")


@dataclass(frozen=True, slots=True)
class FrozenVisionCache:
    global_embeddings: Tensor
    dense_flat: Tensor
    dense_offsets: Tensor
    spatial_shapes: Tensor
    name_to_idx: dict[str, int]
    global_manifest: ImageCacheManifest
    dense_manifest: ImageCacheManifest

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        expected_global: ImageCacheManifest | None = None,
        expected_dense: ImageCacheManifest | None = None,
    ) -> FrozenVisionCache:
        root = Path(directory)
        required = (
            "global.pt",
            "dense_flat.pt",
            "dense_offsets.pt",
            "spatial_shapes.pt",
            "name_to_idx.json",
            "manifest_global.json",
            "manifest_dense.json",
        )
        for filename in required:
            if not (root / filename).is_file():
                raise FileNotFoundError(f"Missing frozen vision cache file: {root / filename}")
        global_manifest = ImageCacheManifest.read(root / "manifest_global.json")
        dense_manifest = ImageCacheManifest.read(root / "manifest_dense.json")
        if expected_global is not None:
            global_manifest.require_exact(expected_global)
        if expected_dense is not None:
            dense_manifest.require_exact(expected_dense)
        name_to_idx = json.loads((root / "name_to_idx.json").read_text(encoding="utf-8"))
        cache = cls(
            global_embeddings=torch.load(root / "global.pt", map_location="cpu", weights_only=True),
            dense_flat=torch.load(root / "dense_flat.pt", map_location="cpu", weights_only=True),
            dense_offsets=torch.load(root / "dense_offsets.pt", map_location="cpu", weights_only=True),
            spatial_shapes=torch.load(root / "spatial_shapes.pt", map_location="cpu", weights_only=True),
            name_to_idx={str(key): int(value) for key, value in name_to_idx.items()},
            global_manifest=global_manifest,
            dense_manifest=dense_manifest,
        )
        cache.validate()
        return cache

    def validate(self) -> None:
        count = len(self.name_to_idx)
        if self.global_manifest.image_count != count or self.dense_manifest.image_count != count:
            raise RuntimeError("Cache manifest image_count disagrees with ID mapping")
        mapping_hash = stable_json_hash(self.name_to_idx)
        if (
            self.global_manifest.image_id_mapping_hash != mapping_hash
            or self.dense_manifest.image_id_mapping_hash != mapping_hash
        ):
            raise RuntimeError("Cache image ID mapping hash mismatch")
        if self.global_embeddings.shape != (count, self.global_manifest.feature_dim):
            raise RuntimeError("Global cache tensor/manifest shape mismatch")
        if self.dense_flat.ndim != 2 or self.dense_flat.shape[1] != self.dense_manifest.feature_dim:
            raise RuntimeError("Dense cache tensor/manifest shape mismatch")
        if self.dense_offsets.shape != (count + 1,) or self.spatial_shapes.shape != (count, 2):
            raise RuntimeError("Dense offsets/spatial shapes mismatch")
        if int(self.dense_offsets[0]) != 0 or int(self.dense_offsets[-1]) != self.dense_flat.shape[0]:
            raise RuntimeError("Dense offsets do not span dense_flat")
        if sorted(self.name_to_idx.values()) != list(range(count)):
            raise RuntimeError("Image ID mapping is not a contiguous bijection")
        norms = self.global_embeddings.float().norm(dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=2e-3, rtol=2e-3):
            raise RuntimeError("Cached global embeddings are not L2-normalized")

    def global_by_ids(self, image_ids: tuple[str, ...] | list[str]) -> Tensor:
        indices = [self.name_to_idx[image_id] for image_id in image_ids]
        return self.global_embeddings[indices]

    def dense_by_ids(
        self, image_ids: tuple[str, ...] | list[str]
    ) -> tuple[Tensor, Tensor, Tensor]:
        rows = []
        shapes = []
        for image_id in image_ids:
            index = self.name_to_idx[image_id]
            start, end = self.dense_offsets[index : index + 2].tolist()
            rows.append(self.dense_flat[start:end])
            shapes.append(self.spatial_shapes[index])
        padded = nn.utils.rnn.pad_sequence(rows, batch_first=True)
        lengths = torch.tensor([row.shape[0] for row in rows])
        mask = torch.arange(padded.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
        return padded, mask, torch.stack(shapes)
