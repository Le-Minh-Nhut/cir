from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn


GLOBAL_DIRECTORY = "global"
DENSE_DIRECTORY = "dense_reference"


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ImageCacheManifest:
    schema_version: int
    cache_kind: str
    image_scope: str
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

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError(f"Unsupported mmap cache schema_version={self.schema_version}")
        valid = {"global": "complete_split", "dense_reference": "reference_only"}
        if self.cache_kind not in valid:
            raise ValueError(f"Unsupported image cache kind: {self.cache_kind}")
        if self.image_scope != valid[self.cache_kind]:
            raise ValueError(
                f"cache_kind={self.cache_kind} requires image_scope={valid[self.cache_kind]}"
            )
        if self.spatial_shapes_present != (self.cache_kind == "dense_reference"):
            raise ValueError("spatial_shapes_present disagrees with cache_kind")

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


def _read_mapping(path: Path) -> dict[str, int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    mapping = {str(key): int(value) for key, value in raw.items()}
    if len(mapping) != len(raw) or sorted(mapping.values()) != list(range(len(mapping))):
        raise RuntimeError(f"Image ID mapping is not a contiguous bijection: {path}")
    return mapping


def _finite_by_chunks(array: NDArray[np.generic], rows: int = 65_536) -> bool:
    return all(
        np.isfinite(array[start : start + rows]).all()
        for start in range(0, len(array), rows)
    )


@dataclass(frozen=True, slots=True)
class GlobalImageCache:
    embeddings: NDArray[np.floating[Any]]
    name_to_idx: dict[str, int]
    manifest: ImageCacheManifest

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        expected: ImageCacheManifest | None = None,
    ) -> GlobalImageCache:
        root = Path(directory)
        for filename in ("global.npy", "name_to_idx.json", "manifest.json"):
            if not (root / filename).is_file():
                raise FileNotFoundError(f"Missing global image cache file: {root / filename}")
        manifest = ImageCacheManifest.read(root / "manifest.json")
        if expected is not None:
            manifest.require_exact(expected)
        cache = cls(
            embeddings=np.load(root / "global.npy", mmap_mode="r", allow_pickle=False),
            name_to_idx=_read_mapping(root / "name_to_idx.json"),
            manifest=manifest,
        )
        cache.validate()
        return cache

    def validate(self) -> None:
        count = len(self.name_to_idx)
        if self.manifest.cache_kind != "global" or self.manifest.image_scope != "complete_split":
            raise RuntimeError("Global cache manifest has the wrong cache kind/scope")
        if not self.manifest.complete_split:
            raise RuntimeError("Partial/debug global cache cannot be used for full training")
        if self.manifest.image_count != count:
            raise RuntimeError("Global manifest image_count disagrees with ID mapping")
        if self.manifest.image_id_mapping_hash != stable_json_hash(self.name_to_idx):
            raise RuntimeError("Global image ID mapping hash mismatch")
        if self.embeddings.shape != (count, self.manifest.feature_dim):
            raise RuntimeError("Global mmap/manifest shape mismatch")
        if not np.issubdtype(self.embeddings.dtype, np.floating):
            raise RuntimeError("Global mmap must use a floating dtype")
        if self.embeddings.dtype != np.dtype(self.manifest.dtype):
            raise RuntimeError("Global mmap dtype disagrees with manifest")
        for start in range(0, count, 65_536):
            block = np.asarray(self.embeddings[start : start + 65_536], dtype=np.float32)
            if not np.isfinite(block).all():
                raise RuntimeError("Global mmap contains NaN/Inf")
            norms = np.linalg.norm(block, axis=-1)
            if not np.allclose(norms, 1.0, atol=2e-3, rtol=2e-3):
                raise RuntimeError("Cached global embeddings are not L2-normalized")

    def by_ids(self, image_ids: tuple[str, ...] | list[str]) -> Tensor:
        try:
            indices = [self.name_to_idx[image_id] for image_id in image_ids]
        except KeyError as error:
            raise KeyError(f"Image ID absent from complete global cache: {error.args[0]}") from error
        return torch.from_numpy(np.array(self.embeddings[indices], copy=True))


@dataclass(frozen=True, slots=True)
class DenseReferenceCache:
    values: NDArray[np.floating[Any]]
    offsets: NDArray[np.integer[Any]]
    spatial_shapes: NDArray[np.integer[Any]]
    reference_name_to_idx: dict[str, int]
    manifest: ImageCacheManifest

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        expected: ImageCacheManifest | None = None,
    ) -> DenseReferenceCache:
        root = Path(directory)
        required = (
            "dense_values.npy",
            "dense_offsets.npy",
            "spatial_shapes.npy",
            "reference_name_to_idx.json",
            "manifest.json",
        )
        for filename in required:
            if not (root / filename).is_file():
                raise FileNotFoundError(f"Missing dense reference cache file: {root / filename}")
        manifest = ImageCacheManifest.read(root / "manifest.json")
        if expected is not None:
            manifest.require_exact(expected)
        cache = cls(
            values=np.load(root / "dense_values.npy", mmap_mode="r", allow_pickle=False),
            offsets=np.load(root / "dense_offsets.npy", mmap_mode="r", allow_pickle=False),
            spatial_shapes=np.load(root / "spatial_shapes.npy", mmap_mode="r", allow_pickle=False),
            reference_name_to_idx=_read_mapping(root / "reference_name_to_idx.json"),
            manifest=manifest,
        )
        cache.validate()
        return cache

    def validate(self) -> None:
        count = len(self.reference_name_to_idx)
        if self.manifest.cache_kind != "dense_reference" or self.manifest.image_scope != "reference_only":
            raise RuntimeError("Dense cache manifest must explicitly declare reference_only scope")
        if not self.manifest.complete_split:
            raise RuntimeError("Partial/debug dense reference cache cannot be used for full training")
        if self.manifest.image_count != count:
            raise RuntimeError("Dense manifest image_count disagrees with reference mapping")
        if self.manifest.image_id_mapping_hash != stable_json_hash(self.reference_name_to_idx):
            raise RuntimeError("Dense reference ID mapping hash mismatch")
        if self.values.ndim != 2 or self.values.shape[1] != self.manifest.feature_dim:
            raise RuntimeError("Dense values mmap/manifest shape mismatch")
        if self.offsets.shape != (count + 1,) or self.spatial_shapes.shape != (count, 2):
            raise RuntimeError("Dense offsets/spatial shapes mismatch")
        if not np.issubdtype(self.offsets.dtype, np.integer) or not np.issubdtype(
            self.spatial_shapes.dtype, np.integer
        ):
            raise RuntimeError("Dense offsets/spatial shapes must use integer dtypes")
        if self.values.dtype != np.dtype(self.manifest.dtype):
            raise RuntimeError("Dense values mmap dtype disagrees with manifest")
        offsets = np.asarray(self.offsets, dtype=np.int64)
        if offsets[0] != 0 or offsets[-1] != self.values.shape[0]:
            raise RuntimeError("Dense offsets do not span dense_values")
        if np.any(offsets[1:] < offsets[:-1]):
            raise RuntimeError("Dense offsets are not monotonic")
        lengths = offsets[1:] - offsets[:-1]
        shape_counts = np.asarray(self.spatial_shapes, dtype=np.int64).prod(axis=1)
        if np.any(np.asarray(self.spatial_shapes) <= 0) or not np.array_equal(lengths, shape_counts):
            raise RuntimeError("Dense row lengths disagree with spatial_shapes")
        if not np.issubdtype(self.values.dtype, np.floating) or not _finite_by_chunks(self.values):
            raise RuntimeError("Dense values mmap must be finite floating values")

    def by_ids(
        self, image_ids: tuple[str, ...] | list[str]
    ) -> tuple[Tensor, Tensor, Tensor]:
        rows: list[Tensor] = []
        shapes: list[Tensor] = []
        for image_id in image_ids:
            try:
                index = self.reference_name_to_idx[image_id]
            except KeyError as error:
                raise KeyError(
                    f"Image ID absent from reference-only dense cache: {image_id}"
                ) from error
            start = int(self.offsets[index])
            end = int(self.offsets[index + 1])
            rows.append(torch.from_numpy(np.array(self.values[start:end], copy=True)))
            shapes.append(torch.from_numpy(np.array(self.spatial_shapes[index], copy=True)))
        if not rows:
            raise ValueError("Dense cache lookup requires at least one reference ID")
        padded = nn.utils.rnn.pad_sequence(rows, batch_first=True)
        lengths = torch.tensor([row.shape[0] for row in rows])
        mask = torch.arange(padded.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
        return padded, mask, torch.stack(shapes).long()


@dataclass(frozen=True, slots=True)
class FrozenVisionCache:
    """Composed mmap cache preserving the model-facing reference/global API."""

    global_store: GlobalImageCache
    dense_store: DenseReferenceCache

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        expected_global: ImageCacheManifest | None = None,
        expected_dense: ImageCacheManifest | None = None,
    ) -> FrozenVisionCache:
        root = Path(directory)
        cache = cls(
            GlobalImageCache.load(root / GLOBAL_DIRECTORY, expected=expected_global),
            DenseReferenceCache.load(root / DENSE_DIRECTORY, expected=expected_dense),
        )
        missing_global = set(cache.reference_name_to_idx).difference(cache.name_to_idx)
        if missing_global:
            first = sorted(missing_global)[0]
            raise RuntimeError(
                f"Dense reference cache contains an ID absent from global cache: {first}"
            )
        return cache

    @property
    def global_manifest(self) -> ImageCacheManifest:
        return self.global_store.manifest

    @property
    def dense_manifest(self) -> ImageCacheManifest:
        return self.dense_store.manifest

    @property
    def name_to_idx(self) -> dict[str, int]:
        return self.global_store.name_to_idx

    @property
    def reference_name_to_idx(self) -> dict[str, int]:
        return self.dense_store.reference_name_to_idx

    @property
    def global_embeddings(self) -> NDArray[np.floating[Any]]:
        return self.global_store.embeddings

    def global_by_ids(self, image_ids: tuple[str, ...] | list[str]) -> Tensor:
        return self.global_store.by_ids(image_ids)

    def dense_by_ids(
        self, image_ids: tuple[str, ...] | list[str]
    ) -> tuple[Tensor, Tensor, Tensor]:
        return self.dense_store.by_ids(image_ids)
