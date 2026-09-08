from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Subset

from datasets.common import CIRSample


def sample_ids_fingerprint(sample_ids: list[str]) -> str:
    """Stable fingerprint of an ordered cohort."""

    payload = json.dumps(sample_ids, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def teacher_batch_fingerprint(sample_ids: list[str], batch_size: int) -> str:
    """Stable fingerprint that also locks the in-batch teacher grouping."""

    groups = [
        sample_ids[start : start + batch_size] for start in range(0, len(sample_ids), batch_size)
    ]
    payload = json.dumps(groups, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_processed_manifest(
    manifest: dict[str, Any], processed_sample_ids: list[str], *, batch_size: int
) -> dict[str, Any]:
    """Require a normal replay to process every manifest row in stored order."""

    expected = [str(record["sample_id"]) for record in manifest["samples"]]
    if processed_sample_ids != expected:
        raise ValueError(
            "processed diagnostic cohort differs from the manifest: "
            f"expected {len(expected)} ordered samples, got {len(processed_sample_ids)}"
        )
    return {
        "manifest_sample_count": len(expected),
        "processed_sample_count": len(processed_sample_ids),
        "processed_sample_ids": processed_sample_ids,
        "processed_sample_ids_sha256": sample_ids_fingerprint(processed_sample_ids),
        "teacher_batch_grouping_sha256": teacher_batch_fingerprint(
            processed_sample_ids, batch_size
        ),
    }


def matched_intersection_ids(old_ids: list[str], strong_ids: list[str]) -> list[str]:
    """Return the stable OLD-ordered live intersection, rejecting ambiguous IDs."""

    if len(set(old_ids)) != len(old_ids) or len(set(strong_ids)) != len(strong_ids):
        raise ValueError("matched timestep comparison requires unique sample IDs")
    strong = set(strong_ids)
    return [sample_id for sample_id in old_ids if sample_id in strong]


def caption_change_masks(
    original_texts: list[str], shuffled_indices: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Separate changed indices from rows whose actual caption string changed."""

    if shuffled_indices.shape != (len(original_texts),):
        raise ValueError("shuffle indices must have one entry per caption")
    cpu_indices = shuffled_indices.detach().long().cpu()
    index_changed = cpu_indices.ne(torch.arange(len(original_texts)))
    text_changed = torch.tensor(
        [
            original_texts[index] != original_texts[int(cpu_indices[index])]
            for index in range(len(original_texts))
        ],
        dtype=torch.bool,
    )
    return {
        "index_changed": index_changed,
        "text_changed": text_changed,
        "duplicate_caption_unchanged": index_changed & ~text_changed,
    }


def _record(sample: CIRSample, dataset_index: int) -> dict[str, Any]:
    return {
        "dataset_index": dataset_index,
        "sample_id": sample.sample_id,
        "category": sample.category,
        "reference_id": sample.reference_id,
        "target_id": sample.target_id,
        "modification_text": sample.modification_text,
    }


def load_or_create_manifest(
    dataset: Dataset[CIRSample],
    path: str | Path,
    *,
    sample_count: int,
    batch_size: int,
    seed: int,
    split: str,
    caption_policy: str,
) -> tuple[Subset[CIRSample], dict[str, Any]]:
    """Create once, then strictly replay a stable diagnostic cohort and ordering."""

    manifest_path = Path(path)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("split") != split or manifest.get("caption_policy") != caption_policy:
            raise ValueError(
                "diagnostic manifest split/caption policy differs from the current dataset"
            )
        if manifest.get("teacher_batch_size") != batch_size:
            raise ValueError(
                "diagnostic manifest teacher batch size differs; changing it would "
                "change the in-batch target/negative pool"
            )
        if manifest.get("seed") != seed:
            raise ValueError("diagnostic manifest seed differs from the configured seed")
        records = manifest.get("samples")
        if not isinstance(records, list) or not records:
            raise ValueError("diagnostic manifest has no sample records")
        stored_request = int(manifest.get("requested_sample_count", len(records)))
        if sample_count != stored_request:
            raise ValueError(
                "diagnostic manifest sample count differs from the requested matched replay: "
                f"manifest_request={stored_request}, requested={sample_count}. "
                "Persistent manifests cannot be silently truncated."
            )
    else:
        count = min(sample_count, len(dataset))
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
        records = [_record(dataset[index], index) for index in indices]
        manifest = {
            "version": 1,
            "split": split,
            "caption_policy": caption_policy,
            "seed": seed,
            "requested_sample_count": sample_count,
            "teacher_batch_size": batch_size,
            "samples": records,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    indices = []
    for expected in records:
        index = int(expected["dataset_index"])
        actual = _record(dataset[index], index)
        if actual != expected:
            raise ValueError(
                f"diagnostic manifest no longer matches dataset at index {index}: "
                f"stored={expected}, current={actual}"
            )
        indices.append(index)
    return Subset(dataset, indices), manifest


def paired_rows(
    before_ids: list[str], before_values: torch.Tensor, after_by_id: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Align before/after values by stable ID; never compare mismatched cohorts."""

    if len(before_ids) != before_values.shape[0]:
        raise ValueError("before IDs and values have different row counts")
    positions = [index for index, sample_id in enumerate(before_ids) if sample_id in after_by_id]
    ids = [before_ids[index] for index in positions]
    if not positions:
        empty = before_values[:0]
        return empty, empty.clone(), []
    before = before_values.index_select(0, torch.tensor(positions, device=before_values.device))
    after = torch.stack([after_by_id[sample_id].to(before_values.device) for sample_id in ids])
    return before, after, ids


def category_shuffle_indices(categories: list[str | None], *, seed: int) -> torch.Tensor:
    """Deterministic non-identity caption permutation within each category."""

    result = torch.arange(len(categories))
    generator = torch.Generator().manual_seed(seed)
    for category in sorted(set(categories), key=str):
        rows = torch.tensor(
            [index for index, value in enumerate(categories) if value == category],
            dtype=torch.long,
        )
        if rows.numel() < 2:
            continue
        order = torch.randperm(rows.numel(), generator=generator)
        shuffled = rows.index_select(0, order)
        if torch.equal(shuffled, rows):
            shuffled = rows.roll(1)
        result[rows] = shuffled
    return result
