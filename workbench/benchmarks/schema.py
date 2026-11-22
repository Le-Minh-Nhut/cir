"""Schema definitions and zero-dependency split loaders for EDIR and PinPoint."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_SPLIT_TAG = "70_10_20_seed42"
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"

DATASET_DIR_NAMES = {
    "edir": "EDIR",
    "pinpoint": "PinPoint",
}


@dataclass(frozen=True)
class CIRQueryRecord:
    """Unified query record shared across EDIR and PinPoint."""

    dataset: str
    split: str
    query_id: str
    normalized_query_id: str
    ref_id: str
    ref_image_rel: str
    text: str
    positive_ids: Tuple[str, ...]
    negative_ids: Tuple[str, ...]
    category: str
    is_single: bool = True
    ref2_id: Optional[str] = None
    ref2_image_rel: Optional[str] = None
    positive_ids_raw: Tuple[str, ...] = ()
    negative_ids_raw: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def ref_path(self, data_dir: Path) -> Path:
        return data_dir / self.ref_image_rel

    def ref2_path(self, data_dir: Path) -> Optional[Path]:
        if not self.ref2_image_rel:
            return None
        return data_dir / self.ref2_image_rel


@dataclass(frozen=True)
class CIRCorpusRecord:
    """Unified gallery/corpus image record."""

    image_id: str
    image_rel: str

    def image_path(self, data_dir: Path) -> Path:
        return data_dir / self.image_rel


@dataclass(frozen=True)
class SplitBundle:
    """Loaded canonical split bundle for a benchmark."""

    dataset: str
    split_tag: str
    split_name: str
    data_dir: Path
    split_dir: Path
    manifest: Dict[str, Any]
    queries: Tuple[CIRQueryRecord, ...]
    corpus: Tuple[CIRCorpusRecord, ...]
    corpus_scope: str
    corpus_index: Dict[str, CIRCorpusRecord]

    def resolve_image_path(self, image_id: str) -> Path:
        rec = self.corpus_index.get(image_id)
        if rec is not None:
            return rec.image_path(self.data_dir)
        # Fallback for EDIR reference images (not in corpus)
        if self.dataset == "edir":
            return self.data_dir / "imgs_mc" / image_id
        # Fallback for PinPoint signatures
        return self.data_dir / "images" / image_id[:2] / f"{image_id}.jpg"


def normalize_dataset_name(dataset: str) -> str:
    key = dataset.strip().lower()
    if key not in DATASET_DIR_NAMES:
        raise ValueError(f"Unsupported dataset {dataset!r}. Expected one of {sorted(DATASET_DIR_NAMES)}.")
    return key


def resolve_dataset_dir(dataset: str, data_root: Optional[Path] = None) -> Path:
    key = normalize_dataset_name(dataset)
    base = Path(data_root) if data_root is not None else DATA_ROOT
    cand = base / DATASET_DIR_NAMES[key]
    if not cand.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {cand}")
    return cand


def resolve_split_dir(
    dataset: str,
    split_tag: str = DEFAULT_SPLIT_TAG,
    data_root: Optional[Path] = None,
) -> Path:
    data_dir = resolve_dataset_dir(dataset, data_root=data_root)
    split_dir = data_dir / "splits" / split_tag
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Split directory not found: {split_dir}. "
            f"Run workbench/scripts/build_edir_pinpoint_splits.py first."
        )
    return split_dir


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_split_integrity(
    dataset: str,
    split_tag: str = DEFAULT_SPLIT_TAG,
    data_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Verify SHA256 checksums of every exported split file against manifest.json."""
    split_dir = resolve_split_dir(dataset, split_tag=split_tag, data_root=data_root)
    manifest_path = split_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest.json in {split_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checksums = manifest.get("file_sha256", {})
    if not checksums:
        raise ValueError(f"manifest.json in {split_dir} has no file_sha256 entries")

    mismatches: List[str] = []
    for rel_name, expected in sorted(checksums.items()):
        fp = split_dir / rel_name
        if not fp.is_file():
            mismatches.append(f"MISSING {rel_name}")
            continue
        actual = sha256_file(fp)
        if actual != expected:
            mismatches.append(f"SHA256_MISMATCH {rel_name}: expected={expected} actual={actual}")
    if mismatches:
        raise RuntimeError("Split integrity verification failed:\n" + "\n".join(mismatches))
    return manifest


def load_split_ids(
    dataset: str,
    split_name: str,
    split_tag: str = DEFAULT_SPLIT_TAG,
    data_root: Optional[Path] = None,
) -> List[str]:
    """Load exact ordered query IDs for a split from `<split_name>_ids.txt`."""
    split_dir = resolve_split_dir(dataset, split_tag=split_tag, data_root=data_root)
    id_file = split_dir / f"{split_name}_ids.txt"
    if not id_file.is_file():
        raise FileNotFoundError(f"Split ID file not found: {id_file}")
    return [line.strip() for line in id_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parse_query_record(row: Dict[str, Any], dataset: str, split_name: str) -> CIRQueryRecord:
    return CIRQueryRecord(
        dataset=dataset,
        split=split_name,
        query_id=str(row["query_id"]),
        normalized_query_id=str(row["normalized_query_id"]),
        ref_id=str(row["ref_id"]),
        ref_image_rel=str(row["ref_image_rel"]),
        text=str(row["text"]),
        positive_ids=tuple(row["positive_ids"]),
        negative_ids=tuple(row["negative_ids"]),
        category=str(row["category"]),
        is_single=bool(row.get("is_single", True)),
        ref2_id=row.get("ref2_id"),
        ref2_image_rel=row.get("ref2_image_rel"),
        positive_ids_raw=tuple(row.get("positive_ids_raw", row["positive_ids"])),
        negative_ids_raw=tuple(row.get("negative_ids_raw", row["negative_ids"])),
        metadata=dict(row.get("metadata", {})),
    )


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_split_bundle(
    dataset: str,
    split_name: str,
    corpus_scope: str = "full",
    split_tag: str = DEFAULT_SPLIT_TAG,
    data_root: Optional[Path] = None,
    verify_checksums: bool = False,
) -> SplitBundle:
    """Load canonical queries + corpus for a dataset split.

    Args:
        dataset: ``"edir"`` or ``"pinpoint"``.
        split_name: ``"train"``, ``"val"``, ``"test"``, or (for PinPoint) ``"multi_image_eval"``.
        corpus_scope: ``"full"`` (full benchmark gallery) or ``"split"`` (gallery restricted to this split).
        split_tag: Versioned split tag (default ``"70_10_20_seed42"``).
        data_root: Optional override for ``<repo>/data``.
        verify_checksums: If True, verifies SHA256 of every file in the split directory.
    """
    ds_key = normalize_dataset_name(dataset)
    data_dir = resolve_dataset_dir(ds_key, data_root=data_root)
    split_dir = resolve_split_dir(ds_key, split_tag=split_tag, data_root=data_root)

    if verify_checksums:
        manifest = verify_split_integrity(ds_key, split_tag=split_tag, data_root=data_root)
    else:
        manifest = json.loads((split_dir / "manifest.json").read_text(encoding="utf-8"))

    query_file = split_dir / f"{split_name}.jsonl"
    if not query_file.is_file():
        raise FileNotFoundError(f"Split file not found: {query_file}")

    if corpus_scope == "full":
        corpus_file = split_dir / "corpus_full.jsonl"
    elif corpus_scope == "split":
        corpus_file = split_dir / f"corpus_{split_name}.jsonl"
    else:
        raise ValueError(f"Invalid corpus_scope {corpus_scope!r}; expected 'full' or 'split'.")
    if not corpus_file.is_file():
        raise FileNotFoundError(f"Corpus file not found: {corpus_file}")

    raw_queries = _load_jsonl(query_file)
    queries = tuple(_parse_query_record(r, ds_key, split_name) for r in raw_queries)

    raw_corpus = _load_jsonl(corpus_file)
    corpus = tuple(
        CIRCorpusRecord(image_id=str(c["image_id"]), image_rel=str(c["image_rel"]))
        for c in raw_corpus
    )
    corpus_index = {c.image_id: c for c in corpus}

    return SplitBundle(
        dataset=ds_key,
        split_tag=split_tag,
        split_name=split_name,
        data_dir=data_dir,
        split_dir=split_dir,
        manifest=manifest,
        queries=queries,
        corpus=corpus,
        corpus_scope=corpus_scope,
        corpus_index=corpus_index,
    )
