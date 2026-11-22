from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _v3_run_receipt(**kwargs: Any) -> dict[str, Any]:
    path = (Path(__file__).resolve().parents[1] / "docs" / "reviews" /
            "20261109-native15-composition-limn" / "scripts" / "t4_parity.py")
    spec = importlib.util.spec_from_file_location("cir_t4_parity_v3", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load V3 receipt helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.v3_run_receipt(**kwargs)
def _v3_artifact_block(plan_path: Path, plan_sha256: str, *, scopes: dict[str, Any],
                       receipts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    path = (Path(__file__).resolve().parents[1] / "docs" / "reviews" /
            "20261109-native15-composition-limn" / "scripts" / "t4_parity.py")
    spec = importlib.util.spec_from_file_location("cir_t4_parity_v3_artifacts", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load V3 artifact helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.v3_artifact_block(plan_path, plan_sha256, scopes=scopes, receipts=receipts)
IDENTITY_FIELDS = frozenset({
    "checkpoint", "configuration", "environment", "protocol", "category", "cohort_id",
    "precision", "seed", "query_order_sha256", "gallery_order_sha256",
    "reference_exclusions_sha256", "query_count", "gallery_count",
})
RANK_POLICY = "1 + #scores_greater + #equal_scores_at_earlier_gallery_indices; best positive; reference eligible"



def _canonical_digest(values: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(list(values), ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported artifact metadata value: {type(value).__name__}")


def _write_streaming_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write uncompressed .npy members without materializing memmap-backed arrays."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, value in arrays.items():
                if not name.isidentifier() or value.dtype.hasobject:
                    raise ValueError(f"invalid NPZ member {name!r}")
                with archive.open(f"{name}.npy", "w") as member:
                    np.lib.format.write_array(member, np.asarray(value), allow_pickle=False)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class AdapterArtifactSink:
    """Persist actual runner memmaps and ranks before runner cleanup."""

    def __init__(self, root: Path, *, parity_run_for: Any) -> None:
        self.root = Path(root)
        self.parity_run_for = parity_run_for

    def __call__(self, *, model_id: str, category: str | None, condition: str,
                 run_id: str, plan_path: Path, plan_sha256: str, started_at: str,
                 identity: Mapping[str, Any], source_id: str, query_ids: Sequence[str],
                 gallery_ids: Sequence[str], query_features: np.ndarray,
                 gallery_features: np.ndarray, ranks: Mapping[str, int | None],
                 metrics: Mapping[str, Any], execution: Mapping[str, Any]) -> None:
        if category is None:
            raise ValueError("adapter parity artifacts require one measured category")
        role = self.parity_run_for(model_id, category, condition)
        if (set(identity) != IDENTITY_FIELDS or dict(role["identity"]) != dict(identity) or
                identity["category"] != category):
            raise ValueError("adapter identity differs from its frozen role")
        if (role["source_id"] != source_id or role["run_id"] != run_id or
                role["plan_sha256"] != plan_sha256 or
                Path(role["plan_path"]).resolve() != Path(plan_path).resolve()):
            raise ValueError("adapter callback differs from preregistered role")
        if (len(query_ids) != identity["query_count"] or
                len(gallery_ids) != identity["gallery_count"] or
                _canonical_digest(query_ids) != identity["query_order_sha256"] or
                _canonical_digest(gallery_ids) != identity["gallery_order_sha256"]):
            raise ValueError("adapter IDs differ from frozen V3 population")
        if (len(set(query_ids)) != len(query_ids) or len(set(gallery_ids)) != len(gallery_ids) or
                query_features.shape[0] != len(query_ids) or
                gallery_features.shape[0] != len(gallery_ids) or set(ranks) != set(query_ids) or
                any(rank is None for rank in ranks.values())):
            raise ValueError("adapter arrays/ranks do not cover unique scored population")
        rank_values = np.asarray([ranks[qid] for qid in query_ids], dtype=np.int64)
        if np.any(rank_values < 1) or np.any(rank_values > len(gallery_ids)):
            raise ValueError("adapter rank falls outside the eligible gallery")
        receipt = _v3_run_receipt(
            run_id=run_id, side="adapter", source_id=source_id,
            plan_sha256=plan_sha256, started_at=started_at,
            arrays={"query_features": (query_features, list(query_ids)),
                    "gallery_features": (gallery_features, list(gallery_ids)),
                    "ranks": (rank_values, list(query_ids))},
            execution=dict(execution), identity=dict(identity))
        directory = self.root / "adapter" / model_id
        directory.mkdir(parents=True, exist_ok=True)
        archive = directory / f"{category}.{condition}.npz"
        document = directory / f"{category}.{condition}.json"
        if archive.exists() or document.exists():
            raise FileExistsError(f"refusing to overwrite adapter evidence for {model_id}/{category}/{condition}")
        arrays = {"query_ids": np.asarray(query_ids, dtype=np.str_),
                  "gallery_ids": np.asarray(gallery_ids, dtype=np.str_),
                  "query_features": query_features, "gallery_features": gallery_features,
                  "rank_qids": np.asarray(query_ids, dtype=np.str_), "ranks": rank_values}
        metadata = {name: {"dtype": str(value.dtype), "shape": list(value.shape),
                           "ordered_ids_sha256": _canonical_digest(
                               gallery_ids if name in {"gallery_ids", "gallery_features"}
                               else query_ids)}
                    for name, value in arrays.items()}
        payload = {"standard": "parity-absolute-relative/v3", "model_id": model_id,
                   "category": category, "condition": condition,
                   "run_id": run_id, "plan_path": Path(plan_path).name,
                   "plan_sha256": plan_sha256, "started_at": started_at, "source_id": source_id,
                   "identity": dict(identity), "protocol": identity["protocol"],
                   "query_order_sha256": identity["query_order_sha256"],
                   "gallery_order_sha256": identity["gallery_order_sha256"],
                   "checkpoint_sha256": identity["checkpoint"],
                   "execution": _jsonable(execution), "metrics": _jsonable(metrics),
                   "rank_policy": "1 + #scores_greater + #equal_scores_at_earlier_gallery_indices; best positive; reference eligible",
                   "arrays": metadata}
        temporary = document.with_suffix(".json.tmp")
        try:
            _write_streaming_npz(archive, arrays)
            temporary.write_text(json.dumps(_jsonable(payload), sort_keys=True, indent=2) + "\n",
                                 encoding="utf-8")
            os.replace(temporary, document)
        except BaseException:
            temporary.unlink(missing_ok=True)
            archive.unlink(missing_ok=True)
            raise
def write_native_artifact(*, root: Path, model_id: str, category: str,
                          condition: str, run_id: str, plan_path: Path,
                          plan_sha256: str, started_at: str, identity: Mapping[str, Any],
                          source_id: str, query_ids: Sequence[str], gallery_ids: Sequence[str],
                          query_features: np.ndarray, gallery_features: np.ndarray,
                          ranks: Mapping[str, int | None], metrics: Mapping[str, Any],
                          execution: Mapping[str, Any], rank_policy: str) -> None:
    """Persist native full arrays under adapter-compatible V3 artifact schema."""
    if set(identity) != IDENTITY_FIELDS or identity.get("category") != category:
        raise ValueError("native identity differs from frozen role")
    if (len(query_ids) != identity["query_count"] or len(gallery_ids) != identity["gallery_count"] or
            _canonical_digest(query_ids) != identity["query_order_sha256"] or
            _canonical_digest(gallery_ids) != identity["gallery_order_sha256"]):
        raise ValueError("native IDs differ from frozen V3 population")
    if (len(set(query_ids)) != len(query_ids) or len(set(gallery_ids)) != len(gallery_ids) or
            query_features.shape[0] != len(query_ids) or gallery_features.shape[0] != len(gallery_ids) or
            set(ranks) != set(query_ids) or any(rank is None for rank in ranks.values())):
        raise ValueError("native arrays/ranks do not cover unique scored population")
    rank_values = np.asarray([ranks[qid] for qid in query_ids], dtype=np.int64)
    if np.any(rank_values < 1) or np.any(rank_values > len(gallery_ids)):
        raise ValueError("native rank falls outside the eligible gallery")
    receipt = _v3_run_receipt(
        run_id=run_id, side="native", source_id=source_id, plan_sha256=plan_sha256,
        started_at=started_at,
        arrays={"query_features": (query_features, list(query_ids)),
                "gallery_features": (gallery_features, list(gallery_ids)),
                "ranks": (rank_values, list(query_ids))},
        execution=dict(execution), identity=dict(identity))
    arrays = {"query_ids": np.asarray(query_ids, dtype=np.str_),
              "gallery_ids": np.asarray(gallery_ids, dtype=np.str_),
              "query_features": query_features, "gallery_features": gallery_features,
              "rank_qids": np.asarray(query_ids, dtype=np.str_), "ranks": rank_values}
    directory = Path(root) / "native" / model_id
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / f"{category}.{condition}.npz"
    document = directory / f"{category}.{condition}.json"
    if archive.exists() or document.exists():
        raise FileExistsError(f"refusing to overwrite native evidence for {model_id}/{category}/{condition}")
    metadata = {name: {"dtype": str(value.dtype), "shape": list(value.shape),
                       "ordered_ids_sha256": _canonical_digest(
                           gallery_ids if name in {"gallery_ids", "gallery_features"} else query_ids)}
                for name, value in arrays.items()}
    payload = {"standard": "parity-absolute-relative/v3", "model_id": model_id,
               "category": category, "condition": condition,
               "run_id": run_id, "plan_path": Path(plan_path).name,
               "plan_sha256": plan_sha256, "started_at": started_at, "source_id": source_id,
               "identity": dict(identity), "protocol": identity["protocol"],
               "query_order_sha256": identity["query_order_sha256"],
               "gallery_order_sha256": identity["gallery_order_sha256"],
               "checkpoint_sha256": identity["checkpoint"], "execution": _jsonable(execution),
               "metrics": _jsonable(metrics), "rank_policy": rank_policy, "arrays": metadata,
               "parity_v3": _v3_artifact_block(
                   Path(plan_path.name), plan_sha256,
                   scopes={"category": category, "condition": condition,
                           "query_count": len(query_ids), "gallery_count": len(gallery_ids)},
                   receipts={"native": receipt})}
    temporary = document.with_suffix(".json.tmp")
    try:
        _write_streaming_npz(archive, arrays)
        temporary.write_text(json.dumps(_jsonable(payload), sort_keys=True, indent=2) + "\n",
                             encoding="utf-8")
        os.replace(temporary, document)
    except BaseException:
        temporary.unlink(missing_ok=True)
        archive.unlink(missing_ok=True)
        raise

def native_artifact_from_capture(*, capture_path: Path, root: Path, model_id: str,
                                  category: str, condition: str, run_id: str,
                                  plan_path: Path, plan_sha256: str, started_at: str,
                                  identity: Mapping[str, Any], source_id: str,
                                  metrics: Mapping[str, Any], execution: Mapping[str, Any],
                                  rank_policy: str) -> None:
    """Convert a completed native capture into immutable V3 evidence."""
    required = {"query_ids", "gallery_ids", "query_features", "gallery_features", "ranks"}
    with np.load(capture_path, allow_pickle=False) as capture:
        missing = required - set(capture.files)
        if missing:
            raise ValueError(f"native capture missing V3 arrays: {sorted(missing)}")
        query_ids = capture["query_ids"].astype(np.str_, copy=False).tolist()
        gallery_ids = capture["gallery_ids"].astype(np.str_, copy=False).tolist()
        ranks = np.asarray(capture["ranks"], dtype=np.int64)
        if ranks.shape != (len(query_ids),):
            raise ValueError("native capture ranks do not align with query IDs")
        write_native_artifact(
            root=root, model_id=model_id, category=category, condition=condition,
            run_id=run_id, plan_path=plan_path, plan_sha256=plan_sha256,
            started_at=started_at, identity=identity, source_id=source_id,
            query_ids=query_ids, gallery_ids=gallery_ids,
            query_features=np.asarray(capture["query_features"]),
            gallery_features=np.asarray(capture["gallery_features"]),
            ranks=dict(zip(query_ids, ranks.tolist(), strict=True)), metrics=metrics,
            execution=execution, rank_policy=rank_policy)
def _read_document(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read parity artifact manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"parity artifact manifest is not an object: {path}")
    return value


def compare_artifacts(native_json: Path, adapter_json: Path, *, atol: float, rtol: float) -> dict[str, Any]:
    native = _read_document(native_json)
    adapter = _read_document(adapter_json)
    for key in ("model_id", "category", "condition", "identity", "rank_policy"):
        if native.get(key) != adapter.get(key):
            raise ValueError(f"native and adapter {key} differ")
    if native["rank_policy"] != "1 + #scores_greater + #equal_scores_at_earlier_gallery_indices; best positive; reference eligible":
        raise ValueError("unsupported native/adapter rank policy")
    if native.get("source_id") == adapter.get("source_id"):
        raise ValueError("native and adapter must have independent source IDs")
    identity = native["identity"]
    if set(identity) != IDENTITY_FIELDS:
        raise ValueError("artifact identity does not match the frozen V3 fields")
    arrays_required = {"query_ids", "gallery_ids", "query_features", "gallery_features", "rank_qids", "ranks"}
    with tempfile.TemporaryDirectory(prefix="cir-parity-compare-") as temporary:
        roots = (Path(temporary) / "native", Path(temporary) / "adapter")
        for root in roots:
            root.mkdir()
        loaded = []
        for document_path, root in zip((native_json, adapter_json), roots):
            with zipfile.ZipFile(Path(document_path).with_suffix(".npz")) as archive:
                members = {info.filename for info in archive.infolist()}
                expected = {f"{name}.npy" for name in arrays_required}
                if members != expected:
                    raise ValueError("artifact NPZ members differ from required schema")
                for name in arrays_required:
                    with archive.open(f"{name}.npy") as source, (root / f"{name}.npy").open("wb") as destination:
                        shutil.copyfileobj(source, destination, length=1 << 20)
            loaded.append({name: np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                           for name in arrays_required})
        native_arrays, adapter_arrays = loaded
        for name in arrays_required:
            left, right = native_arrays[name], adapter_arrays[name]
            if left.shape != right.shape or left.dtype != right.dtype:
                raise ValueError(f"native and adapter {name} shape or dtype differs")
        for arrays in (native_arrays, adapter_arrays):
            if (_canonical_digest(arrays["query_ids"].tolist()) != identity["query_order_sha256"] or
                    _canonical_digest(arrays["gallery_ids"].tolist()) != identity["gallery_order_sha256"]):
                raise ValueError("artifact IDs do not match frozen identity order digests")
        expected_shapes = {"query_ids": (identity["query_count"],),
                           "rank_qids": (identity["query_count"],),
                           "ranks": (identity["query_count"],),
                           "gallery_ids": (identity["gallery_count"],)}
        for name, shape in expected_shapes.items():
            if native_arrays[name].shape != shape:
                raise ValueError(f"native {name} does not match frozen population")
        if native_arrays["query_features"].shape[0] != identity["query_count"] or native_arrays["gallery_features"].shape[0] != identity["gallery_count"]:
            raise ValueError("native feature arrays do not match frozen population")
        for key in ("query_ids", "gallery_ids", "rank_qids", "ranks"):
            if not np.array_equal(native_arrays[key], adapter_arrays[key]):
                raise ValueError(f"native and adapter ordered {key} differ")
        for arrays in (native_arrays, adapter_arrays):
            if not np.array_equal(arrays["query_ids"], arrays["rank_qids"]):
                raise ValueError("rank IDs do not align with query IDs")
        for metadata, arrays in ((native.get("arrays"), native_arrays),
                                 (adapter.get("arrays"), adapter_arrays)):
            if not isinstance(metadata, dict) or set(metadata) != arrays_required:
                raise ValueError("artifact array metadata does not match required schema")
            for name, value in arrays.items():
                ordered_ids = (arrays["gallery_ids"] if name in {"gallery_ids", "gallery_features"}
                               else arrays["query_ids"])
                expected = {"dtype": str(value.dtype), "shape": list(value.shape),
                            "ordered_ids_sha256": _canonical_digest(ordered_ids.tolist())}
                if metadata[name] != expected:
                    raise ValueError(f"artifact {name} metadata differs from its array")
        features = {}
        for key in ("query_features", "gallery_features"):
            reference, actual = native_arrays[key], adapter_arrays[key]
            if reference.ndim != 2:
                raise ValueError(f"invalid {key} arrays")
            mismatches = total = 0
            max_abs = 0.0
            for start in range(0, len(reference), 8192):
                left = np.asarray(reference[start:start + 8192], dtype=np.float64)
                right = np.asarray(actual[start:start + 8192], dtype=np.float64)
                delta = np.abs(right - left)
                mismatches += int(np.count_nonzero(~(delta <= atol + rtol * np.abs(left))))
                max_abs = max(max_abs, float(delta.max(initial=0.0)))
                total += delta.size
            features[key] = {"components_total": total, "mismatch_count": mismatches,
                             "max_abs": max_abs, "passed": mismatches == 0}
        rank_changed = int(np.count_nonzero(native_arrays["ranks"] != adapter_arrays["ranks"]))
        metrics_equal = native.get("metrics") == adapter.get("metrics")
        return {"model_id": native["model_id"], "category": native["category"],
                "condition": native["condition"], "identity": identity,
                "query_count": len(native_arrays["ranks"]),
                "rank_changed_count": rank_changed, "metrics_equal": metrics_equal,
                "native_metrics": native.get("metrics"), "adapter_metrics": adapter.get("metrics"),
                "feature_comparison": features,
                "status": "PASS" if all(item["passed"] for item in features.values()) and
                rank_changed == 0 and metrics_equal else "FAIL"}
