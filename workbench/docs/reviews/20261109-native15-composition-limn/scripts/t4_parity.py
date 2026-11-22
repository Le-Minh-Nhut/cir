"""Task 4: G1 input parity + G2 forward/scoring parity for LIMN.

Two phases, run under the interpreter each side actually uses in production:

  phase `native`  (LIMN interpreter)  — pinned `datasets.py` / `model.py` / `test.py`; records
                  identities, tokens, decoded tensors and native features for a fixed real sample,
                  and the population identities the adapter phase will re-encode.
  phase `adapter` (CONTROL interpreter) — `workbench.composition.model_limn.Adapter`, i.e. the exact
                  gate + subprocess-worker path the CIRCUS runner uses. Compares against the native
                  artifacts and scores/ranks the whole category population.

Splitting on the interpreter is not cosmetic: the adapter refuses to run when its own interpreter is
the controller (`UNSAFE_INTERPRETER`), so "same process" would mean a path production never takes.

Tolerances: declared in INPUT_FORWARD_PARITY.vi.md before this ran.
"""
from __future__ import annotations

import base64
import zlib
import hashlib
import contextlib
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, "/home/heheboiz/data/cir")

import numpy as np

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.backend.operator_config import resolve_config

OUT = Path("/home/heheboiz/data/cir/workbench/docs/reviews/20261109-native15-composition-limn")
RAW = OUT / "raw"

TOL = {
    "forward_max_abs": 1e-5,
    "forward_max_rel": 1e-4,
    "batch_max_abs": 1e-4,
    "batch_max_rel": 1e-3,
    "score_max_abs": 1e-6,
    "tie_epsilon": 1e-6,
}


def correct_text(text: str, correction: dict[str, str]) -> str:
    """Verbatim copy of `datasets.FashionIQ.correct_text`; used only to re-derive the model text."""
    import string

    translation = str.maketrans({key: " " for key in string.punctuation})
    tokens = str(text).lower().translate(translation).strip().split()
    return " ".join(correction.get(word, word) for word in tokens)

# --------------------------------------------------------------------------- v2 evidence writer

#: The declared standard this producer emits from now on. Written FIRST in
#: `workbench/docs/reviews/20261112-parity-repair/V2_STANDARD.md`, then here, then read by
#: `parity_gate._component_checks`. Both phases declare it: the gate refuses a pair whose two sides
#: disagree, and the block itself is written by the phase that holds BOTH arrays.
STANDARD_V2 = "parity-absolute-relative/v2"
#: Decides the coverage form. At or below it the whole array is one `full` record; above it the array
#: is emitted as `chunks`, chunked at `COMPONENT_CHUNK_SIZE`. Named here, declared in the record, and
#: read by `parity_gate` as a documented constant -- not a tunable that could shrink silently.
COMPONENT_FULL_LIMIT = 4096
#: Components per chunk in the `chunks` form. Only the FILE SIZE depends on this: the reader requires
#: the chunks to tile `[0, components_total)` with no gap and no overlap whatever the size, so a
#: record cannot buy an easier verdict by cutting its array into more pieces.
COMPONENT_CHUNK_SIZE = 4096
#: At or below this many components the values are written as JSON numbers so a reviewer can read a
#: small sample directly; above it they travel base64-encoded. Both are the same float64 bytes.
COMPONENT_READABLE_LIMIT = 2048
#: The preview form's head. NO LONGER EMITTED -- kept so an older record can still be read, and
#: because `parity_gate` refuses a `chunked` record a verdict either way.
COMPONENT_HEAD = 512

def _component_digest(array: np.ndarray) -> str:
    """Hash canonical little-endian float64 bytes without materializing full memmaps."""
    digest = hashlib.sha256()
    if not array.size:
        return digest.hexdigest()
    chunks = np.nditer(array, flags=["external_loop", "buffered"],
                       op_flags=[["readonly"]], op_dtypes=[np.dtype("<f8")],
                       order="C", buffersize=65536)
    for chunk in chunks:
        digest.update(np.ascontiguousarray(chunk, dtype="<f8").tobytes())
    return digest.hexdigest()

def _encode_canonical(raw: bytes) -> str:
    return base64.b64encode(zlib.compress(raw, 9)).decode("ascii")


def _encode_components(array: np.ndarray) -> str:
    return _encode_canonical(np.ascontiguousarray(array, dtype="<f8").tobytes())
def component_scope(name: str, dimension: str, actual: np.ndarray, reference: np.ndarray,
                    *, atol: float, rtol: float) -> dict:
    """Emit complete component evidence with bounded temporary memory."""
    if actual.shape != reference.shape:
        raise SystemExit(f"{name}: {actual.shape} vs {reference.shape} are not aligned; refusing to record a "
                         "component block for arrays that do not correspond")
    total = int(actual.size)
    reference_digest, actual_digest, delta_digest = (hashlib.sha256() for _ in range(3))
    max_abs = 0.0
    worst_normalized = 0.0
    offenders: list[dict] = []
    chunks = []
    readable_reference: list[float] = []
    readable_actual: list[float] = []
    if total:
        iterator = np.nditer([actual, reference], flags=["external_loop", "buffered"],
                             op_flags=[["readonly"], ["readonly"]],
                             op_dtypes=[np.dtype("<f8"), np.dtype("<f8")],
                             order="C", buffersize=COMPONENT_CHUNK_SIZE)
    else:
        iterator = ()
    start = 0
    for actual_values, reference_values in iterator:
        a = np.asarray(actual_values, dtype="<f8")
        b = np.asarray(reference_values, dtype="<f8")
        delta = np.abs(a - b)
        denominator = atol + rtol * np.abs(b)
        normalized = np.zeros(delta.shape, dtype=np.float64)
        np.divide(delta, denominator, out=normalized, where=denominator > 0.0)
        raw_actual = np.ascontiguousarray(a, dtype="<f8").tobytes()
        raw_reference = np.ascontiguousarray(b, dtype="<f8").tobytes()
        raw_delta = np.ascontiguousarray(delta, dtype="<f8").tobytes()
        reference_digest.update(raw_reference)
        actual_digest.update(raw_actual)
        delta_digest.update(raw_delta)
        stop = start + len(a)
        if delta.size:
            max_abs = max(max_abs, float(np.max(delta)))
            worst_normalized = max(worst_normalized, float(np.max(normalized)))
            for local in np.flatnonzero(~(delta <= denominator)):
                index = start + int(local)
                offenders.append({"index": index, "reference": float(b[local]),
                                  "actual": float(a[local]), "normalized": float(normalized[local])})
        if total <= COMPONENT_FULL_LIMIT:
            readable_reference.extend(float(value) for value in b)
            readable_actual.extend(float(value) for value in a)
        else:
            chunks.append({"start": start, "reference": _encode_canonical(raw_reference),
                           "actual": _encode_canonical(raw_actual),
                           "sha256": {"reference": hashlib.sha256(raw_reference).hexdigest(),
                                      "actual": hashlib.sha256(raw_actual).hexdigest(),
                                      "delta": hashlib.sha256(raw_delta).hexdigest()}})
        start = stop
    record = {"dimension": dimension, "shape": list(actual.shape),
              "components_total": total, "components_emitted": total,
              "mismatch_count": len(offenders), "max_abs": max_abs,
              "worst_normalized": worst_normalized, "offenders": offenders}
    if total <= COMPONENT_FULL_LIMIT:
        record.update({"coverage": "full", "indices": list(range(total)),
                       "reference": (_encode_canonical(np.asarray(readable_reference, dtype="<f8").tobytes())
                                     if total > COMPONENT_READABLE_LIMIT else readable_reference),
                       "actual": (_encode_canonical(np.asarray(readable_actual, dtype="<f8").tobytes())
                                  if total > COMPONENT_READABLE_LIMIT else readable_actual)})
    else:
        record.update({"coverage": "chunks", "chunk_size": COMPONENT_CHUNK_SIZE, "chunks": chunks})
    record["sha256"] = {"reference": reference_digest.hexdigest(),
                        "actual": actual_digest.hexdigest(), "delta": delta_digest.hexdigest()}
    return record


def _native_pixel_payload(*, capture_full_pixels: bool, query_pixels: np.ndarray,
                          gallery_pixels: np.ndarray | None, sample_indices: list[int],
                          sample_gallery_pixels: np.ndarray) -> dict[str, np.ndarray]:
    if capture_full_pixels:
        if gallery_pixels is None:
            raise ValueError("full native pixel capture requires gallery pixels")
        return {"query_pixels": query_pixels, "gallery_pixels": gallery_pixels,
                "source_tensor": query_pixels[sample_indices]}
    return {"sample_query_pixels": query_pixels, "sample_gallery_pixels": sample_gallery_pixels,
            "source_tensor": query_pixels}

def _source_concat_observer(native_numpy, argsort_observer, query_count: int, gallery_count: int):
    """Retain exact native scorer operands after source-owned row normalization."""
    operands: dict[str, np.ndarray] = {}
    original_concatenate = native_numpy.concatenate

    def capture_concatenate(values, *args, **kwargs):
        result = original_concatenate(values, *args, **kwargs)
        if isinstance(result, np.ndarray) and result.ndim == 2:
            key = "query" if not operands else "gallery"
            if key in operands:
                raise RuntimeError("LIMN source scorer concatenated duplicate feature operand")
            expected_rows = query_count if key == "query" else gallery_count
            if result.shape[0] != expected_rows:
                raise RuntimeError("LIMN source scorer concatenation order/count differs from declared operands")
            operands[key] = result
        return result

    class NativeNumpyProxy:
        def __getattr__(self, name):
            if name == "argsort":
                return argsort_observer
            if name == "concatenate":
                return capture_concatenate
            return getattr(native_numpy, name)

    return NativeNumpyProxy(), operands


def _observe_source_rank(values: np.ndarray, ordered: np.ndarray, gallery_ids: list[str],
                        reference_id: str, target_id: str, *, reference_excluded: bool
                        ) -> tuple[np.ndarray, np.ndarray, int]:
    if values.ndim != 1 or ordered.shape != values.shape or len(values) != len(gallery_ids):
        raise RuntimeError("LIMN source rank observer received misaligned score row/order/gallery IDs")
    if not np.array_equal(np.sort(ordered), np.arange(len(gallery_ids))):
        raise RuntimeError("LIMN source argsort returned invalid gallery order")
    reference_position = gallery_ids.index(reference_id)
    target_position = gallery_ids.index(target_id)
    if reference_excluded and values[reference_position] != np.asarray(10e10, dtype=values.dtype):
        raise RuntimeError("LIMN source reference score is not excluded by pinned sentinel")
    rank = int(np.flatnonzero(ordered == target_position)[0]) + 1
    return -values.copy(), ordered.copy(), rank

def _native_query_image_ids(images: list[str], query: dict) -> tuple[str, str]:
    reference_position, target_position = query["source_img_id"], query["target_img_id"]
    if not (0 <= reference_position < len(images) and 0 <= target_position < len(images)):
        raise RuntimeError("LIMN source query positions exceed the category image split")
    return images[reference_position], images[target_position]


def write_v3_plan(path: Path, plan: dict) -> str:
    """Persist immutable V3 preregistration before starting named runs; return its SHA-256."""
    if path.exists():
        raise FileExistsError(f"V3 preregistration already exists: {path}")
    if plan.get("standard") != "parity-absolute-relative/v3":
        raise ValueError("V3 plan must declare parity-absolute-relative/v3")
    raw = json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def v3_run_receipt(*, run_id: str, side: str, source_id: str, plan_sha256: str,
                   started_at: str, arrays: dict[str, tuple[np.ndarray, list[str]]],
                   execution: dict[str, str | int], identity: dict) -> dict:
    """Capture execution start, actual identity, batch partition, and tensor metadata."""
    identity_fields = {"checkpoint", "configuration", "environment", "protocol", "category", "cohort_id",
                       "precision", "seed", "query_order_sha256", "gallery_order_sha256",
                       "reference_exclusions_sha256", "query_count", "gallery_count"}
    if not all((run_id, source_id, plan_sha256, started_at)) or side not in ("native", "adapter"):
        raise ValueError("V3 receipt needs run ID, source ID, plan digest, start time, and side")
    if not isinstance(identity, dict) or set(identity) != identity_fields:
        raise ValueError("V3 receipt needs exact actual run identity fields")
    if any(not isinstance(identity[key], str) or not identity[key]
           for key in identity_fields - {"seed", "query_count", "gallery_count"}):
        raise ValueError("V3 receipt identity string fields must be observed and non-empty")
    if any(isinstance(identity[key], bool) or not isinstance(identity[key], int) or identity[key] <= 0
           for key in ("query_count", "gallery_count")):
        raise ValueError("V3 receipt identity population counts must be positive integers")
    if isinstance(identity["seed"], bool) or not isinstance(identity["seed"], (int, str)):
        raise ValueError("V3 receipt identity seed must be an integer or explicit string")
    if any(len(identity[key]) != 64 or any(c not in "0123456789abcdef" for c in identity[key])
           for key in ("query_order_sha256", "gallery_order_sha256", "reference_exclusions_sha256")):
        raise ValueError("V3 receipt identity order/exclusion fields must be SHA256 hex digests")
    from datetime import datetime, timezone
    try:
        execution_started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("V3 receipt started_at must be an actual ISO-8601 execution start") from error
    if execution_started.tzinfo is None or execution_started > datetime.now(timezone.utc):
        raise ValueError("V3 receipt started_at must be aware and not future-dated")
    for key in ("partition_id", "query_partition_sha256", "gallery_partition_sha256"):
        value = execution.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"V3 receipt execution lacks {key}")
    query_batch_size = execution.get("query_batch_size")
    gallery_batch_size = execution.get("gallery_batch_size")
    if isinstance(query_batch_size, bool) or not isinstance(query_batch_size, int) or query_batch_size <= 0:
        raise ValueError("V3 receipt execution query_batch_size must be a positive integer")
    if isinstance(gallery_batch_size, bool) or not isinstance(gallery_batch_size, int) or gallery_batch_size <= 0:
        raise ValueError("V3 receipt execution gallery_batch_size must be a positive integer")
    captured = {}
    for name, (value, ordered_ids) in arrays.items():
        if value.dtype.hasobject or not value.size or not ordered_ids or len(ordered_ids) != value.shape[0]:
            raise ValueError(f"V3 receipt array {name!r} needs concrete data and aligned ordered IDs")
        id_bytes = json.dumps(ordered_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        captured[name] = {"dtype": str(value.dtype), "shape": list(value.shape),
                          "component_sha256": _component_digest(value),
                          "ordered_ids_sha256": hashlib.sha256(id_bytes).hexdigest()}
    captured_at = datetime.now(timezone.utc)
    if captured_at < execution_started:
        raise ValueError("V3 capture time precedes execution start")
    return {"run_id": run_id, "side": side, "source_id": source_id,
            "plan_sha256": plan_sha256, "started_at": execution_started.isoformat(),
            "captured_at": captured_at.isoformat(), "execution": execution,
            "identity": identity, "arrays": captured}


def v3_artifact_block(plan_path: Path, plan_sha256: str, *, scopes: dict,
                      receipts: dict[str, dict]) -> dict:
    """Attach captures to final artifact; receipts stay embedded, plan remains immutable sidecar."""
    if not plan_sha256 or not scopes or not receipts:
        raise ValueError("V3 final artifact needs plan digest, scopes, and actual run receipts")
    plan_path = Path(plan_path)
    plan_name = plan_path.name
    if plan_path.is_absolute() or plan_path != Path(plan_name) or not plan_name.endswith(".v3-plan.json"):
        raise ValueError("V3 artifact plan_path must be exactly a relative sibling preregistration filename")
    return {"standard": "parity-absolute-relative/v3",
            "plan_path": plan_name, "plan_sha256": plan_sha256,
            "scopes": scopes, "receipts": receipts}


def _validate_v3_generation(path: Path, *, require_v3_complete: bool) -> None:
    from collections import defaultdict
    groups = defaultdict(set)
    for artifact_path in path.glob("g1_g2.*.json"):
        parts = artifact_path.name.removesuffix(".json").split(".")
        if len(parts) != 4 or parts[0] != "g1_g2" or parts[3] not in ("native", "adapter"):
            continue
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        model, category, side = parts[1:]
        declared_standard = artifact.get("standard", "parity-relative/v1")
        if declared_standard == "parity-absolute-relative/v3" and side == "native":
            block = artifact.get("parity_v3", {})
            proof_name = block.get("scoring_proof_path")
            proof_path = path / proof_name if isinstance(proof_name, str) else None
            if (proof_path is None or proof_path.parent != path or not proof_path.is_file()
                    or hashlib.sha256(proof_path.read_bytes()).hexdigest() != block.get("scoring_proof_sha256")):
                raise ValueError(f"published V3 scoring proof missing or mismatched: {artifact_path.name}")
        if declared_standard == "parity-absolute-relative/v3":
            groups[(model, "v3")].add((category, side))
        elif declared_standard in ("parity-relative/v1", "parity-absolute-relative/v2"):
            groups[(model, "legacy")].add((category, side))
        else:
            raise ValueError(f"unsupported published parity standard: {artifact_path.name}")
    if require_v3_complete:
        models = {model for model, _ in groups}
        for model in models:
            v3 = groups.get((model, "v3"), set())
            legacy = groups.get((model, "legacy"), set())
            expected = {(category, side) for category in ("dress", "shirt", "toptee")
                        for side in ("native", "adapter")}
            if v3 and v3 != expected:
                raise ValueError(f"incomplete V3 category publication for {model}: {sorted(v3)}")


def _current_generation(publication_root: Path) -> Path | None:
    """Resolve only publisher-owned generation IDs through the fixed pointer contract."""
    pointer = publication_root / "CURRENT.json"
    try:
        current = json.loads(pointer.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"V3 CURRENT.json is invalid: {error}") from error
    generation = current.get("generation") if isinstance(current, dict) else None
    if (not isinstance(generation, str) or not generation or generation in (".", "..")
            or Path(generation).name != generation or "/" in generation or "\\" in generation):
        raise ValueError("V3 CURRENT.json has unsafe generation ID")
    generations = publication_root / "generations"
    resolved = (generations / generation).resolve()
    if not resolved.is_relative_to(generations.resolve()):
        raise ValueError("V3 CURRENT.json generation escapes trusted root")
    return resolved


def publish_v3_generation(publication_root: Path, generation: str, artifacts: dict[str, dict],
                          *, plan_paths: dict[str, Path], proof_paths: dict[str, Path] | None = None) -> Path:
    """Serialize publishers while staging complete generations and switching trusted CURRENT.json."""
    import fcntl
    publication_root.mkdir(parents=True, exist_ok=True)
    with (publication_root / ".publish.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _publish_v3_generation_locked(publication_root, generation, artifacts,
                                             plan_paths=plan_paths, proof_paths=proof_paths or {})




def _publish_v3_generation_locked(publication_root: Path, generation: str,
                                  artifacts: dict[str, dict], *,
                                  plan_paths: dict[str, Path], proof_paths: dict[str, Path]) -> Path:
    """Publish one complete generation; caller holds the publication lock."""
    import os
    import shutil
    import tempfile

    if (not generation or Path(generation).name != generation or generation in (".", "..")
            or "/" in generation or "\\" in generation or not artifacts):
        raise ValueError("V3 publication needs safe generation ID and artifacts")
    generations = publication_root / "generations"
    generations.mkdir(parents=True, exist_ok=True)
    destination = generations / generation
    if destination.exists():
        raise FileExistsError(f"V3 generation already published: {destination}")
    additions = set(artifacts)
    if additions:
        model_ids = set()
        for name in additions:
            parts = name.removesuffix(".json").split(".")
            if len(parts) == 4 and parts[0] == "g1_g2" and parts[3] in ("native", "adapter"):
                model_ids.add(parts[1])
            elif len(parts) != 4 or parts[0] != "g1_g2" or parts[3] != "v3-plan":
                raise ValueError(f"invalid V3 publication filename: {name!r}")
        required = {f"g1_g2.{model}.{category}.{side}.json"
                    for model in model_ids for category in ("dress", "shirt", "toptee")
                    for side in ("native", "adapter")}
        required |= {f"g1_g2.{model}.{category}.v3-plan.json"
                     for model in model_ids for category in ("dress", "shirt", "toptee")}
        carried = set()
        previous = _current_generation(publication_root)
        if previous is not None and previous.is_dir():
            carried = {path.name for path in previous.iterdir() if path.is_file()}
        legacy = Path(__file__).resolve().parents[1]
        carried.update(path.name for path in legacy.glob("g1_g2.*.json"))
        if not model_ids or not required.issubset(additions | set(plan_paths) | carried):
            raise ValueError("V3 publication must include both sides and preregistration for all three categories")
    for name in additions:
        parts = name.removesuffix(".json").split(".")
        if len(parts) == 4 and parts[3] in ("native", "adapter"):
            payload = artifacts[name]
            if payload.get("standard") != "parity-absolute-relative/v3":
                raise ValueError(f"new V3 publication cannot relabel or carry non-V3 record {name!r}")
            if payload.get("model_id") != parts[1] or payload.get("category") != parts[2]:
                raise ValueError(f"V3 artifact identity differs from filename {name!r}")
    for name, source in plan_paths.items():
        parts = name.removesuffix(".v3-plan.json").split(".")
        if len(parts) != 3 or parts[0] != "g1_g2":
            raise ValueError(f"invalid V3 plan identity filename {name!r}")
        plan = json.loads(source.read_text(encoding="utf-8"))
        if plan.get("standard") != "parity-absolute-relative/v3" or (plan.get("model_id"), plan.get("category")) != (parts[1], parts[2]):
            raise ValueError(f"V3 plan identity differs from filename {name!r}")
    expected_proofs = set()
    for artifact_name, payload in artifacts.items():
        block = payload.get("parity_v3", {})
        proof_name = block.get("scoring_proof_path")
        if proof_name is None:
            continue
        proof = proof_paths.get(proof_name)
        if (not isinstance(proof_name, str) or Path(proof_name).name != proof_name
                or not isinstance(proof, Path) or not proof.is_file()):
            raise ValueError(f"V3 scoring proof is missing for {artifact_name!r}")
        if hashlib.sha256(proof.read_bytes()).hexdigest() != block.get("scoring_proof_sha256"):
            raise ValueError(f"V3 scoring proof digest differs for {artifact_name!r}")
        expected_proofs.add(proof_name)
    if set(proof_paths) != expected_proofs:
        raise ValueError("V3 publication received undeclared scoring proof sidecars")
    temporary = Path(tempfile.mkdtemp(prefix=f".{generation}.", dir=generations))
    try:
        legacy = Path(__file__).resolve().parents[1]
        if legacy.is_dir():
            for prior in legacy.glob("g1_g2.*.json"):
                os.link(prior, temporary / prior.name)
        current_path = _current_generation(publication_root)
        if current_path is not None and current_path.is_dir():
            for prior in current_path.glob("g1_g2.*.json"):
                carried = temporary / prior.name
                if carried.exists():
                    carried.unlink()
                os.link(prior, carried)
            for prior in current_path.glob("g1_g2.*.score-ranks.jsonl"):
                carried = temporary / prior.name
                carried.unlink(missing_ok=True)
                os.link(prior, carried)
        for name, source in plan_paths.items():
            if Path(name).name != name or not name.startswith("g1_g2.") or not name.endswith(".v3-plan.json"):
                raise ValueError(f"invalid immutable V3 plan filename: {name!r}")
            target = temporary / name
            target.unlink(missing_ok=True)
            shutil.copy2(source, target)
        for name, source in proof_paths.items():
            if Path(name).name != name or not name.startswith("g1_g2.") or not name.endswith(".score-ranks.jsonl"):
                raise ValueError(f"invalid V3 scoring proof filename: {name!r}")
            target = temporary / name
            target.unlink(missing_ok=True)
            shutil.copy2(source, target)
        for name, payload in artifacts.items():
            if Path(name).name != name or not name.startswith("g1_g2.") or not name.endswith(".json"):
                raise ValueError(f"invalid V3 artifact filename: {name!r}")
            if name in plan_paths:
                raise ValueError(f"V3 plan supplied as both artifact and plan source: {name!r}")
            target = temporary / name
            target.unlink(missing_ok=True)
            with target.open("xb") as stream:
                stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                        ensure_ascii=False).encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
        _validate_v3_generation(temporary, require_v3_complete=True)
        os.replace(temporary, destination)
        pointer_tmp = publication_root / f".CURRENT.{os.getpid()}.tmp"
        try:
            with pointer_tmp.open("xb") as stream:
                stream.write(json.dumps({"generation": generation}, separators=(",", ":")).encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(pointer_tmp, publication_root / "CURRENT.json")
        finally:
            pointer_tmp.unlink(missing_ok=True)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination


def component_block(scopes: dict[str, dict]) -> dict:
    """The `component_parity` document, as recorded beside `relation_state` at the top level."""
    return {"standard": STANDARD_V2,
            "declared_rule": "abs(adapter - native) <= atol + rtol * abs(native)",
            "reference_side": "native",
            "chunk_rule": (f"full when components_total <= {COMPONENT_FULL_LIMIT}, else every "
                           f"component in chunks of at most {COMPONENT_CHUNK_SIZE}: the chunks tile "
                           "[0, components_total) with no gap and no overlap, so the coverage is "
                           "complete at any size"),
            "scopes": scopes}


@contextlib.contextmanager
def native_session(category: str, config):
    """Keep one verified native LIMN model resident for bounded controls."""
    from workbench.replay import limn as replay
    checkpoint_root = Path(config.WORKBENCH_CHECKPOINT_ROOT) / "limn"
    dataset_root = Path(config.FASHIONIQ_ROOT)
    source_root = Path(config.WORKBENCH_THIRD_PARTY_ROOT) / "LIMN"
    problems = replay.preflight(checkpoint_root, dataset_root, source_root, (category,))
    if problems:
        raise SystemExit(f"preflight blocked: {problems}")
    with replay.native_category(category, checkpoint_root, dataset_root, source_root) as loaded:
        yield loaded


def native_selected_phase(category: str, config, loaded: dict, *, artifact_root: Path,
                          query_indices: list[int], gallery_indices: list[int], batch_size: int) -> dict:
    """Run source compose_query/compose_target on exact selected dataset rows."""
    import torch
    artifact_root.mkdir(parents=True, exist_ok=True)
    model, dataset = loaded["model"], loaded["dataset"]
    qrows, grows = dataset.test_queries, dataset.test_targets
    if not query_indices or not gallery_indices:
        raise ValueError("native selected phase requires non-empty query/gallery indices")
    dataset_type = {"dress": 0, "shirt": 1, "toptee": 2}[category]
    native_capture: dict[str, list] = {"query_pixels": [], "gallery_pixels": [], "token_ids": [], "padding_mask": [], "texts": []}
    tokenizer = model.backbone.tokenizer
    transformer = model.backbone.clip.transformer
    transformer_forward = transformer.forward
    causal_masks = []
    def capture_tokenizer(texts):
        token_ids = tokenizer(texts)
        native_capture["token_ids"].append(token_ids.detach().cpu().numpy())
        native_capture["padding_mask"].append((token_ids != 0).detach().cpu().numpy())
        native_capture["texts"].extend(texts)
        return token_ids
    def capture_transformer(*args, **kwargs):
        mask = kwargs.get("attn_mask")
        if mask is None:
            raise RuntimeError("LIMN native tokenizer did not expose actual causal attention mask")
        value = mask.detach().cpu().numpy()
        if not causal_masks:
            causal_masks.append(value.copy())
        elif not np.array_equal(causal_masks[0], value, equal_nan=True):
            raise RuntimeError("LIMN causal attention mask changed within capture")
        return transformer_forward(*args, **kwargs)
    query_rows = [qrows[i] for i in query_indices]
    gallery_rows = [grows[i] for i in gallery_indices]
    try:
        model.backbone.tokenizer = capture_tokenizer
        transformer.forward = capture_transformer
        queries = []
        gallery = []
        with torch.no_grad():
            for start in range(0, len(query_rows), batch_size):
                chunk = query_rows[start:start + batch_size]
                images = torch.stack([entry["source_img_data"] for entry in chunk]).float().cuda()
                native_capture["query_pixels"].append(images.detach().cpu().numpy())
                types = torch.ones(len(chunk), device=images.device) * dataset_type
                queries.append(model.compose_query(images, [entry["mod"]["str"] for entry in chunk], types)[0].detach().cpu().numpy())
            for start in range(0, len(gallery_rows), batch_size):
                chunk = gallery_rows[start:start + batch_size]
                images = torch.stack([entry["target_img_data"] for entry in chunk]).float().cuda()
                native_capture["gallery_pixels"].append(images.detach().cpu().numpy())
                types = torch.ones(len(chunk)).cuda() * int(dataset.dataset_type)
                gallery.append(model.compose_target(images, types)[0].detach().cpu().numpy())
    finally:
        model.backbone.tokenizer = tokenizer
        transformer.forward = transformer_forward
    queries = np.concatenate(queries).astype(np.float32, copy=False)
    gallery = np.concatenate(gallery).astype(np.float32, copy=False)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
    if len(native_capture["token_ids"]) == 0 or len(causal_masks) != 1:
        raise RuntimeError("LIMN native selected G1 capture omitted tokens or causal mask")
    split_path = Path(config.FASHIONIQ_ROOT) / "image_splits" / f"split.{category}.val.json"
    image_names = json.loads(split_path.read_text(encoding="utf-8"))
    np.savez_compressed(artifact_root / f"g1_{category}_selected.npz",
                        query_pixels=np.concatenate(native_capture["query_pixels"]),
                        gallery_pixels=np.concatenate(native_capture["gallery_pixels"]),
                        query_ids=np.asarray([image_names[dataset.test_queries[i]["source_img_id"]]
                                              for i in query_indices]),
                        gallery_ids=np.asarray([image_names[dataset.test_targets[i]["target_img_id"]]
                                                for i in gallery_indices]),
                        texts=np.asarray(native_capture["texts"]),
                        token_ids=np.concatenate(native_capture["token_ids"]),
                        text_padding_mask=np.concatenate(native_capture["padding_mask"]),
                        causal_attn_mask_actual=causal_masks[0],
                        queries=queries, gallery=gallery)
    np.savez_compressed(artifact_root / f"g2_{category}_selected.npz", queries=queries, gallery=gallery,
                        query_ids=np.asarray([image_names[dataset.test_queries[i]["source_img_id"]]
                                              for i in query_indices]),
                        gallery_ids=np.asarray([image_names[dataset.test_targets[i]["target_img_id"]]
                                                for i in gallery_indices]),
                        query_indices=np.asarray(query_indices, dtype=np.int32),
                        gallery_indices=np.asarray(gallery_indices, dtype=np.int32))
    import open_clip
    return {"query_count": len(query_indices), "gallery_count": len(gallery_indices),
            "query_batch_size": batch_size, "gallery_batch_size": batch_size,
            "native_runtime": {"python": sys.version, "executable": sys.executable,
                               "torch": torch.__version__,
                               "open_clip": getattr(open_clip, "__version__", "unknown"),
                               "model_training": bool(model.training),
                               "parameter_dtypes": sorted({str(parameter.dtype) for parameter in model.parameters()}),
                               "seed": 0, "batch_size": batch_size, "precision": "float32",
                               "model_mode": "eval", "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                               "cudnn_benchmark": bool(torch.backends.cudnn.benchmark)},
            "g1_path": str(artifact_root / f"g1_{category}_selected.npz")}


def native_phase(category: str, config, *, artifact_root: Path = RAW,
                 batch_size: int = 32, capture_full_pixels: bool = True,
                 loaded: dict | None = None) -> dict:
    from workbench.replay import limn as replay
    artifact_root.mkdir(parents=True, exist_ok=True)

    import torch
    checkpoint_root = Path(config.WORKBENCH_CHECKPOINT_ROOT) / "limn"
    dataset_root = Path(config.FASHIONIQ_ROOT)
    source_root = Path(config.WORKBENCH_THIRD_PARTY_ROOT) / "LIMN"
    problems = replay.preflight(checkpoint_root, dataset_root, source_root, (category,))
    if problems:
        raise RuntimeError("LIMN native preflight failed: " + "; ".join(problems))
    import random
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    began = time.time()
    session = contextlib.nullcontext(loaded) if loaded is not None else replay.native_category(
        category, checkpoint_root, dataset_root, source_root)

    with session as loaded:
        model, dataset, native_test = loaded["model"], loaded["dataset"], loaded["native_test"]
        if model.training or any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise RuntimeError("native LIMN model must be eval-mode float32")
        if batch_size != 32:
            raise ValueError("primary native diagnostic requires preregistered batch size 32")
        images = json.loads((dataset_root / "image_splits" / f"split.{category}.val.json").read_text())
        captions = json.loads((dataset_root / "captions" / f"cap.{category}.val.json").read_text())
        correction = json.loads((dataset_root / "captions" / f"correction_dict_{category}.json").read_text())

        gallery_positions = [entry["target_img_id"] for entry in dataset.test_targets]
        if any(not 0 <= position < len(images) for position in gallery_positions):
            raise RuntimeError("LIMN gallery image positions exceed the category image split")
        gallery_ids = [images[position] for position in gallery_positions]
        rebuilt: list[str] = []
        for row in captions:
            for key in ("candidate", "target"):
                if row[key] not in rebuilt:
                    rebuilt.append(row[key])

        native_texts = [entry["mod"]["str"] for entry in dataset.test_queries]
        composed = [f"{correct_text(row['captions'][0], correction)} and "
                    f"{correct_text(row['captions'][1], correction)}" for row in captions]
        native_refs = [images[entry["source_img_id"]] for entry in dataset.test_queries]
        native_targets = [images[entry["target_img_id"]] for entry in dataset.test_queries]

        # Fixed small REAL sample covering all three categories plus caption edge cases and the
        # final short batch: first row, first row whose FIRST caption is capitalized, first row
        # whose second caption is shorter than 12 characters, and the last row.
        capitalized = next((i for i, row in enumerate(captions)
                            if row["captions"][0][:1].isupper()), 0)
        short_half = next((i for i, row in enumerate(captions)
                           if len(row["captions"][1]) < 12), 0)
        longest = max(range(len(native_texts)), key=lambda i: len(native_texts[i]))
        chosen = sorted({0, capitalized, short_half, longest, len(captions) - 1})

        identity = {
            "query_count_native": len(dataset.test_queries),
            "query_count_annotations": len(captions),
            "gallery_count_native": len(dataset.test_targets),
            "gallery_count_rebuilt_from_annotations": len(rebuilt),
            "gallery_order_identical": gallery_ids == rebuilt,
            "reference_ids_identical": native_refs == [row["candidate"] for row in captions],
            "target_ids_identical": native_targets == [row["target"] for row in captions],
            "model_text_identical_to_author_composition": native_texts == composed,
            "model_text_mismatches": [
                {"index": i, "native": native_texts[i], "recomposed": composed[i]}
                for i in range(len(composed)) if native_texts[i] != composed[i]][:5],
            "duplicate_caption_pairs_within_category":
                len(captions) - len({(row["candidate"], composed[i]) for i, row in enumerate(captions)}),
            "reference_equals_target_rows": sum(1 for row in captions if row["candidate"] == row["target"]),
            "tokenizer": {
                "callable": f"{getattr(model.backbone.tokenizer, '__module__', '?')}."
                            f"{getattr(model.backbone.tokenizer, '__qualname__', type(model.backbone.tokenizer).__name__)}",
                "vocab_size": int(model.backbone.clip.token_embedding.num_embeddings),
                "context_length": int(model.backbone.clip.positional_embedding.shape[0]),
                "source": "open_clip.get_tokenizer('ViT-L-14') inside pinned Backbone.__init__",
            },
            "transform": {
                "image_size": (list(model.backbone.clip.visual.image_size)
                               if isinstance(model.backbone.clip.visual.image_size, (tuple, list))
                               else int(model.backbone.clip.visual.image_size)),
                "mean": [float(v) for v in model.backbone.clip.visual.image_mean],
                "std": [float(v) for v in model.backbone.clip.visual.image_std],
                "source": "open_clip.image_transform(..., is_train=False) in replay.limn.load_native",
            },
            "model_in_eval": not model.training,
            "dataset_type_index": int(dataset.dataset_type),
            "sample_indices": chosen,
            "checkpoint_id": loaded["checkpoint"]["checkpoint_id"],
            "checkpoint_sha256": loaded["checkpoint"]["expected_sha256"],
        }
        identity["sample_tokens"] = [
            {"index": i, "text": native_texts[i],
             "token_ids": model.backbone.tokenizer([native_texts[i]]).numpy().tolist()[0]}
            for i in chosen]

        from PIL import Image

        def image_path(name: str) -> str:
            # NOT `dataset.image_dir`: that is the isolated replay cache, removed when this context
            # exits, so the adapter phase could never open it. The cache only symlinks the real
            # iLearn files, and those are what the native loader read.
            return str(dataset_root / "resized_image" / category / f"{name}.jpg")

        sample_inputs = {
            "category": category,
            "queries": [[image_path(native_refs[i]), image_path(native_targets[i])]
                        for i in chosen],
            "query_texts": [native_texts[i] for i in chosen],
            "reference_ids": [native_refs[i] for i in chosen],
            "target_ids": [native_targets[i] for i in chosen],
            "sample_indices": chosen,
            "full_query_paths": [image_path(name) for name in native_refs],
            "full_query_texts": native_texts,
            "full_reference_ids": native_refs,
            "full_target_ids": native_targets,
            "gallery_ids": gallery_ids,
            "gallery_paths": [image_path(name) for name in gallery_ids],
            "dataset_type_index": int(dataset.dataset_type),
        }
        modes: dict[str, int] = {}
        for pair in sample_inputs["queries"]:
            for path in pair:
                with Image.open(path) as handle:
                    modes[handle.mode] = modes.get(handle.mode, 0) + 1
        identity["sample_pil_modes"] = modes
        identity["sample_convert_to_rgb_is_noop"] = set(modes) <= {"RGB"}
        import torch
        from types import SimpleNamespace

        sample_count = len(chosen)
        native_capture: dict[str, list] = {"query_pixels": [], "query_features": [],
                                           "gallery_features": [], "token_ids": [], "texts": []}

        tokenizer = model.backbone.tokenizer
        transformer = model.backbone.clip.transformer
        transformer_forward = transformer.forward
        native_causal_mask = None

        token_offset = 0

        def capture_tokenizer(value):
            nonlocal token_offset
            token_ids = tokenizer(value)
            native_capture["token_ids"].append(token_ids.detach().cpu().numpy())
            native_capture.setdefault("padding_mask", []).append((token_ids != 0).detach().cpu().numpy())
            native_capture["texts"].extend(value)
            token_offset += len(value)
            return token_ids

        def capture_transformer(*args, **kwargs):
            nonlocal native_causal_mask
            mask = kwargs.get("attn_mask")
            if mask is None:
                raise RuntimeError("LIMN native transformer omitted actual attn_mask")
            observed = mask.detach().cpu().numpy()
            if native_causal_mask is None:
                native_causal_mask = observed.copy()
            elif not np.array_equal(native_causal_mask, observed, equal_nan=True):
                raise RuntimeError("LIMN native causal attention mask changed during capture")
            return transformer_forward(*args, **kwargs)

        compose_query, compose_target = model.compose_query, model.compose_target
        query_offset = 0

        def capture_query(images, texts, dataset_type):
            nonlocal query_offset
            if capture_full_pixels:
                native_capture["query_pixels"].append(images.detach().cpu().numpy())
            else:
                selected_positions = [position for position in
                                      range(query_offset, query_offset + len(images)) if position in chosen]
                if selected_positions:
                    local_indices = [position - query_offset for position in selected_positions]
                    native_capture["query_pixels"].append(images[local_indices].detach().cpu().numpy())
            result = compose_query(images, texts, dataset_type)
            native_capture["query_features"].append(result[0].detach().cpu().numpy())
            query_offset += len(images)
            return result
        query_target_positions = [gallery_ids.index(native_targets[i]) for i in chosen]
        target_positions = set(query_target_positions)
        captured_target_positions: list[int] = []
        target_offset = 0

        def capture_target(images, dataset_type):
            nonlocal target_offset
            selected_positions = [position for position in
                                  range(target_offset, target_offset + len(images))
                                  if position in target_positions]
            if selected_positions:
                local_indices = [position - target_offset for position in selected_positions]
                native_capture.setdefault("target_pixels", []).append(
                    images[local_indices].detach().cpu().numpy())
                captured_target_positions.extend(selected_positions)
            result = compose_target(images, dataset_type)
            if capture_full_pixels:
                native_capture.setdefault("gallery_pixels", []).append(images.detach().cpu().numpy())
            native_capture["gallery_features"].append(result[0].detach().cpu().numpy())
            target_offset += len(images)
            return result

        model.backbone.tokenizer = capture_tokenizer
        transformer.forward = capture_transformer
        model.compose_query, model.compose_target = capture_query, capture_target
        native_argsort = native_test.np.argsort
        source_target_ranks: list[int] = []
        source_score_rows: list[np.ndarray] = []
        source_orders: list[np.ndarray] = []

        def capture_source_argsort(values, *args, **kwargs):
            ordered = native_argsort(values, *args, **kwargs)
            if isinstance(values, np.ndarray) and values.ndim == 1 and len(values) == len(dataset.test_targets):
                row = len(source_target_ranks)
                if row >= len(dataset.test_queries):
                    raise RuntimeError("LIMN source rank observer saw excess score rows")
                query = dataset.test_queries[row]
                source_id, target_id = _native_query_image_ids(images, query)
                observed_scores, observed_order, rank = _observe_source_rank(
                    values, ordered, gallery_ids, source_id, target_id,
                    reference_excluded=category != "birds")
                if source_id != native_refs[row] or target_id != native_targets[row]:
                    raise RuntimeError("LIMN source observed query/gallery IDs differ from captured inputs")
                source_score_rows.append(observed_scores)
                source_orders.append(observed_order)
                source_target_ranks.append(rank)
            return ordered

        native_numpy = native_test.np
        source_numpy, native_score_operands = _source_concat_observer(
            native_numpy, capture_source_argsort,
            len(dataset.test_queries), len(dataset.test_targets))
        native_test.np = source_numpy
        try:
            metrics = dict(native_test.test(SimpleNamespace(
                local_rank=0, batch_size=(batch_size or replay.resolved_batch_size())),
                model, dataset, category))
        finally:
            native_test.np = native_numpy
            model.backbone.tokenizer = tokenizer
            transformer.forward = transformer_forward
            model.compose_query, model.compose_target = compose_query, compose_target
        if (set(native_score_operands) != {"query", "gallery"}
                or native_score_operands["query"].shape[0] != len(dataset.test_queries)
                or native_score_operands["gallery"].shape[0] != len(dataset.test_targets)
                or not np.isfinite(native_score_operands["query"]).all()
                or not np.isfinite(native_score_operands["gallery"]).all()):
            raise RuntimeError("LIMN source scorer operands were not captured after native normalization")
        if len(source_target_ranks) != len(dataset.test_queries):
            raise RuntimeError("LIMN source rank observer did not capture every native query")
        if native_causal_mask is None:
            raise RuntimeError("LIMN native tokenizer did not expose actual causal attention mask")
        gallery_features = np.concatenate(native_capture["gallery_features"])
        native_query_features = np.concatenate(native_capture["query_features"])
        native_query_pixels = np.concatenate(native_capture["query_pixels"])
        captured_count = len(native_query_features)
        captured_target_ids = [native_targets[i] for i in chosen]
        sample_target_features = gallery_features[query_target_positions]
        sampled_pixels_in_gallery_order = np.concatenate(native_capture["target_pixels"])
        sample_position_index = {position: index
                                 for index, position in enumerate(captured_target_positions)}
        sample_target_pixels = sampled_pixels_in_gallery_order[
            [sample_position_index[position] for position in query_target_positions]]
        token_ids = np.concatenate(native_capture["token_ids"])
        if not (len(native_query_features) == len(token_ids) == len(native_capture["texts"])
                == len(native_targets)
                and len(native_query_pixels) == (len(native_refs) if capture_full_pixels else sample_count)
                and len(sample_target_features) == len(sample_target_pixels)
                == len(captured_target_ids) == sample_count
                and len(gallery_features) == len(gallery_ids)
                and query_offset == len(native_refs) and token_offset == len(native_refs)
                and target_offset == len(gallery_ids)):
            raise RuntimeError("LIMN native capture arrays are not aligned to actual ordered queries/gallery")
        native_query_features_pre_norm = native_query_features.copy()
        gallery_features_pre_norm = gallery_features.copy()
        # Preserve legacy independent feature captures; source GEMM operands are persisted separately.
        gallery_features /= np.linalg.norm(gallery_features, axis=1, keepdims=True)
        sample_target_features = gallery_features[query_target_positions]
        native_sample = {
            **sample_inputs,
            "queries": [[image_path(native_refs[i]), image_path(native_targets[i])]
                        for i in chosen],
            "query_texts": [native_texts[i] for i in chosen],
            "reference_ids": [native_refs[i] for i in chosen],
            "target_ids": [native_targets[i] for i in chosen],
            "sample_indices": chosen,
        }
        gallery_pixels = (np.concatenate(native_capture["gallery_pixels"])
                          if capture_full_pixels else None)
        pixel_payload = _native_pixel_payload(
            capture_full_pixels=capture_full_pixels, query_pixels=native_query_pixels,
            gallery_pixels=gallery_pixels,
            sample_indices=(chosen if capture_full_pixels else list(range(sample_count))),
            sample_gallery_pixels=sample_target_pixels)
        np.savez_compressed(artifact_root / f"g2_{category}_native.npz",
                            native_query=native_query_features,
                            native_query_pre_norm=native_query_features_pre_norm,
                            sample_query=native_query_features[chosen],
                            native_gallery=gallery_features,
                            native_gallery_pre_norm=gallery_features_pre_norm,
                            native_target=sample_target_features,
                            **pixel_payload,
                            target_tensor=sample_target_pixels,
                            text_padding_mask=(np.concatenate(native_capture["padding_mask"])
                                               if native_capture.get("padding_mask") else np.empty((0, 0), dtype=bool)),
                            unsupported_sample_conditions=np.asarray(json.dumps({
                                "first_caption_capitalized": any(
                                    captions[i]["captions"][0][:1].isupper() for i in chosen),
                                "second_caption_under_12_chars": any(
                                    len(captions[i]["captions"][1]) < 12 for i in chosen),
                                "longest_composed_text_included": longest in chosen,
                                "final_query_included": len(captions) - 1 in chosen,
                                "selected_query_count": len(chosen),
                            }, sort_keys=True)),
                            sample_query_texts=np.asarray([native_capture["texts"][i] for i in chosen]),
                            sample_text_padding_mask=np.concatenate(native_capture["padding_mask"])[chosen],
                            sample_indices=np.asarray(chosen, dtype=np.int32),
                            causal_attn_mask_actual=native_causal_mask,
                            causal_attn_mask_semantics=np.asarray(
                                "actual native OpenCLIP additive attention mask passed to transformer"),
                            query_texts=np.asarray(native_capture["texts"]),
                            query_ids=np.asarray(native_refs),
                            sample_query_ids=np.asarray([native_refs[i] for i in chosen]),
                            sample_target_ids=np.asarray([native_targets[i] for i in chosen]),
                            sample_token_ids=token_ids[chosen],
                            gallery_ids=np.asarray(gallery_ids))
        reference_positions = [gallery_ids.index(reference_id) for reference_id in native_refs]
        target_positions_all = [gallery_ids.index(target_id) for target_id in native_targets]
        if not (len(source_target_ranks) == len(source_score_rows) == len(source_orders)
                == len(native_refs) == len(native_targets) == len(dataset.test_queries)):
            raise RuntimeError("Observed LIMN source scores, orders, ranks, and IDs have different row counts")
        source_hit_counts = {k: sum(rank <= k for rank in source_target_ranks) for k in (1, 10, 50)}
        source_metrics = {f"{category}_r{k}": 100.0 * source_hit_counts[k] / len(source_target_ranks)
                          for k in source_hit_counts}
        if set(metrics) != set(source_metrics) or any(
                round(float(metrics[key]) * len(source_target_ranks) / 100) != source_hit_counts[k]
                or not np.isclose(float(metrics[key]), source_metrics[key], rtol=0.0, atol=1e-10)
                for k, key in ((k, f"{category}_r{k}") for k in source_hit_counts)):
            raise RuntimeError("Observed LIMN source hit counts do not reproduce pinned aggregate metrics")
        np.savez_compressed(artifact_root / f"g2_{category}_native_ranking.npz",
                            source_target_ranks=np.asarray(source_target_ranks, dtype=np.int32),
                            source_hit_counts=np.asarray([source_hit_counts[k] for k in (1, 10, 50)], dtype=np.int64),
                            source_topk_membership=np.asarray(
                                [[int(rank <= k) for k in (1, 10, 50)] for rank in source_target_ranks],
                                dtype=np.uint8),
                            source_score_rows=np.stack(source_score_rows),
                            source_orders=np.stack(source_orders).astype(np.int32),
                            source_gemm_queries=native_score_operands["query"],
                            source_gemm_gallery=native_score_operands["gallery"],
                            **{f"source_topk_k{k}": np.stack(source_orders)[:, :min(k, len(gallery_ids))].astype(np.int32)
                               for k in (1, 10, 50)},
                            target_positions=np.asarray(target_positions_all, dtype=np.int32),
                            reference_positions=np.asarray(reference_positions, dtype=np.int32),
                            query_ids=np.asarray(native_refs), target_ids=np.asarray(native_targets),
                            gallery_ids=np.asarray(gallery_ids),
                            reference_exclusion_sentinel=np.asarray(-10e10, dtype=np.float64),
                            metrics=np.asarray([metrics[f"{category}_r{k}"] for k in (1, 10, 50)],
                                               dtype=np.float64))
        (artifact_root / f"g2_{category}_inputs.json").write_text(
            json.dumps(native_sample), encoding="utf-8")
        identity["native_metrics_recheck"] = metrics
        import open_clip
        identity["native_runtime"] = {
            "python": sys.version, "executable": sys.executable,
            "torch": torch.__version__, "open_clip": getattr(open_clip, "__version__", "unknown"),
            "model_training": bool(model.training),
            "parameter_dtypes": sorted({str(parameter.dtype) for parameter in model.parameters()}),
            "seed": 0, "batch_size": batch_size, "precision": "float32", "model_mode": "eval",
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        }
        del model, dataset
    torch.cuda.empty_cache()
    identity["phase_seconds"] = round(time.time() - began, 1)
    return identity


@contextlib.contextmanager
def adapter_session(category: str, config, artifact_root: Path, batch_size: int):
    from workbench.composition.model_limn import Adapter
    adapter = Adapter(config)
    adapter.batch_size = batch_size
    adapter.work_root = artifact_root / "parity-work" / category
    adapter.work_root.mkdir(parents=True, exist_ok=True)
    try:
        yield adapter
    finally:
        adapter.close()


def adapter_selected_phase(adapter, *, query_paths: list[str], query_texts: list[str],
                           gallery_paths: list[str], categories: list[str],
                           capture_root: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    if len(query_paths) != len(query_texts) or len(categories) != len(query_paths):
        raise ValueError("selected Adapter query paths/texts/categories must align")
    query_parts, gallery_parts = [], []
    if capture_root is not None:
        capture_root.mkdir(parents=True, exist_ok=True)
    for label, paths, texts, cats, method in (
            ("query", query_paths, query_texts, categories, adapter.encode_queries),
            ("gallery", gallery_paths, [], [categories[0]] * len(gallery_paths), adapter.encode_targets)):
        parts = query_parts if label == "query" else gallery_parts
        for start in range(0, len(paths), 32):
            stop = min(start + 32, len(paths))
            capture_path = (capture_root / f"{label}-{start:04d}.npz") if capture_root else None
            if label == "query":
                encoded = method(paths[start:stop], texts[start:stop], cats[start:stop],
                                 capture_path=capture_path)
            else:
                encoded = method(paths[start:stop], cats[start:stop], capture_path=capture_path)
            parts.append(encoded.astype(np.float32, copy=False))
    return np.concatenate(query_parts), np.concatenate(gallery_parts)

def adapter_population_phase(adapter, *, artifact_root: Path, category: str, inputs: dict,
                             config, capture_root: Path | None = None,
                             native_ranking_path: Path | None = None) -> dict:
    queries = adapter.encode_queries(inputs["full_query_paths"], inputs["full_query_texts"],
                                    [category] * len(inputs["full_query_texts"]))
    gallery = adapter.encode_targets(inputs["gallery_paths"], [category] * len(inputs["gallery_paths"]))
    worker_observation = adapter._worker_observation
    if not isinstance(worker_observation, dict):
        raise RuntimeError("production Adapter omitted actual worker runtime observation")
    np.savez(artifact_root / f"g2_{category}_population.npz", queries=queries, gallery=gallery,
             query_features_ids=np.asarray(inputs["full_reference_ids"]),
             gallery_features_ids=np.asarray(inputs["gallery_ids"]))
    if capture_root is not None:
        query_pairs = inputs["queries"]
        query_texts = inputs["query_texts"]
        reference_ids = inputs["reference_ids"]
        target_ids = inputs["target_ids"]
        if not (len(query_pairs) == len(query_texts) == len(reference_ids) == len(target_ids)):
            raise ValueError("native Adapter sample paths, texts, and IDs must align")
        g1_queries, g1_gallery = adapter_selected_phase(
            adapter, query_paths=[pair[0] for pair in query_pairs], query_texts=query_texts,
            gallery_paths=[pair[1] for pair in query_pairs],
            categories=[category] * len(query_pairs), capture_root=capture_root)
        np.savez_compressed(capture_root / "g1_features.npz", queries=g1_queries, gallery=g1_gallery,
                            query_ids=np.asarray(reference_ids), target_ids=np.asarray(target_ids),
                            query_texts=np.asarray(query_texts),
                            sample_indices=np.asarray(inputs["sample_indices"], dtype=np.int32))
    ranking = score_and_rank(category, inputs, config, artifact_root=artifact_root, adapter=adapter,
                             native_ranking_path=native_ranking_path)
    return {"population": {"query_count": int(queries.shape[0]), "gallery_count": int(gallery.shape[0]),
                           "finite": bool(np.isfinite(queries).all() and np.isfinite(gallery).all())},
            "worker_observation": worker_observation, "scoring_ranking": ranking}

 # --------------------------------------------------------------------------- adapter phase
 # --------------------------------------------------------------------------- adapter phase

def adapter_phase(category: str, config, *, artifact_root: Path = RAW,
                  batch_size: int = 32) -> dict:
    from workbench.composition.model_limn import Adapter

    inputs = json.loads((artifact_root / f"g2_{category}_inputs.json").read_text())
    native = np.load(artifact_root / f"g2_{category}_native.npz")
    began = time.time()
    adapter = Adapter(config)
    adapter.batch_size = batch_size
    adapter.work_root = artifact_root / "parity-work" / category
    adapter.work_root.mkdir(parents=True, exist_ok=True)
    result: dict = {}
    try:
        # Full population through the production worker path.
        population_queries = adapter.encode_queries(
            inputs["full_query_paths"], inputs["full_query_texts"],
            [category] * len(inputs["full_query_texts"]))
        population_gallery = adapter.encode_targets(
            inputs["gallery_paths"], [category] * len(inputs["gallery_paths"]))
        result["worker_observation"] = adapter._worker_observation
        np.savez(artifact_root / f"g2_{category}_population.npz",
                 queries=population_queries, gallery=population_gallery)
        result["population"] = {
            "query_count": int(population_queries.shape[0]),
            "gallery_count": int(population_gallery.shape[0]),
            "finite": bool(np.isfinite(population_queries).all()
                           and np.isfinite(population_gallery).all()),
        }
    finally:
        adapter.close()

    result["scoring_ranking"] = score_and_rank(category, inputs, config,
                                                artifact_root=artifact_root,
                                                batch_size=batch_size)
    result["phase_seconds"] = round(time.time() - began, 1)
    return result


def _native_population_ranks(category: str, artifact_root: Path,
                            native_ranking_path: Path | None = None) -> list[int] | None:
    path = native_ranking_path or artifact_root / f"g2_{category}_native_ranking.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as ranking:
        return [int(rank) for rank in ranking["source_target_ranks"]]




def score_and_rank(category: str, inputs: dict, config, *, artifact_root: Path = RAW,
                   batch_size: int = 32, adapter=None,
                   native_ranking_path: Path | None = None) -> dict:
    """G2 scoring/ranking: native convention vs the adapter's own score/rank, same features."""
    from workbench.composition.model_limn import Adapter

    blob = np.load(artifact_root / f"g2_{category}_population.npz")
    # float32 END TO END, because pinned `test.py` normalizes the float32 feature arrays in place and
    # then dots them: the native ranking is computed on float32 similarities. Comparing a float64
    # recompute against the adapter's float32 path would confound the tie convention with a dtype
    # difference, which is exactly the kind of rank drift this gate must not accept silently.
    queries = blob["queries"].astype(np.float32)
    gallery = blob["gallery"].astype(np.float32)
    queries_n = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    gallery_n = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)
    unmasked = queries_n.dot(gallery_n.T)                     # what score() must reproduce
    gallery_ids = inputs["gallery_ids"]
    position = {name: index for index, name in enumerate(gallery_ids)}
    reference_positions = [position[name] for name in inputs["full_reference_ids"]]
    target_positions = [position[name] for name in inputs["full_target_ids"]]
    owned = adapter is None
    if owned:
        adapter = Adapter(config)
        adapter.batch_size = batch_size
        adapter.work_root = artifact_root / "parity-work" / f"{category}-rank"
        adapter.work_root.mkdir(parents=True, exist_ok=True)
    if native_ranking_path is not None:
        try:
            native_ranks = _native_population_ranks(category, artifact_root, native_ranking_path)
            if native_ranks is None:
                raise FileNotFoundError(native_ranking_path)
            with np.load(native_ranking_path, allow_pickle=False) as source:
                source_scores = source["source_score_rows"]
                source_orders = source["source_orders"]
                source_refs = source["reference_positions"].tolist()
                source_targets = source["target_positions"].tolist()
                if (source_scores.shape != unmasked.shape or source_orders.shape != unmasked.shape
                        or len(native_ranks) != len(queries)
                        or source["query_ids"].tolist() != inputs["full_reference_ids"]
                        or source["target_ids"].tolist() != inputs["full_target_ids"]
                        or source["gallery_ids"].tolist() != gallery_ids
                        or source_refs != reference_positions or source_targets != target_positions):
                    raise ValueError("native source ranking evidence does not align with Adapter inputs")
            scores_from_adapter = adapter.score(
                queries, {category: gallery}, [category] * len(queries)).astype(np.float32)
            if scores_from_adapter.shape != source_scores.shape:
                raise ValueError("Adapter score matrix shape differs from captured native source scores")
            masked_adapter = scores_from_adapter.copy()
            if category != "birds":
                masked_adapter[np.arange(len(reference_positions)), reference_positions] = -10e10
            adapter_source_orders = np.argsort(-masked_adapter, axis=1)
            adapter_source_ranks = [int(np.flatnonzero(row == target)[0]) + 1
                                    for row, target in zip(adapter_source_orders, target_positions, strict=True)]
            rank_flips = [i for i, (a, b) in enumerate(zip(adapter_source_ranks, native_ranks, strict=True))
                          if a != b]
            order_flips = [i for i, (a, b) in enumerate(zip(adapter_source_orders, source_orders, strict=True))
                           if not np.array_equal(a, b)]
            score_delta = np.abs(masked_adapter.astype(np.float64) - source_scores.astype(np.float64))
            boundary_source = [
                float(abs(source_scores[row, target_positions[row]] - source_scores[row, source_orders[row,
                    native_ranks[row] - 2 if native_ranks[row] > 1 else 1]]))
                if len(gallery_ids) > 1 else None for row in range(len(native_ranks))]
            boundary_adapter = [
                float(abs(masked_adapter[row, target_positions[row]] - masked_adapter[row, adapter_source_orders[row,
                    adapter_source_ranks[row] - 2 if adapter_source_ranks[row] > 1 else 1]]))
                if len(gallery_ids) > 1 else None for row in range(len(adapter_source_ranks))]
            topk_widths = {f"k{k}": min(k, len(gallery_ids)) for k in (1, 10, 50)}
            source_topk_orders = {key: source_orders[:, :width].astype(np.int32)
                                  for key, width in topk_widths.items()}
            adapter_topk_orders = {key: adapter_source_orders[:, :width].astype(np.int32)
                                   for key, width in topk_widths.items()}
            topk_order_flips = {key: [int(row) for row in np.flatnonzero(
                np.any(source_topk_orders[key] != adapter_topk_orders[key], axis=1))]
                                for key in source_topk_orders}
            per_query_score_max_delta = np.max(score_delta, axis=1).tolist()
            if not np.isfinite(source_scores).all() or not np.isfinite(scores_from_adapter).all():
                raise ValueError("source or Adapter masked score rows contain non-finite values")
            target_rank_crossings = {}
            for row in rank_flips:
                target_position = target_positions[row]
                source_ahead = set(source_orders[row, :native_ranks[row] - 1].tolist())
                adapter_ahead = set(adapter_source_orders[row, :adapter_source_ranks[row] - 1].tolist())
                crossed = sorted(source_ahead.symmetric_difference(adapter_ahead))
                unexplained = any(
                    not (abs(float(source_scores[row, position] - source_scores[row, target_position]))
                         <= TOL["tie_epsilon"]
                         and abs(float(masked_adapter[row, position] - masked_adapter[row, target_position]))
                         <= TOL["tie_epsilon"])
                    for position in crossed)
                target_rank_crossings[row] = {"candidate_positions": crossed,
                                              "tie_explained": not unexplained}
            unexplained_rank_differences = [row for row, evidence in target_rank_crossings.items()
                                            if not evidence["tie_explained"]]
            topk_membership_flips = {key: [int(row) for row in np.flatnonzero(
                np.any(np.sort(source_topk_orders[key], axis=1) !=
                       np.sort(adapter_topk_orders[key], axis=1), axis=1))]
                                     for key in source_topk_orders}
            target_topk_hit_flips = {key: [int(row) for row in np.flatnonzero(
                np.asarray([rank <= int(key[1:]) for rank in native_ranks]) !=
                np.asarray([rank <= int(key[1:]) for rank in adapter_source_ranks]))]
                                     for key in source_topk_orders}
            adapter_evidence_path = artifact_root / f"g2_{category}_adapter_ranking.npz"
            np.savez_compressed(
                adapter_evidence_path,
                adapter_masked_score_rows=masked_adapter,
                adapter_source_orders=adapter_source_orders.astype(np.int32),
                adapter_source_target_ranks=np.asarray(adapter_source_ranks, dtype=np.int32),
                **{f"adapter_source_topk_k{k}": adapter_source_orders[:, :topk_widths[f"k{k}"]].astype(np.int32)
                   for k in (1, 10, 50)},
                query_ids=np.asarray(inputs["full_reference_ids"]),
                target_ids=np.asarray(inputs["full_target_ids"]),
                gallery_ids=np.asarray(gallery_ids),
                reference_exclusion_sentinel=np.asarray(-10e10, dtype=np.float64))
            rankings = {
                "ranking_source": "captured pinned native test.py np.argsort(-sims[row])",
                "native_source_target_ranks": native_ranks,
                "adapter_source_target_ranks": adapter_source_ranks,
                "native_source_topk": {f"k{k}": [int(rank <= k) for rank in native_ranks]
                                       for k in (1, 10, 50)},
                "adapter_source_topk": {f"k{k}": [int(rank <= k) for rank in adapter_source_ranks]
                                        for k in (1, 10, 50)},
                "native_source_hit_counts": {f"k{k}": sum(rank <= k for rank in native_ranks)
                                             for k in (1, 10, 50)},
                "adapter_source_hit_counts": {f"k{k}": sum(rank <= k for rank in adapter_source_ranks)
                                              for k in (1, 10, 50)},
                "target_rank_flips": rank_flips,
                "target_rank_flip_count": len(rank_flips),
                "topk_order_flip_query_indices": topk_order_flips,
                "order_flip_query_indices": order_flips,
                "order_flip_count": len(order_flips),
                "score_max_abs_delta": per_query_score_max_delta,
                "score_max_abs_delta_overall": float(score_delta.max(initial=0.0)),
                "score_max_abs_delta_vs_dot": float(score_delta.max(initial=0.0)),
                "score_delta_comparison": "masked upstream native dot scores vs identically masked Adapter.score",
                "score_pairs_compared": int(score_delta.size),
                "unexplained_rank_differences": len(unexplained_rank_differences),
                "rank_difference_indices_head": unexplained_rank_differences[:20],
                "target_rank_crossings": target_rank_crossings,
                "topk_membership_flip_query_indices": topk_membership_flips,
                "target_topk_hit_flip_query_indices": target_topk_hit_flips,
                "source_target_boundary_gap": boundary_source,
                "adapter_target_boundary_gap": boundary_adapter,
                "source_topk_orders": {key: value.tolist() for key, value in source_topk_orders.items()},
                "adapter_topk_orders": {key: value.tolist() for key, value in adapter_topk_orders.items()},
                "adapter_ranking_artifact": str(adapter_evidence_path),
                "reference_exclusion_sentinel": float(-10e10),
                "source_argsort": "numpy default argsort on negative masked scores; exact order retained",
            }
            return rankings
        finally:
            if owned:
                adapter.close()
    sims = unmasked.copy()
    if category != "birds":
        sims[np.arange(len(reference_positions)), reference_positions] = -10e10
    native_style = [int(np.flatnonzero(np.argsort(-row) == target)[0]) + 1
                    for row, target in zip(sims, target_positions, strict=True)]
    argsort_unmasked = [int(np.flatnonzero(np.argsort(-unmasked[row]) == target)[0]) + 1
                        for row, target in enumerate(target_positions)]
    positions = [[position] for position in target_positions]
    try:
        adapter_style = adapter.rank(queries, {category: gallery}, positions,
                                     [category] * len(positions))
        scores = adapter.score(queries, {category: gallery}, [category] * len(queries))
    finally:
        if owned:
            adapter.close()
    score_delta = np.abs(scores.astype(np.float64) - unmasked.astype(np.float64))

    native_ranks = _native_population_ranks(category, artifact_root)
    unmasked_differences = [i for i, (a, b) in enumerate(zip(adapter_style, argsort_unmasked))
                            if a != b]
    mask_explained = [i for i, (a, b) in enumerate(zip(adapter_style, native_style)) if a != b]
    rankings = {"query_count": len(native_style),
                "gallery_count": len(inputs["gallery_ids"]),
                # tie-convention comparison: adapter vs argsort, BOTH unmasked
                "adapter_vs_argsort_unmasked_identical": not unmasked_differences,
                "adapter_vs_argsort_unmasked_differences": unmasked_differences[:30],
                "adapter_vs_argsort_unmasked_difference_count": len(unmasked_differences),
                # the mask difference is a protocol difference, quantified rather than hidden
                "mask_explained_rank_shift_count": len(mask_explained),
                "mask_explained_rank_shift_all_positive": all(
                    adapter_style[i] - native_style[i] >= 0 for i in mask_explained),
                "mask_shift_is_exactly_one": all(
                    adapter_style[i] - native_style[i] == 1 for i in mask_explained),
                "adapter_style_recall": {f"r{k}": 100 * sum(1 for r in adapter_style if r <= k) / len(adapter_style)
                                        for k in (1, 10, 50)},
                "native_style_recall": {f"r{k}": 100 * sum(1 for r in native_style if r <= k) / len(native_style)
                                        for k in (1, 10, 50)},
                "adapter_vs_native_masked_identical": adapter_style == native_style,
                "score_max_abs_delta_vs_dot": float(score_delta.max()),
                "score_pairs_compared": int(score_delta.size),
                "native_source_target_ranks": native_ranks,
                "native_recomputed_target_ranks": native_style,
                "adapter_production_target_ranks": adapter_style,
                "reference_excluded": True,
                "ranking_policy": "production Adapter.rank tie-stable; native source np.argsort default"}
    if native_ranks is not None:
        differing = [i for i, (a, b) in enumerate(zip(native_style, native_ranks)) if a != b]
        unexplained = []
        for index in differing:
            row = sims[index]
            low, high = sorted((native_style[index], native_ranks[index]))
            between = np.argsort(-row)[low - 1:high]
            if not np.all(np.abs(row[between] - row[target_positions[index]]) <= TOL["tie_epsilon"]):
                unexplained.append(index)
        rankings.update({
            "native_aggregate_ranks_compared": len(native_ranks),
            "rank_differences": len(differing),
            "rank_difference_indices_head": differing[:20],
            "unexplained_rank_differences": len(unexplained),
            "first_unexplained_index": unexplained[0] if unexplained else None,
            "max_abs_rank_delta": int(max([abs(a - b) for a, b in
                                          zip(native_style, native_ranks)] or [0])),
        })
    else:
        rankings["native_aggregate_ranks_compared"] = None
    return rankings


def refresh_metrics_block(categories: tuple[str, ...]) -> None:
    """Re-derive the `metrics_parity` block on the artifacts this script just rewrote.

    `main()` writes each native artifact from scratch, so the hit-count block added by
    `20261112-parity-repair/scripts/limn_hits.py` would be lost. Deriving it needs only stored data,
    so it is re-run here rather than left to a manual step that will be forgotten.
    """
    import importlib.util

    path = Path("/home/heheboiz/data/cir/workbench/docs/reviews/20261112-parity-repair"
                "/scripts/limn_hits.py")
    if not path.is_file():
        print(f"# metrics block NOT refreshed: {path} is missing; "
              "`metrics_parity` will report missing", flush=True)
        return
    spec = importlib.util.spec_from_file_location("limn_hits", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for line in module.refresh(categories):
        print(f"# metrics: {line}", flush=True)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    phase = argv[argv.index("--phase") + 1] if "--phase" in argv else "native"
    categories = CATEGORIES
    if "--category" in argv:
        categories = (argv[argv.index("--category") + 1],)
    config = resolve_config()
    RAW.mkdir(parents=True, exist_ok=True)

    for category in categories:
        target = OUT / f"g1_g2.{category}.{phase}.json"
        print(f"# {category}/{phase}: start", flush=True)
        if phase == "native":
            payload = {"category": category, "phase": phase, "tolerances": TOL,
                       "standard": STANDARD_V2,
                       "layer1_identity": native_phase(category, config)}
        elif phase == "adapter":
            body = adapter_phase(category, config)
            components = body.pop("component_parity", None)
            payload = {"category": category, "phase": phase, "tolerances": TOL,
                       "standard": STANDARD_V2,
                       "layer2_4_forward": body}
            # Beside the body, at the top level: `parity_gate._component_checks` reads the WHOLE
            # document, exactly as it reads `relation_state` there.
            if components is not None:
                payload["component_parity"] = components
        else:
            raise SystemExit(f"unknown phase {phase!r}; use native or adapter")
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"# {category}/{phase}: wrote {target.name}", flush=True)
        if phase == "native":
            print(json.dumps({
                "gallery_order_identical": payload["layer1_identity"]["gallery_order_identical"],
                "reference_ids_identical": payload["layer1_identity"]["reference_ids_identical"],
                "target_ids_identical": payload["layer1_identity"]["target_ids_identical"],
                "model_text_identical": payload["layer1_identity"]["model_text_identical_to_author_composition"],
                "sample_indices": payload["layer1_identity"]["sample_indices"],
                "rgb_noop": payload["layer1_identity"]["sample_convert_to_rgb_is_noop"],
            }, indent=1))
        else:
            print(json.dumps({
                "sample_query": payload["layer2_4_forward"]["sample_forward"]["query"],
                "sample_target": payload["layer2_4_forward"]["sample_forward"]["target"],
                "batch": payload["layer2_4_forward"]["batch_size_1_vs_n"],
                "restart": payload["layer2_4_forward"]["cache_off_restart"],
                "rankings": payload["layer2_4_forward"]["scoring_ranking"],
            }, indent=1))
    # Both phases rewrite at least one side of the pair, so the block is re-derived after the loop.
    refresh_metrics_block(categories)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
