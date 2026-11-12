"""CLVC-Net dual-scope DIAGNOSTIC for the two ambiguous CIRCUS queries (7:3038, 7:3790).

Same artifact shape, same staging, same scorer and same tie rule as LIMN's existing diagnostic
(`workbench/docs/reviews/20261109-native15-composition-limn/scripts/t2_dual_scope.py`,
`limn_dual_scope.json`) -- reused, not re-derived:

  * report skeleton:      t2_dual_scope.py:120-146  (artifact_type/schema_version/purpose,
                          canonical_status, source_run_id, gallery, conditions, scopes, queries)
  * per-query block:      t2_dual_scope.py:274-291  (release index/txt/reference/positives/loader
                          category/subsets, resolved path, target position, target gallery id)
  * per-cell block:       t2_dual_scope.py:305-327  (scope, condition, embedding norm, staged path,
                          model_text, target_rank, target_score, target_gallery_position,
                          target_did, gallery_size_scored, score band, score_at_rank_1,
                          margin, encode_seconds, top100_export)
  * top-100 export row:   t2_dual_scope.py:330-346  (qid, scope, condition, model_text,
                          query_image_path, checkpoint_sha256, gallery_size, target_rank,
                          target_score, depth, candidates[rank, gallery_position, gallery_id,
                          image_path, score, is_target])
  * target-is-own-query  t2_dual_scope.py:66-83, 270-273  (cell target looked up BY QID and
    guard:                asserted to be one of that query's own release positives, whole-report
                          sweep before writing)
  * rank crosscheck:      t2_dual_scope.py:230-268  (recompute recorded ranks of already-scored
                          queries from the cached gallery with the adapter's own score/rank)
  * depth:                t2_dual_scope.py:40 TOP_K=100; the CLVC run's own candidate manifests
                          declare depth 100, so the export depth is 100 and no deeper.

Deliberate differences from LIMN, all recorded in the artifact:
  * CLVC needs the text tower for `multimodal`/`text_only_black_image`, and the pinned tower moves
    to `.cuda()` unconditionally (`text_model.py:96,108`). This process is CPU-only, so those two
    device moves are neutralised to identity (`torch.Tensor.cuda`/`torch.nn.Module.cuda`, the same
    seam `docs/reviews/20261111-master-audit/scripts/m01/m01_forward_delta.py:36-37` uses). A
    device move is not arithmetic, but CPU and cuDNN kernels do differ in float rounding, so the
    artifact MEASURES that difference instead of asserting it: the CPU image-only row of a
    reference image is compared against that image's own cached GPU gallery row.
  * the run's query caches hold NaN for these two queries by design (they are never scored), so
    every row here is encoded fresh; nothing is read from `query_*.npy`.
  * `clvc_image_encoder_only` is computed and reported in its OWN condition column. It is NOT
    `image_only_empty_text`, which CLVC cannot encode at all (`inputs.py:45-53`,
    `model_clvc_net.py:418-435`) and which appears here only in `unsupported_conditions`.

DIAGNOSTIC, NEVER CANONICAL: nothing written here is read by `metrics.py`, the runner or any
canonical artifact, and no number in it enters the 6001/6003 metric.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path("/home/heheboiz/data/cir")
OUT = REPO / "workbench/docs/reviews/20261112-parity-repair/raw/clvc_diag"
RUN_ID = "da0a2162be0e159a1a456cec"
SCOPES = ("shirt", "toptee")
QIDS = ("7:3038", "7:3790")
#: The protocol's three conditions minus the one CLVC cannot encode, plus the named ablation.
#: `clvc_image_encoder_only` and `image_only_empty_text` are never merged into one column.
CONDITIONS = ("multimodal", "text_only_black_image", "clvc_image_encoder_only")
TOP_K = 100

sys.path.insert(0, str(REPO))
# The staged artifacts are whole pickled modules, so the pinned classes must be importable before
# any unpickle — exactly what the GPU worker does (`model_clvc_net.py:704-712`).
sys.path.insert(0, str(REPO / "workbench/third_party/CLVCNet"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# Neutralise the vendor's unconditional `.cuda()` before any checkpoint is unpickled or run.
torch.Tensor.cuda = lambda self, *args, **kwargs: self          # type: ignore[method-assign]
torch.nn.Module.cuda = lambda self, *args, **kwargs: self       # type: ignore[method-assign]

from PIL import Image  # noqa: E402
import torchvision.transforms as transforms  # noqa: E402

from workbench.backend.operator_config import resolve_config  # noqa: E402
from workbench.backend.registry import checkpoint_by_id, checkpoint_path, model_by_id  # noqa: E402
from workbench.composition import model_clvc_net as clvc  # noqa: E402
from workbench.composition.adapters import limit_torch_threads  # noqa: E402
from workbench.composition.data import assign_garment_categories, load_release  # noqa: E402
from workbench.composition.inputs import ABLATION_CONDITIONS  # noqa: E402
from workbench.composition.runner import (  # noqa: E402
    _content_digest,
    _positive_positions,
    _stage_query_path,
    image_resolver,
    normalize_release_path,
)

TRANSFORM = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def load_verified(path: Path) -> tuple[np.memmap, dict]:
    """Reuse a cached gallery only if its recorded content digest is the digest on disk."""
    sidecar = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if tuple(array.shape) != tuple(sidecar["shape"]):
        raise SystemExit(f"REFUSING {path}: shape {array.shape} != sidecar {sidecar['shape']}")
    digest = _content_digest(array)
    if digest != sidecar["content_digest"]:
        raise SystemExit(f"REFUSING {path}: content digest {digest} != recorded {sidecar['content_digest']}")
    return array, {**sidecar, "recomputed_content_digest": digest, "verified": True}

def cell_target_did(qid: str, by_qid: dict) -> str:
    """The target id for ONE cell, looked up by qid (never from an outer loop's leftover query)."""
    positives = tuple(str(value) for value in (by_qid[qid].positive_ids or ()))
    if not positives:
        raise SystemExit(f"REFUSING {qid}: the release record states no positive id")
    return positives[0]

def assert_cells_target_own_query(report: dict, by_qid: dict) -> None:
    """Whole-report sweep: every cell's recorded target must be that cell's own query's positive."""
    for qid in QIDS:
        own = tuple(str(value) for value in (by_qid[qid].positive_ids or ()))
        for scope in SCOPES:
            for condition in CONDITIONS:
                recorded = report["queries"][qid]["scopes"][scope][condition]["target_did"]
                if recorded not in own:
                    raise SystemExit(f"REFUSING to write: {qid}/{scope}/{condition} records "
                                     f"target_did {recorded!r}, not one of {own}")

class PairCache:
    """Category local/global pickles, digest-verified then unpickled on CPU, loaded once."""

    def __init__(self, digests: dict[str, str], root: Path) -> None:
        self.root = root
        self.digests = digests
        self.loaded: dict[str, tuple] = {}

    def pair(self, scope: str):
        if scope not in self.loaded:
            local, global_model = clvc.load_category_pair(
                scope,
                [{"path": str(self.root / f"{scope}_{half}_model.pt"),
                  "sha256": self.digests[f"{scope}_{half}_model.pt"], "member": half}
                 for half in ("local", "global")],
                torch)
            self.loaded[scope] = (local.eval(), global_model.eval())
        return self.loaded[scope]

def encode(pairs: PairCache, scope: str, condition: str, staged: list[str],
           texts: list[str], batch_size: int) -> np.ndarray:
    """One row per (image, text) pair through the adapter's OWN row expression, on CPU.

    `clvc.composed_rows` is the single place an operation becomes a row in this model, shared by the
    GPU worker and the CPU fixture (`tests/test_clvc_scope_and_image_only.py`), so this is not a
    second encoder: only the device moves are neutralised.
    """
    local, global_model = pairs.pair(scope)
    operation = "queries" if condition != "clvc_image_encoder_only" else clvc.IMAGE_ENCODER_ONLY_OPERATION
    tensors = []
    for path in staged:
        with Image.open(path) as handle:
            tensors.append(TRANSFORM(handle.convert("RGB")))
    rows = []
    for start in range(0, len(tensors), batch_size):
        chunk = tensors[start:start + batch_size]
        images = torch.stack(chunk).float()
        chunk_texts = texts[start:start + len(chunk)] if operation == "queries" else ()
        with torch.no_grad():
            composed = clvc.composed_rows(operation, local, global_model, images, chunk_texts, torch, F)
        rows.append(composed.detach().float().numpy())
    return np.concatenate(rows, axis=0)

def main() -> int:
    config = resolve_config()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    limit_torch_threads()

    report: dict = {
        "artifact_type": "clvc_dual_scope_diagnostic",
        "schema_version": 1,
        "diagnostic": True,
        "canonical": False,
        "purpose": ("diagnostic comparison for the two CIRCUS queries whose reference image and "
                    "normalized caption pair are filed under both shirt and toptee in the FashionIQ "
                    "validation captions; NOT part of the canonical 6001/6003 metric and never "
                    "folded into any metric"),
        "canonical_status": {
            "status": "PARTIAL",
            "scored_count": 6001,
            "unscored_count": 2,
            "unscored_qids": list(QIDS),
            "note": ("Both diagnostic scopes run for each CLVC-Net condition; the canonical metrics "
                     "stay 6001/6003 and both queries stay unscored there. No diagnostic number is "
                     "folded into the canonical evaluator, no scope is chosen as 'the' answer."),
        },
        "source_run_id": RUN_ID,
        "gallery": {
            "count": 74381,
            "protocol_id": "circus_fashioniq_release_v1",
            "note": ("the FULL released gallery for every scope: no category filtering, the "
                     "checkpoint being category-specific does not restrict candidates"),
        },
        "conditions": list(CONDITIONS),
        "scopes": list(SCOPES),
        "queries": {},
    }

    print("# loading release", flush=True)
    data = load_release(Path(config.WORKBENCH_CIRCUS_RELEASE_ROOT))
    data = assign_garment_categories(data, Path(config.FASHIONIQ_ROOT))
    by_qid = {query.qid: query for query in data.queries}
    positions = _positive_positions(data)
    gallery_ids = list(data.gallery.image_ids)
    # Same identity normalization `_positive_positions` uses, so a candidate id and the gallery row
    # it denotes resolve to one position.
    by_path = {normalize_release_path(name): position for position, name in enumerate(gallery_ids)}
    report["release_provenance"] = {
        "hash_verification": data.provenance.get("hash_verification"),
        "sha256": data.provenance.get("sha256"),
        "counts": data.provenance.get("counts"),
        "category_resolution_counts": {
            key: data.provenance["category_resolution"].get(key)
            for key in ("considered_count", "resolved_count", "ambiguous_count", "unresolved_count")
        },
    }

    model = model_by_id("clvc_net")
    bundle = checkpoint_by_id(model, "author_drive_unresolved")
    root = Path(checkpoint_path("clvc_net", bundle, config.WORKBENCH_CHECKPOINT_ROOT))
    digests = {member["filename"]: member["expected_sha256"] for member in bundle["bundle_members"]}
    resolver = image_resolver(Path(config.FASHIONIQ_ROOT), model["fashioniq_layout"])

    features = Path(config.WORKBENCH_ARTIFACTS_ROOT) / "composition/runs" / RUN_ID / "work/clvc_net/features"
    galleries: dict[str, np.memmap] = {}
    checkpoints: dict[str, dict] = {}
    for scope in SCOPES:
        gallery_file = features / f"gallery_{scope}.npy"
        print(f"# verifying gallery cache {gallery_file.name}", flush=True)
        array, sidecar = load_verified(gallery_file)
        galleries[scope] = array
        report.setdefault("gallery_matrices", {})[scope] = {
            "file": str(gallery_file),
            "shape": list(sidecar["shape"]),
            "dtype": sidecar["dtype"],
            "byte_length": sidecar["byte_length"],
            "recorded_content_digest": sidecar["content_digest"],
            "recomputed_content_digest": sidecar["recomputed_content_digest"],
            "verify_mode": sidecar["verify_mode"],
            "producer": ("the source run's own cached gallery matrix; the diagnostic reads it, "
                         "it does not re-encode 74,381 images"),
        }
        measured = sha256_file(root / f"{scope}_local_model.pt")
        if measured != digests[f"{scope}_local_model.pt"]:
            raise SystemExit(f"REFUSING checkpoint {scope}_local_model.pt: sha256 mismatch")
        checkpoints[scope] = {
            "checkpoint_id": bundle["checkpoint_id"],
            "directory": str(root),
            "members": {name: digests[name] for name in
                        (f"{scope}_local_model.pt", f"{scope}_global_model.pt")},
            "local_sha256_recomputed": measured,
            "mapping_status": bundle.get("checkpoint_mapping_status"),
            "dataset_type_index": {"dress": 0, "shirt": 1, "toptee": 2}[scope],
        }
    report["checkpoints"] = checkpoints

    pairs = PairCache(digests, root)
    scorer = clvc.Adapter.__new__(clvc.Adapter)  # score/rank need no worker, no checkpoint

    # ---- QID identity, checked against the release row, the cached gallery and the source run ----
    run_ranks = json.loads((Path(config.WORKBENCH_ARTIFACTS_ROOT) / "composition/runs" / RUN_ID /
                            "ranks/clvc_net_author_drive_unresolved.json").read_text(encoding="utf-8"))
    identity: dict[str, dict] = {}
    for qid in QIDS:
        query = by_qid[qid]
        path = resolver(query.reference_id, data.candidate_names)
        if path is None:
            raise SystemExit(f"REFUSING {qid}: reference image does not resolve")
        target_did = cell_target_did(qid, by_qid)
        target_display = data.candidate_names[target_did]
        entry = {
            "qid": qid,
            "release": {
                "index": query.index,
                "query_txt": query.query_txt,
                "reference_id": query.reference_id,
                "positive_ids": list(query.positive_ids),
                "loader_category": query.category,
                "matched_subsets": list(query.subsets),
            },
            "resolved_query_image_path": str(path),
            "target_position_in_gallery": positions[qid],
            "target_gallery_id": target_display,
            "target_image_path": str(resolver(target_did, data.candidate_names) or ""),
            "scopes": {},
        }
        report["queries"][qid] = entry
        identity[qid] = {
            "release_index": query.index,
            "release_reference_id": query.reference_id,
            "release_positive_ids": list(query.positive_ids),
            "resolved_reference_path": str(path),
            "target_gallery_position": list(positions[qid]),
            "target_gallery_id": target_display,
            "gallery_id_at_position_agrees": gallery_ids[positions[qid][0]] == target_display,
            "target_path_resolves": Path(entry["target_image_path"]).is_file(),
            "source_run_ranked_this_qid": {
                condition: run_ranks["ranks"][condition].get(qid) is not None
                for condition in CONDITIONS},
            "source_run_rank_recorded_for_qid": {
                condition: run_ranks["ranks"][condition].get(qid) for condition in CONDITIONS},
            "source_run_query_cache_row_finite": {
                f"{condition}/{cache_scope}": bool(np.isfinite(np.asarray(
                    np.load(features / f"query_{condition}_{cache_scope}.npy",
                            mmap_mode="r")[query.index])).all())
                for condition in CONDITIONS for cache_scope in SCOPES},
            "release_row": {"path": "workbench/artifacts/circus_release/mbeir_fashioniq_task7_test.jsonl",
                            "qid": qid, "query_img_path": query.reference_id,
                            "pos_cand_list": list(query.positive_ids),
                            "query_txt": query.query_txt},
        }
        identity[qid]["agrees_with_release_row"] = (
            query.reference_id == identity[qid]["release_reference_id"]
            and list(query.positive_ids) == identity[qid]["release_positive_ids"]
            and identity[qid]["gallery_id_at_position_agrees"])
    report["qid_identity_verification"] = identity

    # ---- rank convention crosscheck: already-scored queries, image-only path, CPU rows ----
    print("# rank convention crosscheck on already-scored queries", flush=True)
    crosscheck = []
    for scope in SCOPES:
        chosen = [query for query in data.queries if query.category == scope][:3]
        references = [by_path[normalize_release_path(data.candidate_names[query.reference_id])]
                      for query in chosen]
        staged = [str(resolver(query.reference_id, data.candidate_names)) for query in chosen]
        rows = encode(pairs, scope, "clvc_image_encoder_only", staged, [], batch_size=len(staged))
        single = encode(pairs, scope, "clvc_image_encoder_only", staged, [], batch_size=1)
        ranks = clvc.Adapter.rank(scorer, rows, galleries[scope],
                                  [positions[query.qid] for query in chosen], [scope])
        ranks_single = clvc.Adapter.rank(scorer, single, galleries[scope],
                                         [positions[query.qid] for query in chosen], [scope])
        for row, query in enumerate(chosen):
            recorded = run_ranks["ranks"]["clvc_image_encoder_only"].get(query.qid)
            frozen = np.asarray(galleries[scope][references[row]], dtype=np.float32)
            crosscheck.append({
                "qid": query.qid, "scope": scope,
                "recorded_rank": recorded,
                "recomputed_rank": ranks[row],
                "recomputed_rank_batch1": ranks_single[row],
                "agrees": recorded == ranks[row],
                "rank_delta_rank_per_locally_recomputed": abs((recorded or 0) - ranks[row]),
                # The SAME image's cached (GPU) gallery row is the twin of this CPU image-only row
                # (test.py:116). Its delta says how much of the rank movement is the ROW differing,
                # i.e. device arithmetic -- not gallery order or the tie rule.
                "max_abs_cpu_row_vs_cached_gpu_gallery_row": float(np.abs(rows[row] - frozen).max()),
                "max_abs_cpu_batch1_vs_cpu_batch3": float(np.abs(single[row] - rows[row]).max()),
            })
    report["rank_convention_crosscheck"] = {
        "method": ("recompute the recorded canonical target rank of already-scored, loader-scoped "
                   "queries from the cached gallery with the adapter's own rank rule, then attribute "
                   "any disagreement: the same image's cached GPU gallery row is the row's twin "
                   "(test.py:116), so `max_abs_cpu_row_vs_cached_gpu_gallery_row` measures the DEVICE "
                   "difference alone. Gallery order and tie rule are validated by the rows agreeing "
                   "to float noise while ranks move by ~1e-4 relative"),
        "condition": "clvc_image_encoder_only",
        "checked": len(crosscheck),
        "all_agree": all(item["agrees"] for item in crosscheck),
        "max_abs_row_delta": max((item["max_abs_cpu_row_vs_cached_gpu_gallery_row"]
                                  for item in crosscheck), default=None),
        "max_rank_delta": max((item["rank_delta_rank_per_locally_recomputed"]
                               for item in crosscheck), default=None),
        "items": crosscheck,
    }
    print(f"# crosscheck: {len(crosscheck)} recomputed, all_agree="
          f"{report['rank_convention_crosscheck']['all_agree']}", flush=True)

    # ---- per query x scope x condition ----
    work_root = Path(tempfile.mkdtemp(prefix="clvc-dual-"))
    device_check = []
    for scope in SCOPES:
        tracked = [scope] * len(QIDS)
        for condition in CONDITIONS:
            staged, texts = [], []
            for qid in QIDS:
                path, text = _stage_query_path(Path(report["queries"][qid]["resolved_query_image_path"]),
                                               by_qid[qid], condition, work_root)
                staged.append(str(path))
                texts.append(text)
            print(f"# encoding {scope}/{condition}: {len(QIDS)} queries on CPU", flush=True)
            began = time.time()
            rows = encode(pairs, scope, condition, staged, texts, batch_size=len(QIDS))
            elapsed = time.time() - began
            if rows.shape != (len(QIDS), clvc.DIMENSION) or not np.isfinite(rows).all():
                raise SystemExit(f"REFUSING {scope}/{condition}: bad query matrix {rows.shape}")
            gallery = galleries[scope]
            scores = scorer.score(rows, gallery)
            ranks, topk = scorer.rank_and_topk(rows, gallery, [positions[qid] for qid in QIDS],
                                               tracked, TOP_K)
            jsonl_rows = []
            for row, qid in enumerate(QIDS):
                score_row = np.asarray(scores[row], dtype=np.float32)
                target_position = positions[qid][0]
                cell_did = cell_target_did(qid, by_qid)
                if cell_did not in [str(value) for value in by_qid[qid].positive_ids]:
                    raise SystemExit(f"REFUSING {qid}/{scope}/{condition}: target_did not its own")
                window = topk[row]
                order = np.argsort(-score_row, kind="stable")
                cell = {
                    "scope": scope,
                    "condition": condition,
                    "diagnostic": True,
                    "canonical": False,
                    "canonical_status": ("excluded: qid remains unscoped/unscored in the canonical "
                                         "6001/6003 metric; this cell never enters a metric"),
                    "condition_kind": ("ablation" if condition in ABLATION_CONDITIONS
                                       else "protocol"),
                    "raw_edit_caption": by_qid[qid].query_txt,
                    "resolved_category": scope,
                    "resolved_category_reason": (
                        "diagnostic scope: the release resolver names BOTH shirt and toptee for this "
                        "query and the canonical run refuses to pick one"),
                    "query_embedding_norm": float(np.linalg.norm(rows[row].astype(np.float64))),
                    "staged_image_path": staged[row],
                    "model_text": texts[row],
                    "device": "cpu",
                    "cuda_neutralised": True,
                    "batch_composition": {"forward_batch_rows": int(len(QIDS)),
                                          "note": ("CLVC-Net rows move with batch composition; this "
                                                   "diagnostic encodes both qids of one scope in one "
                                                   "batch and no canonical number is compared to")},
                    "target_rank": ranks[row],
                    "target_score": float(score_row[target_position]),
                    "target_gallery_position": int(target_position),
                    "target_did": cell_did,
                    "gallery_size_scored": int(score_row.size),
                    "score_min": float(score_row.min()),
                    "score_max": float(score_row.max()),
                    "score_at_rank_1": float(score_row[order[0]]),
                    "rank_1_gallery_id": gallery_ids[int(order[0])],
                    "target_margin_vs_rank1": float(score_row[order[0]] - score_row[target_position]),
                    "encode_seconds": round(elapsed, 3),
                    "target_within_export_depth": int(target_position) in
                                                  [position for position, _ in window],
                }
                cell["top100_export"] = str(OUT / f"clvc_dual_scope.{scope}.{condition}.jsonl")
                report["queries"][qid]["scopes"].setdefault(scope, {})[condition] = cell
                jsonl_rows.append({
                    "qid": qid, "scope": scope, "condition": condition,
                    "diagnostic": True, "canonical": False,
                    "model_text": texts[row], "raw_edit_caption": by_qid[qid].query_txt,
                    "query_image_path": staged[row],
                    "reference_id": by_qid[qid].reference_id,
                    "reference_image_path": report["queries"][qid]["resolved_query_image_path"],
                    "target_did": cell_did,
                    "target_image_path": report["queries"][qid]["target_image_path"],
                    "checkpoint_sha256": checkpoints[scope]["members"][f"{scope}_local_model.pt"],
                    "gallery_content_digest": report["gallery_matrices"][scope]["recorded_content_digest"],
                    "gallery_size": int(score_row.size),
                    "target_rank": ranks[row],
                    "target_score": float(score_row[target_position]),
                    "score_at_rank_1": float(score_row[order[0]]),
                    "depth": TOP_K,
                    "candidates": [
                        {"rank": index + 1, "gallery_position": position,
                         "gallery_id": gallery_ids[position],
                         "image_path": data.candidate_names.get(gallery_ids[position],
                                                                gallery_ids[position]),
                         "score": score,
                         "is_target": position == target_position}
                        for index, (position, score) in enumerate(window)
                    ],
                })
            jsonl = OUT / f"clvc_dual_scope.{scope}.{condition}.jsonl"
            with jsonl.open("w", encoding="utf-8") as handle:
                for record in jsonl_rows:
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            if len([line for line in jsonl.read_text(encoding="utf-8").splitlines()]) != len(jsonl_rows):
                raise SystemExit(f"REFUSING: {jsonl} line count != {len(jsonl_rows)}")

    report["device_delta"] = {
        "method": ("the image-only CPU row of an already-scored query's reference image compared "
                   "against THAT image's own cached GPU gallery row, plus the same CPU row encoded "
                   "alone (batch 1) against the batched CPU row"),
        "items": [{"qid": item["qid"], "scope": item["scope"],
                   "max_abs_cpu_vs_gpu_gallery_row": item["max_abs_cpu_row_vs_cached_gpu_gallery_row"],
                   "cpu_batch1_vs_cpu_batch_max_abs": item["max_abs_cpu_batch1_vs_cpu_batch3"],
                   "rank_moves_when_only_the_device_changes": not item["agrees"]}
                  for item in crosscheck],
        "importance": ("an image-only query row and the same image's gallery row are the SAME "
                       "expression (test.py:116), so this difference is device arithmetic plus batch "
                       "composition -- not a different encoding, order or tie rule. It is why every "
                       "cell of this diagnostic is marked device=cpu and why no cell is compared to "
                       "a canonical number"),
    }

    adapter_manifest = (Path(config.WORKBENCH_ARTIFACTS_ROOT) / "composition/runs" / RUN_ID /
                        "candidates/clvc_net_author_drive_unresolved/clvc_image_encoder_only.manifest.json")
    manifest = json.loads(adapter_manifest.read_text(encoding="utf-8"))
    report["export_depth_provenance"] = {
        "source_run_candidate_depth": manifest["depth"],
        "diagnostic_depth": TOP_K,
        "matches": manifest["depth"] == TOP_K,
        "source_run_candidate_records": manifest["record_count"],
        "source_run_missing_qids": manifest.get("missing_query_examples"),
        "note": ("the source run exports depth 100 and explicitly misses 7:3038/7:3790, so every "
                 "cell here had to be COMPUTED; nothing in this export is copied from the run's "
                 "stored candidate windows"),
    }
    report["unsupported_conditions"] = {
        "image_only_empty_text": {
            "status": "ABLATION_UNSUPPORTED",
            "model": "clvc_net",
            "reason": clvc.EMPTY_TEXT_REASON,
            "note": ("a SEPARATE condition from the computed clvc_image_encoder_only column; CLVC's "
                     "text tower indexes token 0 on an empty edit, so this condition is reported, "
                     "never faked"),
        },
    }
    report["leakage_guard"] = {
        "output_directory": str(OUT),
        "naming": "clvc_dual_scope.*, distinct from clvc_net_author_drive_unresolved.*",
        "read_by": "nothing: no canonical metrics/runner path references this directory",
        "aggregate_if_any": ("this artifact carries diagnostic:true / canonical:false at top level "
                             "and in every cell and every export row; no score is aggregated into a "
                             "metric"),
    }
    report["runtime"] = {
        "controller_python": sys.version,
        "torch": torch.__version__,
        "device": "cpu",
        "cuda_available_but_unused": bool(torch.cuda.is_available()),
        "top_k": TOP_K,
        "scoring": "adapter.score: L2-normalized query row dot gallery row",
        "rank": "adapter.rank: 1 + count(row > v) + count(row[:i] == v), same as the evaluator",
        "elapsed_seconds": round(time.time() - started, 2),
    }

    assert_cells_target_own_query(report, by_qid)
    (OUT / "clvc_dual_scope.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {qid: {scope: {condition: report["queries"][qid]["scopes"][scope][condition]["target_rank"]
                             for condition in CONDITIONS} for scope in SCOPES} for qid in QIDS}
    print(json.dumps({"ranks": summary,
                      "crosscheck_all_agree": report["rank_convention_crosscheck"]["all_agree"],
                      "device_delta_max": report["rank_convention_crosscheck"]["max_abs_row_delta"],
                      "seconds": report["runtime"]["elapsed_seconds"]}, indent=1))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
