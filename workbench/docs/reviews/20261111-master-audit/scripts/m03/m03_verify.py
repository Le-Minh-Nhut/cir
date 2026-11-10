"""M-03: independently verify the LIMN dual-scope artifact (read-only).

Recomputes the 12 diagnostic cells by JOINING ON CANDIDATE ID: each cell's target DID comes from
that qid's own release positive_ids, its gallery position is the row whose release gallery_id equals
that DID's candidate asset. Every stored (rank, score, gallery_position, gallery_id) is checked for
mutual consistency, the summary cell is checked against the export row and the release, the r5
correction is checked as a structural leaf diff, and the served validator is probed with corrupted
COPIES of the evidence tree.

No GPU, no query re-encoding: the run cache holds NaN query rows for these two qids by design
(confirmed below), so ordering is verified from the stored window instead of recomputed.

Usage: workbench/envs/limn310-adapted/bin/python scripts/m03/m03_verify.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parents[2]            # workbench/docs/reviews/<audit>
REPO = HERE.parents[3]                                # repo root
sys.path.insert(0, str(REPO))

from workbench.backend import limn_diagnostic as diag          # noqa: E402
from workbench.backend.operator_config import resolve_config   # noqa: E402
from workbench.composition.data import load_release            # noqa: E402
from workbench.composition.runner import _content_digest       # noqa: E402

EVIDENCE = REPO / "workbench/docs/reviews/20261109-native15-composition-limn"
OUT_JSON = HERE / "raw/m03/m03_verify.json"
OUT_TXT = HERE / "raw/m03/m03_verify.txt"
SUMMARY_ROOT = REPO / "workbench/artifacts/composition/summary"
RESULTS_ROOT = REPO / "workbench/artifacts/composition/results/circus_fashioniq_release_v1/limn"
FEATURES = (REPO / "workbench/artifacts/composition/runs/9e4918a6b5c243412cbc58c9"
            / "work/limn/features")
CKPT_DOWNLOADS = REPO / "workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads/limn"

QIDS = ("7:3038", "7:3790")
SCOPES = ("shirt", "toptee")
CONDITIONS = ("multimodal", "text_only_black_image", "image_only_empty_text")
DEPTH_EXPECTED = 100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def flat_leaves(value: Any, prefix: str = "") -> dict[str, Any]:
    """Dotted leaf paths; lists stay leaves (a list is a value, not a set of paths)."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            out.update(flat_leaves(item, f"{prefix}.{key}" if prefix else key))
        return out
    return {prefix: json.dumps(value, sort_keys=True) if isinstance(value, list) else value}


def check(checks: dict[str, Any], name: str, ok: bool, detail: str | None = None) -> None:
    checks[name] = {"ok": bool(ok)} | ({"detail": detail} if detail else {})


def export_name(scope: str, condition: str) -> str:
    return f"limn_dual_scope.{scope}.{condition}.jsonl"


def duplicated(values: list[Any]) -> list[Any]:
    return sorted({value for value in values if values.count(value) > 1})


# --------------------------------------------------------------------------- release + exports
def main() -> int:
    config = resolve_config()
    release = load_release(Path(config.WORKBENCH_CIRCUS_RELEASE_ROOT))
    by_qid = {query.qid: query for query in release.queries}
    gallery_ids = list(release.gallery.image_ids)
    position_of = {gid: index for index, gid in enumerate(gallery_ids)}
    # the release candidate table maps DID -> gallery asset path; invert it to learn which DID a
    # gallery row belongs to without ever going through a position or a tensor.
    did_of_asset = {asset: did for did, asset in release.candidate_names.items()}
    assets = set(did_of_asset)
    summary = json.loads((EVIDENCE / "limn_dual_scope.json").read_text(encoding="utf-8"))

    report: dict[str, Any] = {
        "artifact": "m03_verify",
        "target": "workbench/docs/reviews/20261109-native15-composition-limn (LIMN dual-scope artifact)",
        "join": "candidate id: qid -> own positive_ids -> candidate_names[did] -> gallery position",
        "release": {
            "root": str(config.WORKBENCH_CIRCUS_RELEASE_ROOT),
            "gallery_count": len(gallery_ids),
            "gallery_id_unique": len(position_of) == len(gallery_ids),
            "candidate_count": len(release.candidate_names),
            "hash_verification": release.provenance.get("hash_verification"),
            "query_count": len(release.queries),
        },
        "score_recompute": ("NOT performed: no query embeddings available. The run's query cache "
                            "holds NaN rows for both qids (verified below), and recomputing them "
                            "requires the GPU. Ordering verified from the stored window."),
    }

    # ---------------------------------------------------------------- query cache NaN proof
    cache_nan: dict[str, Any] = {}
    for condition in CONDITIONS:
        for scope in SCOPES:
            path = FEATURES / f"query_{condition}_{scope}.npy"
            entry: dict[str, Any] = {"file": str(path), "exists": path.is_file()}
            if path.is_file():
                array = np.load(path, mmap_mode="r")
                entry["shape"] = list(array.shape)
                entry["qid_row_all_nan"] = {
                    qid: bool(np.isnan(np.asarray(array[by_qid[qid].index])).all()) for qid in QIDS}
            cache_nan[f"{scope}/{condition}"] = entry
    report["query_cache_nan_rows"] = cache_nan
    report["query_cache_cannot_recompute"] = all(
        all(v.get("qid_row_all_nan", {}).get(qid, False) for qid in QIDS)
        for v in cache_nan.values() if v.get("qid_row_all_nan"))

    # ---------------------------------------------------------------- gallery matrices
    matrices: dict[str, Any] = {}
    for scope in SCOPES:
        declared = summary["gallery_matrices"][scope]
        path = Path(declared["file"])
        array = np.load(path, mmap_mode="r")
        recomputed = _content_digest(array)
        sidecar = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
        matrices[scope] = {
            "file": str(path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "byte_length": path.stat().st_size,
            "sidecar_digest": sidecar["content_digest"],
            "declared_digest": declared["recomputed_content_digest"],
            "recomputed_digest": recomputed,
            "digest_matches": (recomputed == sidecar["content_digest"]
                               == declared["recomputed_content_digest"] == declared["recorded_content_digest"]),
            "declared_shape_matches": (list(array.shape) == list(declared["shape"])
                                       == list(sidecar["shape"])),
            "declared_byte_length_matches": path.stat().st_size == declared["byte_length"],
        }
    report["gallery_matrices"] = matrices

    # ---------------------------------------------------------------- checkpoints
    checkpoints: dict[str, Any] = {}
    for scope in SCOPES:
        declared = summary["checkpoints"][scope]
        path = Path(declared["path"])
        measured = sha256(path) if path.is_file() else None
        checkpoints[scope] = {
            "path": str(path),
            "exists": path.is_file(),
            "declared_sha256": declared["sha256"],
            "recomputed_sha256": measured,
            "sha256_matches": measured == declared["sha256"],
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "declared_size_matches": (path.is_file() and path.stat().st_size == declared["size_bytes"]),
            "checkpoint_id": declared["checkpoint_id"],
        }
    report["checkpoints"] = checkpoints

    # ---------------------------------------------------------------- exports
    exports: dict[str, Any] = {}
    rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for scope in SCOPES:
        for condition in CONDITIONS:
            path = EVIDENCE / export_name(scope, condition)
            parsed = read_jsonl(path) if path.is_file() else []
            rows[(scope, condition)] = parsed
            qid_list = [row.get("qid") for row in parsed]
            exports[export_name(scope, condition)] = {
                "exists": path.is_file(),
                "sha256": sha256(path) if path.is_file() else None,
                "row_count": len(parsed),
                "row_count_expected": 2,
                "row_count_ok": len(parsed) == 2,
                "qids": qid_list,
                "duplicate_qids": duplicated(qid_list),
                "missing_qids": [qid for qid in QIDS if qid not in qid_list],
                "unexpected_qids": [qid for qid in qid_list if qid not in QIDS],
                "declared_depth": sorted({row.get("depth") for row in parsed}),
                "stored_candidate_counts": {row["qid"]: len(row.get("candidates") or []) for row in parsed},
                "truncated": sorted(row["qid"] for row in parsed
                                    if isinstance(row.get("depth"), int)
                                    and len(row.get("candidates") or []) < row["depth"]),
                "window_longer_than_depth": sorted(row["qid"] for row in parsed
                                                   if isinstance(row.get("depth"), int)
                                                   and len(row.get("candidates") or []) > row["depth"]),
                "scope_condition_declared": sorted({(row.get("scope"), row.get("condition")) for row in parsed},
                                                   key=str),
            }
    report["exports"] = exports

    # ---------------------------------------------------------------- 12 cells
    cells: list[dict[str, Any]] = []
    for qid in QIDS:
        query = by_qid[qid]
        own_positives = [str(value) for value in query.positive_ids]
        did = own_positives[0] if own_positives else None
        asset = release.candidate_names.get(did) if did else None
        joined_position = position_of.get(asset) if asset else None
        for scope in SCOPES:
            for condition in CONDITIONS:
                declared = summary["queries"][qid]["scopes"][scope][condition]
                row = next((item for item in rows[(scope, condition)] if item.get("qid") == qid), None)
                checks: dict[str, Any] = {}
                failures: list[str] = []
                cell = {
                    "qid": qid, "scope": scope, "condition": condition,
                    "join": {"own_positives": own_positives, "did": did, "candidate_asset": asset,
                             "gallery_position": joined_position},
                }

                check(checks, "own_positives_present", bool(own_positives), f"{own_positives}")
                check(checks, "did_has_candidate_asset", asset is not None, f"did={did} asset={asset}")
                check(checks, "asset_found_in_release_gallery", joined_position is not None,
                      f"{asset} -> {joined_position}")

                if row is None:
                    check(checks, "export_row_found", False, export_name(scope, condition))
                    cell |= {"checks": checks, "verdict": "FAIL", "failures": ["export_row_missing"]}
                    cells.append(cell)
                    continue
                check(checks, "export_row_found", True, export_name(scope, condition))
                check(checks, "export_row_identity", row.get("scope") == scope and row.get("condition") == condition,
                      f"scope={row.get('scope')} condition={row.get('condition')}")
                check(checks, "export_row_qid_unique",
                      [item.get("qid") for item in rows[(scope, condition)]].count(qid) == 1)

                candidates = row.get("candidates") or []
                depth = row.get("depth")
                check(checks, "declared_depth_is_int", isinstance(depth, int) and not isinstance(depth, bool),
                      f"{depth!r}")
                check(checks, "declared_depth_equals_expected", depth == DEPTH_EXPECTED, f"{depth}")
                check(checks, "stored_count_equals_declared", len(candidates) == depth,
                      f"stored={len(candidates)} declared={depth}")

                ranks = [c.get("rank") for c in candidates]
                check(checks, "ranks_sequential_1_n", ranks == list(range(1, len(candidates) + 1)),
                      f"{ranks[:3]}..{ranks[-1:] if ranks else []}")
                scores = [c.get("score") for c in candidates]
                check(checks, "scores_all_finite", all(isinstance(s, float) and np.isfinite(s) for s in scores))
                increases = [(i, scores[i], scores[i + 1]) for i in range(len(scores) - 1)
                             if scores[i] < scores[i + 1]]
                check(checks, "scores_monotone_nonincreasing", not increases, f"{increases[:3]}")

                gids = [c.get("gallery_id") for c in candidates]
                gpositions = [c.get("gallery_position") for c in candidates]
                check(checks, "window_gallery_id_unique", not duplicated(gids), f"{duplicated(gids)[:3]}")
                check(checks, "window_position_unique", not duplicated(gpositions), f"{duplicated(gpositions)[:3]}")

                # position <-> gallery_id must be the release's own ordering, for every stored row.
                bad_pairs = [(c.get("rank"), c.get("gallery_position"), c.get("gallery_id"))
                             for c in candidates
                             if not isinstance(c.get("gallery_position"), int)
                             or not 0 <= c["gallery_position"] < len(gallery_ids)
                             or gallery_ids[c["gallery_position"]] != c.get("gallery_id")]
                check(checks, "every_window_position_matches_release_gallery_id", not bad_pairs, f"{bad_pairs[:3]}")

                bad_paths = [(c.get("rank"), c.get("gallery_id"), c.get("image_path"))
                             for c in candidates
                             if c.get("image_path") is not None and c.get("image_path") != c.get("gallery_id")]
                check(checks, "every_window_image_path_equals_gallery_id", not bad_paths, f"{bad_paths[:3]}")
                not_in_release = [c.get("gallery_id") for c in candidates
                                  if c.get("gallery_id") not in assets]
                check(checks, "every_window_gallery_id_is_a_release_candidate_asset",
                      not not_in_release, f"{not_in_release[:3]}")

                # target identity: joined position must be in the window when it is inside depth
                flagged = [c for c in candidates if c.get("is_target")]
                check(checks, "is_target_flag_count_leq_1", len(flagged) <= 1, f"{len(flagged)}")
                target_in_window = (isinstance(declared["target_rank"], int)
                                    and 1 <= declared["target_rank"] <= len(candidates))
                if target_in_window:
                    check(checks, "target_flagged_in_window", len(flagged) == 1, f"{len(flagged)}")
                else:
                    check(checks, "target_flagged_in_window", True,
                          f"target_rank {declared['target_rank']} outside depth {len(candidates)}")
                if flagged:
                    flagged_row = flagged[0]
                    check(checks, "flagged_gallery_position_equals_join",
                          flagged_row.get("gallery_position") == joined_position,
                          f"flagged={flagged_row.get('gallery_position')} join={joined_position}")
                    check(checks, "flagged_gallery_id_equals_release_asset",
                          flagged_row.get("gallery_id") == asset,
                          f"flagged={flagged_row.get('gallery_id')} asset={asset}")
                    check(checks, "flagged_rank_equals_summary_rank",
                          flagged_row.get("rank") == declared["target_rank"],
                          f"flagged={flagged_row.get('rank')} summary={declared['target_rank']}")
                    check(checks, "flagged_score_equals_summary_score",
                          flagged_row.get("score") == declared["target_score"],
                          f"flagged={flagged_row.get('score')} summary={declared['target_score']}")
                    window_at_join = [c["rank"] for c in candidates
                                      if c.get("gallery_position") == joined_position]
                    check(checks, "summary_rank_is_rank_of_join_position_in_window",
                          window_at_join == [declared["target_rank"]],
                          f"window_rank={window_at_join} summary={declared['target_rank']}")
                else:
                    check(checks, "summary_rank_beyond_depth",
                          isinstance(declared["target_rank"], int)
                          and declared["target_rank"] > len(candidates),
                          f"rank={declared['target_rank']} depth={len(candidates)}")

                if scores:
                    check(checks, "score_at_rank_1_equals_window_rank1",
                          declared.get("score_at_rank_1") == scores[0],
                          f"summary={declared.get('score_at_rank_1')} window={scores[0]}")
                    check(checks, "score_max_equals_window_rank1",
                          declared.get("score_max") == scores[0],
                          f"summary_max={declared.get('score_max')} window_rank1={scores[0]}")
                    check(checks, "target_margin_equals_rank1_minus_target_score",
                          _close(declared.get("target_margin_vs_rank1"),
                                 scores[0] - declared["target_score"]),
                          f"declared={declared.get('target_margin_vs_rank1')} "
                          f"derived={scores[0] - declared['target_score']}")
                check(checks, "target_score_within_min_max",
                      _inside(declared), str(declared.get("score_min")))
                check(checks, "gallery_size_scored_equals_release",
                      declared.get("gallery_size_scored") == len(gallery_ids) == row.get("gallery_size"),
                      f"summary={declared.get('gallery_size_scored')} export={row.get('gallery_size')} "
                      f"release={len(gallery_ids)}")

                # summary identity vs release, by join (never by position or vector)
                check(checks, "summary_target_did_is_own_positive",
                      declared.get("target_did") in own_positives,
                      f"summary={declared.get('target_did')} own={own_positives}")
                check(checks, "summary_target_did_equals_release_join",
                      declared.get("target_did") == did,
                      f"summary={declared.get('target_did')} release={did}")
                check(checks, "query_record_target_gallery_id_equals_release_asset",
                      summary["queries"][qid].get("target_gallery_id") == asset,
                      f"query_record={summary['queries'][qid].get('target_gallery_id')} asset={asset}")
                check(checks, "summary_target_position_equals_join",
                      declared.get("target_gallery_position") == joined_position,
                      f"summary={declared.get('target_gallery_position')} join={joined_position}")
                check(checks, "summary_position_in_query_target_positions",
                      declared.get("target_gallery_position") in (
                          summary["queries"][qid].get("target_position_in_gallery") or [None]),
                      str(summary["queries"][qid].get("target_position_in_gallery")))
                check(checks, "summary_rank_equals_export_rank",
                      declared.get("target_rank") == row.get("target_rank"),
                      f"summary={declared.get('target_rank')} export={row.get('target_rank')}")
                check(checks, "summary_score_equals_export_score",
                      declared.get("target_score") == row.get("target_score"),
                      f"summary={declared.get('target_score')} export={row.get('target_score')}")
                check(checks, "export_checkpoint_equals_summary_checkpoint",
                      row.get("checkpoint_sha256") == summary["checkpoints"][scope]["sha256"])
                check(checks, "export_checkpoint_equals_disk_checkpoint",
                      row.get("checkpoint_sha256") == checkpoints[scope]["recomputed_sha256"])
                check(checks, "top100_export_path_is_this_file",
                      Path(str(declared.get("top100_export"))).name == export_name(scope, condition),
                      str(declared.get("top100_export")))
                check(checks, "query_embedding_norm_is_unit",
                      _close(declared.get("query_embedding_norm"), 1.0, tol=1e-6),
                      str(declared.get("query_embedding_norm")))

                # gallery matrix: the joined row exists and is L2-normalized (score = q . row)
                if joined_position is not None:
                    array = np.load(Path(summary["gallery_matrices"][scope]["file"]), mmap_mode="r")
                    norm = float(np.linalg.norm(np.asarray(array[joined_position], dtype=np.float64)))
                    check(checks, "joined_gallery_row_is_unit_norm", _close(norm, 1.0, tol=1e-5), f"{norm:.9f}")
                    check(checks, "joined_gallery_row_finite",
                          bool(np.isfinite(np.asarray(array[joined_position])).all()))

                failures = [name for name, item in checks.items() if not item["ok"]]
                cell |= {"checks": checks, "failures": failures,
                         "verdict": "PASS" if not failures else "FAIL",
                         "observed": {
                             "target_rank": declared["target_rank"], "target_score": declared["target_score"],
                             "score_at_rank_1": declared.get("score_at_rank_1"),
                             "target_gallery_position": declared.get("target_gallery_position"),
                             "target_did": declared.get("target_did"),
                             "target_gallery_id": declared.get("target_gallery_id"),
                             "in_window": target_in_window, "depth": depth,
                             "stored_candidates": len(candidates)}}
                cells.append(cell)

    report["cells"] = cells
    report["cells_summary"] = {
        "total": len(cells),
        "pass": sum(1 for c in cells if c["verdict"] == "PASS"),
        "fail": sum(1 for c in cells if c["verdict"] == "FAIL"),
        "failing_cells": [f"{c['qid']}/{c['scope']}/{c['condition']}" for c in cells if c["verdict"] == "FAIL"],
        "failed_checks": sorted({name for c in cells for name in c["failures"]}),
        "checks_per_cell_min": min(len(c["checks"]) for c in cells),
        "target_ranks": {f"{c['qid']}/{c['scope']}/{c['condition']}": c["observed"]["target_rank"] for c in cells},
    }

    # ---------------------------------------------------------------- r5 structural leaf diff
    current_path = EVIDENCE / "limn_dual_scope.json"
    pre_path = EVIDENCE / "limn_dual_scope.json.pre-r5"
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    leaves_now, leaves_pre = flat_leaves(summary), flat_leaves(pre)
    added = sorted(set(leaves_now) - set(leaves_pre))
    removed = sorted(set(leaves_pre) - set(leaves_now))
    changed = sorted(path for path in set(leaves_now) & set(leaves_pre)
                     if leaves_now[path] != leaves_pre[path])
    declared_correction = summary["provenance_correction_r5"]
    expected_changed = {
        f"queries.{item['qid']}.scopes.{item['scope']}.{item['condition']}.target_did"
        for item in declared_correction["changed"]}
    did_changed = {path for path in changed if path.endswith(".target_did")}
    non_did_changed = [path for path in changed if not path.endswith(".target_did")]
    r5 = {
        "files": {"current": str(current_path), "pre_r5": str(pre_path)},
        "leaf_counts": {"current": len(leaves_now), "pre_r5": len(leaves_pre)},
        "added_count": len(added), "removed_count": len(removed), "changed_count": len(changed),
        "added_paths": added, "removed_paths": removed, "changed_paths": changed,
        "added_all_under_provenance_correction_r5": all(
            path.startswith("provenance_correction_r5") for path in added),
        "removed_none": not removed,
        "changed_target_did_cells": len(did_changed),
        "changed_non_target_did": non_did_changed,
        "changed_set_equals_declared_cells": did_changed == expected_changed,
        "declared_cells_changed_field": declared_correction["cells_changed"],
        "declared_cells_count_matches": declared_correction["cells_changed"] == len(did_changed) == 6,
        "declared_changed_values_match": all(
            leaves_pre[f"queries.{item['qid']}.scopes.{item['scope']}.{item['condition']}.target_did"]
            == item["was"]
            and leaves_now[f"queries.{item['qid']}.scopes.{item['scope']}.{item['condition']}.target_did"]
            == item["now"] for item in declared_correction["changed"]),
        "declared_unchanged_fields_changed": sorted({
            name for name in declared_correction["unchanged"]
            for path in changed if path.endswith(f".{name}")}),
        "top_level_target_did_present_now": any(
            path.endswith(f"queries.{qid}.target_did") for path in leaves_now for qid in QIDS),
        "top_level_target_did_present_pre_r5": any(
            path.endswith(f"queries.{qid}.target_did") for path in leaves_pre for qid in QIDS),
        "sha256": {"current": sha256(current_path), "pre_r5": sha256(pre_path)},
        "declared_artifact_sha256_before": declared_correction["artifact_sha256_before"],
        "artifact_sha256_before_matches_pre_r5": (
            declared_correction["artifact_sha256_before"] == sha256(pre_path)),
        "rescored_claimed": declared_correction["rescored"],
        "reranked_claimed": declared_correction["reranked"],
        "rank_score_position_leaves_unchanged": not non_did_changed,
    }
    r5["verdict"] = ("PASS" if r5["changed_count"] == 6 and not non_did_changed
                     and not removed and r5["added_all_under_provenance_correction_r5"]
                     and r5["changed_set_equals_declared_cells"]
                     and r5["artifact_sha256_before_matches_pre_r5"] else "FAIL")
    report["r5"] = r5

    # ---------------------------------------------------------------- depth + top100 export map
    depth_rows = []
    for scope in SCOPES:
        for condition in CONDITIONS:
            for row in rows[(scope, condition)]:
                depth_rows.append({
                    "export": export_name(scope, condition), "qid": row.get("qid"),
                    "declared_depth": row.get("depth"),
                    "stored_candidates": len(row.get("candidates") or []),
                    "truncated": (isinstance(row.get("depth"), int)
                                  and len(row.get("candidates") or []) < row["depth"]),
                    "window_longer_than_depth": (isinstance(row.get("depth"), int)
                                                 and len(row.get("candidates") or []) > row["depth"]),
                })
    report["depth"] = {
        "rows": depth_rows,
        "rows_total": len(depth_rows),
        "truncated_rows": [r for r in depth_rows if r["truncated"]],
        "window_longer_than_depth_rows": [r for r in depth_rows if r["window_longer_than_depth"]],
        "declared_depth_values": sorted({r["declared_depth"] for r in depth_rows}, key=str),
        "stored_count_values": sorted({r["stored_candidates"] for r in depth_rows}),
        "runtime_top_k": summary["runtime"]["top_k"],
    }

    referenced: dict[str, list[str]] = {}
    for qid in QIDS:
        for scope in SCOPES:
            for condition in CONDITIONS:
                path = Path(str(summary["queries"][qid]["scopes"][scope][condition]["top100_export"]))
                referenced.setdefault(path.name, []).append(f"{qid}/{scope}/{condition}")
    expected_names = sorted(export_name(scope, condition) for scope in SCOPES for condition in CONDITIONS)
    report["top100_exports"] = {
        "distinct_files_referenced": len(referenced),
        "expected_distinct_files": len(expected_names),
        "names_match_expected_set": sorted(referenced) == expected_names,
        "each_file_has_two_cells": {name: len(cells_) for name, cells_ in sorted(referenced.items())},
        "cells_to_one_file_ok": all(len(cells_) == 2 for cells_ in referenced.values()),
        "all_reference_paths_resolve": all((EVIDENCE / name).is_file() for name in referenced),
        "missing": [name for name in expected_names if name not in referenced],
        "unexpected": [name for name in referenced if name not in expected_names],
        "absolute_paths_under_evidence_dir": all(
            Path(str(summary["queries"][qid]["scopes"][scope][condition]["top100_export"])).parent
            == EVIDENCE for qid in QIDS for scope in SCOPES for condition in CONDITIONS),
    }

    # ---------------------------------------------------------------- served validator
    public = diag.load_diagnostic()
    internal = diag._load_record()
    report["validator"] = {
        "status": public["status"],
        "complete": public["complete"],
        "standard": public["standard"],
        "defect_count": len(public["defects"]),
        "defects": public["defects"],
        "defect_codes": sorted({item["code"] for item in public["defects"]}),
        "manifest_digest": (public["manifest"] or {}).get("digest"),
        "manifest_files": (public["manifest"] or {}).get("files"),
        "manifest_file_count": len((public["manifest"] or {}).get("files") or []),
        "manifest_declared_root": (public["manifest"] or {}).get("declared_root"),
        "depth_stored_per_combination": {
            f"{key[0]}/{key[1]}/{key[2]}": value["depth_stored"]
            for key, value in sorted(internal["combinations"].items())},
        "depth_stored_values": sorted({value["depth_stored"]
                                       for value in internal["combinations"].values()}),
        "public_payload_exposes_depth_stored": "depth_stored" in json.dumps(public["queries"]),
        "defect_target_did_codes": sorted({item["code"] for item in public["defects"]
                                           if "target_did" in item["code"]}),
    }

    # ---------------------------------------------------------------- manifest binding
    files = [EVIDENCE / "limn_dual_scope.json"] + [
        EVIDENCE / export_name(scope, condition) for scope in SCOPES for condition in CONDITIONS]
    independent_files = [{"name": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
                         for path in files if path.is_file()]
    independent_digest = hashlib.sha256(
        json.dumps(independent_files, sort_keys=True).encode("utf-8")).hexdigest()
    served_manifest = public["manifest"] or {}
    served_files = {item["name"]: item for item in (served_manifest.get("files") or [])}
    report["manifest_binding"] = {
        "file_count": len(independent_files),
        "expected_file_count": 7,
        "files": independent_files,
        "independent_digest": independent_digest,
        "served_digest": served_manifest.get("digest"),
        "digest_matches_served": independent_digest == served_manifest.get("digest"),
        "served_file_entries_match_disk": all(
            served_files.get(item["name"], {}).get("sha256") == item["sha256"]
            and served_files.get(item["name"], {}).get("bytes") == item["bytes"]
            for item in independent_files),
        "sha256_recorded_in_summary": {
            "limn_dual_scope.json": summary["provenance_correction_r5"]["artifact_sha256_before"],
            "note": ("the summary records only the PRE-r5 digest; the served manifest digest is "
                     "computed from the bytes on disk and is not written into the artifact"),
        },
    }

    # ---------------------------------------------------------------- corrupt-copy probes
    probes: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="m03-probe-") as tmpdir:
        tmp_evidence = Path(tmpdir).joinpath(*diag.EVIDENCE_RELATIVE)
        tmp_evidence.mkdir(parents=True)

        def copy_tree() -> None:
            for name in ["limn_dual_scope.json"] + [export_name(s, c) for s in SCOPES for c in CONDITIONS]:
                shutil.copy2(EVIDENCE / name, tmp_evidence / name)

        def probe_config() -> SimpleNamespace:
            return SimpleNamespace(CIR_REPO_ROOT=Path(tmpdir), FASHIONIQ_ROOT=config.FASHIONIQ_ROOT)

        def run_probe() -> dict[str, Any]:
            diag._RECORD_CACHE.clear()
            try:
                result = diag.load_diagnostic(probe_config())
            except Exception as error:  # noqa: BLE001 - the refusal IS the observation
                return {"raised": type(error).__name__, "message": str(error),
                        "status": None, "complete": None, "defects": [], "defect_codes": []}
            return {"raised": None, "status": result["status"], "complete": result["complete"],
                    "defect_count": len(result["defects"]),
                    "defects": result["defects"],
                    "defect_codes": sorted({item["code"] for item in result["defects"]}),
                    "manifest_digest": (result["manifest"] or {}).get("digest")}

        # A: candidates window cut to 1 row while depth keeps declaring 100.
        copy_tree()
        target = tmp_evidence / export_name("shirt", "multimodal")
        lines_ = read_jsonl(target)
        for row in lines_:
            # Depth keeps declaring 100; the stored window keeps its own rank-1 row untouched.
            row["candidates"] = row["candidates"][:1]
        target.write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in lines_) + "\n",
                          encoding="utf-8")
        tampered_a = read_jsonl(tmp_evidence / export_name("shirt", "multimodal"))
        probes["A_window_truncated_to_1_depth_100"] = run_probe() | {
            "tampered": export_name("shirt", "multimodal"),
            "tamper": "candidates[:1] with depth still 100",
            "tampered_rows": {r["qid"]: {"declared_depth": r["depth"],
                                         "stored_candidates": len(r["candidates"]),
                                         "target_rank": r["target_rank"]} for r in tampered_a},
            "detected": None}

        # B: file truncated to a single row (one qid missing), depth untouched.
        copy_tree()
        target = tmp_evidence / export_name("toptee", "multimodal")
        target.write_text(json.dumps(read_jsonl(target)[0], separators=(",", ":")) + "\n",
                          encoding="utf-8")
        probes["B_export_truncated_to_1_row"] = run_probe() | {
            "tampered": export_name("toptee", "multimodal"),
            "tamper": "file cut to 1 of 2 qid rows",
            "detected": None}

        # C: duplicate qid row appended (identical).
        copy_tree()
        target = tmp_evidence / export_name("shirt", "text_only_black_image")
        lines_ = read_jsonl(target)
        target.write_text("\n".join(json.dumps(r, separators=(",", ":")) for r in lines_ + [lines_[0]]) + "\n",
                          encoding="utf-8")
        probes["C_duplicate_qid_row_appended"] = run_probe() | {
            "tampered": export_name("shirt", "text_only_black_image"),
            "tamper": "row_count 2 -> 3, duplicate qid identical",
            "qids_after": [r["qid"] for r in read_jsonl(target)],
            "detected": None}

        # D: duplicate qid row appended with a CONTRADICTORY target_rank (next(...) keeps the first).
        copy_tree()
        target = tmp_evidence / export_name("toptee", "image_only_empty_text")
        lines_ = read_jsonl(target)
        conflicting = json.loads(json.dumps(lines_[0]))
        conflicting["target_rank"] = conflicting["target_rank"] + 1000
        conflicting["target_score"] = -1.0
        target.write_text("\n".join(json.dumps(r, separators=(",", ":"))
                                    for r in lines_ + [conflicting]) + "\n", encoding="utf-8")
        probes["D_conflicting_duplicate_qid_row"] = run_probe() | {
            "tampered": export_name("toptee", "image_only_empty_text"),
            "tamper": "row_count 2 -> 3, duplicate qid with target_rank+1000",
            "rows_after": [{"qid": r["qid"], "target_rank": r["target_rank"],
                            "target_score": r["target_score"]} for r in read_jsonl(target)],
            "detected": None}

        # E: missing export file, as a control that the validator CAN refuse.
        copy_tree()
        (tmp_evidence / export_name("toptee", "text_only_black_image")).unlink()
        probes["E_export_file_missing_control"] = run_probe() | {
            "tampered": export_name("toptee", "text_only_black_image"),
            "tamper": "file deleted",
            "detected": None}
    diag._RECORD_CACHE.clear()

    for name, probe in probes.items():
        probe["detected"] = bool(probe.get("raised") or probe.get("defect_count"))
        probe["silent_pass"] = not probe["detected"]
    report["validator_probes"] = {
        "note": ("each probe runs load_diagnostic() against a COPY of the evidence tree in a temp "
                 "dir; the real tree is never modified"),
        "probes": probes,
        "silently_accepted": sorted(name for name, p in probes.items() if p["silent_pass"]),
        "refused_or_flagged": sorted(name for name, p in probes.items() if not p["silent_pass"]),
    }

    # ---------------------------------------------------------------- generation mixing
    generations: dict[str, Any] = {}
    for directory in sorted(SUMMARY_ROOT.glob("*limn*")):
        if not directory.is_dir():
            continue
        files: dict[str, Any] = {}
        for subset in ("full", "sf", "validated"):
            path = directory / f"{subset}.json"
            if not path.is_file():
                files[subset] = {"exists": False}
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            metrics = payload.get("metrics") or {}
            files[subset] = {
                "exists": True,
                "path": str(path.relative_to(REPO)),
                "sha256": sha256(path),
                "run_id": payload.get("run_id"),
                "subset": payload.get("subset"),
                "query_count": payload.get("query_count"),
                "scored_count": (metrics.get("multimodal") or {}).get("scored_count"),
                "multimodal_recall_at_10": (metrics.get("multimodal") or {}).get("recall_at_10"),
                "checkpoint_identity": payload.get("checkpoint_identity"),
                "gallery_identity_sha256": payload.get("gallery_identity_sha256"),
                "smoke": payload.get("smoke"),
                "execution_status": payload.get("execution_status"),
                "results_dir": str((RESULTS_ROOT / "base_iter0_all_categories"
                                    / str(payload.get("run_id")) / subset).relative_to(REPO)),
                "results_dir_exists": (RESULTS_ROOT / "base_iter0_all_categories"
                                       / str(payload.get("run_id")) / subset).is_dir(),
            }
        generations[directory.name] = files

    canonical_run = summary["source_run_id"]
    canonical_summary_path = SUMMARY_ROOT / f"limn_base_iter0_all_categories_{canonical_run}" / "full.json"
    canonical_summary = json.loads(canonical_summary_path.read_text(encoding="utf-8"))
    derived_ids = {name: {subset: (item.get("run_id") if item.get("exists") else None)
                          for subset, item in files.items()}
                   for name, files in generations.items()}
    report["generations"] = {
        "artifact": "limn (base_iter0_all_categories)",
        "canonical_run_id_in_dual_scope": canonical_run,
        "generations": generations,
        "run_ids_by_generation": derived_ids,
        "canonical_generation_has": {subset: generations.get(
            f"limn_base_iter0_all_categories_{canonical_run}", {}).get(subset, {"exists": False})["exists"]
            for subset in ("full", "sf", "validated")},
        "sf_validated_same_run_as_full": {
            subset: (generations.get(f"limn_base_iter0_all_categories_{canonical_run}", {})
                     .get(subset, {}).get("run_id") == canonical_run)
            for subset in ("sf", "validated")},
        "sf_validated_absent_in_canonical_generation": [
            subset for subset in ("sf", "validated")
            if not generations.get(f"limn_base_iter0_all_categories_{canonical_run}", {})
            .get(subset, {"exists": False})["exists"]],
        "mixing_risk": ("canonical generation has full only; its sf/validated rows do not exist. "
                        "Any sf/validated figure shown beside it comes from another run id."),
        "source_run_id_consistency": {
            "limn_dual_scope.json": canonical_run,
            "canonical_summary_full.json": canonical_summary.get("run_id"),
            "canonical_summary_path": str(canonical_summary_path.relative_to(REPO)),
            "equal": canonical_run == canonical_summary.get("run_id"),
        },
        "canonical_cohort_claim": summary["canonical_cohort"],
        "canonical_cohort_matches_summary": {
            "query_count": summary["canonical_cohort"]["query_count"] == canonical_summary.get("query_count"),
            "scored_count": summary["canonical_cohort"]["scored_count"]
            == (canonical_summary["metrics"]["multimodal"]).get("scored_count"),
            "multimodal_recall_at_10": _close(
                summary["canonical_cohort"]["multimodal_recall_at_10"],
                (canonical_summary["metrics"]["multimodal"]).get("recall_at_10"), tol=1e-4),
            "source_field": summary["canonical_cohort"]["source"],
            "query_count_per_subset": summary["canonical_cohort"]["query_count_per_subset"]
            == canonical_summary.get("query_count_per_subset"),
            "unscored_are_the_two_qids": sorted(summary["canonical_status"]["unscored_qids"]) == sorted(QIDS),
            "summary_scored_plus_unscored_equals_full": (
                summary["canonical_status"]["scored_count"] + summary["canonical_status"]["unscored_count"]
                == canonical_summary.get("query_count")),
        },
        "other_generations_ids": {
            name: {subset: (item.get("run_id") if item.get("exists") else None)
                   for subset, item in files.items()} for name, files in generations.items()},
        "checkpoint_identity_shared_across_generations": sorted({
            item.get("checkpoint_identity") for files in generations.values()
            for item in files.values() if item.get("exists")}),
        "gallery_identity_shared_across_generations": sorted({
            item.get("gallery_identity_sha256") for files in generations.values()
            for item in files.values() if item.get("exists")}),
    }

    # ---------------------------------------------------------------- findings
    findings = []
    if report["cells_summary"]["fail"]:
        for cell in cells:
            if cell["verdict"] == "FAIL":
                findings.append({"severity": "FAIL", "what": f"cell {cell['qid']}/{cell['scope']}/"
                                                             f"{cell['condition']}",
                                 "detail": cell["failures"]})
    if r5["verdict"] != "PASS":
        findings.append({"severity": "FAIL", "what": "r5 leaf diff", "detail": r5})
    for name, probe in probes.items():
        if probe["silent_pass"]:
            findings.append({"severity": "GAP", "what": f"validator silently accepts {name}",
                             "detail": probe.get("tamper")})
    if not report["validator"]["complete"]:
        findings.append({"severity": "FAIL", "what": "served validator not complete",
                         "detail": report["validator"]["defect_codes"]})
    absent = report["generations"]["sf_validated_absent_in_canonical_generation"]
    if absent:
        findings.append({"severity": "GAP",
                         "what": f"canonical limn generation lacks {absent} summary rows",
                         "detail": report["generations"]["other_generations_ids"]})
    if not report["top100_exports"]["each_file_has_two_cells"]:
        findings.append({"severity": "FAIL", "what": "top100_export cell/file map"})
    report["findings"] = findings
    report["verdict"] = "PASS" if not [f for f in findings if f["severity"] == "FAIL"] else "FAIL"

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    OUT_TXT.write_text(render_text(report), encoding="utf-8")
    print(render_console(report))
    return 0


def _close(left: Any, right: Any, tol: float = 1e-9) -> bool:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return False
    return abs(float(left) - float(right)) <= tol


def _inside(declared: dict[str, Any]) -> bool:
    low, high, score = declared.get("score_min"), declared.get("score_max"), declared.get("target_score")
    if not all(isinstance(value, (int, float)) for value in (low, high, score)):
        return False
    return low <= score <= high


def render_text(report: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add("M-03 LIMN DUAL-SCOPE VERIFICATION")
    add("=" * 100)
    add(f"target      : {report['target']}")
    add(f"join        : {report['join']}")
    add(f"release     : gallery={report['release']['gallery_count']} candidates="
        f"{report['release']['candidate_count']} hash={report['release']['hash_verification']} "
        f"gallery_id_unique={report['release']['gallery_id_unique']}")
    add(f"verdict     : {report['verdict']}")
    add("")

    add("1. PER-CELL (12 = 2 qids x 2 scopes x 3 conditions)")
    add("-" * 100)
    add(f"{'qid':<8}{'scope':<8}{'condition':<25}{'rank':>6}{'depth':>6}{'inWin':>6}"
        f"{'pos':>7}{'did':>8}{'checks':>7}{'fail':>5}  verdict")
    for cell in report["cells"]:
        observed = cell["observed"]
        add(f"{cell['qid']:<8}{cell['scope']:<8}{cell['condition']:<25}"
            f"{observed['target_rank']:>6}{observed['depth']:>6}{str(observed['in_window']):>6}"
            f"{observed['target_gallery_position']:>7}{observed['target_did']:>8}"
            f"{len(cell['checks']):>7}{len(cell['failures']):>5}  {cell['verdict']}"
            + ("" if not cell["failures"] else f"  {';'.join(cell['failures'])}"))
    summary_cells = report["cells_summary"]
    add(f"cells pass={summary_cells['pass']}/{summary_cells['total']} fail={summary_cells['fail']} "
        f"checks_per_cell>={summary_cells['checks_per_cell_min']} "
        f"failed_checks={summary_cells['failed_checks'] or 'none'}")
    add(f"score recompute: {report['score_recompute']}")
    add(f"query cache NaN rows for both qids: {report['query_cache_cannot_recompute']}")
    add("")

    add("2. r5 CORRECTION (structural leaf diff)")
    add("-" * 100)
    r5 = report["r5"]
    add(f"leaves current={r5['leaf_counts']['current']} pre-r5={r5['leaf_counts']['pre_r5']} | "
        f"added={r5['added_count']} removed={r5['removed_count']} changed={r5['changed_count']} "
        f"| verdict={r5['verdict']}")
    add(f"added all under provenance_correction_r5: {r5['added_all_under_provenance_correction_r5']}")
    add(f"changed are exactly target_did cells: {r5['changed_target_did_cells']}/6, "
        f"declared set equal={r5['changed_set_equals_declared_cells']}, "
        f"non-target_did changes={r5['changed_non_target_did'] or 'none'}")
    for path in r5["changed_paths"]:
        add(f"  ~ {path}")
    for path in r5["added_paths"]:
        add(f"  + {path}")
    add(f"sha256 current   = {r5['sha256']['current']}")
    add(f"sha256 pre-r5    = {r5['sha256']['pre_r5']}")
    add(f"declared before  = {r5['declared_artifact_sha256_before']} -> matches pre-r5: "
        f"{r5['artifact_sha256_before_matches_pre_r5']}")
    add(f"rank/score/position leaves unchanged: {r5['rank_score_position_leaves_unchanged']} "
        f"(rescored={r5['rescored_claimed']} reranked={r5['reranked_claimed']})")
    add("")

    add("3. EXPORT FILES / DUPLICATE QIDS")
    add("-" * 100)
    add(f"{'file':<52}{'rows':>5}{'depth':>7}{'stored':>10}{'dupQid':>8}{'dupPos':>7}  sha256[:16]")
    for name, item in report["exports"].items():
        stored = ",".join(str(v) for v in item["stored_candidate_counts"].values())
        duplicate_positions = [c["qid"] + "/" + c["scope"] + "/" + c["condition"]
                               for c in report["cells"]
                               if c["scope"] + "." + c["condition"] in name
                               and not c["checks"]["window_position_unique"]["ok"]]
        add(f"{name:<52}{item['row_count']:>5}{str(item['declared_depth'])[1:-1]:>7}"
            f"{stored:>10}{str(item['duplicate_qids'] or '-'):>8}"
            f"{str(len(duplicate_positions) or '-'):>7}  {str(item['sha256'])[:16]}")
    dup_inside = {cell["qid"] + "/" + cell["scope"] + "/" + cell["condition"]: cell["checks"]["window_gallery_id_unique"]["ok"]
                  for cell in report["cells"]}
    add(f"duplicate qid per file: {[ (n, i['duplicate_qids']) for n, i in report['exports'].items() if i['duplicate_qids']] or 'none'}")
    add(f"duplicate gallery_id inside windows: "
        f"{[k for k, ok in dup_inside.items() if not ok] or 'none'}")
    add(f"duplicate gallery_position inside windows: "
        f"{[c['qid'] + '/' + c['scope'] + '/' + c['condition'] for c in report['cells'] if not c['checks']['window_position_unique']['ok']] or 'none'}")
    add("")

    add("4. DEPTH")
    add("-" * 100)
    depth = report["depth"]
    add(f"rows={depth['rows_total']} declared_depth={depth['declared_depth_values']} "
        f"stored={depth['stored_count_values']} runtime_top_k={depth['runtime_top_k']}")
    add(f"truncated rows (stored<declared): {depth['truncated_rows'] or 'none'}")
    add(f"windows longer than declared depth: {depth['window_longer_than_depth_rows'] or 'none'}")
    top = report["top100_exports"]
    add(f"top100_export files: distinct={top['distinct_files_referenced']} "
        f"expected={top['expected_distinct_files']} names_match={top['names_match_expected_set']}")
    add(f"cells per file: {top['each_file_has_two_cells']} all_resolve={top['all_reference_paths_resolve']} "
        f"missing={top['missing'] or 'none'} unexpected={top['unexpected'] or 'none'}")
    add("")

    add("5. SERVED VALIDATOR + CORRUPT-COPY PROBES")
    add("-" * 100)
    validator = report["validator"]
    add(f"status={validator['status']} complete={validator['complete']} defects={validator['defect_count']} "
        f"codes={validator['defect_codes'] or 'none'} standard={validator['standard']}")
    add(f"manifest digest={validator['manifest_digest']} files={validator['manifest_file_count']} "
        f"root={validator['manifest_declared_root']}")
    add(f"depth_stored per combination: {validator['depth_stored_per_combination']}")
    add(f"public load_diagnostic() exposes depth_stored: {validator['public_payload_exposes_depth_stored']}")
    binding = report["manifest_binding"]
    add(f"manifest binding: files={binding['file_count']}/{binding['expected_file_count']} "
        f"independent_digest={binding['independent_digest']} matches_served="
        f"{binding['digest_matches_served']} entries_match_disk={binding['served_file_entries_match_disk']}")
    for name, probe in report["validator_probes"]["probes"].items():
        add(f"  {name}: detected={probe['detected']} raised={probe['raised']} "
            f"status={probe['status']} codes={probe.get('defect_codes') or 'none'} "
            f"[{probe.get('tamper')}]")
    add(f"silently accepted: {report['validator_probes']['silently_accepted'] or 'none'}")
    add("")

    add("6. GENERATION MIXING")
    add("-" * 100)
    for name, files in report["generations"]["generations"].items():
        present = {subset: (item.get("run_id") if item.get("exists") else "ABSENT")
                   for subset, item in files.items()}
        add(f"{name}: {present}")
        for subset, item in files.items():
            if item.get("exists"):
                add(f"    {subset:<10} q={item['query_count']:<5} scored={item['scored_count']:<5} "
                    f"mm_r10={item['multimodal_recall_at_10']} ckpt={str(item['checkpoint_identity'])[:16]} "
                    f"smoke={bool((item.get('smoke') or {}).get('limited'))} sha256={item['sha256'][:16]}")
    gens = report["generations"]
    add(f"canonical run={gens['canonical_run_id_in_dual_scope']} has="
        f"{gens['canonical_generation_has']} absent={gens['sf_validated_absent_in_canonical_generation']}")
    add(f"source_run_id consistency: dual_scope={gens['source_run_id_consistency']['limn_dual_scope.json']} "
        f"summary={gens['source_run_id_consistency']['canonical_summary_full.json']} "
        f"equal={gens['source_run_id_consistency']['equal']}")
    add(f"canonical cohort vs summary: {gens['canonical_cohort_matches_summary']}")
    add("")

    add("FINDINGS")
    add("-" * 100)
    for finding in report["findings"]:
        add(f"[{finding['severity']}] {finding['what']} :: {json.dumps(finding.get('detail'))[:220]}")
    if not report["findings"]:
        add("none")
    add("")
    add(f"json: {OUT_JSON}")
    add(f"txt : {OUT_TXT}")
    return "\n".join(lines) + "\n"


def render_console(report: dict[str, Any]) -> str:
    cells = report["cells_summary"]
    r5 = report["r5"]
    validator = report["validator"]
    lines = [
        f"VERDICT {report['verdict']}",
        f"cells {cells['pass']}/{cells['total']} pass, fail={cells['failing_cells'] or 'none'}, "
        f"failed_checks={cells['failed_checks'] or 'none'}",
        f"r5 changed={r5['changed_count']} added={r5['added_count']} removed={r5['removed_count']} "
        f"sha_before_matches={r5['artifact_sha256_before_matches_pre_r5']}",
        f"exports rows={ {k: v['row_count'] for k, v in report['exports'].items()} }",
        f"depth truncated={report['depth']['truncated_rows'] or 'none'} top100_files="
        f"{report['top100_exports']['distinct_files_referenced']}",
        f"validator status={validator['status']} complete={validator['complete']} "
        f"defects={validator['defect_count']} digest={str(validator['manifest_digest'])[:16]} "
        f"digest_matches_independent={report['manifest_binding']['digest_matches_served']}",
        f"probes silent={report['validator_probes']['silently_accepted']}",
        f"generations absent_in_canonical={report['generations']['sf_validated_absent_in_canonical_generation']}",
        f"findings {[(f['severity'], f['what']) for f in report['findings']]}",
        f"wrote {OUT_JSON} and {OUT_TXT}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
