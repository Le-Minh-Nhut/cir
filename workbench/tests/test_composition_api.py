"""Test read-only CIRCUS composition API routes and safety invariants."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from workbench.backend.main import app
from workbench.backend.composition import (
    resolve_composition_path,
    validate_summary_dict,
    determine_query_outcome,
)


def test_resolve_composition_path_rejects_traversal(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "composition").mkdir()

    valid = resolve_composition_path(root, "summary/run1/full.json")
    assert str(valid).endswith("summary/run1/full.json")

    with pytest.raises(ValueError, match="Path traversal refused"):
        resolve_composition_path(root, "../../etc/passwd")

    with pytest.raises(ValueError, match="Path traversal refused"):
        resolve_composition_path(root, "../sibling")


def test_validate_summary_dict():
    valid = {
        "model_id": "encoder",
        "checkpoint_id": "fashioniq",
        "run_id": "run123",
        "subset": "full",
        "conditions": ["multimodal"],
        "metrics": {"coverage": 1.0},
    }
    ok, err = validate_summary_dict(valid)
    assert ok is True
    assert err is None

    missing = dict(valid)
    del missing["run_id"]
    ok, err = validate_summary_dict(missing)
    assert ok is False
    assert "run_id" in err

    invalid_sub = dict(valid, subset="invalid")
    ok, err = validate_summary_dict(invalid_sub)
    assert ok is False
    assert "subset" in err


def test_determine_query_outcome():
    # K = 10
    assert determine_query_outcome({"multimodal": None, "text_only_black_image": 1, "image_only_empty_text": 1}) == "unscored"
    assert determine_query_outcome({"multimodal": 5, "text_only_black_image": 20, "image_only_empty_text": 30}) == "composition_required"
    assert determine_query_outcome({"multimodal": 5, "text_only_black_image": 8, "image_only_empty_text": 25}) == "text_only"
    assert determine_query_outcome({"multimodal": 5, "text_only_black_image": 25, "image_only_empty_text": 8}) == "image_only"
    assert determine_query_outcome({"multimodal": 5, "text_only_black_image": 6, "image_only_empty_text": 8}) == "both"
    assert determine_query_outcome({"multimodal": 50, "text_only_black_image": 60, "image_only_empty_text": 80}) == "unresolved"


def test_composition_routes():
    client = TestClient(app)
    # 1. GET /api/composition/runs (fixtures opt-in keeps this smoke test independent
    #    of whether a real measurement exists on disk)
    res = client.get("/api/composition/runs?subset=full&include_fixtures=true")
    assert res.status_code == 200
    runs = res.json()
    assert isinstance(runs, list)
    assert len(runs) > 0
    first = runs[0]
    assert "run_id" in first
    assert "model_id" in first
    assert "metrics" in first
    assert "training_or_transfer_label" in first

    # 2. GET /api/composition/runs/{run_id}
    run_id = first["run_id"]
    detail_res = client.get(f"/api/composition/runs/{run_id}?subset=full")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["run_id"] == run_id

    # 3. GET /api/composition/runs/nonexistent -> 404
    missing_res = client.get("/api/composition/runs/nonexistent_run_id_404")
    assert missing_res.status_code == 404

    # 4. GET /api/composition/runs/{run_id}/queries
    queries_res = client.get(f"/api/composition/runs/{run_id}/queries?subset=full&limit=10")
    assert queries_res.status_code == 200
    q_data = queries_res.json()
    assert "items" in q_data
    assert "total" in q_data


def test_real_encoder_summary_metrics_normalization():
    import json
    from workbench.backend.operator_config import resolve_config
    from workbench.backend.composition import resolve_composition_path

    cfg = resolve_config()
    try:
        summary_dir = resolve_composition_path(cfg.WORKBENCH_ARTIFACTS_ROOT, "summary")
    except ValueError:
        pytest.skip("composition summary root unavailable")

    encoder_files = list(summary_dir.glob("encoder_*/full.json"))
    if not encoder_files:
        pytest.skip("real encoder composition summary absent")

    raw_data = json.loads(encoder_files[0].read_text(encoding="utf-8"))
    run_id = raw_data["run_id"]
    nested_m = raw_data.get("metrics", {})

    client = TestClient(app)
    res = client.get(f"/api/composition/runs/{run_id}?subset=full")
    assert res.status_code == 200
    detail = res.json()
    metrics = detail["metrics"]

    # Flat keys are non-null and match nested values
    assert metrics["ndcg_multimodal"] is not None
    assert metrics["ndcg_multimodal"] == nested_m["multimodal"]["ndcg_full"]

    assert metrics["recall_at_10_multimodal"] is not None
    assert metrics["recall_at_10_multimodal"] == nested_m["multimodal"]["recall_at_10"]

    assert metrics["ndcg_text_only"] is not None
    assert metrics["ndcg_text_only"] == nested_m["text_only"]["ndcg_full"]

    assert metrics["ndcg_image_only"] is not None
    assert metrics["ndcg_image_only"] == nested_m["image_only"]["ndcg_full"]

    assert metrics["coverage"] is not None
    assert metrics["coverage"] == nested_m["multimodal"]["coverage"]

    assert metrics["composition_gap_ndcg"] == nested_m["composition_gap_ndcg"]

    # Assert fixture-shaped summaries still render with flat keys
    list_res = client.get("/api/composition/runs?subset=full&include_fixtures=true")
    assert list_res.status_code == 200
    runs = list_res.json()
    fixtures = [r for r in runs if r.get("is_synthetic")]
    if fixtures:
        f_m = fixtures[0]["metrics"]
        assert f_m["ndcg_multimodal"] is not None
        assert f_m["recall_at_10_multimodal"] is not None
        assert f_m["ndcg_text_only"] is not None
        assert f_m["ndcg_image_only"] is not None
        assert f_m["composition_gap_ndcg"] is not None
        assert f_m["coverage"] is not None


def _real_summaries():
    """Every real, full (non-fixture, non-smoke) summary as (path, payload), or skip."""
    import json

    from workbench.backend.operator_config import resolve_config

    cfg = resolve_config()
    try:
        summary_dir = resolve_composition_path(cfg.WORKBENCH_ARTIFACTS_ROOT, "summary")
    except ValueError:
        pytest.skip("composition summary root unavailable")
    found = []
    for path in sorted(summary_dir.glob("*/full.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if (data.get("is_synthetic") or data.get("synthetic")
                or str(data.get("run_id") or "").startswith("SYNTHETIC_")):
            continue
        if isinstance(data.get("smoke"), dict) and data["smoke"].get("limited"):
            continue
        found.append((path, data))
    if not found:
        pytest.skip("real composition summary absent")
    return found


def _real_summary_file():
    """Path and payload of one real, full (non-fixture, non-smoke) summary, or skip."""
    return _real_summaries()[0]


def _has_numbers(run):
    """True when the row carries at least one non-null score."""
    metrics = run.get("metrics") or {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and (
                str(key).startswith("ndcg_") or str(key).startswith("recall_at_")):
            return True
        if isinstance(value, dict) and any(
                k in {"ndcg_full", "recall_at_10", "recall_at_50", "mrr"}
                and isinstance(v, (int, float)) and not isinstance(v, bool)
                for k, v in value.items()):
            return True
    return False


def _tier(run):
    """Sort tier the API guarantees: 0 measured, 1 failed/no-numbers, 2 smoke, 3 fixture."""
    if run.get("is_synthetic"):
        return 3
    if str(run.get("execution_status") or "").upper() == "FAILED" or not _has_numbers(run):
        return 1
    if (run.get("smoke") or {}).get("limited"):
        return 2
    return 0

def test_default_list_excludes_fixtures_and_ranks_real_first():
    real_ids = {data["run_id"] for _, data in _real_summaries()}
    client = TestClient(app)
    runs = client.get("/api/composition/runs?subset=full").json()
    ids = [r["run_id"] for r in runs]

    assert runs, "default composition list must not be empty when a real summary exists"
    assert not [r for r in runs if r.get("is_synthetic")], (
        "synthetic fixtures must stay hidden by default: "
        f"{[r['run_id'] for r in runs if r.get('is_synthetic')]}"
    )
    assert not [r for r in runs if (r.get("smoke") or {}).get("limited")], (
        "smoke-limited runs must stay hidden by default: "
        f"{[r['run_id'] for r in runs if (r.get('smoke') or {}).get('limited')]}"
    )
    # every returned row is a real measurement, and all real summaries are reachable
    assert set(ids) == real_ids, (
        f"default list must be exactly the real measurements; got {ids}, expected {sorted(real_ids)}"
    )
    assert [r["run_id"] for r in client.get("/api/composition/runs?subset=full").json()] == ids, (
        f"default list must be deterministic across calls, got {ids}"
    )

def test_include_fixtures_true_keeps_real_first():
    real_ids = {data["run_id"] for _, data in _real_summaries()}
    client = TestClient(app)
    url = "/api/composition/runs?subset=full&include_fixtures=true"
    runs = client.get(url).json()
    ids = [r["run_id"] for r in runs]

    fixtures = [r for r in runs if r.get("is_synthetic")]
    assert fixtures, "include_fixtures=true must still reach the layout fixtures"
    assert real_ids <= set(ids), f"real measurements must survive include_fixtures, got {ids}"
    first_fixture = next(i for i, r in enumerate(runs) if r.get("is_synthetic"))
    assert all(_tier(r) < 3 for r in runs[:first_fixture]), (
        f"every measurement must precede every fixture, got {ids}"
    )
    assert all(_tier(r) == 3 for r in runs[first_fixture:]), (
        f"no measurement may sort below a fixture, got {ids}"
    )
    assert [r["run_id"] for r in client.get(url).json()] == ids, (
        f"include_fixtures=true list must be deterministic across calls, got {ids}"
    )

def test_smoke_runs_sort_below_measurements():
    _real_summaries()  # skip cleanly when no real measurement exists
    client = TestClient(app)
    url = "/api/composition/runs?subset=full&include_smoke=true&include_fixtures=true"
    runs = client.get(url).json()
    ids = [r["run_id"] for r in runs]

    assert runs, "include_smoke=true list must not be empty"
    assert all(_tier(later) >= _tier(earlier)
               for earlier, later in zip(runs, runs[1:])), (
        f"tiers must be non-decreasing measured -> failed -> smoke -> fixture, got {ids}"
    )
    assert [r["run_id"] for r in client.get(url).json()] == ids, (
        f"include_smoke=true list must be deterministic across calls, got {ids}"
    )

def test_failed_runs_rank_below_measured_runs():
    real = _real_summaries()
    failed_runs = [
        data for _, data in real
        if str(data.get("execution_status") or "").upper() == "FAILED"
        or not _has_numbers({"metrics": data.get("metrics")})
    ]
    if not failed_runs:
        pytest.skip("no failed composition summary on disk")

    client = TestClient(app)
    runs = client.get("/api/composition/runs?subset=full").json()
    ids = [r["run_id"] for r in runs]

    failed_ids = {data["run_id"] for data in failed_runs}
    # a failed run is information: it stays visible in the default list
    assert failed_ids <= {r["run_id"] for r in runs}, (
        f"failed runs must stay visible in the default list, got {ids}"
    )
    measured = [i for i, r in enumerate(runs)
                if r["run_id"] not in failed_ids and _has_numbers(r)
                and str(r.get("execution_status") or "").upper() in {"COMPLETED", "PARTIAL", "REUSED"}]
    failed_positions = [i for i, r in enumerate(runs) if r["run_id"] in failed_ids]
    assert measured, f"expected at least one scored run to compare against, got {ids}"
    assert max(measured) < min(failed_positions), (
        "a run with no numbers must never outrank a scored run: "
        f"measured at {[ids[i] for i in measured]}, failed at {[ids[i] for i in failed_positions]}"
    )
    for i in failed_positions:
        assert not _has_numbers(runs[i]) or str(runs[i].get("execution_status") or "").upper() == "FAILED", (
            f"row ranked as failed still carries numbers: {ids[i]}"
        )

def _write_summary(root, run_dir, *, run_id, execution_status, metrics, smoke_limited=False):
    import json
    path = root / "composition" / "summary" / run_dir
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": run_dir.split("_")[0], "checkpoint_id": run_dir.split("_")[1],
        "run_id": run_id, "subset": "full",
        "conditions": ["multimodal", "text_only_black_image", "image_only_empty_text"],
        "execution_status": execution_status, "metrics": metrics,
        "smoke": {"limited": smoke_limited}, "is_synthetic": False,
        "condition_status": {c: {"status": execution_status} for c in
                             ("multimodal", "text_only_black_image", "image_only_empty_text")},
    }
    (path / "full.json").write_text(json.dumps(payload), encoding="utf-8")

_NULL_METRICS = {
    "ndcg_multimodal": None, "ndcg_text_only": None, "ndcg_image_only": None,
    "recall_at_10_multimodal": None, "composition_gap_ndcg": None, "coverage": 0.0,
    "multimodal": {"ndcg_full": None, "recall_at_10": None, "recall_at_50": None, "mrr": None},
}
_SCORED_METRICS = {
    "ndcg_multimodal": 0.29, "ndcg_text_only": 0.13, "ndcg_image_only": 0.09,
    "recall_at_10_multimodal": 30.0, "composition_gap_ndcg": 0.54, "coverage": 0.98,
    "multimodal": {"ndcg_full": 0.29, "recall_at_10": 30.0, "recall_at_50": 48.0, "mrr": 0.2},
}

def test_failed_runs_sort_after_scored_runs_regardless_of_run_id(tmp_path):
    """A run with no numbers must never sort as a measurement, even when its run_id sorts first."""
    from workbench.backend.composition import list_composition_summaries

    _write_summary(tmp_path, "aaa_model_failed", run_id="aaa_failed", execution_status="FAILED",
                   metrics=_NULL_METRICS)
    _write_summary(tmp_path, "aab_model_nullmetrics", run_id="aab_nullmetrics",
                   execution_status="COMPLETED", metrics=_NULL_METRICS)
    _write_summary(tmp_path, "zzz_model_scored", run_id="zzz_scored", execution_status="COMPLETED",
                   metrics=_SCORED_METRICS)
    _write_summary(tmp_path, "zzz_model_smoke", run_id="zzz_smoke", execution_status="COMPLETED",
                   metrics=_SCORED_METRICS, smoke_limited=True)

    ids = [r["run_id"] for r in list_composition_summaries(artifacts_root=tmp_path, subset="full",
                                                           include_fixtures=False,
                                                           include_smoke=True)]
    assert ids == ["zzz_scored", "aaa_failed", "aab_nullmetrics", "zzz_smoke"], (
        f"tier order must be scored -> failed/no-numbers -> smoke, got {ids}"
    )
    assert [r["run_id"] for r in list_composition_summaries(artifacts_root=tmp_path, subset="full",
                                                            include_fixtures=False,
                                                            include_smoke=True)] == ids, (
        "summary ordering must be deterministic"
    )

def test_composition_queries_nonexistent_run_404():
    client = TestClient(app)
    res = client.get("/api/composition/runs/no_such_run_xyz_bogus/queries?subset=full")
    assert res.status_code == 404
    data = res.json()
    assert data["detail"]["error"] == "composition_run_not_found"

    # And real run returns 200
    _, real = _real_summary_file()
    res_real = client.get(f"/api/composition/runs/{real['run_id']}/queries?subset=full")
    assert res_real.status_code == 200

def test_failed_run_exposes_is_failed_flag():
    """The FAILED row must be classifiable by the UI, so its zero-coverage cell blanks."""
    failed_run_id = "e0dd04a46ef7a58334b9e353"
    client = TestClient(app)
    runs = client.get("/api/composition/runs?subset=full").json()
    failed = [r for r in runs if r["run_id"] == failed_run_id]
    if not failed:
        pytest.skip(f"failed composition run {failed_run_id} absent")
    row = failed[0]
    assert row["execution_status"] == "FAILED"
    assert row["is_failed"] is True
    # coverage stays the computed value; only its rendering is blanked
    assert row["metrics"]["coverage"] == 0.0

    detail = client.get(f"/api/composition/runs/{failed_run_id}?subset=full").json()
    assert detail["is_failed"] is True

    scored = [r for r in runs if r["execution_status"] in {"COMPLETED", "PARTIAL", "REUSED"}]
    assert scored, "expected at least one scored run"
    assert all(r["is_failed"] is False for r in scored)

def test_failure_message_never_shows_python_none_sentinel():
    """Unresolved categories must read as prose, in new and legacy failure records alike."""
    from workbench.backend.composition import readable_failure_message

    legacy = ("query 7:2179: category None has no adapter embedding scope ('dress', 'shirt', 'toptee'); "
              "batch_indices=(2178,), qids=['7:2179']")
    readable = readable_failure_message(legacy)
    assert "None" not in readable
    assert "category unresolved has no adapter embedding scope ('dress', 'shirt', 'toptee')" in readable
    assert "query 7:2179" in readable and "batch_indices=(2178,)" in readable
    assert readable_failure_message(None) is None
    assert readable_failure_message("") == ""
    assert readable_failure_message("query q1: query features unavailable") == "query q1: query features unavailable"
