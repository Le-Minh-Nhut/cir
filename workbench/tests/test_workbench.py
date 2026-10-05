from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from workbench.backend.analysis import aligned_queries, analysis_rows, cohort_metrics, require_top_k
from workbench.backend.errors import WorkbenchError
from workbench.backend.index import load_runs, rebuild_index
from workbench.backend.main import app, resolve_image
from workbench.backend.registry import checkpoint_availability, load_registry
from workbench.backend.schemas.results import ResultRun

FIXTURES = Path(__file__).parent / "fixtures" / "results"


def fixture(name: str) -> ResultRun:
    return ResultRun.model_validate_json((FIXTURES / name).read_text())


def originals() -> list[ResultRun]:
    return [fixture("original_model_a.json"), fixture("original_model_b.json")]


def copied_results(tmp_path: Path, names: list[str] | None = None) -> Path:
    root = tmp_path / "results"; root.mkdir()
    for source in (names or ["original_model_a.json", "original_model_b.json", "val_model_a.json", "val_model_b.json"]):
        (root / source).write_text((FIXTURES / source).read_text())
    return root


def test_registry_keeps_noise_dimensions_and_source_dir_separate() -> None:
    registry = load_registry()
    airknow = next(model for model in registry["models"] if model["model_id"] == "airknow")
    assert airknow["checkpoint_variants"][0]["checkpoint_training_noise_pct"] == 50
    assert airknow["checkpoint_variants"][0]["evaluation_noise_pct"] == 0
    assert airknow["source_dir"] == "AirKnow"


def test_exact_query_alignment_allows_model_text_difference() -> None:
    left, right = originals()
    assert left.queries[0].model_input_text != right.queries[0].model_input_text
    assert len(aligned_queries([left, right])) == 24

@pytest.mark.parametrize("mutation,code", [("missing", "query_alignment_mismatch"), ("extra", "query_alignment_mismatch"), ("target", "canonical_query_mismatch"), ("reference", "canonical_query_mismatch")])
def test_query_alignment_fails_closed(mutation: str, code: str) -> None:
    left, right = originals()
    if mutation == "missing": right.queries.pop()
    if mutation == "extra": right.queries[0].query_id = "dress:999:other:other"
    if mutation == "target": right.queries[0].target_id = "wrong-target"
    if mutation == "reference": right.queries[0].reference_id = "wrong-reference"
    with pytest.raises(WorkbenchError) as raised: aligned_queries([left, right])
    assert raised.value.code == code


def test_run_id_identity_keeps_checkpoint_variants_distinct() -> None:
    left, right = originals()
    assert left.run.model_id == right.run.model_id == "mock_habit"
    rows = analysis_rows([left, right], 200)
    assert set(rows[0]["target_rank"]) == {left.run.run_id, right.run.run_id}


def test_top_k_200_supported_and_insufficient_depth_fails() -> None:
    left, right = originals()
    assert len(analysis_rows([left, right], 200)) == 24
    left.run.top_k_saved = 50
    with pytest.raises(WorkbenchError, match="enough top-K") as raised: require_top_k([left, right], 100)
    assert raised.value.code == "insufficient_top_k_depth"


def test_target_rank_analysis_needs_no_top_k_when_not_retrieving_lists() -> None:
    metrics = cohort_metrics(originals(), [query.query_id for query in originals()[0].queries])
    assert metrics[0]["r100"] is not None
    assert metrics[0]["query_count"] == 24


def test_schema_v2_rejects_invalid_saved_depth() -> None:
    payload = json.loads((FIXTURES / "original_model_a.json").read_text())
    payload["run"]["top_k_saved"] = 199
    with pytest.raises(ValueError, match="top_k_saved"): ResultRun.model_validate(payload)


def test_duplicate_run_id_is_rejected_with_sources(tmp_path: Path) -> None:
    root = copied_results(tmp_path, ["original_model_a.json"])
    (root / "second.json").write_text((FIXTURES / "original_model_a.json").read_text())
    with pytest.raises(WorkbenchError) as raised: load_runs(root)
    assert raised.value.code == "duplicate_run_id"
    assert "first_source" in raised.value.details


def test_rebuild_is_deterministic_and_failure_keeps_existing_index(tmp_path: Path) -> None:
    root = copied_results(tmp_path, ["original_model_a.json", "original_model_b.json"])
    database = tmp_path / "workbench.duckdb"
    assert rebuild_index(root, database) == 2
    original = database.read_bytes()
    (root / "bad.json").write_text("not json")
    with pytest.raises(Exception): rebuild_index(root, database)
    assert database.read_bytes() == original


def test_manual_checkpoint_is_runnable_without_automatic_download(monkeypatch, tmp_path: Path) -> None:
    model = next(item for item in load_registry()["models"] if item["model_id"] == "encoder")
    checkpoint = model["checkpoint_variants"][0]
    path = tmp_path / "checkpoints" / "encoder" / checkpoint["filename"]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"manual")
    (tmp_path / "third_party" / "ENCODER").mkdir(parents=True)
    monkeypatch.setattr("workbench.backend.registry.CHECKPOINT_ROOT", tmp_path / "checkpoints")
    monkeypatch.setattr("workbench.backend.registry.ROOT", tmp_path)
    status = checkpoint_availability(model, checkpoint)
    assert status["checkpoint_downloaded"] and not status["automatic_download_available"]
    assert status["runnable"]

def test_image_resolver_supports_png_jpg_and_rejects_traversal(tmp_path: Path) -> None:
    root = tmp_path / "FashionIQ"; (root / "images").mkdir(parents=True)
    (root / "images" / "png-id.png").write_bytes(b"x")
    (root / "dress").mkdir(); (root / "dress" / "jpg-id.jpg").write_bytes(b"x")
    assert resolve_image("dress", "png-id", root).suffix == ".png"
    assert resolve_image("dress", "jpg-id", root).suffix == ".jpg"
    assert resolve_image("dress", "../secret", root) is None


def test_annotation_update_cohort_metrics_and_cross_protocol_guard(monkeypatch, tmp_path: Path) -> None:
    import workbench.backend.main as main
    root = copied_results(tmp_path)
    database = tmp_path / "workbench.duckdb"; rebuild_index(root, database)
    monkeypatch.setattr("workbench.backend.storage.DATABASE_PATH", database)
    monkeypatch.setattr(main, "ANNOTATIONS_PATH", tmp_path / "annotations.json")
    monkeypatch.setattr(main, "COHORTS_PATH", tmp_path / "cohorts.json")
    client = TestClient(app)
    query = originals()[0].queries[0].query_id
    body = {"query_id":query,"protocol_id":"fashioniq_original_split","labels":["under_edit"],"note":"first"}
    assert client.put("/api/annotations", json=body).status_code == 200
    body["note"] = "updated"; assert client.put("/api/annotations", json=body).status_code == 200
    assert client.get(f"/api/annotations?query_id={query}&protocol_id=fashioniq_original_split").json()[0]["note"] == "updated"
    cohort = {"cohort_id":"mock-cohort","protocol_id":"fashioniq_original_split","query_ids":[query],"run_ids":["mock-habit-fiql-n02","mock-habit-fiql-n05"],"definition":{},"notes":"mock"}
    assert client.put("/api/cohorts", json=cohort).status_code == 200
    assert "R@10" in client.get("/api/cohorts/mock-cohort/export/markdown").json()["content"]
    cohort["run_ids"].append("mock-pair-b1")
    response = client.put("/api/cohorts", json=cohort)
    assert response.status_code == 400 and response.json()["error"] == "cross_protocol"


def test_queries_are_paginated_and_structured_analysis_errors(monkeypatch, tmp_path: Path) -> None:
    root = copied_results(tmp_path)
    database = tmp_path / "workbench.duckdb"; rebuild_index(root, database)
    monkeypatch.setattr("workbench.backend.storage.DATABASE_PATH", database)
    client = TestClient(app)
    response = client.get("/api/queries?run_id=mock-habit-fiql-n02&limit=5&offset=5")
    assert response.status_code == 200 and response.json()["total"] == 24 and len(response.json()["items"]) == 5
    response = client.get("/api/queries?run_id=mock-habit-fiql-n02&query_id=dress%3A0%3Afas-ref-00%3Afas-target-00&limit=1")
    assert response.status_code == 200 and response.json()["total"] == 1
    response = client.get("/api/analysis?run_ids=mock-habit-fiql-n02,mock-pair-b1&k=10")
    assert response.status_code == 400 and response.json()["error"] == "cross_protocol"


def test_downloader_resume_modes(tmp_path: Path, monkeypatch) -> None:
    spec = importlib.util.spec_from_file_location("downloader", Path("workbench/scripts/download_checkpoints.py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Response:
        def __init__(self, status: int, content_range: str | None, data: bytes) -> None:
            self.status, self.headers, self.data = status, {"Content-Range": content_range} if content_range else {}, data
        def getcode(self) -> int: return self.status
        def read(self, size: int = -1) -> bytes:
            data, self.data = self.data, b""
            return data
        def __enter__(self): return self
        def __exit__(self, *_): return None

    partial = tmp_path / "weight.pt.part"
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda _: Response(200, None, b"fresh"))
    module.download("https://example.invalid/weight", partial)
    assert partial.read_bytes() == b"fresh"
    partial.write_bytes(b"old-")
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda _: Response(206, "bytes 4-7/8", b"tail"))
    module.download("https://example.invalid/weight", partial)
    assert partial.read_bytes() == b"old-tail"
    partial.write_bytes(b"stale")
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda _: Response(200, None, b"replacement"))
    module.download("https://example.invalid/weight", partial)
    assert partial.read_bytes() == b"replacement"
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda _: Response(206, "bytes 1-3/4", b"bad"))
    with pytest.raises(RuntimeError, match="Content-Range"):
        module.download("https://example.invalid/weight", partial)
