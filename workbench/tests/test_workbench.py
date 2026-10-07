from __future__ import annotations

import importlib.util
import subprocess
import urllib.error
from pathlib import Path

import pytest

from workbench.backend.index import rebuild_index
from workbench.backend.analysis import aligned_queries, analysis_rows
from workbench.backend.errors import WorkbenchError
from workbench.backend import main
from workbench.backend.main import resolve_image
from workbench.backend.schemas.results import ResultRun
from workbench.backend.storage import get_analysis_input, get_run, list_runs, query_page
from workbench.tests.mock_data import build_mock_runs


def write_results(root: Path) -> None:
    for result in build_mock_runs():
        path = root / f"{result.run.run_id}.json"
        path.write_text(result.model_dump_json())


def indexed(monkeypatch, tmp_path: Path) -> Path:
    root = tmp_path / "results"; root.mkdir(); write_results(root)
    database = tmp_path / "workbench.duckdb"; rebuild_index(root, database)
    monkeypatch.setattr("workbench.backend.storage.DATABASE_PATH", database)
    return database


def test_mock_generation_is_deterministic_and_schema_v2_valid() -> None:
    left, right = build_mock_runs(), build_mock_runs()
    assert [item.model_dump_json() for item in left] == [item.model_dump_json() for item in right]
    assert len(left) == 8 and all(item.schema_version == 2 for item in left)
    assert {query.category for query in left[0].queries} == {"dress", "shirt", "toptee"}
    assert {item.run.literature_split_label for item in left if item.run.protocol_id != "fashioniq_val_split"} == {"original"}
    assert max(query.target_rank for query in left[0].queries) > 200


def test_schema_rejects_top_results_beyond_gallery() -> None:
    payload = build_mock_runs()[0].model_dump()
    payload["run"]["gallery_size"] = 199
    with pytest.raises(ValueError, match="effective saved retrieval depth"):
        ResultRun.model_validate(payload)


def test_schema_rejects_literature_label_mismatch() -> None:
    payload = build_mock_runs()[0].model_dump()
    payload["run"]["literature_split_label"] = "val"

    with pytest.raises(ValueError, match="literature_split_label does not match protocol_id"):
        ResultRun.model_validate(payload)


def test_schema_preserves_repeated_reference_target_annotations() -> None:
    payload = build_mock_runs()[0].model_dump()
    template = payload["queries"][0].copy()
    template["annotation_index"] = 99
    template["query_id"] = f"{template['category']}:99:{template['reference_id']}:{template['target_id']}"
    payload["queries"].append(template)

    result = ResultRun.model_validate(payload)

    assert len(result.queries) == 25


def test_sql_list_runs_reads_metadata(monkeypatch, tmp_path: Path) -> None:
    database = indexed(monkeypatch, tmp_path)
    runs = list_runs(database_path=database)
    assert len(runs) == 8
    assert {run["run_id"] for run in runs} == {item.run.run_id for item in build_mock_runs()}
    assert all("queries" not in run for run in runs)


def test_sql_query_page_filters_and_exact_lookup(monkeypatch, tmp_path: Path) -> None:
    database = indexed(monkeypatch, tmp_path)
    run_id = build_mock_runs()[0].run.run_id
    page = query_page(run_id, category="dress", failed_at=10, min_rank=11, max_rank=250, limit=2, offset=0, database_path=database)
    assert page["total"] and len(page["items"]) == 2
    assert all(item["category"] == "dress" and 11 <= item["target_rank"] <= 250 for item in page["items"])
    query_id = build_mock_runs()[0].queries[0].query_id
    exact = query_page(run_id, query_id=query_id, limit=1, database_path=database)
    assert exact["total"] == 1 and exact["items"][0]["query_id"] == query_id


def test_get_run_returns_metadata_and_rejects_missing_run(monkeypatch, tmp_path: Path) -> None:
    database = indexed(monkeypatch, tmp_path)
    assert get_run("mock_csmcir-fiq_a", database)["checkpoint_id"] == "fiq_a"
    with pytest.raises(WorkbenchError) as error:
        get_run("missing-run", database)
    assert error.value.code == "run_not_found"


def test_queries_reject_missing_run(monkeypatch, tmp_path: Path) -> None:
    indexed(monkeypatch, tmp_path)
    with pytest.raises(WorkbenchError) as error:
        main.queries("missing-run")
    assert error.value.code == "run_not_found"


def indexed_with_saved_depth(monkeypatch, tmp_path: Path, depth: int) -> Path:
    root = tmp_path / "results"; root.mkdir()
    for result in build_mock_runs()[:2]:
        payload = result.model_dump()
        payload["run"]["top_k_saved"] = depth
        for query in payload["queries"]:
            query["top_results"] = query["top_results"][:depth]
        (root / f"{payload['run']['run_id']}.json").write_text(ResultRun.model_validate(payload).model_dump_json())
    database = tmp_path / "workbench.duckdb"; rebuild_index(root, database)
    monkeypatch.setattr("workbench.backend.storage.DATABASE_PATH", database)
    return database


def test_compare_rejects_unavailable_top_k(monkeypatch, tmp_path: Path) -> None:
    indexed_with_saved_depth(monkeypatch, tmp_path, 50)
    query_id = build_mock_runs()[0].queries[0].query_id
    with pytest.raises(WorkbenchError) as error:
        main.compare(query_id, "mock_csmcir-fiq_a,mock_csmcir-fiq_b", 100)
    assert error.value.code == "insufficient_top_k_depth"


def test_failure_jaccard_uses_exact_target_rank_without_top_k(monkeypatch, tmp_path: Path) -> None:
    indexed_with_saved_depth(monkeypatch, tmp_path, 50)
    rows = main.overlap("mock_csmcir-fiq_a,mock_csmcir-fiq_b", 100)
    assert len(rows) == 4 and rows[0]["jaccard"] >= 0


def test_exact_alignment_is_map_based_and_fails_closed() -> None:
    left, right = build_mock_runs()[:2]
    assert len(aligned_queries([left, right])) == 24
    right.queries.pop()
    with pytest.raises(WorkbenchError, match="identical query universe"):
        aligned_queries([left, right])


def test_common_distractor_has_distinct_run_provenance() -> None:
    rows = analysis_rows(build_mock_runs()[:2], 10)
    distractor = rows[0]["common_distractors"][0]
    assert distractor["run_count"] == 2
    assert distractor["run_ids"] == ["mock_csmcir-fiq_a", "mock_csmcir-fiq_b"]


def test_duplicate_run_id_and_atomic_rebuild(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "results"; root.mkdir(); write_results(root)
    database = tmp_path / "workbench.duckdb"; rebuild_index(root, database); prior = database.read_bytes()
    (root / "duplicate.json").write_text((root / "mock_csmcir-fiq_a.json").read_text())
    with pytest.raises(WorkbenchError, match="Duplicate run_id"): rebuild_index(root, database)
    assert database.read_bytes() == prior


def test_csmcir_cannot_share_analysis_with_full_gallery_reference_excluded_models(monkeypatch, tmp_path: Path) -> None:
    indexed(monkeypatch, tmp_path)
    response = main.queries("mock_csmcir-fiq_a", limit=5, offset=5, category="dress", min_rank=1, max_rank=250)
    assert len(response["items"]) <= 5
    for model_id in ("mock_airknow", "mock_conesep", "mock_habit", "mock_intent"):
        with pytest.raises(WorkbenchError, match="Cross-protocol analysis is invalid"):
            main.analysis(f"mock_csmcir-fiq_a,{model_id}-fiq_n05", 10)
    runs, maps = get_analysis_input(["mock_csmcir-fiq_a", "mock_csmcir-fiq_b"], 200)
    assert len(runs) == 2 and len(maps["mock_csmcir-fiq_a"]) == 24


def test_image_resolver_supports_extensions_and_rejects_traversal(tmp_path: Path) -> None:
    root = tmp_path / "FashionIQ"; (root / "images").mkdir(parents=True); (root / "images" / "png.png").write_bytes(b"x"); (root / "dress").mkdir(); (root / "dress" / "jpg.jpg").write_bytes(b"x")
    assert resolve_image("dress", "png", root).suffix == ".png"
    assert resolve_image("dress", "jpg", root).suffix == ".jpg"
    assert resolve_image("dress", "../secret", root) is None


def test_downloader_lists_and_skips_unresolved_without_network(tmp_path: Path) -> None:
    listed = subprocess.run(["python", "workbench/scripts/download_checkpoints.py", "--list"], text=True, capture_output=True, check=True)
    assert "pair_b1" in listed.stdout and "pair_b2" in listed.stdout and "CHECKPOINT_MAPPING_UNVERIFIED" in listed.stdout
    bulk = subprocess.run(["python", "workbench/scripts/download_checkpoints.py", "--all", "--dry-run", "--output-root", str(tmp_path)], text=True, capture_output=True, check=True)
    assert "SKIPPED: unresolved checkpoint mapping (pair / pair_b1)" in bulk.stdout
    assert "SKIPPED: direct official download URL unavailable (encoder / fashioniq)" in bulk.stdout
    assert not list(tmp_path.rglob("*.pt"))


def test_downloader_handles_fresh_and_resumed_responses(tmp_path: Path, monkeypatch) -> None:
    spec = importlib.util.spec_from_file_location("downloader", Path("workbench/scripts/download_checkpoints.py")); assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    class Response:
        def __init__(self, status: int, content_range: str | None, data: bytes): self.status,self.headers,self.data=status,{"Content-Range":content_range} if content_range else {},data
        def getcode(self): return self.status
        def read(self, _: int = -1): data,self.data=self.data,b""; return data
        def __enter__(self): return self
        def __exit__(self,*_): return None
    partial = tmp_path / "weight.part"
    monkeypatch.setattr(module.urllib.request,"urlopen",lambda _:Response(200,None,b"fresh")); module.download("https://example.test/weight",partial); assert partial.read_bytes()==b"fresh"
    partial.write_bytes(b"old-"); monkeypatch.setattr(module.urllib.request,"urlopen",lambda _:Response(206,"bytes 4-7/8",b"tail")); module.download("https://example.test/weight",partial); assert partial.read_bytes()==b"old-tail"


def test_verifier_distinguishes_local_and_official_hashes(tmp_path: Path, capsys) -> None:
    spec = importlib.util.spec_from_file_location("downloader_verify", Path("workbench/scripts/download_checkpoints.py")); assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    destination = tmp_path / "checkpoint.pt"; destination.write_bytes(b"checkpoint")
    assert module.verify({}, {"expected_sha256": None}, destination)
    assert "LOCAL SHA256 COMPUTED — official SHA256 unavailable" in capsys.readouterr().out
    assert module.verify({}, {"expected_sha256": module.sha256_file(destination)}, destination)
    assert "OFFICIAL SHA256 MATCH" in capsys.readouterr().out
    assert not module.verify({}, {"expected_sha256": "0" * 64}, destination)
    assert "OFFICIAL SHA256 MISMATCH" in capsys.readouterr().out



def test_downloader_retries_cleanly_after_416(tmp_path: Path, monkeypatch) -> None:
    spec = importlib.util.spec_from_file_location("downloader_retry", Path("workbench/scripts/download_checkpoints.py")); assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    class Response:
        headers: dict[str, str] = {}
        def __init__(self): self.pending = b"replacement"
        def getcode(self): return 200
        def read(self, _: int = -1): data, self.pending = self.pending, b""; return data
        def __enter__(self): return self
        def __exit__(self,*_): return None
    partial = tmp_path / "weight.part"; partial.write_bytes(b"stale")
    calls = 0
    def urlopen(_):
        nonlocal calls; calls += 1
        if calls == 1: raise urllib.error.HTTPError("https://example.test/weight", 416, "range", {}, None)
        return Response()
    monkeypatch.setattr(module.urllib.request, "urlopen", urlopen)
    module.download("https://example.test/weight", partial)
    assert calls == 2 and partial.read_bytes() == b"replacement"
