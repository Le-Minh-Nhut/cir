from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from workbench.backend.analysis import analysis_rows, require_same_protocol
from workbench.backend.index import rebuild_index
from workbench.backend.registry import load_registry
from workbench.backend.adapters.base import EvalRequest
from workbench.backend.adapters.models import ADAPTERS
from workbench.backend.schemas.results import ResultRun

FIXTURES = Path(__file__).parent / "fixtures" / "results"


def test_registry_keeps_noise_dimensions_separate() -> None:
    registry = load_registry()
    airknow = next(model for model in registry["models"] if model["model_id"] == "airknow")
    assert airknow["checkpoint_variants"][0]["checkpoint_training_noise_pct"] == 50
    assert airknow["checkpoint_variants"][0]["evaluation_noise_pct"] == 0


def test_query_identity_ignores_model_text() -> None:
    left = ResultRun.model_validate_json((FIXTURES / "original_split_model_a.json").read_text())
    right = ResultRun.model_validate_json((FIXTURES / "original_split_model_b.json").read_text())
    assert left.queries[0].query_id == right.queries[0].query_id
    assert left.queries[0].model_input_text != right.queries[0].model_input_text


def test_schema_rejects_non_contiguous_ranks() -> None:
    payload = json.loads((FIXTURES / "original_split_model_a.json").read_text())
    payload["queries"][0]["top_results"][1]["rank"] = 3
    with pytest.raises(ValueError, match="contiguous"):
        ResultRun.model_validate(payload)


def test_cross_protocol_analysis_fails_closed() -> None:
    original = ResultRun.model_validate_json((FIXTURES / "original_split_model_a.json").read_text())
    reduced = ResultRun.model_validate_json((FIXTURES / "val_split_model_a.json").read_text())
    with pytest.raises(ValueError, match="Cross-protocol"):
        require_same_protocol([original, reduced])


def test_analysis_exposes_common_distractor() -> None:
    runs = [ResultRun.model_validate_json((FIXTURES / name).read_text()) for name in ("original_split_model_a.json", "original_split_model_b.json")]
    rows = analysis_rows(runs, 10)
    assert next(row for row in rows if row["query_id"].startswith("toptee"))["common_distractors"][0]["image_id"] == "wrong-common"


def test_rebuild_index_is_deterministic(tmp_path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    for fixture in FIXTURES.glob("*.json"):
        (results / fixture.name).write_text(fixture.read_text())
    database = tmp_path / "workbench.duckdb"
    assert rebuild_index(results, database) == 4
    assert rebuild_index(results, database) == 4


def test_downloader_dry_run_never_creates_artifact(tmp_path) -> None:
    command = ["python", "workbench/scripts/download_checkpoints.py", "--model", "csmcir", "--dry-run", "--output-root", str(tmp_path)]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    assert "DRY RUN" in result.stdout
    assert not list(tmp_path.rglob("*.pt"))


def test_adapters_expose_explicit_protocol_compatibility(tmp_path) -> None:
    request = EvalRequest("csmcir", "fashioniq", "fashioniq_val_split", tmp_path, tmp_path / "output.json")
    with pytest.raises(ValueError, match="does not support"):
        ADAPTERS["csmcir"]().validate_request(request)
    assert {"csmcir", "encoder", "hint", "pair", "airknow", "conesep", "habit", "intent"} <= set(ADAPTERS)
