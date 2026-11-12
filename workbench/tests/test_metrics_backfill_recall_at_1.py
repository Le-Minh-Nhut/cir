"""The R@1 backfill reads the rank table, refuses a cohort mismatch, and refuses a summary write."""
from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict
import sys
from pathlib import Path

import pytest

from workbench.composition.metrics import compose

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

def _load():
    spec = importlib.util.spec_from_file_location("backfill_recall_at_1_test",
                                                  SCRIPTS / "backfill_recall_at_1.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

RANKS = {
    "multimodal": {"7:1": 1, "7:2": 2, "7:3": None},
    "text_only_black_image": {"7:1": 4, "7:2": 1, "7:3": 9},
    "image_only_empty_text": {"7:1": None, "7:2": None, "7:3": None},
}

def _fixture(tmp_path: Path, ranks=None, query_count: int = 3) -> Path:
    composition = tmp_path / "artifacts" / "composition"
    (composition / "runs" / "abc123" / "ranks").mkdir(parents=True)
    (composition / "summary" / "fixture_model_ckpt_abc123").mkdir(parents=True)
    ranks = RANKS if ranks is None else ranks
    (composition / "runs" / "abc123" / "ranks" / "fixture_model_ckpt.json").write_text(
        json.dumps({"attempt": "abc123", "ranks": ranks,
                    "condition_status": {"multimodal": {"status": "COMPLETED"}}}), encoding="utf-8")
    published = {"metrics": json.loads(json.dumps(asdict(compose(ranks, query_count=query_count))))}
    published["model_id"], published["checkpoint_id"] = "fixture_model", "ckpt"
    published["run_id"], published["subset"] = "abc123", "full"
    published["conditions"] = list(ranks)
    published["query_count"] = query_count
    (composition / "summary" / "fixture_model_ckpt_abc123" / "full.json").write_text(
        json.dumps(published), encoding="utf-8")
    return composition

def test_backfill_emits_r1_with_a_consistency_guard(tmp_path: Path) -> None:
    module = _load()
    composition = _fixture(tmp_path)
    record = module.derive(composition, "fixture_model", "abc123")
    assert record["conditions"]["multimodal"]["recall_at_1"] == 50.0
    assert record["conditions"]["multimodal"]["rank_1_count"] == 1
    assert record["conditions"]["text_only_black_image"]["recall_at_1"] == pytest.approx(100 / 3)
    assert record["conditions"]["image_only_empty_text"]["recall_at_1"] is None
    assert record["conditions"]["image_only_empty_text"]["unavailable_reason"] == \
        "no queries have an exact rank"
    assert "R@1 <= R@10 <= R@50" in record["consistency"]
    assert record["derived_from"]["ranks"]["sha256"]

    output = composition / "derived" / "recall_at_1" / "out.json"
    assert module.main(["--artifacts-root", str(tmp_path / "artifacts"),
                        "--run", "fixture_model:abc123", "--output", str(output)]) == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["conditions"]["multimodal"]["recall_at_1"] == 50.0


def test_backfill_refuses_when_the_summary_is_not_reproducible(tmp_path: Path) -> None:
    module = _load()
    composition = _fixture(tmp_path)
    summary_path = composition / "summary" / "fixture_model_ckpt_abc123" / "full.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["metrics"]["multimodal"]["recall_at_10"] = 99.0
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(SystemExit, match="not reproducible"):
        module.derive(composition, "fixture_model", "abc123")


def test_backfill_refuses_a_cohort_mismatch(tmp_path: Path) -> None:
    module = _load()
    composition = _fixture(tmp_path, query_count=4)
    with pytest.raises(SystemExit, match="refusing to compare cohorts"):
        module.derive(composition, "fixture_model", "abc123")


def test_backfill_refuses_to_write_into_a_published_summary(tmp_path: Path) -> None:
    module = _load()
    _fixture(tmp_path)
    target = tmp_path / "artifacts" / "composition" / "summary" / "fixture_model_ckpt_abc123" / "full.json"
    with pytest.raises(SystemExit, match="published summary"):
        module.main(["--artifacts-root", str(tmp_path / "artifacts"),
                     "--run", "fixture_model:abc123", "--output", str(target)])
