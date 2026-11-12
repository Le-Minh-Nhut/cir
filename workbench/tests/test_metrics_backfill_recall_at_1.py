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

VARIANT_RANKS = {
    "multimodal": {"7:1": 1, "7:2": 2, "7:3": None},
    "text_only_black_image": {"7:1": 4, "7:2": 8, "7:3": 3},
    "clvc_image_encoder_only": {"7:1": 1, "7:2": 9, "7:3": 6},
}
VARIANT_TRIPLE = ("multimodal", "text_only_black_image", "clvc_image_encoder_only")

def _variant_fixture(tmp_path: Path, *, drop_gap_conditions: bool = False,
                     extra_condition: str | None = None) -> Path:
    """A summary whose image role is the ablation, i.e. the shape `clvc_net` now publishes."""
    composition = _fixture(tmp_path, ranks=VARIANT_RANKS)
    summary_path = composition / "summary" / "fixture_model_ckpt_abc123" / "full.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["metrics"] = json.loads(json.dumps(
        asdict(compose(VARIANT_RANKS, query_count=3, conditions=VARIANT_TRIPLE))))
    if drop_gap_conditions:
        del summary["metrics"]["gap_conditions"]
    summary["conditions"] = list(VARIANT_RANKS)
    if extra_condition is not None:
        summary["conditions"] = summary["conditions"] + [extra_condition]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return composition

def test_a_variant_triple_is_mapped_from_gap_conditions_and_agrees_with_the_summary(tmp_path: Path) -> None:
    """The image role's numbers live in `metrics.image_only`, whatever condition filled the role."""
    module = _load()
    composition = _variant_fixture(tmp_path)
    record = module.derive(composition, "fixture_model", "abc123")
    assert record["role_mapping"]["source"] == "gap_conditions"
    assert record["role_mapping"]["mapping"]["clvc_image_encoder_only"] == "image_only"
    image = record["conditions"]["clvc_image_encoder_only"]
    assert image["metric_block"] == "image_only"
    assert image["recall_at_1"] == pytest.approx(100 / 3)
    assert image["rank_1_count"] == 1
    assert image["already_present"] is True
    assert image["agrees_with_summary"] is True
    summary = json.loads((composition / "summary" / "fixture_model_ckpt_abc123" / "full.json")
                         .read_text(encoding="utf-8"))
    for condition, block in record["conditions"].items():
        assert block["summary_recall_at_1"] == \
            summary["metrics"][block["metric_block"]]["recall_at_1"], condition
        assert block["recall_at_1"] == block["summary_recall_at_1"], condition
    assert "role" in record["role_mapping"]["explanation"]

def test_a_summary_without_gap_conditions_falls_back_to_the_protocol_triple(tmp_path: Path) -> None:
    """Older artifacts keep working, and the record says which path was taken."""
    module = _load()
    composition = _variant_fixture(tmp_path, drop_gap_conditions=True)
    summary_path = composition / "summary" / "fixture_model_ckpt_abc123" / "full.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    # Without gap_conditions the role blocks still exist; the legacy map names the protocol
    # conditions, so the ablation is unplaceable and must be REFUSED, not skipped.
    with pytest.raises(SystemExit, match="cannot place condition"):
        module.derive(composition, "fixture_model", "abc123")

    # The same artifact with the protocol names is placeable and takes the legacy path.
    summary["conditions"] = ["multimodal", "text_only_black_image", "image_only_empty_text"]
    summary["metrics"]["image_only"]["condition"] = "image_only_empty_text"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    ranks_path = composition / "runs" / "abc123" / "ranks" / "fixture_model_ckpt.json"
    payload = json.loads(ranks_path.read_text(encoding="utf-8"))
    payload["ranks"]["image_only_empty_text"] = payload["ranks"].pop("clvc_image_encoder_only")
    ranks_path.write_text(json.dumps(payload), encoding="utf-8")
    record = module.derive(composition, "fixture_model", "abc123")
    assert record["role_mapping"]["source"] == "legacy_protocol_triple"
    assert record["role_mapping"]["mapping"]["image_only_empty_text"] == "image_only"
    assert record["conditions"]["image_only_empty_text"]["metric_block"] == "image_only"

def test_an_unmappable_condition_is_refused_by_name(tmp_path: Path) -> None:
    module = _load()
    composition = _variant_fixture(tmp_path, extra_condition="mystery_condition")
    with pytest.raises(SystemExit, match=r"cannot place condition\(s\) \['mystery_condition'\]"):
        module.derive(composition, "fixture_model", "abc123")

def test_a_contradicting_r1_in_the_summary_refuses_the_backfill(tmp_path: Path) -> None:
    """A derived file that disagrees with the artifact is worse than none."""
    module = _load()
    composition = _variant_fixture(tmp_path)
    summary_path = composition / "summary" / "fixture_model_ckpt_abc123" / "full.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["metrics"]["image_only"]["recall_at_1"] = 99.0
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(SystemExit, match="not reproducible"):
        module.derive(composition, "fixture_model", "abc123")

def test_a_summary_whose_gap_conditions_disagree_with_its_blocks_is_refused(tmp_path: Path) -> None:
    module = _load()
    composition = _variant_fixture(tmp_path)
    summary_path = composition / "summary" / "fixture_model_ckpt_abc123" / "full.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["metrics"]["image_only"]["condition"] = "some_other_condition"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(SystemExit, match="not reproducible"):
        module.derive(composition, "fixture_model", "abc123")
