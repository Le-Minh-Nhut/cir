"""Verification tests for F05, F08, and F09.

F05: FashionIQ constructor needs TRAIN data, but validators don't require it.
F08: DCNet dataset_root passed to adapter vs native data_dir resolution.
F09: DCNet metric parsing for native evaluator output.
"""
from __future__ import annotations
import json
import re
import sys
import tempfile
from pathlib import Path
import pytest
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workbench.backend.fashioniq_layout import CATEGORIES, ilearn_resized_training_paths
from workbench.backend.metrics_extraction import MetricExtractionResult, _check_parity, _is_valid_recall


# ==============================================================================
# F05 Tests: FashionIQ constructor train-data requirement & validation
# ==============================================================================

def test_f05_hint_constructor_reads_train_captions_for_all_categories():
    hint_datasets_path = REPO_ROOT / "workbench" / "third_party" / "HINT" / "datasets.py"
    content = hint_datasets_path.read_text(encoding="utf-8")
    assert "def train_init_process(self):" in content
    assert 'for name in [\'dress\', \'shirt\', \'toptee\']:' in content
    assert 'cap.{}.{}.json".format(name, \'train\')' in content


def test_f05_encoder_constructor_reads_train_captions_for_all_categories():
    encoder_datasets_path = REPO_ROOT / "workbench" / "third_party" / "ENCODER" / "datasets.py"
    content = encoder_datasets_path.read_text(encoding="utf-8")
    assert "def train_init_process(self):" in content
    assert 'for name in [\'dress\', \'shirt\', \'toptee\']:' in content
    assert 'cap.{}.{}.json".format(name, \'train\')' in content


def test_f05_pair_constructor_reads_train_captions_for_all_categories():
    pair_datasets_path = REPO_ROOT / "workbench" / "third_party" / "PAIR" / "data" / "datasets.py"
    content = pair_datasets_path.read_text(encoding="utf-8")
    assert "def train_init_process(self):" in content
    assert 'for name in [\'dress\', \'shirt\', \'toptee\']:' in content
    assert 'cap.{}.{}.json".format(name, \'train\')' in content


def test_f05_pair_orchestrator_contract_already_requires_train():
    contract_path = REPO_ROOT / "workbench" / "registry" / "preparation_contracts.yaml"
    content = contract_path.read_text(encoding="utf-8")
    assert "preparation_id: pair_fashioniq" in content
    for cat in CATEGORIES:
        assert f"captions/cap.{cat}.train.json" in content


def test_f05_isolated_cache_root_starts_fresh_without_cached_tensors():
    from workbench.replay.common import isolated_cache_root

    with tempfile.TemporaryDirectory() as tmp:
        dataset_root = Path(tmp)
        (dataset_root / "captions").mkdir()
        (dataset_root / "image_splits").mkdir()
        (dataset_root / "resized_image").mkdir()
        # Stash dummy cache files in the canonical root
        (dataset_root / "fashion_iq_data.json").write_text('{"cached": true}')
        (dataset_root / "test_queries_dress.pkl").write_text("dummy-pickle")

        with isolated_cache_root(dataset_root) as cache_root:
            assert not (cache_root / "fashion_iq_data.json").exists()
            assert not (cache_root / "test_queries_dress.pkl").exists()
            assert (cache_root / "captions").is_symlink()
            assert (cache_root / "image_splits").is_symlink()
            assert (cache_root / "resized_image").is_symlink()


def test_f05_extended_validator_detects_missing_train_captions():
    def validate_dataset_with_train(dataset_root: Path, categories: tuple[str, ...] = CATEGORIES) -> list[str]:
        blockers: list[str] = []
        for cat in CATEGORIES:
            for rel in (
                f"captions/cap.{cat}.train.json",
                f"captions/cap.{cat}.val.json",
                f"captions/correction_dict_{cat}.json",
                f"image_splits/split.{cat}.val.json",
            ):
                if not (dataset_root / rel).is_file():
                    blockers.append(f"dataset input missing: {dataset_root / rel}")
            cat_resized = dataset_root / "resized_image" / cat
            if not cat_resized.is_dir():
                blockers.append(f"dataset input missing: {cat_resized}")
        return blockers

    with tempfile.TemporaryDirectory() as tmp:
        ds_root = Path(tmp)
        (ds_root / "captions").mkdir()
        (ds_root / "image_splits").mkdir()
        (ds_root / "resized_image").mkdir()
        for cat in CATEGORIES:
            (ds_root / "resized_image" / cat).mkdir()
            (ds_root / "captions" / f"cap.{cat}.val.json").write_text("[]")
            (ds_root / "captions" / f"correction_dict_{cat}.json").write_text("{}")
            (ds_root / "image_splits" / f"split.{cat}.val.json").write_text("[]")
            # NOTE: train captions intentionally omitted

        blockers = validate_dataset_with_train(ds_root)
        assert len(blockers) == 3
        for cat in CATEGORIES:
            assert any(f"cap.{cat}.train.json" in b for b in blockers)


# ==============================================================================
# F08 Tests: DCNet dataset_root mismatch and preflight resolution
# ==============================================================================

def test_f08_dcnet_config_data_dir_resolution():
    dcnet_source = REPO_ROOT / "workbench" / "third_party" / "DCNet"
    cfg_path = dcnet_source / "configs" / "ce" / "fashioniq_dcnet.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    raw_data_dir = cfg["data_loader"]["args"]["data_dir"]
    assert raw_data_dir == "dataset/fashioniq"

    resolved_data_dir = (dcnet_source / raw_data_dir).resolve()
    assert resolved_data_dir == dcnet_source / "dataset" / "fashioniq"

    # Mismatch check with workbench dataset root
    mock_workbench_dataset = REPO_ROOT / "data" / "fashionIQ_dataset"
    assert resolved_data_dir != mock_workbench_dataset.resolve()


def test_f08_preflight_refusal_on_data_dir_mismatch():
    def validate_dcnet_data_dir_alignment(checkpoint_root: Path, source_root: Path, dataset_root: Path) -> list[str]:
        cfg_file = checkpoint_root / "config.json"
        if not cfg_file.is_file():
            return [f"DCNet config missing: {cfg_file}"]
        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        data_dir_str = cfg.get("data_loader", {}).get("args", {}).get("data_dir")
        if not data_dir_str:
            return ["DCNet config missing data_loader.args.data_dir"]
        configured_path = (source_root / data_dir_str).resolve() if not Path(data_dir_str).is_absolute() else Path(data_dir_str).resolve()
        if configured_path != dataset_root.resolve():
            return [
                f"DCNet dataset root mismatch: checkpoint config.json data_loader.args.data_dir '{data_dir_str}' "
                f"resolves to '{configured_path}', but evaluator received dataset_root '{dataset_root.resolve()}'. "
                f"To resolve without modifying author config.json, stage a symlink at {source_root / data_dir_str} -> {dataset_root.resolve()}."
            ]
        return []

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = Path(tmp) / "ckpt"
        ckpt_dir.mkdir()
        (ckpt_dir / "config.json").write_text(json.dumps({"data_loader": {"args": {"data_dir": "dataset/fashioniq"}}}))
        src_dir = Path(tmp) / "DCNet"
        src_dir.mkdir()
        ds_dir = Path(tmp) / "data" / "fashionIQ_dataset"
        ds_dir.mkdir(parents=True)

        # 1. Unlinked: mismatch refused
        blockers = validate_dcnet_data_dir_alignment(ckpt_dir, src_dir, ds_dir)
        assert len(blockers) == 1
        assert "DCNet dataset root mismatch" in blockers[0]

        # 2. Linked: matches, no blockers
        (src_dir / "dataset").mkdir()
        (src_dir / "dataset" / "fashioniq").symlink_to(ds_dir, target_is_directory=True)
        assert validate_dcnet_data_dir_alignment(ckpt_dir, src_dir, ds_dir) == []


# ==============================================================================
# F09 Tests: DCNet native output metric parser
# ==============================================================================

def parse_dcnet_native_output(stdout_text: str, paper_metrics: dict[str, float | None] | None = None) -> MetricExtractionResult:
    """Proposed parser for DCNet native stdout (test.py --mode val)."""
    parser_id = "dcnet_native_logger"
    unparsed = MetricExtractionResult(
        extraction_status="EVALUATION_COMPLETED_METRICS_UNPARSED",
        parser_id=parser_id,
        parser_version=1,
        observed_metrics=None,
        category_metrics=None,
        parity_status="PARITY_NOT_EVALUATED",
        metric_source="stdout",
    )

    categories = ("dress", "shirt", "toptee")
    category_metrics: dict[str, dict[str, float]] = {}

    # 1. Search for combined metrics line:
    # "comb: <dress_r10> <dress_r50> <shirt_r10> <shirt_r50> <toptee_r10> <toptee_r50>"
    comb_match = re.search(
        r"(?:^|\n)\s*(?:.* - INFO - )?comb:\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)",
        stdout_text,
    )

    # 2. Search for component lines (comp corr r@10, r@50)
    # "[<category>] comp corr r@10, r@50: <c10>\t<c50>\t<k10>\t<k50>"
    comp_corr_matches = re.findall(
        r"(?:^|\n)\s*(?:.* - INFO - )?\[(dress|shirt|toptee)\]\s+comp\s+corr\s+r@10,\s*r@50:\s*([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)",
        stdout_text,
    )

    # 3. Search for 2-number single branch fallback
    # "[<category>] r@10, r@50: <r10>\t<r50>"
    single_matches = re.findall(
        r"(?:^|\n)\s*(?:.* - INFO - )?\[(dress|shirt|toptee)\]\s+r@10,\s*r@50:\s*([0-9.eE+-]+)\s+([0-9.eE+-]+)",
        stdout_text,
    )

    raw_output: dict[str, object] = {}

    if comb_match:
        # Native order from configs/ce/fashioniq_dcnet.json: categories = ["dress", "shirt", "toptee"]
        vals = [float(x) * 100.0 for x in comb_match.groups()]
        for idx, cat in enumerate(categories):
            r10 = vals[idx * 2]
            r50 = vals[idx * 2 + 1]
            if not (_is_valid_recall(r10) and _is_valid_recall(r50)):
                return unparsed
            category_metrics[cat] = {"r10": round(r10, 4), "r50": round(r50, 4)}
        raw_output["branch"] = "combined"
        if comp_corr_matches:
            raw_output["components"] = {
                m[0]: {
                    "comp_r10": round(float(m[1]) * 100.0, 4),
                    "comp_r50": round(float(m[2]) * 100.0, 4),
                    "corr_r10": round(float(m[3]) * 100.0, 4),
                    "corr_r50": round(float(m[4]) * 100.0, 4),
                }
                for m in comp_corr_matches
            }
    elif len(single_matches) == 3:
        for cat, r10_s, r50_s in single_matches:
            r10 = float(r10_s) * 100.0
            r50 = float(r50_s) * 100.0
            if not (_is_valid_recall(r10) and _is_valid_recall(r50)):
                return unparsed
            category_metrics[cat] = {"r10": round(r10, 4), "r50": round(r50, 4)}
        raw_output["branch"] = "single"
    else:
        return unparsed

    if set(category_metrics.keys()) != set(categories):
        return unparsed

    macro_r10 = sum(category_metrics[c]["r10"] for c in categories) / 3.0
    macro_r50 = sum(category_metrics[c]["r50"] for c in categories) / 3.0
    macro_mean = (macro_r10 + macro_r50) / 2.0

    # Validate against comb_avg if present in stdout
    comb_avg_match = re.search(
        r"(?:^|\n)\s*(?:.* - INFO - )?\[comb_avg\]\s*:\s*([0-9.eE+-]+)",
        stdout_text,
    )
    if comb_avg_match:
        decl_mean = float(comb_avg_match.group(1)) * 100.0
        if abs(decl_mean - macro_mean) > 0.05:
            return unparsed

    observed = {
        "r10": round(macro_r10, 4),
        "r50": round(macro_r50, 4),
        "mean": round(macro_mean, 4),
    }

    return MetricExtractionResult(
        extraction_status="EXTRACTED_VERIFIED",
        parser_id=parser_id,
        parser_version=1,
        observed_metrics=observed,
        category_metrics=category_metrics,
        parity_status=_check_parity(observed, paper_metrics),
        metric_source="stdout",
        raw_output=raw_output,
    )


def test_f09_dcnet_parser_extracts_combined_metrics_and_multiplies_by_100():
    # Emulated stdout from test.py with TrainerJoint
    stdout_sample = """
Running test on cuda
save att
[dress] comp corr r@10, r@50: 0.2500\t0.5000\t0.2100\t0.4200
[shirt] comp corr r@10, r@50: 0.2000\t0.4000\t0.1800\t0.3600
[toptee] comp corr r@10, r@50: 0.2700\t0.5200\t0.2300\t0.4600
comb: 0.2895 0.5607 0.2395 0.4730 0.3044 0.5829
[Avg Recall]     : 0.3567
[Avg Recall corr]: 0.3100
[comb_avg]       : 0.4083
Saving score matrix: val/test_score.pt ...
Done in 12.345s
"""
    paper_metrics = {"mean": 40.83, "r10": 27.78, "r50": 53.89}
    res = parse_dcnet_native_output(stdout_sample, paper_metrics)

    assert res.extraction_status == "EXTRACTED_VERIFIED"
    assert res.category_metrics["dress"] == {"r10": 28.95, "r50": 56.07}
    assert res.category_metrics["shirt"] == {"r10": 23.95, "r50": 47.30}
    assert res.category_metrics["toptee"] == {"r10": 30.44, "r50": 58.29}

    assert res.observed_metrics["r10"] == pytest.approx(27.78, abs=1e-2)
    assert res.observed_metrics["r50"] == pytest.approx(53.8867, abs=1e-2)
    assert res.observed_metrics["mean"] == pytest.approx(40.83, abs=1e-2)
    assert res.parity_status == "PARITY_VERIFIED"


def test_f09_dcnet_parser_fallback_for_single_branch():
    stdout_single = """
[dress] r@10, r@50: 0.2895\t0.5607
[shirt] r@10, r@50: 0.2395\t0.4730
[toptee] r@10, r@50: 0.3044\t0.5829
"""
    res = parse_dcnet_native_output(stdout_single)
    assert res.extraction_status == "EXTRACTED_VERIFIED"
    assert res.category_metrics["dress"] == {"r10": 28.95, "r50": 56.07}
    assert res.raw_output["branch"] == "single"


def test_f09_dcnet_parser_rejects_inconsistent_macro_mean():
    stdout_bad_macro = """
comb: 0.2895 0.5607 0.2395 0.4730 0.3044 0.5829
[comb_avg]       : 0.9999
"""
    res = parse_dcnet_native_output(stdout_bad_macro)
    assert res.extraction_status == "EVALUATION_COMPLETED_METRICS_UNPARSED"
