"""F05, F07, F09, F10, F14: verifications that changed shared declarations or loaders.

Each test asserts the property that was actually wrong, using the real producer or the real
declaration rather than a restatement of it:

* F05 — the native dataset constructor needs TRAIN captions, so the preflight must demand them and
  the layout must say so. A missing train file must be reported as a blocker, not raised later by the
  constructor after an expensive model load.
* F07 — the pinned `clip` dependency must be the module the evaluator imports, not a same-named
  package that lacks the API.
* F09 — DCNet prints metrics as FRACTIONS; the parser must convert once, keep the branch the paper row
  describes, and keep the composition/correction splits separate.
* F10 — the ENCODER checkpoint is a real author file with published metadata, and the loader must
  refuse to evaluate a partially-loaded model instead of discarding what `strict=False` reports.
* F14 — SPRC selects its vision encoder from a DECLARED backbone, never from a filename substring.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from workbench.backend.adapters.base import EvalRequest  # noqa: E402
from workbench.backend.adapters.models import ADAPTERS  # noqa: E402
from workbench.backend.metrics_extraction import extract_aggregate_metrics, parse_dcnet_output  # noqa: E402
from workbench.backend.registry import model_by_id  # noqa: E402

CATEGORIES = ("dress", "shirt", "toptee")


# --------------------------------------------------------------------------------------- F05

def _val_only_root(tmp_path: Path) -> Path:
    root = tmp_path / "FashionIQ"
    (root / "captions").mkdir(parents=True)
    (root / "image_splits").mkdir(parents=True)
    for category in CATEGORIES:
        (root / "resized_image" / category).mkdir(parents=True)
        (root / "captions" / f"cap.{category}.val.json").write_text("[]")
        (root / "captions" / f"correction_dict_{category}.json").write_text("{}")
        (root / "image_splits" / f"split.{category}.val.json").write_text("[]")
    return root


@pytest.mark.parametrize("module_name,label", [("hint", "HINT"), ("encoder", "ENCODER"), ("pair", "PAIR")])
def test_f05_a_val_only_root_is_reported_as_missing_train(module_name, label, tmp_path) -> None:
    """The constructor reads TRAIN captions first; a val-only root must be caught before loading."""
    import importlib

    module = importlib.import_module(f"workbench.replay.{module_name}")
    root = _val_only_root(tmp_path)
    blockers = module.validate_dataset(root)
    assert blockers, f"{label} accepted a root its own dataset constructor cannot open"
    assert any(".train.json" in blocker for blocker in blockers), blockers

    for category in CATEGORIES:
        (root / "captions" / f"cap.{category}.train.json").write_text("[]")
    assert module.validate_dataset(root) == [], "a complete root must pass"


@pytest.mark.parametrize("model_id", ["hint", "encoder", "pair"])
def test_f05_the_declared_layout_requires_the_train_paths(model_id) -> None:
    """The registry's layout is what the preflight derives its checks from."""
    from workbench.backend.fashioniq_layout import paths_for_layout

    model = model_by_id(model_id)
    assert model["fashioniq_layout"] == "fashioniq_ilearn_resized_training", (
        "a constructor that reads train captions belongs to the training layout")
    names = {path.name for path in paths_for_layout(Path("/FashionIQ"), model["fashioniq_layout"])}
    for category in CATEGORIES:
        assert f"cap.{category}.train.json" in names


def test_f05_the_native_constructors_really_read_train() -> None:
    """Guard the premise: if upstream stopped reading train, the requirement would be wrong."""
    for relative in ("HINT/datasets.py", "ENCODER/datasets.py", "PAIR/data/datasets.py"):
        source = (REPO / "workbench" / "third_party" / relative).read_text()
        assert "train_init_process" in source, relative
        assert "cap.{}.{}.json\".format(name, 'train')" in source or "'train'" in source, relative


# --------------------------------------------------------------------------------------- F07

def test_f07_tgcir_pins_the_module_its_evaluator_imports() -> None:
    """PyPI's `clip` has no `load`; the evaluator calls one, so the pin must provide it."""
    model = model_by_id("tgcir")
    packages = model["environment"]["packages"]
    clip_pins = [p for p in packages if p.split("==")[0].split(" ")[0].strip().lower() == "clip"
                 or p.strip().lower().startswith("clip")]
    assert clip_pins, packages
    pin = clip_pins[0]
    assert "openai/CLIP" in pin, "the pin must be OpenAI CLIP, not a same-named PyPI package"
    import re

    commit = re.search(r"@([0-9a-f]{40})", pin)
    assert commit, f"a revision must be pinned, not a branch: {pin}"

    evaluator = (REPO / "workbench" / "third_party" / "TGCIR" / "model.py").read_text()
    assert "clip.load(" in evaluator, "the pinned module must expose the API the evaluator calls"


# --------------------------------------------------------------------------------------- F09

def _dcnet_log(comp=(0.301, 0.501), corr=(0.291, 0.491), comb=(0.4010, 0.6010, 0.4020, 0.6020, 0.4030, 0.6030)):
    return "\n".join([
        f"2026-10-19 10:00:00,000 - INFO - [dress] comp corr r@10, r@50: {comp[0]} {comp[1]} {corr[0]} {corr[1]}",
        "2026-10-19 10:00:00,001 - INFO - [shirt] comp corr r@10, r@50: 0.311 0.511 0.301 0.501",
        "2026-10-19 10:00:00,002 - INFO - [toptee] comp corr r@10, r@50: 0.321 0.521 0.311 0.511",
        "2026-10-19 10:00:00,003 - INFO - comb: " + " ".join(str(v) for v in comb),
        "2026-10-19 10:00:00,004 - INFO - [comb_avg]       : 0.502",
    ])


def test_f09_dcnet_fractions_are_converted_once() -> None:
    """`recall()` returns 0..1, so the % schema needs exactly one multiplication."""
    result = extract_aggregate_metrics("dcnet", _dcnet_log(), "", 0)
    assert result.extraction_status == "EXTRACTED_VERIFIED"
    assert result.observed_metrics["r10"] == pytest.approx(40.2, abs=0.05)
    assert result.observed_metrics["r50"] == pytest.approx(60.2, abs=0.05)
    assert result.observed_metrics["mean"] == pytest.approx(50.2, abs=0.05)
    assert result.parser_id == "dcnet_native_logger"


def test_f09_the_combined_branch_is_the_one_read() -> None:
    """The registry's paper row defines the reported mean as the mean of the six `comb:` cells."""
    result = parse_dcnet_output(_dcnet_log())
    assert result.raw_output == {"branch": "combined"}


def test_f09_composition_and_correction_are_not_merged_into_the_combined_result() -> None:
    """They are different metrics; conflating them would be a silent scientific change."""
    result = parse_dcnet_output(_dcnet_log())
    assert result.category_metrics["dress"] == {"r10": 40.1, "r50": 60.1}
    assert result.category_metrics["dress__composition"] == {"r10": 30.1, "r50": 50.1}
    assert result.category_metrics["dress__correction"] == {"r10": 29.1, "r50": 49.1}


def test_f09_the_two_cell_branch_is_accepted_only_when_complete() -> None:
    complete = "\n".join(
        f"2026-10-19 10:00:00,00{i} - INFO - [{c}] r@10, r@50: 0.4{i} 0.6{i}"
        for i, c in enumerate(CATEGORIES))
    assert parse_dcnet_output(complete).observed_metrics["r10"] == pytest.approx(41.0, abs=0.05)

    truncated = "\n".join(
        f"2026-10-19 10:00:00,00{i} - INFO - [{c}] r@10, r@50: 0.4{i} 0.6{i}"
        for i, c in enumerate(CATEGORIES[:2]))
    assert parse_dcnet_output(truncated).extraction_status != "EXTRACTED_VERIFIED"


def test_f09_a_stream_that_contradicts_its_own_average_is_refused() -> None:
    """With one conversion the mean of the six cells IS the macro mean, so a mismatch means misread."""
    log = _dcnet_log()
    lines = log.splitlines()
    lines[-1] = "2026-10-19 10:00:00,004 - INFO - [comb_avg]       : 0.99"
    assert parse_dcnet_output("\n".join(lines)).extraction_status != "EXTRACTED_VERIFIED"


def test_f09_ranges_and_completeness_still_hold() -> None:
    assert parse_dcnet_output("2026 - INFO - [dress] r@10, r@50: 5.0 0.6").extraction_status != "EXTRACTED_VERIFIED"
    assert parse_dcnet_output("nothing parseable").extraction_status == "EVALUATION_COMPLETED_METRICS_UNPARSED"
    assert extract_aggregate_metrics("dcnet", _dcnet_log(), "", 1).extraction_status == "EVALUATION_FAILED"


# --------------------------------------------------------------------------------------- F10

def test_f10_the_encoder_checkpoint_is_pinned_to_an_author_revision() -> None:
    """Verified against the Hub API: revision, size, and the LFS oid of the file itself."""
    variant = model_by_id("encoder")["checkpoint_variants"][0]
    assert variant["checkpoint_mapping_status"] == "VERIFIED_METADATA"
    assert variant["status"] == "URL_VERIFIED"
    assert variant["expected_size_bytes"] == 631879365
    assert variant["expected_sha256"] == (
        "0f31c50150a7bda3495e3378df13b4cec7924b52efd23b5e5dd9731459da9173")
    assert "iLearn-Lab/AAAI25-ENCODER" in variant["download_url"]
    # A revision, not a moving branch.
    assert "/resolve/main/" not in variant["download_url"]
    assert variant["source_type"] == "author_huggingface"


def test_f10_the_loader_reports_what_strict_false_hid() -> None:
    """`strict=False` is kept, but an unmatched checkpoint must block rather than evaluate."""
    source = (REPO / "workbench" / "replay" / "encoder.py").read_text()
    assert "load_state_dict(ckpt_data, strict=False)" in source, "upstream's choice is preserved"
    assert "getattr(report, \"missing_keys\"" in source
    assert "Refusing to evaluate a partially-loaded model." in source
    # The author's file carries the training temperature, which is not a module parameter.
    assert '!= "loss_T"' in source


# --------------------------------------------------------------------------------------- F14

def test_f14_sprc_backbone_is_declared_not_inferred() -> None:
    """`strict=False` would hide a wrong encoder, so the encoder is declared per variant."""
    variants = model_by_id("sprc")["checkpoint_variants"]
    by_id = {variant["checkpoint_id"]: variant for variant in variants}
    assert by_id["sprc_fiq_unresolved"]["backbone"] == "pretrain"
    assert by_id["sprc_fiq_vitl"]["backbone"] == "pretrain_vitL"
    assert by_id["sprc_fiq_vitl"].get("scientific_variant") is True, (
        "ViT-L is a different configuration from the paper row and must be labelled")


def test_f14_the_adapter_uses_the_declared_backbone(tmp_path) -> None:
    from unittest.mock import patch

    from workbench.backend import runtime as _runtime

    for checkpoint_id, expected in (("sprc_fiq_unresolved", "pretrain"), ("sprc_fiq_vitl", "pretrain_vitL")):
        request = EvalRequest("sprc", checkpoint_id, "fashioniq_original_split", tmp_path, tmp_path / "o.json")
        with patch.object(_runtime, "python_executable", lambda _: "/usr/bin/python3"):
            command = ADAPTERS["sprc"]().command(tmp_path / "SPRC", tmp_path / f"{checkpoint_id}.pt",
                                                 request)
        assert command[command.index("--backbone") + 1] == expected


def test_f14_an_undeclared_backbone_is_refused_rather_than_guessed(tmp_path) -> None:
    """A filename based guess could silently evaluate a partly-initialised model."""
    from unittest.mock import patch

    from workbench.backend import runtime as _runtime
    from workbench.backend.errors import WorkbenchError

    request = EvalRequest("sprc", "no_such_variant", "fashioniq_original_split", tmp_path,
                          tmp_path / "o.json")
    with patch.object(_runtime, "python_executable", lambda _: "/usr/bin/python3"):
        with pytest.raises(WorkbenchError):
            ADAPTERS["sprc"]().command(tmp_path / "SPRC", tmp_path / "sprc_fiq.pt", request)
