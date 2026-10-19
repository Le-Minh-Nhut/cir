"""Regression tests for the four reproduced defects at 914703e, plus the F06 processor work.

Each test drives the PRODUCTION entry point the defect lived behind, so a fix cannot be undone by
removing a helper nobody calls. Where the earlier wave tested a copy of the logic inside the test
file (`test_f05_f08_f09_verification.py`), the replacement here imports the production function.
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.metrics_extraction import extract_aggregate_metrics, parse_dcnet_output
from workbench.backend.preflight import approved_patch_state
from workbench.backend.registry import load_registry
from workbench.replay import common
from workbench.replay.dcnet import binding_mismatch, configured_data_dir, preflight

REPO = Path(__file__).resolve().parents[2]
DCNET_RUN = {"data_loader": {"args": {"data_dir": "dataset/fashioniq"}}}
CATEGORIES = ("dress", "shirt", "toptee")


def _run_directory(root: Path, config: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "trained_model.pth").write_bytes(b"weights")
    return root


# ------------------------------------------------------------------------------------------- A1-A4
# A. The DCNet adapter launches the evaluator itself, so the binding check has to run on that path.

def test_the_binding_refusal_reaches_the_shared_preflight_list(tmp_path: Path) -> None:
    """`preflight_blockers` is what every operator entry point consults; a helper alone is not a fix."""
    from workbench.backend import preflight as module

    source = tmp_path / "DCNet"
    evaluator_cwd = source
    (evaluator_cwd / "dataset" / "fashioniq").mkdir(parents=True)
    run = _run_directory(tmp_path / "run", DCNET_RUN)
    requested = tmp_path / "requested"
    requested.mkdir()

    mismatch = binding_mismatch(run, requested, evaluator_cwd)
    assert mismatch and "binding mismatch" in mismatch
    assert str(evaluator_cwd / "dataset" / "fashioniq") in mismatch
    assert str(requested) in mismatch

    # And the list every entry point reads carries it, for a model dict whose checkpoint path is
    # this run directory.
    model = {**next(m for m in load_registry()["models"] if m["model_id"] == "dcnet"),
             "checkpoint_variants": [{"checkpoint_id": "fashioniq_run_directory",
                                      "filename": str(run), "artifact_type": "run_directory",
                                      "required_files": ["config.json", "trained_model.pth"],
                                      "checkpoint_mapping_status": "VERIFIED_METADATA"}]}

    class _Config:
        pass

    from workbench.backend.operator_config import resolve_config

    config = resolve_config()
    monkey = module.model_requirements
    module.model_requirements = lambda *a, **k: {"blocks": [], "warnings": [], "paths": []}
    try:
        blockers = module.preflight_blockers(model, config, dataset_root=requested)
    finally:
        module.model_requirements = monkey
    assert any("binding mismatch" in blocker for blocker in blockers), blockers


def test_an_absolute_config_path_is_compared_not_trusted(tmp_path: Path) -> None:
    """An absolute `data_dir` used to be accepted unconditionally, whatever it named."""
    other = tmp_path / "fiq-other"
    other.mkdir()
    run = _run_directory(tmp_path / "run", {"data_loader": {"args": {"data_dir": str(other)}}})

    refusal = binding_mismatch(run, tmp_path / "fiq-selected", tmp_path / "unused-cwd")
    assert refusal and str(other) in refusal
    assert binding_mismatch(run, other, tmp_path / "unused-cwd") is None


def test_a_relative_path_that_resolves_to_the_requested_root_passes(tmp_path: Path) -> None:
    """The canonical preparation stages the dataset as a symlink, so this is the expected shape."""
    source = tmp_path / "source"
    real = tmp_path / "real-dataset"
    real.mkdir(parents=True)
    (source / "dataset").mkdir(parents=True)
    (source / "dataset" / "fashioniq").symlink_to(real)
    run = _run_directory(tmp_path / "run", DCNET_RUN)

    resolved, blockers = configured_data_dir(run / "config.json", real, source)
    assert blockers == []
    assert resolved == real.resolve()
    assert binding_mismatch(run, real, source) is None


def test_a_relative_path_resolving_inside_the_checkout_is_refused(tmp_path: Path) -> None:
    """A plain directory the evaluator would open is not the verified dataset root."""
    source = tmp_path / "source"
    (source / "dataset" / "fashioniq").mkdir(parents=True)
    run = _run_directory(tmp_path / "run", DCNET_RUN)

    resolved, blockers = configured_data_dir(run / "config.json", tmp_path / "selected", source)
    assert resolved == (source / "dataset" / "fashioniq").resolve()
    assert any("binding mismatch" in blocker for blocker in blockers)


def test_a_config_without_the_field_is_refused(tmp_path: Path) -> None:
    run = _run_directory(tmp_path / "run", {"data_loader": {"args": {}}})
    _, blockers = configured_data_dir(run / "config.json", tmp_path / "selected", tmp_path / "run")
    assert any("declares no data_loader.args.data_dir" in blocker for blocker in blockers)


def test_no_environment_override_can_change_what_the_evaluator_opens(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R10-01: the decision reads the author's config, because that is what the evaluator reads.

    An override used to be honoured here, which made the validator answer a question the native
    `ConfigParser` never asks: a run could be certified against `selected` while the loader opened
    `wrong`. Both are exercised here, and the environment must move neither.
    """
    checkout = tmp_path / "checkout"
    (checkout / "dataset").mkdir(parents=True)
    wrong = tmp_path / "wrong"
    correct = tmp_path / "correct"
    for root in (wrong, correct):
        root.mkdir()
    # The config says `dataset/fashioniq`; the checkout's link is what decides where that lands.
    (checkout / "dataset" / "fashioniq").symlink_to(wrong)
    run = _run_directory(tmp_path / "run", DCNET_RUN)

    # The override claims the run targets `correct`. The link says otherwise, and the link wins.
    monkeypatch.setenv("WORKBENCH_DCNET_DATA_DIR", str(correct))
    refusal = binding_mismatch(run, correct, checkout)
    assert refusal and "binding mismatch" in refusal
    assert str(wrong) in refusal

    # And with the link pointing where the override claimed, the binding is exact with no override
    # present -- so the environment variable was never load-bearing for a correct staging.
    monkeypatch.delenv("WORKBENCH_DCNET_DATA_DIR", raising=False)
    (checkout / "dataset" / "fashioniq").unlink()
    (checkout / "dataset" / "fashioniq").symlink_to(correct)
    assert binding_mismatch(run, correct, checkout) is None


def test_preflight_reports_the_mismatch_for_a_relative_literal(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "DCNet" / "dataset" / "fashioniq").mkdir(parents=True)
    run = _run_directory(tmp_path / "run", DCNET_RUN)
    dataset = tmp_path / "dataset"
    (dataset / "image_splits").mkdir(parents=True)
    (dataset / "captions").mkdir(parents=True)
    (dataset / "resized_images").mkdir(parents=True)
    for category in CATEGORIES:
        (dataset / "image_splits" / f"split.{category}.val.json").write_text("[]", encoding="utf-8")
        (dataset / "captions" / f"cap.{category}.glove.val.pkl").write_bytes(b"x")

    problems = preflight(run, dataset, source, evaluator_cwd=source / "DCNet")
    assert any("binding mismatch" in problem for problem in problems)


def test_the_staged_run_records_the_config_it_read(tmp_path: Path) -> None:
    """`verify_run` re-parses a run's stdout, so the config it needs must be recorded with it."""
    from workbench.scripts.evaluate_models import EvaluationPlan, read_executed_config

    run = _run_directory(tmp_path / "run", DCNET_RUN)
    plan = EvaluationPlan(model_id="dcnet", checkpoint_id="c", protocol="p", dataset_root=tmp_path,
                          source=tmp_path, checkpoint=run, cwd=tmp_path, command=[], pin=None,
                          actual_source_commit=None)
    assert read_executed_config(plan) == DCNET_RUN

    other = EvaluationPlan(model_id="hint", checkpoint_id="c", protocol="p", dataset_root=tmp_path,
                           source=tmp_path, checkpoint=run, cwd=tmp_path, command=[], pin=None,
                           actual_source_commit=None)
    assert read_executed_config(other) is None


# ------------------------------------------------------------------------------------------- B1-B4
# B. The six anonymous `comb:` cells must be attributed by the order the evaluator printed them in.

def _dcnet_stdout(order: tuple[str, ...], comb: tuple[float, ...]) -> str:
    named = "\n".join(
        f"2026-10-19 10:00:0{index} - INFO - [{category}] comp corr r@10, r@50: "
        f"{0.1 + index / 100} {0.3 + index / 100} {0.2 + index / 100} {0.4 + index / 100}"
        for index, category in enumerate(order))
    cells = " ".join(str(value) for value in comb)
    return (f"{named}\n2026-10-19 10:00:09 - INFO - comb: {cells}\n"
            "2026-10-19 10:00:10 - INFO - [comb_avg]       : 0.5")


def test_a_non_default_category_order_is_attributed_per_category() -> None:
    """The reviewed reproduction: order `toptee, dress, shirt`, cells `0.7 0.9 0.2 0.4 0.3 0.5`."""
    stdout = _dcnet_stdout(("toptee", "dress", "shirt"), (0.7, 0.9, 0.2, 0.4, 0.3, 0.5))
    config = {"data_loader": {"args": {"categories": ["toptee", "dress", "shirt"]}}}
    result = parse_dcnet_output(stdout, None, config)
    assert result.extraction_status == "EXTRACTED_VERIFIED"
    assert result.category_metrics["toptee"] == {"r10": 70.0, "r50": 90.0}
    assert result.category_metrics["dress"] == {"r10": 20.0, "r50": 40.0}
    assert result.category_metrics["shirt"] == {"r10": 30.0, "r50": 50.0}
    # The aggregate is permutation-invariant, which is exactly why it cannot be the check.
    assert result.observed_metrics["mean"] == pytest.approx(50.0, abs=0.05)


def test_the_order_comes_from_the_config_the_evaluator_ran_with() -> None:
    stdout = _dcnet_stdout(("toptee", "dress", "shirt"), (0.7, 0.9, 0.2, 0.4, 0.3, 0.5))
    config = {"data_loader": {"args": {"categories": ["toptee", "dress", "shirt"]}}}
    assert parse_dcnet_output(stdout, None, config).category_metrics["toptee"] == {
        "r10": 70.0, "r50": 90.0}


def test_a_config_that_contradicts_the_stream_is_refused() -> None:
    stdout = _dcnet_stdout(("toptee", "dress", "shirt"), (0.7, 0.9, 0.2, 0.4, 0.3, 0.5))
    contradictory = {"data_loader": {"args": {"categories": ["dress", "shirt", "toptee"]}}}
    assert parse_dcnet_output(stdout, None, contradictory).extraction_status != "EXTRACTED_VERIFIED"


def test_a_config_declaring_another_category_set_states_no_order() -> None:
    """`['dress','shirt','toptee','shoes']` is not a statement about three FashionIQ categories."""
    stdout = _dcnet_stdout(("toptee", "dress", "shirt"), (0.7, 0.9, 0.2, 0.4, 0.3, 0.5))
    config = {"data_loader": {"args": {"categories": ["dress", "shirt", "toptee", "shoes"]}}}
    result = parse_dcnet_output(stdout, None, config)
    assert result.extraction_status == "EXTRACTED_VERIFIED"
    assert result.category_metrics["toptee"] == {"r10": 70.0, "r50": 90.0}


def test_the_extractor_threads_the_recorded_config_through() -> None:
    stdout = _dcnet_stdout(("toptee", "dress", "shirt"), (0.7, 0.9, 0.2, 0.4, 0.3, 0.5))
    order = {"data_loader": {"args": {"categories": ["toptee", "dress", "shirt"]}}}
    extracted = extract_aggregate_metrics("dcnet", stdout, "", 0, executed_config=order)
    assert extracted.category_metrics["toptee"] == {"r10": 70.0, "r50": 90.0}


def test_fractions_are_converted_to_percent_exactly_once() -> None:
    stdout = _dcnet_stdout(CATEGORIES, (0.401, 0.601, 0.402, 0.602, 0.403, 0.603))
    result = parse_dcnet_output(stdout)
    assert result.observed_metrics["r10"] == pytest.approx(40.2, abs=0.001)
    assert result.observed_metrics["r50"] == pytest.approx(60.2, abs=0.001)


# ------------------------------------------------------------------------------------------- C1-C4
# C. CSMCIR declares two required patches; the first is not the set.

def _csmcir() -> dict:
    return next(m for m in load_registry()["models"] if m["model_id"] == "csmcir")


def _staged_csmcir(tmp_path: Path, applied: int) -> Path:
    target = tmp_path / "CSMCIR"
    for patch in _csmcir()["approved_source_patches"]:
        destination = target / patch["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPO / "workbench/third_party/CSMCIR" / patch["path"]).read_bytes())
    for patch in _csmcir()["approved_source_patches"][:applied]:
        result = subprocess.run(["patch", "-p1", "--directory", str(target), "-i",
                                 str(REPO / patch["patch_file"])], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    return target


def test_a_pristine_checkout_is_not_applied(tmp_path: Path) -> None:
    state = approved_patch_state(_csmcir(), _staged_csmcir(tmp_path, 0))
    assert state["applied"] is False
    assert len(state["patches"]) == 2
    assert [entry["applied"] for entry in state["patches"]] == [False, False]


def test_only_the_first_patch_is_not_the_full_repair(tmp_path: Path) -> None:
    """The reviewed defect: this state used to read as `applied=True`."""
    state = approved_patch_state(_csmcir(), _staged_csmcir(tmp_path, 1))
    assert state["applied"] is False
    assert "1 of 2 required patches are not applied" in state["detail"]
    assert [entry["applied"] for entry in state["patches"]] == [True, False]


def test_only_the_second_patch_is_not_the_full_repair(tmp_path: Path) -> None:
    target = _staged_csmcir(tmp_path, 0)
    second = _csmcir()["approved_source_patches"][1]
    result = subprocess.run(["patch", "-p1", "--directory", str(target), "-i",
                             str(REPO / second["patch_file"])], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    state = approved_patch_state(_csmcir(), target)
    assert state["applied"] is False
    assert [entry["applied"] for entry in state["patches"]] == [False, True]


def test_both_patches_together_are_the_approved_variant(tmp_path: Path) -> None:
    state = approved_patch_state(_csmcir(), _staged_csmcir(tmp_path, 2))
    assert state["applied"] is True
    assert all(entry["applied"] for entry in state["patches"])
    assert "not applied" not in state["detail"]


def test_an_edit_after_both_patches_is_still_refused(tmp_path: Path) -> None:
    target = _staged_csmcir(tmp_path, 2)
    edited = target / _csmcir()["approved_source_patches"][0]["path"]
    edited.write_text(edited.read_text(encoding="utf-8") + "\n# unapproved\n", encoding="utf-8")
    state = approved_patch_state(_csmcir(), target)
    assert state["applied"] is False
    assert "neither the published file nor the approved variant" in state["detail"]


def test_a_partial_repair_is_reported_as_partial_by_the_requirements(tmp_path: Path) -> None:
    """The variant note must not claim a repair it did not verify."""
    from workbench.backend import preflight as module
    from workbench.backend.operator_config import resolve_config

    config = resolve_config()
    partial = _staged_csmcir(tmp_path, 1)
    third_party = tmp_path / "third_party"
    third_party.mkdir(exist_ok=True)
    (third_party / "CSMCIR").symlink_to(partial, target_is_directory=True)
    config = replace(config, WORKBENCH_THIRD_PARTY_ROOT=third_party)
    model = _csmcir()
    requirements = module.model_requirements(model, config, canonical_root=config.FASHIONIQ_ROOT)
    messages = " ".join(requirements["blocks"] + requirements["warnings"])
    assert "required patches are not applied" in messages
    assert "1 of 2" in messages


# ------------------------------------------------------------------------------------------- D1-D3
# D. The native constructor opens every category, whatever `--category` selects.

WRAPPERS = ("hint", "habit", "intent", "conesep", "airknow", "pair")
LABELS = {"hint": "HINT", "habit": "HABIT", "intent": "INTENT", "conesep": "ConeSep",
          "airknow": "Air-Know", "pair": "PAIR"}
NATIVE_DIRECTORIES = {"hint": "HINT", "habit": "HABIT", "intent": "INTENT", "conesep": "ConeSep",
                      "airknow": "AirKnow", "pair": "PAIR"}


def _dress_only_root(tmp_path: Path) -> Path:
    root = tmp_path / "fashioniq"
    for directory in ("captions", "image_splits"):
        (root / directory).mkdir(parents=True)
    (root / "resized_image" / "dress").mkdir(parents=True)
    (root / "captions" / "cap.dress.train.json").write_text("[]", encoding="utf-8")
    (root / "captions" / "cap.dress.val.json").write_text("[]", encoding="utf-8")
    (root / "captions" / "correction_dict_dress.json").write_text("{}", encoding="utf-8")
    (root / "image_splits" / "split.dress.val.json").write_text("[]", encoding="utf-8")
    return root


def _complete_root(tmp_path: Path) -> Path:
    root = _dress_only_root(tmp_path)
    for category in CATEGORIES:
        (root / "resized_image" / category).mkdir(parents=True, exist_ok=True)
        (root / "captions" / f"cap.{category}.train.json").write_text("[]", encoding="utf-8")
        (root / "captions" / f"cap.{category}.val.json").write_text("[]", encoding="utf-8")
        (root / "captions" / f"correction_dict_{category}.json").write_text("{}", encoding="utf-8")
        (root / "image_splits" / f"split.{category}.val.json").write_text("[]", encoding="utf-8")
    return root


@pytest.mark.parametrize("wrapper", WRAPPERS)
def test_a_subset_request_still_requires_every_constructor_input(wrapper: str, tmp_path: Path) -> None:
    module = importlib.import_module(f"workbench.replay.{wrapper}")
    blockers = module.validate_dataset(_dress_only_root(tmp_path), ("dress",))
    assert blockers, f"{wrapper}: a dress-only root must not preflight clean for a subset run"
    for category in ("shirt", "toptee"):
        assert any(f"cap.{category}.train.json" in blocker for blocker in blockers), (
            f"{wrapper}: the constructor opens cap.{category}.train.json whatever --category scores")
        assert any(f"split.{category}.val.json" in blocker for blocker in blockers)
    assert all(LABELS[wrapper] in blocker for blocker in blockers)


@pytest.mark.parametrize("wrapper", WRAPPERS)
def test_a_complete_root_still_preflights_clean_for_a_subset(wrapper: str, tmp_path: Path) -> None:
    module = importlib.import_module(f"workbench.replay.{wrapper}")
    assert module.validate_dataset(_complete_root(tmp_path), ("dress",)) == []


@pytest.mark.parametrize("wrapper", WRAPPERS)
def test_the_validated_set_is_the_one_the_native_constructor_opens(wrapper: str, tmp_path: Path) -> None:
    """The claim under test is about the native loop, so it is read from the native source."""
    source = REPO / "workbench" / "third_party" / NATIVE_DIRECTORIES[wrapper]
    candidates = list(source.rglob("datasets.py"))
    assert candidates, f"{wrapper}: no native datasets.py under {source}"
    native = "\n".join(candidate.read_text(encoding="utf-8") for candidate in candidates)
    assert "for name in ['dress', 'shirt', 'toptee']" in native
    assert "get_test_data('shirt')" in native and "get_test_data('toptee')" in native

    module = importlib.import_module(f"workbench.replay.{wrapper}")
    blockers = module.validate_dataset(_dress_only_root(tmp_path), ("dress",))
    for category in ("shirt", "toptee"):
        assert any(f"cap.{category}.train.json" in blocker for blocker in blockers)
        assert any(f"split.{category}.val.json" in blocker for blocker in blockers)


def test_the_shared_helper_is_the_one_the_wrappers_call(tmp_path: Path) -> None:
    """`constructor_input_blockers` must be production code the wrappers use, not a parallel copy."""
    calls = common.constructor_input_blockers(tmp_path, lambda path: f"missing: {path}")
    assert any("cap.shirt.train.json" in str(call) for call in calls)
    for wrapper in WRAPPERS:
        source = (REPO / "workbench" / "replay" / f"{wrapper}.py").read_text(encoding="utf-8")
        assert "constructor_input_blockers" in source, (
            f"{wrapper} must ask the shared question, not re-list the paths")


# ------------------------------------------------------------------------------------------- E1-E3
# E. F06: the processor must come from the pinned config, not from a second BLIP-2 model.

LAVIS_MODELS = {"hint": ("HINT", "HINT.py"), "habit": ("HABIT", "HABIT.py"),
                "intent": ("INTENT", "blip2_cir.py"), "conesep": ("ConeSep", "ConeSep.py"),
                "airknow": ("AirKnow", "blip2_cir.py")}


def _declared_configs(wrapper: str) -> dict[str, str]:
    import ast

    directory, class_file = LAVIS_MODELS[wrapper]
    class_source = (REPO / "workbench" / "third_party" / directory / "lavis" / "models" /
                    "blip2_models" / class_file).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(class_source)):
        if isinstance(node, ast.Assign) and any(
                getattr(target, "id", None) == "PRETRAINED_MODEL_CONFIG_DICT"
                for target in node.targets):
            return {key.value: value.value for key, value in zip(node.value.keys, node.value.values)}
    raise AssertionError(f"{wrapper}: the model class declares no config dict")


@pytest.mark.parametrize("wrapper", sorted(LAVIS_MODELS))
def test_no_wrapper_builds_a_second_model_for_its_processor(wrapper: str) -> None:
    import ast

    source = (REPO / "workbench" / "replay" / f"{wrapper}.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    # The name a call is bound to is not the property under test, so collect BOTH sides of it: the
    # names imported, and the attribute the call reads. `from lavis.models import
    # load_model_and_preprocess as loader; loader(...)` imports the same function under a different
    # name, and `importlib.import_module("lavis.models").load_model_and_preprocess(...)` reaches it
    # through an attribute. Both are uses of the pinned loader, so both are caught.
    imported: set[str] = set()
    origins: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.rsplit(".", 1)[-1])
                origins.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            origins.add(node.module or "")
            for alias in node.names:
                imported.add(alias.name)
    attributes: set[str] = set()
    literals: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            attributes.add(node.attr)
            if isinstance(node.value, ast.Name):
                attributes.add(node.value.id)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            literals.add(node.value)
    offending = {"load_model_and_preprocess"} & (imported | attributes | literals)
    assert not offending, (
        f"{wrapper} must not construct a second pretrained model just to read txt_processors")
    # `from lavis.models import X` and `import lavis.models` both reach the model registry, whatever
    # `X` is called; the module named in the import is what is compared, not the name bound to it.
    declared_paths = origins | literals
    models_module = {origin for origin in declared_paths if origin.startswith("lavis.models")}
    assert not models_module, (
        f"{wrapper} must not import the pinned model registry, whatever name it is bound to: "
        f"{sorted(models_module)}")
    assert "PROCESSOR_CONFIGS" in source


@pytest.mark.parametrize("wrapper", sorted(LAVIS_MODELS))
def test_the_declared_config_is_the_one_the_model_class_publishes(wrapper: str) -> None:
    """Every entry the model class publishes is either carried by the wrapper or named as out of scope."""
    wrapper_source = (REPO / "workbench" / "replay" / f"{wrapper}.py").read_text(encoding="utf-8")
    declared = _declared_configs(wrapper)
    carried = [config for config in declared.values() if f'"{config}"' in wrapper_source]
    assert carried, (
        f"{wrapper}: the model class publishes {sorted(declared.values())} and the wrapper carries "
        "none of them")
    # `pretrain_vitL` selects a different backbone, so a wrapper that does not run it says so rather
    # than silently claiming it.
    missing = [model_type for model_type, config in declared.items()
               if f'"{config}"' not in wrapper_source]
    for model_type in missing:
        assert model_type in wrapper_source or "pretrain_vitL" in wrapper_source, (
            f"{wrapper}: {model_type!r} is published by the model class and neither carried nor "
            "documented")


@pytest.mark.parametrize("wrapper", sorted(LAVIS_MODELS))
def test_the_pinned_processor_loads_without_root_bert_assets(wrapper: str) -> None:
    """The processor must not depend on `/root/bert-base-uncased`, which no operator has."""
    from workbench.replay import lavis_processors

    directory, _ = LAVIS_MODELS[wrapper]
    source_root = REPO / "workbench" / "third_party" / directory
    # Each model owns a vendored `lavis`, and the pinned package names are global to a process, so
    # this comparison drops the previous checkout's modules. A production run never needs to: one
    # model, one subprocess -- which `test_one_process_prepares_one_checkout` asserts separately.
    lavis_processors.reset()
    processors, boundary = lavis_processors.text_processors(
        source_root, "configs/models/blip2/blip2_pretrain.yaml")
    assert type(processors["eval"]).__name__ == "BlipCaptionProcessor"
    # Native punctuation and whitespace behaviour, observed rather than assumed.
    assert processors["eval"]("  A dog, running!  ") == "a dog, running"
    assert processors["eval"]("Dress.") == "dress"
    assert boundary


def test_one_process_prepares_one_checkout() -> None:
    """The property the wrappers rely on: a second checkout in the same process is refused.

    `evaluate_models.py` launches one model per subprocess, so this refusal is what keeps a run from
    silently resolving another checkout's class names. It is asserted here rather than left to the
    parity tool's `reset`, which exists precisely because a comparison tool does want to switch.
    """
    from workbench.replay import lavis_processors

    lavis_processors.reset()
    lavis_processors.prepare(REPO / "workbench" / "third_party" / "HINT")
    with pytest.raises(ImportError, match="already prepared the lavis context"):
        lavis_processors.prepare(REPO / "workbench" / "third_party" / "HABIT")
    # And the same checkout is a no-op rather than an error, so a second call is safe.
    again = lavis_processors.prepare(REPO / "workbench" / "third_party" / "HINT")
    assert again["reused"] is True
    lavis_processors.reset()


def test_the_integrated_loader_returns_the_native_processor(tmp_path: Path) -> None:
    """The real `_load_replay_modules` for HINT, on the real pinned checkout.

    Only the two heavy neighbours are patched -- `torch` and the pinned `datasets`/`data_utils`/`test`
    modules -- so what runs is the wrapper's own import sequence plus the pinned processor. This is
    the integrated path the evaluator takes; a passing unit test of `lavis_processors` alone would not
    have shown that the wrapper wires it up.
    """
    from unittest.mock import MagicMock, patch

    from workbench.replay import hint

    source_root = REPO / "workbench" / "third_party" / "HINT"
    from workbench.replay import lavis_processors

    lavis_processors.reset()
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def pinned_import(name, *args, **kwargs):
        if name in {"datasets", "data_utils", "test"}:
            return MagicMock(name=name)
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=pinned_import):
        torch_module, datasets, pinned, processors = hint._load_replay_modules(source_root, "pretrain")
    txt_processors, boundary = processors
    assert type(txt_processors["eval"]).__name__ == "BlipCaptionProcessor"
    assert txt_processors["eval"]("  A Dog, Running!  ") == "a dog, running"
    assert boundary


def test_an_unknown_model_type_is_refused_rather_than_defaulted() -> None:
    from workbench.replay import hint

    source_root = REPO / "workbench" / "third_party" / "HINT"
    with pytest.raises(ValueError, match="publishes no 'pretrain_t5' processor config"):
        hint._load_replay_modules(source_root, "pretrain_t5")
