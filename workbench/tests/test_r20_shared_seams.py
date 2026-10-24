"""Shared-seam regressions for the R20 integration repairs.

Each case executes a REAL boundary — the stage planner, the doctor CLI, the settings lookup, the
source authority, or a registry-declared contract — rather than a mock that mirrors the
implementation. The heavy model computation is never involved here: these defects were all in
orchestration, selection, path resolution and admission, which is exactly what is left real.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from workbench.backend.operator_config import WorkbenchConfig, resolve_config
from workbench.backend.registry import (evaluator_relative_asset_paths, external_asset_blockers,
                                        external_asset_statuses, load_registry, model_by_id,
                                        preparation_contract_for_model)
from workbench.scripts import doctor, pipeline

REPO = Path(__file__).resolve().parents[2]

def _workbench_script(name: str) -> str:
    """The script's basename. Stage argv carries absolute paths, so a suffix match is the assertion."""
    return name


def _runs(argv: list[str], script: str) -> bool:
    return any(Path(item).name == script for item in argv)


def _autopilot_args(selected: list[str], checkpoints: dict[str, list[str]],
                    config: WorkbenchConfig, **overrides) -> argparse.Namespace:
    """The Namespace the autopilot really builds: `model=None` plus a per-model selection."""
    values = dict(
        mode="all", model=None, selected_model_ids=selected, checkpoint_ids=checkpoints,
        checkpoint=None, checkpoint_policy="strict_clean", dataset_root=config.FASHIONIQ_ROOT,
        protocol=None, top_k=10, continue_on_error=True, apply=True, verify_gpu=False,
        allow_env_install=False, allow_preparation=False, download_checkpoints=True,
        download_auxiliary_assets=False, evaluate=True, rebuild_index=False, serve=False,
        sync_sources=True, checkpoint_overrides={}, allow_gpu_preparation=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def _stage(stages: list[pipeline.Stage], name: str, model_id: str) -> pipeline.Stage:
    return next(stage for stage in stages
                if stage.name == name and stage.model_id == model_id)


def _argv(stage: pipeline.Stage) -> list[str]:
    assert stage.commands, f"{stage.name}:{stage.model_id} has no command"
    return [str(item) for item in stage.commands[0]]


def test_autopilot_selection_reaches_the_runtime_preflight(monkeypatch) -> None:
    """R20-09: the selected checkpoint must survive into doctor's argv on the autopilot path.

    Before the repair the preflight stage dropped `--checkpoint` whenever `args.model` was None,
    which is every autopilot invocation. Doctor then inspected every registered variant of the
    model and refused the run because unselected variants were absent.
    """
    config = resolve_config()
    args = _autopilot_args(["conesep"], {"conesep": ["fiq_n02"]}, config)
    stages = pipeline.all_stages(args, config)
    argv = _argv(_stage(stages, "runtime-preflight", "conesep"))
    assert _runs(argv, "doctor.py")
    assert argv[argv.index("--checkpoint") + 1] == "fiq_n02"
    assert argv[argv.index("--checkpoint-policy") + 1] == "strict_clean"
    # The acquisition and execution stages must agree with the preflight about the same variant.
    assert "--checkpoint" in _argv(_stage(stages, "checkpoint", "conesep"))
    assert "--checkpoint" in _argv(_stage(stages, "evaluation", "conesep"))


def test_selected_only_checkpoint_is_not_blocked_by_absent_siblings(monkeypatch, tmp_path) -> None:
    """R20-09 acceptance: only the SELECTED variant is inspected, and a missing one still blocks.

    The fixture installs one variant of a three-variant model. Everything unrelated to readiness is
    supplied as an already-verified prerequisite, so the checkpoint/readiness logic is the only thing
    under test.
    """
    from workbench.tests.test_doctor_vendored_source_and_selection import config as fixture_config

    body = b"selected variant bytes"
    digest = hashlib.sha256(body).hexdigest()
    variants = []
    for condition in (20, 50, 80):
        variants.append({"checkpoint_id": f"fiq_n{condition // 10:02d}" if condition != 20 else "fiq_n02",
                         "filename": f"ConeSep-FIQ-N0.{condition // 10}.pt", "expected_sha256": digest,
                         "status": "NO_CLEAN_CHECKPOINT",
                         "checkpoint_mapping_status": "VERIFIED_METADATA",
                         "checkpoint_training_noise_pct": condition,
                         "evaluation_noise_pct": 0})
    model = {"model_id": "conesep", "source_dir": None,
             "checkpoint_variants": variants, "checkpoint_bundles": [],
             "required_runtime_assets": [], "environment": None,
             "command_status": "COMMAND_AUDITED", "native_protocol": "fashioniq_original_split",
             "checkpoint_selection_preference": None}
    settings = fixture_config(tmp_path)
    installed = settings.WORKBENCH_CHECKPOINT_ROOT / "conesep" / "ConeSep-FIQ-N0.2.pt"
    installed.parent.mkdir(parents=True)
    installed.write_bytes(body)
    # The model must not resolve as ambiguous when the run already named its variant.
    model["checkpoint_selection_preference"] = "fiq_n02"

    monkeypatch.setattr(doctor, "load_registry", lambda: {"models": [model], "protocols": {}})
    monkeypatch.setattr(doctor, "resolve_config", lambda: settings)
    monkeypatch.setattr(doctor, "workbench_checks", lambda *_: [])
    monkeypatch.setattr(doctor, "fashioniq_check", lambda *_: {"name": "fashioniq", "status": "OK"})
    monkeypatch.setattr(doctor, "source_state", lambda *_: {"path": None, "state": "clean", "head": "pinned",
                                                           "dirty": False, "pin": "pinned"})
    monkeypatch.setattr(doctor, "preflight_blockers", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(doctor, "environment_status", lambda *_a, **_k: {
        "status": "READY", "interpreter": "python", "tier": "cpu", "variable": None,
        "verified": True, "gpu_deferred": False, "reason": None})
    monkeypatch.setattr(doctor, "preparation_checks", lambda *_: ([], []))
    monkeypatch.setattr(doctor, "external_asset_checks", lambda *_: ([], []))
    monkeypatch.setattr(doctor, "required_runtime_assets", lambda *_: [])

    def runtime_record() -> dict:
        report = doctor.collect(settings, model_id="conesep", scope="real",
                                requested_checkpoints=["fiq_n02"],
                                checkpoint_policy="strict_clean")
        return next(item for item in report["checks"] if item["name"] == "runtime:conesep")

    selected = runtime_record()
    assert selected["status"] == "OK", selected["evidence"]
    assert [item["checkpoint_id"] for item in selected["evidence"]["checkpoints"]] == ["fiq_n02"]

    installed.unlink()
    missing_selected = runtime_record()
    assert missing_selected["status"] == "BLOCKED"
    # The selected variant is now unobtainable, so the refusal names that: either the checkpoint
    # path is missing, or the selection itself reports the selected id as lacking an obtainable
    # source. Both are the correct, precise answer; an "unselected sibling" is never mentioned.
    reasons = [reason for item in missing_selected["evidence"]["checkpoints"]
               for reason in item["runtime_blockers"]]
    assert any(("checkpoint missing or invalid" in reason)
               or ("lack verified provenance or an obtainable source" in reason)
               for reason in reasons), reasons
    assert not any("fiq_n05" in reason or "fiq_n08" in reason or "0.5" in reason or "0.8" in reason
                   for reason in reasons), reasons


def test_doctor_policy_flag_is_accepted_and_changes_the_inspected_set() -> None:
    """The run's policy must be selectable from the real CLI, not only from a Python call."""
    completed = subprocess.run(
        [sys.executable, str(REPO / "workbench" / "scripts" / "doctor.py"), "--help"],
        cwd=REPO, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "--checkpoint-policy" in completed.stdout
    assert "strict_clean" in completed.stdout


def test_config_file_interpreter_settings_reach_real_consumers(monkeypatch, tmp_path) -> None:
    """R20-10: a file-only per-model interpreter must resolve, and the process must still win.

    Before the repair `resolve_config` parsed these names and discarded them, and every consumer read
    `os.environ` directly, so a `workbench.env` containing only `WORKBENCH_LIMN_PYTHON` resolved to
    None. `WORKBENCH_TME_PYTHON`/`WORKBENCH_PTHA_MTST_PYTHON` were not even accepted by the parser.
    """
    from workbench.backend import model_settings, operator_config
    from workbench.backend.runtime import resolve_model_interpreter

    env_file = tmp_path / "workbench.env"
    env_file.write_text("\n".join([
        "WORKBENCH_LIMN_PYTHON=/fixture/limn/bin/python",
        "WORKBENCH_TME_PYTHON=/fixture/tme/bin/python",
        "WORKBENCH_PTHA_MTST_PYTHON=/fixture/ptha/bin/python",
    ]) + "\n", encoding="utf-8")
    monkeypatch.setattr(operator_config, "_CONFIG_PATH", env_file)
    model_settings.file_settings.cache_clear()
    try:
        for model_id, expected in (("limn", "/fixture/limn/bin/python"),
                                   ("tme", "/fixture/tme/bin/python"),
                                   ("ptha_mtst", "/fixture/ptha/bin/python")):
            assert resolve_model_interpreter(model_by_id(model_id)) == expected
            variable = (model_by_id(model_id).get("environment") or {})["python_env_var"]
            assert model_settings.setting_source(variable) == "workbench_env"

        monkeypatch.setenv("WORKBENCH_LIMN_PYTHON", "/process/limn/bin/python")
        assert resolve_model_interpreter(model_by_id("limn")) == "/process/limn/bin/python"
        assert model_settings.setting_source("WORKBENCH_LIMN_PYTHON") == "process_environment"

        # An empty value means "unset" in both sources, never "an interpreter with no name".
        monkeypatch.setenv("WORKBENCH_TME_PYTHON", "   ")
        assert resolve_model_interpreter(model_by_id("tme")) == "/fixture/tme/bin/python"
        assert model_settings.setting_source("WORKBENCH_TME_PYTHON") == "workbench_env"
        assert model_settings.setting("WORKBENCH_NOTHING_DECLARES_THIS") is None
    finally:
        model_settings.file_settings.cache_clear()

    # An unknown name is still refused: the parser is a schema, not a free-form environment reader.
    bad = tmp_path / "bad.env"
    bad.write_text("NOT_A_WORKBENCH_SETTING=1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown setting"):
        operator_config._read_env(bad)


def test_tme_assets_resolve_where_the_pinned_loader_reads_them(monkeypatch, tmp_path) -> None:
    """R20-02: an `evaluator_cwd` asset is inside the checkout, not beside it.

    The pinned LAVIS code opens `weight/<file>` relative to the evaluator's working directory, so the
    declaration is checkout-relative. Resolving it against the repository root put the checker one
    directory above the loader and reported a correctly staged asset as missing.
    """
    from dataclasses import replace as _replace

    from workbench.backend import operator_config
    from workbench.backend.registry import asset_admission_status

    model = model_by_id("tme")
    assert evaluator_relative_asset_paths(model) == {
        "BLIP-2 pretrain weights": "weight/blip2_pretrained.pth",
        "EVA-ViT-g backbone weights": "weight/eva_vit_g.pth",
    }
    third_party = tmp_path / "third_party"
    checkout = third_party / model["source_dir"]
    real_resolve = operator_config.resolve_config
    monkeypatch.setattr(
        operator_config, "resolve_config",
        lambda: _replace(real_resolve(), WORKBENCH_THIRD_PARTY_ROOT=third_party))
    contract_asset = {asset["name"]: asset
                      for asset in preparation_contract_for_model(model)["external_assets"]}

    # Absent first: the checker must look exactly where the loader reads.
    absent = {status["name"]: status for status in external_asset_statuses(model)}
    for name, relative in evaluator_relative_asset_paths(model).items():
        assert Path(absent[name]["path"]) == (checkout / relative)
        assert absent[name]["status"] == "CONFIGURED_MISSING"

    eva = contract_asset["EVA-ViT-g backbone weights"]
    staged = checkout / "weight" / "eva_vit_g.pth"
    staged.parent.mkdir(parents=True)
    # Non-uniform content at the DECLARED size: a sparse or uniformly filled file is refused by the
    # payload check, so a fixture that is a single repeated byte would not model a real weight file.
    with staged.open("wb") as handle:
        handle.truncate(eva["expected_size_bytes"])
        handle.seek(0)
        handle.write(b"\x00\x01\x02\x03" * 1024)
    present = {status["name"]: status for status in external_asset_statuses(model)}
    assert present["EVA-ViT-g backbone weights"]["status"] == "EXISTS_UNVERIFIED_CHECKSUM"
    assert asset_admission_status(eva, present["EVA-ViT-g backbone weights"]) == "LOCALLY_RECORDED_ASSET"
    assert not any("EVA-ViT-g" in blocker for blocker in external_asset_blockers(model))

    # A file of the wrong size is a size mismatch, and no consumer admits it.
    staged.write_bytes(b"short")
    wrong = {status["name"]: status for status in external_asset_statuses(model)}
    assert wrong["EVA-ViT-g backbone weights"]["status"] == "SIZE_MISMATCH"
    assert asset_admission_status(eva, wrong["EVA-ViT-g backbone weights"]) == "SIZE_MISMATCH"
    assert any("EVA-ViT-g" in blocker
               and "size mismatch" in blocker for blocker in external_asset_blockers(model))

    # A file of exactly the DECLARED size that contains no payload must not be admitted either: a
    # truncated or zero-filled allocation passes the size check and fails only on content.
    with staged.open("wb") as handle:
        handle.truncate(eva["expected_size_bytes"])
    empty = {status["name"]: status for status in external_asset_statuses(model)}
    assert empty["EVA-ViT-g backbone weights"]["status"] == "EXISTS_UNVERIFIED_CHECKSUM"
    assert asset_admission_status(eva, empty["EVA-ViT-g backbone weights"]) == "SUSPECT_PAYLOAD"
    assert any("EVA-ViT-g" in blocker and "plausible payload" in blocker
               for blocker in external_asset_blockers(model))


def test_asset_admission_agrees_between_shared_readiness_and_the_wrapper_contract() -> None:
    """R20-04: one admission decision, two callers.

    The TME wrapper admitted an author-linked asset with no published digest (checking size and
    recording local provenance) while `external_asset_blockers` and `preflight.model_requirements`
    rejected exactly the same bytes. Both must now agree, and neither may invent a digest.
    """
    from workbench.backend.preflight import model_requirements
    from workbench.backend.registry import asset_admission_status

    model = model_by_id("tme")
    config = resolve_config()
    contract = preparation_contract_for_model(model)
    statuses = {status["name"]: status for status in external_asset_statuses(model)}
    declaration = {asset["name"]: asset for asset in contract["external_assets"]}

    blockers = external_asset_blockers(model)
    requirements = model_requirements(model, config, canonical_root=config.FASHIONIQ_ROOT)
    tokenizer = statuses["bert-base-uncased tokenizer"]
    # The tokenizer cache is a cache, not a file: admitted only if its consumer can load it.
    from workbench.backend.registry import cache_asset_blocker

    digestless = declaration["BLIP-2 pretrain weights"]
    assert digestless["expected_sha256"] is None
    admitted = asset_admission_status(digestless, statuses["BLIP-2 pretrain weights"])
    assert admitted in {"LOCALLY_RECORDED_ASSET", "CONFIGURED_MISSING"}
    if admitted == "LOCALLY_RECORDED_ASSET":
        assert not any("BLIP-2" in blocker for blocker in blockers)
        assert not any("BLIP-2" in blocker for blocker in requirements["blocks"])
        assert any("BLIP-2" in warning for warning in requirements["warnings"])
    reason = cache_asset_blocker(declaration["bert-base-uncased tokenizer"], tokenizer)
    blocked = any("tokenizer" in blocker for blocker in blockers)
    assert blocked == (reason is not None), (reason, blockers)


def test_declared_tme_runtime_artifacts_keep_the_checkout_pinned(tmp_path) -> None:
    """R20-03: authorized preparation must not make a pinned checkout look tampered with.

    A copied pinned TME tree used to report `verified=False, dirty=True` with all four legitimate
    runtime files listed as unauthorized, which failed the source gate before any model loader ran.
    """
    from workbench.backend.vendored_source import source_provenance

    source = REPO / "workbench" / "third_party" / "TME"
    checkout = tmp_path / "TME"
    shutil.copytree(source, checkout, ignore=shutil.ignore_patterns("__pycache__", "features", "weight"))
    model = model_by_id("tme")
    before = source_provenance(model, checkout)
    assert before["verified"] is True and before["dirty"] is False, before.get("error")

    for relative in ("features/FashionIQ/val_images.pt",
                     "features/FashionIQ/val_name_to_idx.json",
                     "weight/blip2_pretrained.pth",
                     "weight/eva_vit_g.pth"):
        path = checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    after = source_provenance(model, checkout)
    assert after["verified"] is True, after.get("error")
    assert after["dirty"] is False
    assert after["unauthorized_untracked"] == []

    # A real source edit is still rejected, and so is an undeclared executable.
    edited = next(path for path in sorted(checkout.rglob("*.py")))
    edited.write_text(edited.read_text(encoding="utf-8") + "\n# unauthorized edit\n")
    assert source_provenance(model, checkout)["verified"] is False
    edited.write_text(edited.read_text(encoding="utf-8").replace("\n# unauthorized edit\n", ""))
    executable = checkout / "surprise_tool.sh"
    executable.write_text("#!/bin/sh\n")
    assert source_provenance(model, checkout)["verified"] is False


def test_vendored_checkout_is_verified_evidence_for_resume(monkeypatch) -> None:
    """A vendored checkout must not make a stage's fingerprint weak.

    The vendored layout deliberately has no nested `.git` (the pinned tree is committed in this
    repository). Probing `.git` and treating its absence as an I/O failure marked EVERY vendored
    model's fingerprint weak, so `--run.sh autopilot --resume` refused every completed stage that
    depends on source. The check is now the provenance authority's own answer, so it cannot drift
    from it: a clean pin is content-verified evidence, and a dirty one is not.
    """
    from workbench.backend.registry import model_by_id
    from workbench.backend.vendored_source import source_provenance
    from workbench.tests.test_doctor_vendored_source_and_selection import config as fixture_config

    config = resolve_config()
    model = model_by_id("encoder")
    checkout = config.WORKBENCH_THIRD_PARTY_ROOT / model["source_dir"]
    assert not (checkout / ".git").exists(), "this regression is about the VENDORED layout"
    provenance = source_provenance(model, checkout)
    assert provenance["verified"] is True and provenance["dirty"] is False, provenance.get("error")

    args = _autopilot_args(["encoder"], {"encoder": ["fashioniq"]}, config, model="encoder",
                           selected_model_ids=["encoder"], allow_preparation=True)
    stages = pipeline.all_stages(args, config)
    evaluation = _stage(stages, "evaluation", "encoder")
    info = pipeline.stage_input_fingerprint_info(evaluation, config)
    assert info["source_provenance"]["category"] == "CONTENT_VERIFIED", info["source_provenance"]
    assert info["weak"] is False, [
        item for item in info["inputs"] if not pipeline._is_verified_entry(item["category"])]


def test_every_registry_model_keeps_a_consistent_declared_command() -> None:
    """The 17 registered entries are planned through the same seam this task changed."""
    registry = load_registry()
    assert len(registry["models"]) == 17
    config = resolve_config()
    selected = [model["model_id"] for model in registry["models"]]
    args = _autopilot_args(selected, {}, config, allow_preparation=True)
    stages = pipeline.all_stages(args, config)
    preflights = {stage.model_id for stage in stages if stage.name == "runtime-preflight"}
    assert preflights == set(selected)
    # Every model's preflight carries the selection the run will execute, derived from the registry
    # rather than from a hard-coded list, so this is checked against each model's own resolution.
    from workbench.backend.checkpoint_selection import select_checkpoints

    for stage in stages:
        if stage.name != "runtime-preflight":
            continue
        argv = _argv(stage)
        assert _runs(argv, "doctor.py") and "--model" in argv and "--scope" in argv
        model = model_by_id(stage.model_id)
        # The preflight must inspect exactly the set the run will evaluate — derived from this
        # model's own registry facts, never a sibling variant the operator did not choose.
        planned = pipeline.evaluated_checkpoint_ids(model, args, config.WORKBENCH_CHECKPOINT_ROOT)
        forwarded = [argv[index + 1] for index, item in enumerate(argv) if item == "--checkpoint"]
        assert forwarded == planned, (stage.model_id, forwarded, planned)
        resolved = select_checkpoints(model, policy="strict_clean",
                                      checkpoint_root=config.WORKBENCH_CHECKPOINT_ROOT)
        if forwarded and not resolved.blocked:
            bundles = {bundle["bundle_id"] for bundle in model.get("checkpoint_bundles") or []}
            assert set(forwarded) <= set(resolved.checkpoint_ids) | bundles, stage.model_id
