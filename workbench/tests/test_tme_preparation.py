from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
import pytest

import torch

from workbench.backend.operator_config import resolve_config
from workbench.backend.registry import load_registry, preparation_contract_for_model
from workbench.scripts import prepare_model

ROOT = Path(__file__).resolve().parents[2]
MODEL = next(model for model in load_registry()["models"] if model["model_id"] == "tme")
CONTRACT = preparation_contract_for_model(MODEL)
PARAMETERS = next(item["parameters"] for item in CONTRACT["generated_artifacts"]
                  if item["generator"] == prepare_model.TME_FEATURE_GENERATOR)


def _fixture(tmp_path: Path, monkeypatch):
    checkout = tmp_path / "TME"
    script = checkout / PARAMETERS["source_script"]
    script.parent.mkdir(parents=True)
    script.write_text("# native runner is replaced by subprocess stub\n")
    dataset = tmp_path / "FashionIQ"
    splits = dataset / "image_splits"
    splits.mkdir(parents=True)
    for category in ("dress", "shirt", "toptee"):
        image_name = f"{category}_0"
        (splits / f"split.{category}.val.json").write_text(json.dumps([image_name]))
        (dataset / "captions").mkdir(exist_ok=True)
        (dataset / "captions" / f"cap.{category}.val.json").write_text("[]")
        (dataset / "images").mkdir(exist_ok=True)
        (dataset / "images" / f"{image_name}.png").write_bytes(b"raw png stand-in")
    config = replace(resolve_config(), FASHIONIQ_ROOT=dataset,
                     WORKBENCH_THIRD_PARTY_ROOT=tmp_path / "third_party",
                     WORKBENCH_ARTIFACTS_ROOT=tmp_path / "artifacts")
    config.WORKBENCH_THIRD_PARTY_ROOT.mkdir()
    (config.WORKBENCH_THIRD_PARTY_ROOT / MODEL["source_dir"]).symlink_to(checkout, target_is_directory=True)
    from workbench.backend import registry
    monkeypatch.setattr(registry, "required_runtime_assets", lambda _: [])
    monkeypatch.setattr(registry, "external_asset_statuses", lambda _: [])
    contract = json.loads(json.dumps(CONTRACT))
    # Tiny tensor dims keep the receipt boundary real without allocating the 257x1408 cache.
    artifact = next(a for a in contract["generated_artifacts"]
                    if a["generator"] == prepare_model.TME_FEATURE_GENERATOR)
    artifact["parameters"].update(tokens_per_image=2, feature_width=3, feature_dtype="float16")
    # RESOLVED parameters, not the raw declaration: the receipt must record the batch the generator
    # is actually launched with, and an operator's `WORKBENCH_TME_BATCH_SIZE` legitimately changes
    # it. A fixture that wrote the pinned 512 built a receipt no environment with an override can
    # match, so the reuse path was untested exactly where the override is configured.
    return config, checkout, dataset, contract, prepare_model.tme_generation_parameters(artifact["parameters"])


def _write_valid_cache(config, checkout, contract, parameters, monkeypatch):
    _, feature_root = prepare_model._tme_artifact(config, checkout, contract)
    feature_root.mkdir(parents=True, exist_ok=True)
    names = [f"{category}_0" for category in ("dress", "shirt", "toptee")]
    features = torch.zeros((3, 2, 3), dtype=torch.float16)
    feature = feature_root / "val_images.pt"
    mapping = feature_root / "val_name_to_idx.json"
    torch.save(features, feature)
    mapping.write_text(json.dumps({name: index for index, name in enumerate(names)}))
    inputs = prepare_model._tme_inputs(config, contract, MODEL)
    files = {"features": {"size_bytes": feature.stat().st_size,
                          "sha256": prepare_model._file_sha256(feature)},
             "name_to_idx": {"size_bytes": mapping.stat().st_size,
                             "sha256": prepare_model._file_sha256(mapping)}}
    receipt = prepare_model.tme_receipt_path(config, checkout, contract)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({
        "kind": "workbench_tme_precompute_receipt", "model_id": "tme",
        "generator": prepare_model.TME_FEATURE_GENERATOR,
        "source_script": parameters["source_script"], "source_commit": MODEL["upstream_commit_sha"],
        "dataset": parameters["eval_dataset"], "split": parameters["split"], "inputs": inputs,
        "batch_size": parameters["batch_size"], "workers": parameters["workers"],
        "files": files, "feature_shape": [3, 2, 3], "feature_dtype": "torch.float16",
        "interpreter": sys.executable,
    }))
    return feature_root, feature, mapping, receipt


def _stub_verified_source(monkeypatch, checkout):
    monkeypatch.setattr("workbench.backend.runtime.source_clean_and_pinned",
                        lambda model, source: (True, MODEL["upstream_commit_sha"], None))
    monkeypatch.setattr("workbench.backend.runtime.require_verified_model_interpreter",
                        lambda model, config=None: sys.executable)


def test_cpu_dispatch_reaches_cuda_blocker_before_native_command(tmp_path, monkeypatch, capsys):
    config, checkout, _, _, _ = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(prepare_model, "resolve_config", lambda: config)
    monkeypatch.setattr(prepare_model, "_stage_source_assets", lambda *args, **kwargs: True)
    _stub_verified_source(monkeypatch, checkout)
    calls = []

    def probe_only(command, **kwargs):
        if not (len(command) > 2 and command[1] == "-c" and "torch.cuda.is_available()" in command[2]):
            # The interpreter-IDENTITY probe (`-I -S`) is a different question asked of the same
            # binary, and this stub answers only the CUDA one. Treating every call as the CUDA probe
            # made the test depend on an earlier blocker short-circuiting before identity was ever
            # resolved: with a valid operator `workbench.env` the run reached identity resolution and
            # the stub failed on a call it was never meant to classify.
            return subprocess.CompletedProcess(command, 0, "", "")
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "False\n", "")

    monkeypatch.setattr(prepare_model.subprocess, "run", probe_only)
    result = prepare_model.main(["--model", "tme", "--execute", "--allow-preparation",
                                 "--allow-gpu-preparation"])
    captured = capsys.readouterr()

    assert result == 1
    assert len(calls) == 1
    assert "requires a CUDA-capable GPU" in captured.err
    assert "manual preparation required" not in captured.err


def test_valid_receipt_cache_reuses_without_native_generator(tmp_path, monkeypatch):
    config, checkout, _, contract, parameters = _fixture(tmp_path, monkeypatch)
    _stub_verified_source(monkeypatch, checkout)
    feature_root, _, _, _ = _write_valid_cache(config, checkout, contract, parameters, monkeypatch)
    assert prepare_model.tme_features_complete(config, MODEL, contract) is True
    called = []
    monkeypatch.setattr(prepare_model.subprocess, "run", lambda *args, **kwargs: called.append(args))

    ok, reason = prepare_model._run_tme_precompute(config, MODEL, contract, dry_run=False)

    assert ok and "reusing validated" in reason
    assert not called


def test_partial_cache_and_changed_split_never_validate_or_reuse(tmp_path, monkeypatch):
    config, checkout, dataset, contract, parameters = _fixture(tmp_path, monkeypatch)
    _stub_verified_source(monkeypatch, checkout)
    feature_root, feature, mapping, receipt = _write_valid_cache(config, checkout, contract, parameters, monkeypatch)
    assert prepare_model.tme_features_complete(config, MODEL, contract) is True
    mapping.unlink()
    assert prepare_model.tme_features_complete(config, MODEL, contract) is False
    mapping.write_text(json.dumps({"wrong": 0}))
    assert prepare_model.tme_features_complete(config, MODEL, contract) is False

    _write_valid_cache(config, checkout, contract, parameters, monkeypatch)
    assert prepare_model.tme_features_complete(config, MODEL, contract) is True
    split = dataset / "image_splits" / "split.dress.val.json"
    split.write_text('["changed"]')
    (dataset / "images" / "changed.png").write_bytes(b"changed raw image")
    assert prepare_model.tme_features_complete(config, MODEL, contract) is False
    ok, reason = prepare_model._run_tme_precompute(config, MODEL, contract, dry_run=False)
    assert not ok and "--force-regenerate-unknown-cache" in reason
    assert feature.is_file() and receipt.is_file()


@pytest.mark.parametrize("change, expected_calls", [("none", 0), ("image", 1), ("batch", 1)])
def test_production_tme_preparation_resume_binds_gallery_and_batch(
        tmp_path, monkeypatch, capsys, change, expected_calls):
    from workbench.backend import runtime
    from workbench.backend import registry
    from workbench.scripts import pipeline

    monkeypatch.setenv("WORKBENCH_TME_BATCH_SIZE", "32")
    config, checkout, dataset, contract, parameters = _fixture(tmp_path, monkeypatch)
    _stub_verified_source(monkeypatch, checkout)
    monkeypatch.setattr(prepare_model, "resolve_config", lambda: config)
    monkeypatch.setattr(registry, "preparation_contract_for_model", lambda model: contract)
    monkeypatch.setattr(runtime, "source_provenance", lambda model, source: {
        "actual_commit": MODEL["upstream_commit_sha"], "dirty": False,
        "provenance_digest": "fixture-source"})
    monkeypatch.setattr(runtime, "verify_environment", lambda *args, **kwargs: object())
    monkeypatch.setattr(runtime, "_cached_gpu_capability", lambda *args, **kwargs: None)
    _, _, _, receipt = _write_valid_cache(config, checkout, contract, parameters, monkeypatch)

    if change == "image":
        (dataset / "images" / "dress_0.png").write_bytes(b"changed canonical gallery bytes")
    elif change == "batch":
        monkeypatch.setenv("WORKBENCH_TME_BATCH_SIZE", "9")

    args = pipeline.parser_for().parse_args([
        "all", "--model", "tme", "--apply", "--allow-preparation", "--allow-gpu-preparation"])
    prep = next(stage for stage in pipeline.all_stages(args, config)
                if stage.name == "preparation" and stage.model_id == "tme")
    state_path = tmp_path / "pipeline_state.json"
    initial = pipeline.stage_input_fingerprint_info(prep, config, {"stages": {}})
    state_path.write_text(json.dumps({"stages": {pipeline.stage_key(prep): {
        "stage_id": pipeline.stage_key(prep), "name": "preparation", "model_id": "tme",
        "status": "COMPLETE", "fingerprint": initial["digest"],
        "output_fingerprints": pipeline.compute_stage_output_fingerprints(prep)}}}))
    if change != "none":
        legacy_calls = []
        monkeypatch.setattr(pipeline, "run_command", lambda command: legacy_calls.append(command) or 1)
        legacy_result = pipeline.run_stages(
            [replace(prep, resume_refusal_reason=None)], apply=True, resume=True,
            authorized_capabilities={"allow_preparation", "allow_gpu_preparation"},
            config=config, mode="all", state_path=state_path)
        legacy_output = capsys.readouterr().out
        print(f"D1_LEGACY_CASE={change} return={legacy_result} command_invocations={len(legacy_calls)} "
              f"output={legacy_output!r}")
        assert legacy_result == 0 and not legacy_calls
        assert "[SKIP] resume: already completed" in legacy_output
        capsys.readouterr()

    calls = []

    def execute(command):
        calls.append(command)
        return 1

    monkeypatch.setattr(pipeline, "run_command", execute)
    result = pipeline.run_stages(
        [prep], apply=True, resume=True,
        authorized_capabilities={"allow_preparation", "allow_gpu_preparation"},
        config=config, mode="all", state_path=state_path)
    output = capsys.readouterr().out
    print(f"D1_CASE={change} return={result} command_invocations={len(calls)} "
          f"receipt={receipt.is_file()} output={output!r}")

    assert len(calls) == expected_calls
    if change == "none":
        assert result == 0
        assert "[SKIP] resume: already completed" in output
    else:
        assert result == 1
        assert "[INFO] resume refused: preparation self-check failed: TME feature receipt does not bind current source, gallery bytes, or resolved generation parameters" in output
        assert "[RUN]" in output
        assert "[SKIP] resume: already completed" not in output
        assert receipt.is_file()


def test_relative_cli_paths_match_absolute_preflight(tmp_path):
    checkout = tmp_path / "TME"
    (checkout / "src").mkdir(parents=True)
    native = ROOT / "workbench" / "third_party" / "TME" / "src"
    (checkout / "src" / "precompute_test.py").write_text("# fixture\n")
    (checkout / "src" / "precompute_evaluation.py").symlink_to(native / "precompute_evaluation.py")
    (checkout / "src" / "data.py").symlink_to(native / "data.py")
    dataset = tmp_path / "FashionIQ"
    (dataset / "captions").mkdir(parents=True)
    (dataset / "image_splits").mkdir()
    for category in ("dress", "shirt", "toptee"):
        (dataset / "captions" / f"cap.{category}.val.json").write_text("[]")
        (dataset / "image_splits" / f"split.{category}.val.json").write_text("[]")
    feature = checkout / "features" / "FashionIQ"
    feature.mkdir(parents=True)
    (feature / "val_images.pt").write_bytes(b"fixture")
    (feature / "val_name_to_idx.json").write_text("{}")
    checkpoint = tmp_path / "checkpoints"
    checkpoint.mkdir()
    checkpoint_name = next(item["filename"] for item in MODEL["checkpoint_variants"]
                           if item["checkpoint_id"] == "fashioniq_0")
    (checkpoint / checkpoint_name).write_bytes(b"fixture")

    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    args = [sys.executable, "-m", "workbench.replay.tme", "--preflight-only",
            "--source-root", str(checkout), "--dataset-root", str(dataset),
            "--checkpoint-root", str(checkpoint)]
    absolute = subprocess.run(args, cwd=ROOT, env=env, text=True, capture_output=True)
    assert absolute.returncode != 0  # required runtime assets remain unavailable in this isolated fixture
    relative_args = [sys.executable, "-m", "workbench.replay.tme", "--preflight-only",
                     "--source-root", checkout.name, "--dataset-root", dataset.name,
                     "--checkpoint-root", checkpoint.name]
    relative = subprocess.run(relative_args, cwd=tmp_path, env=env, text=True, capture_output=True)
    # Both invocations report the same existing checkpoint and cache paths; only missing runtime assets block.
    assert "checkpoint missing" not in absolute.stderr
    assert "feature cache file missing" not in absolute.stderr
    assert "checkpoint missing" not in relative.stderr
    assert "feature cache file missing" not in relative.stderr


def test_tme_generator_child_gets_workbench_hf_home(tmp_path, monkeypatch):
    config, checkout, _, contract, parameters = _fixture(tmp_path, monkeypatch)
    _stub_verified_source(monkeypatch, checkout)
    from workbench.backend import model_settings, registry

    cache_root = tmp_path / "configured-hf"
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(model_settings, "file_settings", lambda: {"HF_HOME": str(cache_root)})
    monkeypatch.setattr(registry, "external_asset_blockers", lambda *args, **kwargs: [])
    output_root, output = prepare_model._tme_artifact(config, checkout, contract)
    names = [f"{category}_0" for category in ("dress", "shirt", "toptee")]
    calls = []

    def launch(command, **kwargs):
        calls.append(kwargs)
        if "torch.cuda.is_available()" in command[-1]:
            return subprocess.CompletedProcess(command, 0, "True\n", "")
        assert kwargs["env"]["HF_HOME"] == str(cache_root)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(torch.zeros((3, 2, 3), dtype=torch.float16), output / "val_images.pt")
        (output / "val_name_to_idx.json").write_text(json.dumps({name: i for i, name in enumerate(names)}))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(prepare_model.subprocess, "run", launch)
    ok, reason = prepare_model._run_tme_precompute(config, MODEL, contract, dry_run=False)
    assert ok, reason
    assert len(calls) == 2
    assert "HF_HOME" not in os.environ
