from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


from workbench.backend.operator_config import WorkbenchConfig
from workbench.backend.registry import load_registry


def load_module():
    spec = importlib.util.spec_from_file_location("evaluate_models", Path("workbench/scripts/evaluate_models.py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module



def config(tmp_path: Path) -> WorkbenchConfig:
    return WorkbenchConfig(tmp_path, tmp_path / "data", tmp_path / "FashionIQ", "127.0.0.1", 8000, 5173, tmp_path / "checkpoints", tmp_path / "results", tmp_path / "third_party")
def records():
    return {model["model_id"]: model for model in load_registry()["models"]}


def prepare_source(monkeypatch, module, tmp_path: Path, model: dict) -> tuple[Path, Path, WorkbenchConfig]:
    settings = config(tmp_path)
    source = settings.WORKBENCH_THIRD_PARTY_ROOT / model["source_dir"]
    source.mkdir(parents=True)
    checkpoint = model["checkpoint_variants"][0]
    checkpoint_file = settings.WORKBENCH_CHECKPOINT_ROOT / model["model_id"] / checkpoint["filename"]
    checkpoint_file.parent.mkdir(parents=True)
    checkpoint_file.write_bytes(b"checkpoint")
    monkeypatch.setattr(module, "resolve_config", lambda: settings)
    monkeypatch.setattr(module, "pinned_revision", lambda _: model["upstream_commit_sha"])
    return source, checkpoint_file, settings

def prepare_ilearn_layout(root: Path) -> None:
    for directory in (
        root / "captions",
        root / "image_splits",
        root / "resized_image",
        *(root / "resized_image" / category for category in ("dress", "shirt", "toptee")),
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for category in ("dress", "shirt", "toptee"):
        (root / "captions" / f"cap.{category}.val.json").write_text("[]")
        (root / "captions" / f"correction_dict_{category}.json").write_text("{}")
        (root / "image_splits" / f"split.{category}.val.json").write_text("[]")


def test_unaudited_adapter_is_skipped() -> None:
    module = load_module()
    model = records()["hint"]
    plan, reasons = module.guarded_plan(model, model["checkpoint_variants"][0], model["native_protocol"], None, 200)
    assert plan is None
    assert reasons == ["SKIPPED: adapter command not audited"]


def test_missing_checkpoint_is_blocked(monkeypatch, tmp_path: Path) -> None:
    module = load_module()
    model = records()["csmcir"]
    settings = config(tmp_path)
    source = settings.WORKBENCH_THIRD_PARTY_ROOT / model["source_dir"]
    (source / "fashionIQ_dataset").mkdir(parents=True)
    monkeypatch.setattr(module, "resolve_config", lambda: settings)
    monkeypatch.setattr(module, "pinned_revision", lambda _: model["upstream_commit_sha"])

    plan, reasons = module.guarded_plan(model, model["checkpoint_variants"][0], model["native_protocol"], source / "fashionIQ_dataset", 200, settings)

    assert plan is None
    assert any(reason.startswith("checkpoint missing:") for reason in reasons)


def test_csmcir_dry_run_constructs_official_command(monkeypatch, tmp_path: Path, capsys) -> None:
    module = load_module()
    model = records()["csmcir"]
    source, checkpoint_file, settings = prepare_source(monkeypatch, module, tmp_path, model)
    settings.FASHIONIQ_ROOT.mkdir()
    dataset_root = source / "fashionIQ_dataset"
    dataset_root.symlink_to(settings.FASHIONIQ_ROOT, target_is_directory=True)
    for path in (
        dataset_root / "captions", dataset_root / "image_splits", dataset_root / "images",
        source / "COT_ours2" / "fashioniq", dataset_root / "qwen_captions",
    ):
        path.mkdir(parents=True)
    for category in ("dress", "shirt", "toptee"):
        (source / "COT_ours2" / "fashioniq" / f"{category}_cot_val.json").write_text("")
        (dataset_root / "qwen_captions" / f"{category}_cot_val.json").write_text("")
    script = source / "src" / "validate_blip_csmcir.py"
    script.parent.mkdir()
    script.write_text("")
    import workbench.backend.adapters.models as adapters

    monkeypatch.setattr(adapters, "checkpoint_path", lambda *_: checkpoint_file)
    plan, reasons = module.guarded_plan(model, model["checkpoint_variants"][0], model["native_protocol"], dataset_root, 200, settings)
    result = module.main([
        "--model", "csmcir", "--checkpoint", "fashioniq", "--protocol", model["native_protocol"],
        "--dataset-root", str(dataset_root), "--dry-run",
    ])

    assert result == 0
    assert reasons == []
    assert plan is not None
    assert plan.cwd == source / "src"
    assert plan.command == ["python", str(script), "--dataset", "fashionIQ", "--blip-model-path", str(checkpoint_file)]
    assert "[RUN] command:" in capsys.readouterr().out


def test_csmcir_main_accepts_link_and_explicit_canonical_root(monkeypatch, tmp_path: Path) -> None:
    module = load_module()
    model = records()["csmcir"]
    source, checkpoint_file, settings = prepare_source(monkeypatch, module, tmp_path, model)
    canonical = tmp_path / "custom-FashionIQ"
    canonical.mkdir()
    layout = source / "fashionIQ_dataset"
    layout.symlink_to(canonical, target_is_directory=True)
    for path in (layout / "captions", layout / "image_splits", layout / "images", source / "COT_ours2" / "fashioniq", layout / "qwen_captions"):
        path.mkdir(parents=True)
    for category in ("dress", "shirt", "toptee"):
        (source / "COT_ours2" / "fashioniq" / f"{category}_cot_val.json").write_text("")
        (layout / "qwen_captions" / f"{category}_cot_val.json").write_text("")
    script = source / "src" / "validate_blip_csmcir.py"
    script.parent.mkdir()
    script.write_text("")
    import workbench.backend.adapters.models as adapters

    monkeypatch.setattr(adapters, "checkpoint_path", lambda *_: checkpoint_file)
    assert module.main(["--model", "csmcir", "--checkpoint", "fashioniq", "--dataset-root", str(layout), "--canonical-dataset-root", str(canonical), "--dry-run"]) == 0


def test_encoder_requires_native_layout_before_openclip(monkeypatch, tmp_path: Path) -> None:
    module = load_module()
    model = records()["encoder"]
    source, _, _ = prepare_source(monkeypatch, module, tmp_path, model)
    (source / "evaluate_model.py").write_text("")
    dataset_root = tmp_path / "FashionIQ"
    dataset_root.mkdir()

    plan, reasons = module.guarded_plan(model, model["checkpoint_variants"][0], model["native_protocol"], dataset_root, 200)

    assert plan is None
    assert reasons == [
        f"FashionIQ fashioniq_ilearn_resized requirement missing: {dataset_root / 'captions'}",
        f"ENCODER asset missing: {source / 'open_clip_pytorch_model.bin'}",
        f"ENCODER evaluator import missing: {source / 'datasets1.py'}",
    ]


def test_encoder_reports_actual_missing_layout_path(monkeypatch, tmp_path: Path) -> None:
    module = load_module()
    model = records()["encoder"]
    source, _, _ = prepare_source(monkeypatch, module, tmp_path, model)
    for name in ("evaluate_model.py", "datasets1.py", "open_clip_pytorch_model.bin"):
        (source / name).write_text("")
    dataset_root = tmp_path / "FashionIQ"
    prepare_ilearn_layout(dataset_root)
    missing = dataset_root / "captions" / "correction_dict_shirt.json"
    missing.unlink()

    plan, reasons = module.guarded_plan(model, model["checkpoint_variants"][0], model["native_protocol"], dataset_root, 200)

    assert plan is None
    assert reasons == [f"FashionIQ fashioniq_ilearn_resized requirement missing: {missing}"]


def test_encoder_constructs_plan_with_complete_native_layout(monkeypatch, tmp_path: Path) -> None:
    module = load_module()
    model = records()["encoder"]
    source, checkpoint, _ = prepare_source(monkeypatch, module, tmp_path, model)
    for name in ("evaluate_model.py", "datasets1.py", "open_clip_pytorch_model.bin"):
        (source / name).write_text("")
    dataset_root = tmp_path / "FashionIQ"
    prepare_ilearn_layout(dataset_root)

    plan, reasons = module.guarded_plan(model, model["checkpoint_variants"][0], model["native_protocol"], dataset_root, 200)

    assert reasons == []
    assert plan is not None
    assert plan.checkpoint == checkpoint
    assert plan.command[plan.command.index("--fashioniq_path") + 1] == f"{dataset_root}/"


def test_encoder_complete_layout_still_blocks_missing_evaluator_import(monkeypatch, tmp_path: Path) -> None:
    module = load_module()
    model = records()["encoder"]
    source, _, _ = prepare_source(monkeypatch, module, tmp_path, model)
    for name in ("evaluate_model.py", "open_clip_pytorch_model.bin"):
        (source / name).write_text("")
    dataset_root = tmp_path / "FashionIQ"
    prepare_ilearn_layout(dataset_root)

    plan, reasons = module.guarded_plan(model, model["checkpoint_variants"][0], model["native_protocol"], dataset_root, 200)

    assert plan is None
    assert reasons == [f"ENCODER evaluator import missing: {source / 'datasets1.py'}"]


def test_encoder_complete_layout_and_import_still_blocks_openclip(monkeypatch, tmp_path: Path) -> None:
    module = load_module()
    model = records()["encoder"]
    source, _, _ = prepare_source(monkeypatch, module, tmp_path, model)
    for name in ("evaluate_model.py", "datasets1.py"):
        (source / name).write_text("")
    dataset_root = tmp_path / "FashionIQ"
    prepare_ilearn_layout(dataset_root)

    plan, reasons = module.guarded_plan(model, model["checkpoint_variants"][0], model["native_protocol"], dataset_root, 200)

    assert plan is None
    assert reasons == [f"ENCODER asset missing: {source / 'open_clip_pytorch_model.bin'}"]

def test_encoder_command_preserves_upstream_trailing_root_separator(tmp_path: Path) -> None:
    from workbench.backend.adapters.models import EncoderAdapter
    from workbench.backend.adapters.base import EvalRequest

    command = EncoderAdapter().command(
        tmp_path / "ENCODER",
        tmp_path / "fashioniq.pt",
        EvalRequest("encoder", "fashioniq", "fashioniq_val_split", tmp_path / "FashionIQ", tmp_path / "result.json"),
    )

    assert command[command.index("--fashioniq_path") + 1] == f"{tmp_path / 'FashionIQ'}/"


def test_execute_writes_reproducible_logs(monkeypatch, tmp_path: Path) -> None:
    module = load_module()
    settings = config(tmp_path)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    plan = module.EvaluationPlan(
        "demo",
        "checkpoint",
        "fashioniq_original_split",
        tmp_path / "FashionIQ",
        tmp_path / "source",
        checkpoint,
        tmp_path,
        ["official", "--eval"],
        "a" * 40,
        "a" * 40,
    )

    class Stream:
        def __iter__(self):
            return iter(("line\n",))

    class Process:
        stdout = Stream()
        stderr = Stream()

        def wait(self):
            return 0

    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(module, "log_dir", lambda config, plan, timestamp: tmp_path / "logs" / "attempt")

    assert module.execute(plan, settings) == 0
    directory = tmp_path / "logs" / "attempt"
    assert (directory / "stdout.log").read_text() == "line\n"
    assert (directory / "stderr.log").read_text() == "line\n"
    metadata = json.loads((directory / "command.json").read_text())
    assert metadata["model_id"] == "demo"
    assert metadata["checkpoint_id"] == "checkpoint"
    assert metadata["protocol_id"] == "fashioniq_original_split"
    assert metadata["argv"] == ["official", "--eval"]
    assert metadata["cwd"] == str(tmp_path)
    assert metadata["upstream_expected_pin"] == "a" * 40
    assert metadata["actual_source_commit"] == "a" * 40
    assert metadata["checkpoint_local_sha256"]
    assert metadata["started_at"] and metadata["finished_at"]
    assert metadata["return_code"] == 0
    assert metadata["dataset_root"] == str(tmp_path / "FashionIQ")
