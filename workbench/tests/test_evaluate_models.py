from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


from workbench.backend.operator_config import WorkbenchConfig
from workbench.backend import registry
from workbench.backend.registry import load_registry


def load_module():
    spec = importlib.util.spec_from_file_location("evaluate_models", Path("workbench/scripts/evaluate_models.py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module



def _fake_isolated_interpreter(tmp_path: Path) -> Path:
    """Create a real lightweight venv so runtime identity verification passes."""
    import venv as _venv
    env_dir = tmp_path / f"fake_venv_{len(list(tmp_path.iterdir()))}"
    _venv.EnvBuilder(with_pip=False, system_site_packages=False).create(env_dir)
    return (env_dir / "bin" / "python").resolve()


def config(tmp_path: Path) -> WorkbenchConfig:
    return WorkbenchConfig(tmp_path, tmp_path / "data", tmp_path / "FashionIQ", "127.0.0.1", 8000, 5173, tmp_path / "checkpoints", tmp_path / "results", tmp_path / "third_party")
def records():
    return {model["model_id"]: model for model in load_registry()["models"]}


def prepare_source(monkeypatch, module, tmp_path: Path, model: dict) -> tuple[Path, Path, WorkbenchConfig]:
    import subprocess as _sp
    settings = config(tmp_path)
    source = settings.WORKBENCH_THIRD_PARTY_ROOT / model["source_dir"]
    source.mkdir(parents=True)
    # Real git checkout pinned at the registry commit, so source verification is honest.
    _sp.run(["git", "init", "-q"], cwd=source, check=True)
    _sp.run(["git", "config", "user.email", "t@t.invalid"], cwd=source, check=True)
    _sp.run(["git", "config", "user.name", "T"], cwd=source, check=True)
    (source / ".keep").write_text("")
    _sp.run(["git", "add", ".keep"], cwd=source, check=True)
    _sp.run(["git", "commit", "-q", "-m", "pin"], cwd=source, check=True)
    _sp.run(["git", "tag", "pin"], cwd=source, check=True)
    checkpoint = model["checkpoint_variants"][0]
    checkpoint_file = settings.WORKBENCH_CHECKPOINT_ROOT / model["model_id"] / checkpoint["filename"]
    checkpoint_file.parent.mkdir(parents=True)
    # A registry entry may now record the author's published size and sha256 (CSMCIR does, taken from
    # the author's Hugging Face LFS metadata). A fixture writing arbitrary bytes would be refused by
    # the very check that exists to catch a substituted file. `main()` loads its own registry, so the
    # declaration is rewritten on every load rather than mutated on the dict this helper was handed.
    payload = b"checkpoint"
    if checkpoint.get("expected_sha256") or checkpoint.get("expected_size_bytes"):
        digest = hashlib.sha256(payload).hexdigest()
        original_loader = registry.load_registry

        def loader_with_fixture_declarations():
            loaded = original_loader()
            for candidate in loaded["models"]:
                if candidate["model_id"] != model["model_id"]:
                    continue
                for variant in candidate.get("checkpoint_variants", []):
                    if variant.get("expected_sha256") or variant.get("expected_size_bytes"):
                        variant["expected_sha256"] = digest
                        variant["expected_size_bytes"] = len(payload)
            return loaded

        monkeypatch.setattr(registry, "load_registry", loader_with_fixture_declarations)
        # `evaluate_models` binds the name at import time (`from ... import load_registry`), so
        # patching only the registry module would not reach the call inside `main()`.
        monkeypatch.setattr(module, "load_registry", loader_with_fixture_declarations, raising=False)
        # The caller holds this model dict and passes its variant straight into `guarded_plan`, which
        # does not re-read the registry. Keep that object consistent with the stub bytes too.
        checkpoint["expected_sha256"] = digest
        checkpoint["expected_size_bytes"] = len(payload)
    checkpoint_file.write_bytes(payload)
    monkeypatch.setattr(module, "resolve_config", lambda: settings)
    monkeypatch.setattr(module, "pinned_revision", lambda _: model["upstream_commit_sha"])
    import workbench.backend.runtime as runtime

    monkeypatch.setattr(runtime, "source_clean_and_pinned", lambda model, _: (True, model["upstream_commit_sha"], None))
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

def prepare_standard_layout(root: Path) -> None:
    for directory in (root / "captions", root / "image_splits", root / "images"):
        directory.mkdir(parents=True, exist_ok=True)
    for category in ("dress", "shirt", "toptee"):
        (root / "captions" / f"cap.{category}.val.json").write_text("[]")
        (root / "image_splits" / f"split.{category}.val.json").write_text("[]")



def test_unaudited_adapter_is_skipped() -> None:
    module = load_module()
    model = records()["hint"]
    plan, reasons = module.guarded_plan(model, model["checkpoint_variants"][0], model["native_protocol"], None, 200)
    assert plan is None
    assert reasons == ["SKIPPED: adapter command not audited"]




def test_limn_candidate_is_complete_checkpoint_bundle_and_all_runnable_schedules_once() -> None:
    module = load_module()
    models = list(records().values())

    limn = module.candidates(module.parser_for(load_registry()).parse_args(["--model", "limn"]), models)
    all_runnable = module.candidates(module.parser_for(load_registry()).parse_args(["--all-runnable"]), models)
    encoder = module.candidates(module.parser_for(load_registry()).parse_args(["--model", "encoder"]), models)

    assert len(limn) == 1
    assert limn[0][1]["checkpoint_id"] == "base_iter0_all_categories"
    assert limn[0][1]["bundle_id"] == "base_iter0_all_categories"
    assert len([item for item in all_runnable if item[0]["model_id"] == "limn"]) == 1
    assert [item[1]["checkpoint_id"] for item in encoder] == ["fashioniq"]


def test_limn_explicit_category_remains_diagnostic_and_bundle_blocked(monkeypatch, tmp_path: Path) -> None:
    module = load_module()
    limn = records()["limn"]
    category_checkpoint = limn["checkpoint_variants"][0]
    bundle = module.candidates(module.parser_for(load_registry()).parse_args(["--model", "limn"]), [limn])[0][1]
    # `guarded_plan` deliberately uses the plan-level view (GPU readiness is a launch
    # concern), so this is the function the production path consults.
    monkeypatch.setattr(module, "runtime_plan_blockers",
                        lambda model, checkpoint, *args: ["checkpoint bundle incomplete: base_iter0_all_categories: missing category"])
    monkeypatch.setattr(module, "preflight_blockers", lambda *args, **kwargs: [])

    _, _, _, diagnostic_reasons = module.blockers(limn, category_checkpoint, limn["native_protocol"], tmp_path, config(tmp_path))
    _, _, _, bundle_reasons = module.blockers(limn, bundle, limn["native_protocol"], tmp_path, config(tmp_path))

    assert diagnostic_reasons == []
    assert bundle_reasons == ["checkpoint bundle incomplete: base_iter0_all_categories: missing category"]


def test_limn_adapter_runs_full_bundle_or_named_category(monkeypatch, tmp_path: Path) -> None:
    from workbench.backend.adapters.base import EvalRequest
    from workbench.backend.adapters.models import LIMNAdapter

    interpreter = _fake_isolated_interpreter(tmp_path)
    monkeypatch.setenv("WORKBENCH_LIMN_PYTHON", str(interpreter))
    # This test covers the LIMN command *shape*; verified interpreter selection is
    # covered on its own by the INT-05 integration test with a real contract.
    monkeypatch.setattr("workbench.backend.runtime.python_executable", lambda model: str(interpreter))
    adapter = LIMNAdapter()
    source = tmp_path / "LIMN"
    checkpoint_root = tmp_path / "checkpoints" / "limn"
    dataset = tmp_path / "FashionIQ"
    bundle = adapter.command(source, checkpoint_root / "0_dress_best_model.pt", EvalRequest("limn", "base_iter0_all_categories", "fashioniq_val_split", dataset, tmp_path / "bundle.json"))
    diagnostic = adapter.command(source, checkpoint_root / "0_dress_best_model.pt", EvalRequest("limn", "base_iter0_dress", "fashioniq_val_split", dataset, tmp_path / "dress.json"))

    assert "--category" not in bundle
    assert bundle[bundle.index("--checkpoint-root") + 1] == str(checkpoint_root)
    assert diagnostic[diagnostic.index("--category") + 1] == "dress"
def test_legacy_adapter_is_skipped_without_guessed_command() -> None:
    module = load_module()
    model = records()["tgcir"]

    plan, reasons = module.guarded_plan(model, model["checkpoint_variants"][0], model["native_protocol"], None, 200)

    assert plan is None
    assert reasons == ["SKIPPED: adapter command not audited"]


def test_dcnet_audited_command_reports_non_command_blockers() -> None:
    module = load_module()
    model = records()["dcnet"]

    plan, reasons = module.guarded_plan(model, model["checkpoint_variants"][0], model["native_protocol"], None, 200)

    assert model["native_protocol"] == "fashioniq_full_gallery_ref_excluded"
    assert plan is None
    assert not any(reason.startswith("SKIPPED:") for reason in reasons)
    assert "checkpoint mapping unresolved" in reasons
    assert any(reason.startswith("external runtime asset file missing: torchvision ResNet-50 ImageNet weights (") for reason in reasons)


def test_dcnet_command_uses_configured_model_interpreter(tmp_path: Path, monkeypatch) -> None:
    import sys
    from workbench.backend.adapters.base import EvalRequest
    from workbench.backend.adapters.models import DCNetAdapter

    interpreter = _fake_isolated_interpreter(tmp_path)
    monkeypatch.setenv("WORKBENCH_DCNET_PYTHON", str(interpreter))
    monkeypatch.setattr("workbench.backend.runtime.python_executable", lambda model: str(interpreter))
    command = DCNetAdapter().command(tmp_path / "DCNet", tmp_path / "fashioniq_dcnet", EvalRequest("dcnet", "fashioniq_run_directory", "fashioniq_full_gallery_ref_excluded", tmp_path / "FashionIQ", tmp_path / "result.json"))

    assert command == [str(interpreter), str(tmp_path / "DCNet" / "test.py"), "--resume", str(tmp_path / "fashioniq_dcnet")]

def test_missing_checkpoint_is_blocked(monkeypatch, tmp_path: Path) -> None:
    module = load_module()
    model = records()["csmcir"]
    settings = config(tmp_path)
    source = settings.WORKBENCH_THIRD_PARTY_ROOT / model["source_dir"]
    (source / "fashionIQ_dataset").mkdir(parents=True)
    monkeypatch.setattr(module, "resolve_config", lambda: settings)
    monkeypatch.setattr(module, "pinned_revision", lambda _: model["upstream_commit_sha"])
    import workbench.backend.runtime as runtime

    monkeypatch.setattr(runtime, "source_clean_and_pinned", lambda model, _: (True, model["upstream_commit_sha"], None))

    plan, reasons = module.guarded_plan(model, model["checkpoint_variants"][0], model["native_protocol"], source / "fashionIQ_dataset", 200, settings)

    assert plan is None
    assert any(reason.startswith("checkpoint missing:") for reason in reasons)


def prepare_csmcir_caption_assets(monkeypatch, source: Path, dataset_root: Path) -> None:
    from workbench.backend import registry

    assets = registry.auxiliary_assets_for_model
    def fixture_assets(model_id):
        items = assets(model_id)
        for asset in items:
            asset = dict(asset)
            if asset["family"] in {"Qwen captions", "COT_ours2 captions"}:
                asset["expected_sha256"] = None
                asset["expected_size_bytes"] = None
                asset["provenance_status"] = "VERIFIED"
                if asset.get("content_contract"):
                    asset["content_contract"] = {**asset["content_contract"], "expected_record_count": 1}
            yield asset
    monkeypatch.setattr(registry, "auxiliary_assets_for_model", lambda model_id: list(fixture_assets(model_id)))
    for index, category in enumerate(("dress", "shirt", "toptee")):
        image_id = f"B00000000{index}"
        (dataset_root / "image_splits" / f"split.{category}.val.json").write_text(json.dumps([image_id]))
        payload = json.dumps({image_id: {
            "Core_Essence": "garment", "Visual_Attributes": "blue",
            "Observation_Process": "visual inspection", "Final_Caption": "a blue garment",
        }})
        (source / "COT_ours2" / "fashioniq" / f"{category}_cot_val.json").write_text(payload)
        (dataset_root / "qwen_captions" / f"{category}_cot_val.json").write_text(payload)
        (dataset_root / "images" / f"{image_id}.png").write_bytes(b"fixture")


def test_csmcir_dry_run_constructs_official_command(monkeypatch, tmp_path: Path, capsys) -> None:
    module = load_module()
    model = records()["csmcir"]
    source, checkpoint_file, settings = prepare_source(monkeypatch, module, tmp_path, model)
    settings.FASHIONIQ_ROOT.mkdir()
    dataset_root = source / "fashionIQ_dataset"
    dataset_root.symlink_to(settings.FASHIONIQ_ROOT, target_is_directory=True)
    prepare_standard_layout(settings.FASHIONIQ_ROOT)
    for path in (source / "COT_ours2" / "fashioniq", dataset_root / "qwen_captions"):
        path.mkdir(parents=True)
    prepare_csmcir_caption_assets(monkeypatch, source, dataset_root)
    script = source / "src" / "validate_blip_csmcir.py"
    script.parent.mkdir()
    script.write_text("")
    import workbench.backend.adapters.models as adapters

    monkeypatch.setattr(adapters, "checkpoint_path", lambda *_: checkpoint_file)
    strict_plan, strict_reasons = module.guarded_plan(model, model["checkpoint_variants"][0], model["native_protocol"], dataset_root, 200, settings)
    plan, reasons = module.guarded_plan(model, model["checkpoint_variants"][0], model["native_protocol"], dataset_root, 200, settings, dry_run=True)
    assert strict_plan is None
    assert any("LAVIS BLIP-2 pretrain weights unverified" in reason for reason in strict_reasons)
    result = module.main([
        "--model", "csmcir", "--checkpoint", "fashioniq", "--protocol", model["native_protocol"],
        "--dataset-root", str(dataset_root), "--dry-run",
    ])

    assert result == 0
    assert reasons == []
    assert plan is not None
    assert plan.cwd == source / "src"
    # CSMCIR requires no isolated environment, so it runs in the workbench interpreter;
    # the command must name that interpreter, never a bare "python".
    assert plan.command[0] == sys.executable
    assert plan.command[1:] == [str(script), "--dataset", "fashionIQ", "--blip-model-path", str(checkpoint_file)]
    output = capsys.readouterr().out
    assert "[RUN] command:" in output
    assert "[PLAN] first-use downloads:" in output
    assert "dry-run cannot verify acquisition or evaluator success" in output


def test_csmcir_main_accepts_link_and_explicit_canonical_root(monkeypatch, tmp_path: Path) -> None:
    module = load_module()
    model = records()["csmcir"]
    source, checkpoint_file, settings = prepare_source(monkeypatch, module, tmp_path, model)
    canonical = tmp_path / "custom-FashionIQ"
    canonical.mkdir()
    layout = source / "fashionIQ_dataset"
    layout.symlink_to(canonical, target_is_directory=True)
    prepare_standard_layout(canonical)
    for path in (source / "COT_ours2" / "fashioniq", layout / "qwen_captions"):
        path.mkdir(parents=True)
    prepare_csmcir_caption_assets(monkeypatch, source, layout)
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

    # The layout check reports every missing requirement it finds, and the image-extension rule
    # is one of them; assert on membership rather than on an exact list, so adding a genuine
    # requirement does not silently pass.
    assert f"FashionIQ fashioniq_ilearn_resized requirement missing: {dataset_root / 'captions'}" in reasons
    assert f"ENCODER evaluator import missing: {source / 'datasets1.py'}" in reasons
    assert f"ENCODER asset missing: {source / 'open_clip_pytorch_model.bin'}" in reasons
    assert any("image extension requirement" in reason for reason in reasons), reasons
    # The extension rule names the directory the iLearn layout actually reads.
    assert any(str(dataset_root / "resized_image") in reason for reason in reasons), reasons


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


def test_encoder_complete_layout_still_blocks_missing_evaluator_import(monkeypatch, tmp_path: Path) -> None:
    module = load_module()
    model = records()["encoder"]
    source, _, _ = prepare_source(monkeypatch, module, tmp_path, model)
    for name in ("evaluate_model.py", "open_clip_pytorch_model.bin"):
        (source / name).write_text("")
    dataset_root = tmp_path / "FashionIQ"
    prepare_ilearn_layout(dataset_root)

    plan, reasons = module.guarded_plan(model, model["checkpoint_variants"][0], model["native_protocol"], dataset_root, 200)

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


def test_encoder_complete_local_prerequisites_yield_a_plan(monkeypatch, tmp_path: Path) -> None:
    """A verified mapping must not be blocked as if it were unresolved.

    ENCODER's registry entry once carried its two state fields transposed, so a complete local
    installation still produced "checkpoint mapping unresolved" and no plan. With the entry
    corrected, every prerequisite satisfied must yield a plan rather than a blocker.
    """
    module = load_module()
    model = records()["encoder"]
    source, _, _ = prepare_source(monkeypatch, module, tmp_path, model)
    for name in ("evaluate_model.py", "datasets1.py", "open_clip_pytorch_model.bin"):
        (source / name).write_text("")
    dataset_root = tmp_path / "FashionIQ"
    prepare_ilearn_layout(dataset_root)

    plan, reasons = module.guarded_plan(model, model["checkpoint_variants"][0], model["native_protocol"], dataset_root, 200)

    assert reasons == []
    assert plan is not None
    assert plan.model_id == "encoder"
    assert plan.checkpoint_id == "fashioniq"

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
    monkeypatch.setattr(module, "log_dir", lambda config, plan, timestamp, run_id=None: tmp_path / "logs" / "attempt")

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
