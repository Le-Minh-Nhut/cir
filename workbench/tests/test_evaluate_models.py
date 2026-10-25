from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from types import SimpleNamespace


from workbench.backend.operator_config import WorkbenchConfig
from workbench.backend import registry
from workbench.backend.registry import load_registry


def load_module():
    spec = importlib.util.spec_from_file_location("evaluate_models", REPO / "workbench/scripts/evaluate_models.py")
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


def _record_verified_auxiliary_asset(monkeypatch, settings: WorkbenchConfig, model: dict,
                                     relative_path: str,
                                     payload: bytes = b"placeholder weights") -> Path:
    """Place an auxiliary asset inside the checkout *and record it as an acquired, verified asset*.

    The registry declares ENCODER's open_clip backbone as a required asset, so a file that merely
    exists is reported `unverified: operator-supplied unverified provenance` and blocks the plan
    before the test's own claim is reached. A real acquisition writes three things this mirrors:
    the file, a manifest entry naming its digest, and — because the registry entry says VERIFIED —
    the declared published digest the file is checked against. `validate_auxiliary_asset` decides
    provenance from the *record*, so the record is what has to say VERIFIED.
    """
    path = settings.WORKBENCH_THIRD_PARTY_ROOT / model["source_dir"] / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    manifest_dir = (settings.WORKBENCH_ARTIFACTS_ROOT
                    or settings.CIR_REPO_ROOT / "workbench" / "artifacts") / "auxiliary"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / "download_manifest.json"
    try:
        entries = json.loads(manifest.read_text())
    except (OSError, ValueError):
        entries = []
    entries = [entry for entry in entries if entry.get("destination") != str(path)]
    entries.append({"destination": str(path), "downloaded_sha256": digest})
    manifest.write_text(json.dumps(entries))

    from workbench.backend import registry as registry_module

    original_loader = registry_module.load_auxiliary_assets

    def loader_with_verified_backbone(*args, **kwargs):
        records = []
        for record in original_loader(*args, **kwargs):
            if record["model_id"] == model["model_id"] and Path(record["destination_path"]).name == relative_path:
                record = {**record, "provenance_status": "VERIFIED",
                          "expected_sha256": digest, "expected_size_bytes": len(payload)}
            records.append(record)
        return records

    # `raising=False`: preflight imports the name at module scope in some paths, and a plain
    # assignment would leak the stub into later tests if it were not undone.
    monkeypatch.setattr(registry_module, "load_auxiliary_assets", loader_with_verified_backbone,
                        raising=False)
    return path


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
    # This fixture builds a synthetic checkout, so a defect the registry declares about the real
    # pinned source does not exist here. Clearing it lets these tests exercise the command path they
    # are actually about. Done unconditionally: not every model declares a checkpoint hash.
    model.pop("source_blocker", None)
    if checkpoint.get("expected_sha256") or checkpoint.get("expected_size_bytes"):
        digest = hashlib.sha256(payload).hexdigest()
        original_loader = registry.load_registry

        def loader_with_fixture_declarations():
            loaded = original_loader()
            for candidate in loaded["models"]:
                if candidate["model_id"] != model["model_id"]:
                    continue
                # This fixture builds a synthetic checkout, so a defect the registry declares about
                # the real pinned source does not exist here. Clearing it lets these tests exercise
                # the command path they are actually about.
                candidate.pop("source_blocker", None)
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
    monkeypatch.setattr(module, "pinned_revision", lambda source, _model=None: model["upstream_commit_sha"])
    # The interpreter gate is part of the same stub set. A model that declares a `python_env_var` is
    # gated on it, and this fixture builds a synthetic checkout with no provisioned environment, so
    # leaving it unstubbed would make every plan test fail on the environment rather than on the
    # thing it is about. It must be stubbed here rather than at each call site: while
    # `environment_required` was missing from the registry for CSMCIR and ENCODER,
    # `python_executable` returned the workbench interpreter and these tests passed for the wrong
    # reason -- they exercised neither branch of the gate.
    monkeypatch.setattr(
        "workbench.backend.runtime.python_executable",
        lambda _model: sys.executable,
    )
    import workbench.backend.runtime as runtime

    monkeypatch.setattr(runtime, "source_clean_and_pinned", lambda model, _: (True, model["upstream_commit_sha"], None))
    # The declared-artifact check resolves the acquisition manifest through `resolve_config()`, which
    # reads the process environment rather than the config object this fixture builds. Without this
    # redirection a staged asset would be looked up under the real artifacts root, find no manifest
    # entry, and be reported as operator-supplied with unverified provenance.
    def runtime_config():
        return SimpleNamespace(WORKBENCH_ARTIFACTS_ROOT=settings.WORKBENCH_ARTIFACTS_ROOT,
                               FASHIONIQ_ROOT=settings.FASHIONIQ_ROOT,
                               WORKBENCH_THIRD_PARTY_ROOT=settings.WORKBENCH_THIRD_PARTY_ROOT)

    monkeypatch.setattr(runtime, "resolve_config", runtime_config, raising=False)
    import workbench.backend.operator_config as operator_config
    monkeypatch.setattr(operator_config, "resolve_config", runtime_config)
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
        # The native constructor builds a fresh cache from the TRAIN captions before reading any
        # validation file, so a "complete" iLearn layout includes them: the registry declares
        # `fashioniq_ilearn_resized_training` for exactly that reason.
        (root / "captions" / f"cap.{category}.train.json").write_text("[]")
        (root / "captions" / f"correction_dict_{category}.json").write_text("{}")
        (root / "image_splits" / f"split.{category}.val.json").write_text("[]")

def prepare_standard_layout(root: Path) -> None:
    for directory in (root / "captions", root / "image_splits", root / "images"):
        directory.mkdir(parents=True, exist_ok=True)
    for category in ("dress", "shirt", "toptee"):
        (root / "captions" / f"cap.{category}.val.json").write_text("[]")
        (root / "image_splits" / f"split.{category}.val.json").write_text("[]")



def test_every_registered_adapter_builds_a_command() -> None:
    """No adapter may inherit the base `command`, which raises NotImplementedError.

    The earlier form of this test asserted that *some* model still inherited it, so that the skip rule
    stayed exercised. That is no longer a reachable state: every one of the fifteen adapters now
    builds a real command, because "upstream has no CLI" is a reason to write a replay wrapper, not a
    reason to leave a model unreachable. What still matters is the rule itself, so the rule is tested
    directly on a synthetic class below, and this test asserts the real registry has no such model.
    """
    module = load_module()
    from workbench.backend.adapters.models import ADAPTERS, OfficialScriptAdapter

    unimplemented = [mid for mid, adapter in sorted(ADAPTERS.items())
                     if getattr(adapter, "command", None) is OfficialScriptAdapter.command]
    assert unimplemented == [], f"these adapters cannot build a command: {unimplemented}"
    for mid, adapter in sorted(ADAPTERS.items()):
        assert module.command_is_audited(adapter), mid


def test_an_unaudited_adapter_is_still_skipped() -> None:
    """The skip rule must stay enforced, even with no real model in that state.

    A synthetic adapter that inherits the base `command` is the honest way to test a rule whose real
    instances no longer exist: if someone reintroduces one, `guarded_plan` must refuse it rather than
    plan a run whose command would raise.
    """
    module = load_module()
    from workbench.backend.adapters.models import ADAPTERS, OfficialScriptAdapter

    class UnauditedAdapter(OfficialScriptAdapter):
        model_id = "tgcir"

    assert not module.command_is_audited(UnauditedAdapter)
    saved = ADAPTERS["tgcir"]
    ADAPTERS["tgcir"] = UnauditedAdapter
    try:
        model = records()["tgcir"]
        plan, reasons = module.guarded_plan(model, model["checkpoint_variants"][0],
                                            model["native_protocol"], None, 200)
    finally:
        ADAPTERS["tgcir"] = saved
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
def test_tgcir_reports_the_archive_mapping_blocker_rather_than_launching() -> None:
    """TG-CIR's member mapping is unresolved, and the reported reason must say so.

    This test previously asserted that the adapter was skipped as unaudited. TG-CIR now has a command
    (its evaluator is a callable with no CLI, so the wrapper supplies one), which moves the model out
    of the skip case and into having to name its real missing prerequisite. "We cannot launch" and
    "we do not know which archive member to load" are different facts, and only the second is true.
    """
    module = load_module()
    model = records()["tgcir"]

    plan, reasons = module.guarded_plan(model, model["checkpoint_variants"][0], model["native_protocol"], None, 200)

    assert plan is None
    joined = " ".join(reasons)
    assert "checkpoint mapping unresolved" in joined, reasons
    assert not any("not audited" in reason for reason in reasons), (
        "tgcir has an audited command now; reporting it as unaudited would be false")


def test_dcnet_audited_command_reports_non_command_blockers() -> None:
    module = load_module()
    model = records()["dcnet"]

    plan, reasons = module.guarded_plan(model, model["checkpoint_variants"][0], model["native_protocol"], None, 200)

    assert model["native_protocol"] == "fashioniq_full_gallery_ref_excluded"
    assert plan is None
    assert not any(reason.startswith("SKIPPED:") for reason in reasons)
    assert any(reason.startswith("external runtime asset file missing: torchvision ResNet-50 ImageNet weights (")
               for reason in reasons)

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
    monkeypatch.setattr(module, "pinned_revision", lambda source, _model=None: model["upstream_commit_sha"])
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

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "huggingface"))
    lavis_weights = tmp_path / "cache" / "lavis" / "blip2_pretrained.pth"
    monkeypatch.setattr(adapters, "checkpoint_path", lambda *_: checkpoint_file)
    strict_plan, missing_reasons = module.guarded_plan(model, model["checkpoint_variants"][0], model["native_protocol"], dataset_root, 200, settings)
    assert strict_plan is None
    assert any("LAVIS BLIP-2 pretrain weights missing" in reason for reason in missing_reasons)
    lavis_weights.parent.mkdir(parents=True)
    lavis_weights.write_bytes(b"unverified fixture weights")
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
    # The registry pins csmcir's checkpoint-to-paper mapping as VERIFIED_METADATA; the plan
    # must carry that fact, because the paper gate certifies a pass only when it is true.
    assert plan.checkpoint_mapping_verified is True
    # CSMCIR requires no isolated environment, so it runs in the workbench interpreter;
    # the command must name that interpreter, never a bare "python".
    assert plan.command[0] == sys.executable
    assert plan.command[1:] == [str(script), "--dataset", "fashionIQ", "--blip-model-path", str(checkpoint_file)]
    output = capsys.readouterr().out
    assert "[RUN] command:" in output
    assert "[PLAN] first-use downloads:" in output
    assert "dry-run cannot verify acquisition or evaluator success" in output

def test_the_plan_carries_the_registry_mapping_verdict_not_a_hardcoded_true(monkeypatch, tmp_path: Path) -> None:
    """The plan-level mapping flag must mirror the registry, in both directions.

    An unverified mapping reaching the paper gate as verified would let the gate certify
    `PASS_WITHIN_5_PERCENT` for a run whose checkpoint-to-paper identity nobody established.
    """
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

    unverified = {**model, "checkpoint_variants": [
        {**model["checkpoint_variants"][0], "checkpoint_mapping_status": "UNVERIFIED"}]}
    # Dry run drops the first-use LAVIS downloads. `runtime_plan_blockers` would refuse the
    # unverified mapping outright — that is the runtime guard, not the flag under test — so the
    # plan-level wiring is isolated by stubbing the runtime blockers and keeping everything else.
    import workbench.backend.runtime as runtime
    monkeypatch.setattr(runtime, "runtime_plan_blockers", lambda *a, **k: [])
    # evaluate_models binds the name directly (`from ... import`), so patch there too.
    monkeypatch.setattr(module, "runtime_plan_blockers", lambda *a, **k: [])
    plan, reasons = module.guarded_plan(unverified, unverified["checkpoint_variants"][0],
                                        unverified["native_protocol"], dataset_root, 200,
                                        settings, dry_run=True)
    assert plan is not None, reasons
    assert plan.checkpoint_mapping_verified is False


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
    assert f"FashionIQ fashioniq_ilearn_resized_training requirement missing: {dataset_root / 'captions'}" in reasons
    assert f"ENCODER evaluator import missing: {source / 'datasets1.py'}" in reasons
    assert f"ENCODER asset missing: {source / 'open_clip_pytorch_model.bin'}" in reasons
    assert any("image extension requirement" in reason for reason in reasons), reasons
    # The extension rule names the directory the iLearn layout actually reads.
    assert any(str(dataset_root / "resized_image") in reason for reason in reasons), reasons


def test_encoder_reports_actual_missing_layout_path(monkeypatch, tmp_path: Path) -> None:
    module = load_module()
    model = records()["encoder"]
    source, _, settings = prepare_source(monkeypatch, module, tmp_path, model)
    # The backbone is staged as an ACQUIRED asset (file + manifest digest), because the registry
    # declares it required: a bare file is reported `unverified: operator-supplied unverified
    # provenance` and blocks the plan before the claim under test is reached. These tests are about
    # the evaluator import, the data layout and the checkpoint mapping -- not the weights' bytes.
    _record_verified_auxiliary_asset(monkeypatch, settings, model, "open_clip_pytorch_model.bin")
    for name in ("evaluate_model.py", "datasets1.py"):
        (source / name).write_text("")
    dataset_root = tmp_path / "FashionIQ"
    prepare_ilearn_layout(dataset_root)
    missing = dataset_root / "captions" / "correction_dict_shirt.json"
    missing.unlink()

    plan, reasons = module.guarded_plan(model, model["checkpoint_variants"][0], model["native_protocol"], dataset_root, 200)
    assert plan is None
    # The mapping is honestly UNVERIFIED, so the unresolved mapping blocks alongside the layout gap.
    # The mapping is no longer part of the reason set: ENCODER's variant is bound by an author-linked
    # revision with a published size and hash, so the layout gap is the only obstacle left here.
    assert reasons == [f"FashionIQ fashioniq_ilearn_resized_training requirement missing: {missing}"]


def test_encoder_complete_layout_still_blocks_missing_evaluator_import(monkeypatch, tmp_path: Path) -> None:
    module = load_module()
    model = records()["encoder"]
    source, _, settings = prepare_source(monkeypatch, module, tmp_path, model)
    _record_verified_auxiliary_asset(monkeypatch, settings, model, "open_clip_pytorch_model.bin")
    (source / "evaluate_model.py").write_text("")
    dataset_root = tmp_path / "FashionIQ"
    prepare_ilearn_layout(dataset_root)

    plan, reasons = module.guarded_plan(model, model["checkpoint_variants"][0], model["native_protocol"], dataset_root, 200)

    assert reasons == [f"ENCODER evaluator import missing: {source / 'datasets1.py'}"]

def test_encoder_complete_layout_and_import_still_blocks_openclip(monkeypatch, tmp_path: Path) -> None:
    """With the evaluator and the data layout present, the missing backbone is what blocks.

    The assertion checks that the backbone is reported *missing* rather than pinning the exact
    producer string: the registry now declares this asset, so it is reported by the auxiliary-asset
    check as well as by the model's own prerequisite list, and two accurate messages must not make
    the test fail. The claim — a missing backbone blocks the plan even when layout and import are
    satisfied — is what is asserted.
    """
    module = load_module()
    model = records()["encoder"]
    source, _, _ = prepare_source(monkeypatch, module, tmp_path, model)
    for name in ("evaluate_model.py", "datasets1.py"):
        (source / name).write_text("")
    dataset_root = tmp_path / "FashionIQ"
    prepare_ilearn_layout(dataset_root)

    plan, reasons = module.guarded_plan(model, model["checkpoint_variants"][0], model["native_protocol"], dataset_root, 200)

    assert plan is None
    backbone = str(source / "open_clip_pytorch_model.bin")
    assert any("missing" in reason.lower() and backbone in reason for reason in reasons), reasons

def test_encoder_complete_prerequisites_produce_a_plan_because_the_mapping_is_verified(monkeypatch, tmp_path: Path) -> None:
    module = load_module()
    model = records()["encoder"]
    source, _, settings = prepare_source(monkeypatch, module, tmp_path, model)
    _record_verified_auxiliary_asset(monkeypatch, settings, model, "open_clip_pytorch_model.bin")
    (source / "evaluate_model.py").write_text("")
    (source / "datasets1.py").write_text("")
    dataset_root = tmp_path / "FashionIQ"
    prepare_ilearn_layout(dataset_root)

    plan, reasons = module.guarded_plan(model, model["checkpoint_variants"][0], model["native_protocol"],
                                        dataset_root, 200)

    assert plan is not None, reasons
    assert plan.checkpoint_mapping_verified is True
    assert reasons == []

def test_encoder_replay_preserves_upstream_trailing_root_separator(tmp_path: Path, monkeypatch) -> None:
    import torch
    from types import SimpleNamespace
    from workbench.replay import encoder

    dataset_root = tmp_path / "FashionIQ"
    for name in ("captions", "image_splits", "resized_image"):
        (dataset_root / name).mkdir(parents=True)
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    source_root = tmp_path / "ENCODER"
    source_root.mkdir()
    paths = []

    class Model:
        def __init__(self, **_kwargs):
            pass
        def to(self, _device):
            return self

        def eval(self):
            return None

    class Dataset:
        def FashionIQ(self, *, path, **_kwargs):
            paths.append(path)
            return object()

    fake_open_clip = SimpleNamespace(create_model_and_transforms=lambda *_a, **_k: (None, None, None))
    fake_model = SimpleNamespace(Encoder=Model)
    monkeypatch.setattr(encoder, "_load_replay_modules",
                        lambda _root: (torch, fake_open_clip, Dataset(), SimpleNamespace(), fake_model))
    monkeypatch.setattr(torch, "load", lambda *_a, **_k: Model())
    monkeypatch.delenv("WORKBENCH_REPLAY_CACHE_ROOT", raising=False)

    encoder.evaluate_categories(checkpoint_root, dataset_root, source_root, categories=(), device="cpu")

    assert len(paths) == 1
    assert paths[0].endswith("/")
    assert Path(paths[0].rstrip("/")) != dataset_root


def test_sprc_command_selects_the_registered_model_and_a_matching_backbone(tmp_path: Path) -> None:
    """SPRC's own defaults are wrong in two ways, and the command must override both.

    `src/blip_validate.py:144` defaults `--blip-model-name` to a class the pinned tree never
    registers, so a default launch dies in `load_model_and_preprocess`; the command passes the
    registered name explicitly. `--backbone` selects the vision encoder, and it comes from the
    VARIANT'S DECLARED `backbone` (the registry states it per variant) rather than from the filename
    or the checkpoint path: the loader uses `strict=False`, so a wrong encoder would evaluate a
    partly-initialised model without raising.
    """
    from unittest.mock import patch

    from workbench.backend import runtime as _runtime
    from workbench.backend.adapters.base import EvalRequest
    from workbench.backend.adapters.models import ADAPTERS

    def command_for(checkpoint_id: str) -> list[str]:
        request = EvalRequest("sprc", checkpoint_id, "fashioniq_original_split", tmp_path,
                              tmp_path / "out.json")
        with patch.object(_runtime, "python_executable", lambda _: "/usr/bin/python3"):
            return ADAPTERS["sprc"]().command(tmp_path / "SPRC", tmp_path / f"{checkpoint_id}.pt", request)

    vitl = command_for("sprc_fiq_vitl")
    assert vitl[vitl.index("--blip-model-name") + 1] == "blip2_cir_align_prompt"
    assert vitl[vitl.index("--backbone") + 1] == "pretrain_vitL"
    assert vitl[vitl.index("--dataset") + 1] == "fashionIQ"

    vitg = command_for("sprc_fiq_unresolved")
    assert vitg[vitg.index("--backbone") + 1] == "pretrain"




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


def _limn_bundle_environment(monkeypatch, module, tmp_path: Path):
    """A LIMN environment whose bundle state the caller controls, with the REAL blocker sources.

    Nothing is stubbed between `runtime_blockers` and the registry: the bundle-incomplete blocker
    is produced by `checkpoint_bundle_missing_paths` reading the checkpoint root, so the test
    exercises the same chain the API does.
    """
    import workbench.backend.runtime as runtime

    settings = config(tmp_path)
    model = records()["limn"]
    bundle = model["checkpoint_bundles"][0]
    payload, digest = b"member-bytes", hashlib.sha256(b"member-bytes").hexdigest()
    checkpoint_root = settings.WORKBENCH_CHECKPOINT_ROOT / "limn"
    checkpoint_root.mkdir(parents=True)
    for name in ("0_dress_best_model.pt", "0_shirt_best_model.pt", "0_toptee_best_model.pt"):
        (checkpoint_root / name).write_bytes(payload)
    source = settings.WORKBENCH_THIRD_PARTY_ROOT / model["source_dir"]
    (source / "LIMN").mkdir(parents=True)
    (source / "LIMN" / "test.py").write_text("")
    dataset_root = settings.FASHIONIQ_ROOT
    for directory in ("captions", "image_splits", "resized_image/dress", "resized_image/shirt",
                      "resized_image/toptee"):
        (dataset_root / directory).mkdir(parents=True, exist_ok=True)
    for category in ("dress", "shirt", "toptee"):
        for name in (f"cap.{category}.val.json", f"cap.{category}.train.json",
                     f"correction_dict_{category}.json"):
            (dataset_root / "captions" / name).write_text("[]")
        (dataset_root / "image_splits" / f"split.{category}.val.json").write_text("[]")
    # The bundle members are real registry entries with author-published digests; the fixture
    # writes its own bytes, so the declarations are rewritten on every registry load rather than
    # left to contradict the files.
    original_loader = registry.load_registry

    def loader_with_fixture_declarations():
        loaded = original_loader()
        for candidate in loaded["models"]:
            if candidate["model_id"] == "limn":
                for variant in candidate["checkpoint_variants"]:
                    variant["expected_sha256"] = digest
                    variant["expected_size_bytes"] = len(payload)
        return loaded

    monkeypatch.setattr(registry, "load_registry", loader_with_fixture_declarations)
    monkeypatch.setattr(module, "load_registry", loader_with_fixture_declarations, raising=False)
    for variant in model["checkpoint_variants"]:
        variant["expected_sha256"] = digest
        variant["expected_size_bytes"] = len(payload)

    interpreter = _fake_isolated_interpreter(tmp_path)
    monkeypatch.setenv("WORKBENCH_LIMN_PYTHON", str(interpreter))
    monkeypatch.setattr(runtime, "python_executable", lambda _model: str(interpreter))
    monkeypatch.setattr(runtime, "source_clean_and_pinned",
                        lambda _model, _source: (True, model["upstream_commit_sha"], None))
    monkeypatch.setattr(module, "resolve_config", lambda: settings)
    monkeypatch.setattr(module, "pinned_revision", lambda source, _model=None: model["upstream_commit_sha"])
    # The fold `POST /api/jobs` performs: the bundle record, keeping the bundle identity. The
    # bundle must stay ONE selection with `checkpoint_id == bundle_id`. Resolved through the
    # registry so this fixture builds exactly the record production builds -- a hand-rebuilt fold
    # that dropped the member facts is what let a complete bundle be reported as a missing
    # directory in production while this test's own fold looked fine.
    from workbench.backend.registry import checkpoint_by_id

    folded = checkpoint_by_id(model, bundle["bundle_id"])
    return settings, model, dataset_root, folded, checkpoint_root, bundle


def test_a_complete_limn_bundle_is_not_refused_by_a_bundle_blocker(monkeypatch, tmp_path: Path) -> None:
    """(a) All three members present -> no bundle-incomplete blocker, and ONE bundle plan."""
    module = load_module()
    settings, model, dataset_root, folded, _, bundle = _limn_bundle_environment(monkeypatch, module, tmp_path)

    plan, reasons = module.guarded_plan(model, folded, model["native_protocol"], dataset_root, 200, settings)

    assert not [r for r in reasons if r.startswith("checkpoint bundle incomplete")], reasons
    assert plan is not None, reasons
    # One selection, not three runs, and all three categories recorded as consumed.
    assert plan.checkpoint_id == bundle["bundle_id"]
    assert sorted(plan.checkpoint_bundle) == ["0_dress_best_model.pt", "0_shirt_best_model.pt",
                                              "0_toptee_best_model.pt"]
    assert plan.artifact_type == "category_specific_whole_models"


def test_an_incomplete_limn_bundle_is_refused_and_the_member_is_named(monkeypatch, tmp_path: Path) -> None:
    """(b) A missing member still refuses the bundle, and the blocker names the member."""
    module = load_module()
    settings, model, dataset_root, folded, checkpoint_root, bundle = _limn_bundle_environment(monkeypatch, module, tmp_path)
    missing = checkpoint_root / "0_toptee_best_model.pt"
    missing.unlink()

    plan, reasons = module.guarded_plan(model, folded, model["native_protocol"], dataset_root, 200, settings)

    assert plan is None
    assert [r for r in reasons if r.startswith("checkpoint bundle incomplete")] == [
        f"checkpoint bundle incomplete: {bundle['bundle_id']}: {missing}"], reasons


# ---------------------------------------------------------------- F-08: observation integration

def test_execute_invokes_the_observer_with_this_invocations_identity(monkeypatch, tmp_path: Path) -> None:
    """F-08: the per-query observer must run as part of the SAME run, not as a separate experiment.

    This drives `execute()` with a stubbed evaluator and a stubbed observation module, and asserts
    the exact argv the observer receives: the authoritative `run_id`, and the AGGREGATE invocation's
    command/environment digests (which the observer cannot derive for itself, because it runs a
    different command).
    """
    module = load_module()
    captured: dict = {}

    def fake_invoke(plan, config, run_id, root):
        directory = tmp_path / "logs" / f"{plan.model_id}__{plan.checkpoint_id}__{run_id}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "stdout.log").write_text('{"dress_r1": 1.0, "dress_r10": 2.0, "dress_r50": 3.0}')
        (directory / "stderr.log").write_text("")
        return 0, directory, {"orchestrator_python": "3", "platform": "test"}

    artifact = tmp_path / "published.json"

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["cwd"] = kwargs.get("cwd")

        class Done:
            returncode = 0
            stdout = json.dumps({"published": str(artifact), "run_id": "x"})
            stderr = ""
        # Write a minimal valid artifact the parent will re-validate.
        return Done()

    monkeypatch.setattr(module, "invoke_evaluator", fake_invoke)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "extract_aggregate_metrics",
                        lambda *a, **k: type("E", (), {
                            "extraction_status": "EXTRACTED_VERIFIED", "parser_id": "limn_structured_json",
                            "parser_version": 1, "observed_metrics": {"r10": 1.0, "r50": 2.0, "mean": 1.5},
                            "category_metrics": None, "parity_status": "PARITY_NOT_EVALUATED",
                            "metric_source": "stdout"})())

    plan = module.EvaluationPlan(
        model_id="limn", checkpoint_id="base_iter0_all_categories", protocol="fashioniq_val_split",
        dataset_root=tmp_path, source=tmp_path, checkpoint=tmp_path / "ckpt",
        cwd=tmp_path, command=[sys.executable, "x"], pin="a" * 40,
        actual_source_commit="a" * 40, paper_metrics=None)
    settings = config(tmp_path)

    run_id = "20260101T000000000000Z_limn_base_iter0_all_categories_deadbeef"
    module.execute(plan, settings, run_id=run_id)

    argv = captured.get("argv")
    assert argv is not None, "the observer was never invoked (F-08)"
    assert "workbench.replay.limn_export" in " ".join(argv)
    # The SAME identity: the artifact is bound to this invocation, not to an identity of its own.
    assert argv[argv.index("--run-id") + 1] == run_id
    # The aggregate invocation's digests, forwarded so the artifact can be verified.
    assert "--command-digest" in argv and "--environment-digest" in argv


def test_the_evaluation_module_does_not_widen_limns_traced_environment() -> None:
    """The regression that broke cold start twice, pinned as a contract.

    `environment_planner._evaluator_imports` walks `workbench/replay/limn.py` to decide which
    interpreter LIMN needs, and it follows imports through every module reachable from there --
    including function-local ones, because it uses `ast.walk`. `backend/runtime` lazily imports this
    module, so ANY new top-level or function-local import added here becomes a declared requirement
    of LIMN's environment unless it is already on the path.

    Adding `workbench.backend.schemas.results` here (for pydantic) made the planner demand an
    interpreter for pydantic, provisioning failed, and the cold-start run stopped before evaluation.
    The parent verifies what it can from raw JSON instead; the admission verdict belongs to the index
    and API, which are not on this graph.
    """
    from pathlib import Path as _Path

    from workbench.backend.environment_planner import _evaluator_imports
    from workbench.backend.registry import model_by_id

    _traced = _evaluator_imports(model_by_id("limn"), _Path("workbench/third_party"))
    traced = _traced[0] if _traced is not None else set()
    # pydantic is not a LIMN runtime dependency: it arrives only through the canonical result schema.
    assert "pydantic" not in traced, f"pydantic leaked into LIMN's traced imports: {sorted(traced)}"
    # And the modules that must remain reachable still are.
    assert {"torch", "open_clip", "numpy"} <= set(traced), sorted(traced)


def test_a_failed_observation_leaves_the_aggregate_result_intact(monkeypatch, tmp_path: Path) -> None:
    """§28: a failed query export must not destroy a valid aggregate evaluation.

    The aggregate is the authoritative result and has already succeeded by the time observation
    runs. Observation is an optional capability, so its failure is recorded as a reason -- and the
    report must say `per_query_export_available: False` rather than claiming an artifact exists.
    """
    module = load_module()
    from workbench.backend.schemas.results import ResultRun

    def fake_invoke(plan, config, run_id, root):
        directory = tmp_path / "logs" / f"{plan.model_id}__{plan.checkpoint_id}__{run_id}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "stdout.log").write_text('{"dress_r1": 1.0, "dress_r10": 2.0, "dress_r50": 3.0}')
        (directory / "stderr.log").write_text("")
        return 0, directory, {"orchestrator_python": "3", "platform": "test"}

    def failing_run(argv, **kwargs):
        class Failed:
            returncode = 1
            stdout = ""
            stderr = "observer could not load the checkpoint"

        return Failed()

    monkeypatch.setattr(module, "invoke_evaluator", fake_invoke)
    monkeypatch.setattr(module.subprocess, "run", failing_run)
    monkeypatch.setattr(module, "extract_aggregate_metrics",
                        lambda *a, **k: type("E", (), {
                            "extraction_status": "EXTRACTED_VERIFIED", "parser_id": "limn_structured_json",
                            "parser_version": 1, "observed_metrics": {"r10": 1.0, "r50": 2.0, "mean": 1.5},
                            "category_metrics": None, "parity_status": "PARITY_NOT_EVALUATED",
                            "metric_source": "stdout"})())

    plan = module.EvaluationPlan(
        model_id="limn", checkpoint_id="base_iter0_all_categories", protocol="fashioniq_val_split",
        dataset_root=tmp_path, source=tmp_path, checkpoint=tmp_path / "ckpt",
        cwd=tmp_path, command=[sys.executable, "x"], pin="a" * 40,
        actual_source_commit="a" * 40, paper_metrics=None)
    settings = config(tmp_path)
    run_id = "20260101T000000000000Z_limn_base_iter0_all_categories_fa11ed"
    assert module.execute(plan, settings, run_id=run_id) == 0, "an observation failure is not fatal"

    # The aggregate report is written under the artifacts root the config names.
    reports = list(tmp_path.rglob("*_aggregate.json"))
    assert reports, f"the aggregate report must still exist; tree: {sorted(p.name for p in tmp_path.rglob('*'))}"
    report = json.loads(reports[0].read_text())
    assert report["run_id"] == run_id
    assert report["execution_status"] == "EXECUTION_COMPLETED"
    assert report["per_query_export_available"] is False, "no artifact was produced"
    assert report.get("per_query_artifact_digest") is None
    assert "observer exited" in (report.get("per_query_export_reason") or "")


# --------------------------------------------- per-query capability and truthful status (P1)

def _run_with(module, monkeypatch, tmp_path: Path, *, model_id: str, observer_run):
    """Drive execute() for one model with a stubbed native command and a stubbed child pipeline."""

    def fake_invoke(plan, config, run_id, root):
        directory = tmp_path / "logs" / f"{plan.model_id}__{plan.checkpoint_id}__{run_id}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "stdout.log").write_text('{"dress_r1": 1.0, "dress_r10": 2.0, "dress_r50": 3.0}')
        (directory / "stderr.log").write_text("")
        return 0, directory, {"orchestrator_python": "3", "platform": "test"}

    monkeypatch.setattr(module, "invoke_evaluator", fake_invoke)
    monkeypatch.setattr(module.subprocess, "run", observer_run)
    monkeypatch.setattr(module, "extract_aggregate_metrics",
                        lambda *a, **k: type("E", (), {
                            "extraction_status": "EXTRACTED_VERIFIED", "parser_id": "p",
                            "parser_version": 1, "observed_metrics": {"r10": 1.0, "r50": 2.0, "mean": 1.5},
                            "category_metrics": None, "parity_status": "PARITY_NOT_EVALUATED",
                            "metric_source": "stdout"})())
    plan = module.EvaluationPlan(
        model_id=model_id, checkpoint_id="base_iter0_all_categories",
        protocol="fashioniq_val_split", dataset_root=tmp_path, source=tmp_path,
        checkpoint=tmp_path / "ckpt", cwd=tmp_path, command=[sys.executable, "x"],
        pin="a" * 40, actual_source_commit="a" * 40, paper_metrics=None)
    settings = config(tmp_path)
    run_id = f"20260101T000000000000Z_{model_id}_ckpt_deadbeef"
    module.execute(plan, settings, run_id=run_id)
    reports = sorted(tmp_path.rglob("*_aggregate.json"))
    assert reports, "the aggregate report must exist"
    return json.loads(reports[-1].read_text()), run_id


def test_a_declared_observer_that_exports_nothing_is_reported_as_export_failed(monkeypatch, tmp_path: Path) -> None:
    """P1: a model that DECLARED a per-query observer and produced no artifact has failed that axis.

    The aggregate stays valid and is preserved -- `execute()` still returns the native exit code --
    but the report must say so plainly, because the launcher's readiness claim is derived from it.
    """

    def failing_observer(argv, **kwargs):
        class Failed:
            returncode = 1
            stdout = ""
            stderr = "observer could not resolve the checkpoint directory"

        return Failed()

    report, _ = _run_with(load_module(), monkeypatch, tmp_path, model_id="limn",
                          observer_run=failing_observer)
    assert report["per_query_capability"] == "LIMN_QUERY_OBSERVER_V1"
    assert report["per_query_state"] == "PER_QUERY_EXPORT_FAILED", report["per_query_state"]
    assert report["per_query_export_available"] is False
    assert report["execution_status"] == "EXECUTION_COMPLETED", "the aggregate must be untouched"


def test_a_model_without_an_observer_is_capability_absent_not_a_failure(monkeypatch, tmp_path: Path) -> None:
    """P1: an aggregate-only model must not be reported as a failed per-query export.

    No observer exists for it by design, so the launcher must not treat its absence as a pilot
    blocker. The capability is read from the registry, not inferred from a model id.
    """
    called = {"observer": False}

    def must_not_run(argv, **kwargs):
        called["observer"] = True

        class Never:
            returncode = 0
            stdout = ""
            stderr = ""

        return Never()

    report, _ = _run_with(load_module(), monkeypatch, tmp_path, model_id="encoder",
                          observer_run=must_not_run)
    assert report["per_query_capability"] is None
    assert report["per_query_state"] == "CAPABILITY_ABSENT", report["per_query_state"]
    assert not called["observer"], "an observer must not run for a model that declares none"
    assert report["execution_status"] == "EXECUTION_COMPLETED"


def test_a_successful_two_stage_export_is_reported_as_exported_but_unverified(monkeypatch, tmp_path: Path) -> None:
    """Only the verifier may say VERIFIED; the run can only report that it produced an artifact.

    Exercises the real two-stage contract: the observer child stages a payload, the publisher child
    (under the workbench interpreter) writes the canonical artifact, and the parent reads it back to
    compute the digest itself.
    """
    canonical = tmp_path / "canonical" / "run.json"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    seen: list[list[str]] = []

    def two_stage(argv, **kwargs):
        seen.append(list(argv))
        is_publisher = "workbench.replay.limn_publish" in argv
        if is_publisher:
            canonical.write_text(json.dumps({
                "schema_version": 2,
                "run": {"run_id": "20260101T000000000000Z_limn_ckpt_deadbeef",
                        "instrumentation_parity": "INSTRUMENTATION_PARITY_VERIFIED",
                        "data_kind": "experiment"},
                "queries": [],
            }))
            out = json.dumps({"published": str(canonical)})
        else:
            staged = tmp_path / "staged.payload.json"
            staged.write_text(json.dumps({"artifact_type": "limn_observation_payload"}))
            out = json.dumps({"payload": str(staged)})

        class Done:
            returncode = 0
            stdout = out
            stderr = ""

        return Done()

    report, run_id = _run_with(load_module(), monkeypatch, tmp_path, model_id="limn",
                               observer_run=two_stage)
    assert report["per_query_state"] == "PER_QUERY_EXPORTED_UNVERIFIED", report["per_query_state"]
    assert report["per_query_export_available"] is True
    assert report["per_query_artifact_digest"], "the parent computes the digest from the bytes"
    # BOTH stages ran, and the publisher ran under the WORKBENCH interpreter (this process's).
    assert any("workbench.replay.limn_export" in call for call in seen)
    publisher = next(call for call in seen if "workbench.replay.limn_publish" in call)
    assert publisher[0] == sys.executable


def test_the_observer_and_publisher_receive_a_sanitized_environment(monkeypatch, tmp_path: Path) -> None:
    """P1: model code must not receive the operator's secrets, least of all the ledger key.

    The observer runs model-supplied code paths, so it gets the same allowlisted environment as the
    official evaluator. This asserts the ACTUAL environment built for both children, not a helper.
    """
    seen_envs: list[dict] = []

    def capture(argv, **kwargs):
        seen_envs.append(dict(kwargs.get("env") or {}))
        is_publisher = "workbench.replay.limn_publish" in argv
        if is_publisher:
            # The parent re-reads whatever the publisher reports, so it must exist.
            reported = tmp_path / "env-probe.json"
            reported.write_text(json.dumps({
                "schema_version": 2,
                "run": {"run_id": "20260101T000000000000Z_limn_ckpt_deadbeef",
                        "instrumentation_parity": "INSTRUMENTATION_PARITY_VERIFIED",
                        "data_kind": "experiment"},
                "queries": [],
            }))
            out = json.dumps({"published": str(reported)})
        else:
            staged = tmp_path / "env-probe.payload.json"
            staged.write_text(json.dumps({"artifact_type": "limn_observation_payload"}))
            out = json.dumps({"payload": str(staged)})

        class Done:
            returncode = 0
            stdout = out
            stderr = ""

        return Done()

    monkeypatch.setenv("WORKBENCH_LEDGER_KEY", "TEST_SECRET")
    monkeypatch.setenv("OPENAI_API_KEY", "TEST_SECRET")
    monkeypatch.setenv("GITHUB_TOKEN", "TEST_SECRET")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    _run_with(load_model_module := load_module(), monkeypatch, tmp_path, model_id="limn",
              observer_run=capture)
    assert seen_envs, "the observer must have been invoked"
    for env in seen_envs:
        assert "WORKBENCH_LEDGER_KEY" not in env, "the ledger signing key must never reach a child"
        assert "OPENAI_API_KEY" not in env and "GITHUB_TOKEN" not in env
    # A required runtime variable is still forwarded: CUDA visibility is a deliberate operator
    # decision the evaluator needs.
    assert any(env.get("CUDA_VISIBLE_DEVICES") == "0" for env in seen_envs), seen_envs


def test_the_evaluation_module_does_not_widen_limns_traced_environment() -> None:
    """The regression that broke cold start twice, pinned as a contract.

    `environment_planner._evaluator_imports` walks `workbench/replay/limn.py` to decide which
    interpreter LIMN needs, and it follows imports through every module reachable from there --
    including function-local ones, because it uses `ast.walk`. `backend/runtime` lazily imports this
    module, so ANY new import added here becomes a declared requirement of LIMN's environment.

    The canonical schema (pydantic) must stay off that graph: the model environment does not have it,
    which is why the observer stages plain JSON and a separate publisher (run under the workbench
    interpreter) writes the artifact.
    """
    from pathlib import Path as _Path

    from workbench.backend.environment_planner import _evaluator_imports
    from workbench.backend.registry import model_by_id

    _traced = _evaluator_imports(model_by_id("limn"), _Path("workbench/third_party"))
    traced = _traced[0] if _traced is not None else set()
    assert "pydantic" not in traced, f"pydantic leaked into LIMN's traced imports: {sorted(traced)}"
    assert {"torch", "open_clip", "numpy"} <= set(traced), sorted(traced)


# ------------------------------- §22: the LIMN bundle path, adapter and observer agreeing

def test_the_adapter_and_the_observer_resolve_the_same_member_directory(monkeypatch, tmp_path: Path) -> None:
    """The regression that survived the previous 1309-test release, pinned end to end.

    A LIMN bundle's members are SIBLINGS inside the directory `checkpoint_path` resolves to. The
    observer used `plan.checkpoint.parent`, which is one level too high for a bundle, so it searched
    the whole checkpoint root for `0_*_best_model.pt` and found nothing while the aggregate
    succeeded. This drives the real resolver, the real adapter, and the real observation command
    builder over a temporary bundle.
    """
    import subprocess as _sp

    from workbench.backend.adapters.models import ADAPTERS
    from workbench.backend.adapters.base import EvalRequest
    from workbench.backend.registry import (
        checkpoint_bundle_missing_paths, checkpoint_member_root, checkpoint_path, load_registry,
    )

    module = load_module()
    model = next(m for m in load_registry()["models"] if m["model_id"] == "limn")

    # Real member files on disk, under the layout the registry declares.
    checkpoints = tmp_path / "checkpoints"
    member_dir = checkpoints / "limn"
    member_dir.mkdir(parents=True)
    members = [f"0_{category}_best_model.pt" for category in ("dress", "shirt", "toptee")]
    for name in members:
        (member_dir / name).write_bytes(b"weights-" + name.encode())

    # A bundle selection dict, exactly as the pipeline builds one.
    bundle = {"bundle_id": "base_iter0_all_categories",
              "required_checkpoint_ids": ["base_iter0_dress", "base_iter0_shirt", "base_iter0_toptee"]}
    resolved = checkpoint_path("limn", bundle, root=checkpoints)
    member_root = checkpoint_member_root(resolved)

    # The declared rule and the filesystem agree, and every member is found there.
    assert member_root == member_dir, (member_root, member_dir)
    assert resolved.is_dir(), "a bundle resolves to the directory its members occupy"
    assert sorted(p.name for p in member_root.glob("*.pt")) == sorted(members)
    # The bundle resolver looks for the members INSIDE the member directory. (The dummy weights do
    # not match the declared digests, so they are reported as unusable -- that is the digest contract
    # working, not a path defect. What this test pins is WHERE it looked.)
    reported = checkpoint_bundle_missing_paths(model, checkpoints)
    assert reported, "dummy bytes cannot satisfy the declared digests"
    for paths in reported.values():
        for path in paths:
            assert path.parent == member_dir, f"looked in the wrong directory: {path}"

    # The old expression names the WRONG directory and would find no members there.
    assert resolved.parent == checkpoints
    assert list(resolved.parent.glob("0_*_best_model.pt")) == []

    # The adapter builds its command from the SHARED rule.
    request = EvalRequest(model_id="limn", checkpoint_id="base_iter0_all_categories",
                          protocol_id="fashioniq_val_split", dataset_root=tmp_path,
                          output_path=tmp_path / "out.json", top_k=200)
    # ADAPTERS maps a model id to the adapter CLASS.
    adapter = ADAPTERS["limn"]()
    import workbench.backend.runtime as _runtime

    # The adapter asks the runtime for the model's verified interpreter; the interpreter itself is
    # not what this test is about, so it is supplied.
    monkeypatch.setattr(_runtime, "python_executable", lambda m: sys.executable, raising=False)
    command = adapter.command(tmp_path, resolved, request)
    root_argument = command[command.index("--checkpoint-root") + 1]
    assert Path(root_argument) == member_dir, command

    # And the observation command the parent builds agrees with both.
    plan = module.EvaluationPlan(
        model_id="limn", checkpoint_id="base_iter0_all_categories",
        protocol="fashioniq_val_split", dataset_root=tmp_path, source=tmp_path,
        checkpoint=resolved, cwd=tmp_path, command=command, pin="a" * 40,
        actual_source_commit="a" * 40, paper_metrics=None, member_root=member_root,
        artifact_type="category_specific_whole_models")
    assert plan.member_root == member_dir
    # The observer's own argument is derived from the plan, so it cannot differ.
    assert str(plan.member_root or checkpoint_member_root(plan.checkpoint)) == str(member_dir)


def test_the_member_root_follows_the_declaration_not_the_filesystem(tmp_path: Path) -> None:
    """A bundle is a member directory even before its weights are downloaded.

    `is_dir()` answers differently before and after a download, so deriving the member root from the
    filesystem made the adapter and the manifest disagree with a freshly-planned run. The plan
    records the root from the DECLARED artifact type instead.
    """
    from workbench.backend.registry import checkpoint_member_root

    module = load_module()
    absent = tmp_path / "checkpoints" / "limn"  # does not exist yet
    plan = module.EvaluationPlan(
        model_id="limn", checkpoint_id="base_iter0_all_categories",
        protocol="fashioniq_val_split", dataset_root=tmp_path, source=tmp_path,
        checkpoint=absent, cwd=tmp_path, command=[sys.executable, "x"], pin=None,
        actual_source_commit=None, paper_metrics=None, member_root=absent,
        artifact_type="category_specific_whole_models")
    assert plan.member_root == absent
    # The shared fallback agrees for a file-shaped artifact and for a directory-shaped one.
    # Declared rules, so the answer is the same before and after the weights arrive.
    assert checkpoint_member_root(tmp_path / "limn" / "0_dress_best_model.pt",
                                  artifact_type="single_file") == tmp_path / "limn"
    assert checkpoint_member_root(absent, artifact_type="category_specific_whole_models") == absent
    # A bare path with no declaration still resolves (filesystem fallback).
    assert checkpoint_member_root(tmp_path / "limn" / "0_dress_best_model.pt") == tmp_path / "limn"


def test_the_plan_and_the_observer_agree_before_the_weights_exist(tmp_path: Path) -> None:
    """The disagreement a reviewer found: the plan and the observer must resolve the SAME directory.

    `guarded_plan` derives the member root from the DECLARED artifact type, so it is correct before a
    download. The adapter is handed an already-resolved path (a directory for a bundle, a file for a
    single selection), so its shared rule is correct there. What must hold is that the two agree --
    the observer is invoked with the plan's answer, and the adapter's answer for the same selection
    is the same directory.
    """
    from workbench.backend.registry import checkpoint_member_root

    module = load_module()
    absent = tmp_path / "checkpoints" / "limn"          # not downloaded yet
    plan = module.EvaluationPlan(
        model_id="limn", checkpoint_id="base_iter0_all_categories",
        protocol="fashioniq_val_split", dataset_root=tmp_path, source=tmp_path,
        checkpoint=absent, cwd=tmp_path, command=[sys.executable, "x"], pin=None,
        actual_source_commit=None, paper_metrics=None, member_root=absent,
        artifact_type="category_specific_whole_models")
    # The observer's argument, exactly as `_run_observation` builds it.
    observer_root = plan.member_root or checkpoint_member_root(plan.checkpoint,
                                                               artifact_type=plan.artifact_type)
    assert observer_root == absent

    # Once downloaded, `checkpoint_path` resolves the bundle to that same directory, and the
    # adapter's shared rule returns it unchanged -- so the two answers coincide.
    absent.mkdir(parents=True)
    for category in ("dress", "shirt", "toptee"):
        (absent / f"0_{category}_best_model.pt").write_bytes(b"w")
    assert checkpoint_member_root(absent) == absent == observer_root

def test_a_vendored_checkout_reports_its_own_pin_not_the_workbench_commit(tmp_path, monkeypatch):
    """The completion proof must name the MODEL's revision, not the workbench's.

    `pinned_revision` ran a bare `git -C <source> rev-parse HEAD`. A vendored `third_party/<MODEL>`
    is delivered with its nested `.git` removed, so Git walks up to the enclosing workbench
    repository and answers with the WORKBENCH's HEAD. The completion proof then compared that
    against the model's `upstream_commit_sha`, refused with

        declared output source commit is '<workbench sha>', expected '<model sha>'

    and no evaluation of any vendored model could produce a valid proof. The two values belong to
    different repositories and can never agree. Passing the model delegates to the canonical
    provenance answer, which reads the recorded manifest for this layout.
    """
    import json
    from pathlib import Path

    from workbench.scripts import evaluate_models

    source = tmp_path / "third_party" / "LIMN"
    source.mkdir(parents=True)
    model_commit = "7d7bc9b116f594a65ac22457491edf28a88d3c3e"
    (source / "provenance.json").write_text(json.dumps({
        "model_ids": ["limn"], "upstream_commit": model_commit,
        "file_manifest": {}, "local_fixes": [],
    }))
    model = {"model_id": "limn", "upstream_commit_sha": model_commit}

    assert evaluate_models.pinned_revision(source, model) == model_commit
    # Without a model there is no declaration to read, and the answer must NOT be the enclosing
    # repository's HEAD: this checkout is vendored (no nested `.git`), so `git rev-parse` would walk
    # up and name the workbench itself. `None` is the honest answer -- the revision is unavailable
    # without the record that states it.
    assert evaluate_models.pinned_revision(source) is None

def _enclosing_git_repo(root: Path) -> str:
    """A throwaway Git repository whose HEAD can be borrowed if the walk-up fires.

    The earlier version of this regression used `tmp_path`, which on most hosts is outside any Git
    repository -- so `git rev-parse` failed and the test passed without ever exercising the walk-up it
    exists to catch. It also FAILED, for the same reason, whenever pytest's basetemp happened to sit
    inside a repository. Building the enclosing repository explicitly makes the test mean the same
    thing everywhere.
    """
    root.mkdir(parents=True, exist_ok=True)
    run = lambda *args: subprocess.run(["git", *args], cwd=root, check=True,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run("init", "-q")
    run("config", "user.email", "t@t.invalid")
    run("config", "user.name", "T")
    (root / ".keep").write_text("")
    run("add", ".keep")
    run("commit", "-q", "-m", "enclosing")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                          capture_output=True, text=True).stdout.strip()


def test_a_vendored_checkout_with_no_recorded_pin_does_not_borrow_the_enclosing_commit(tmp_path):
    """A missing manifest must not silently fall back to the enclosing repository's HEAD.

    Reproduced with a REAL enclosing repository, which is the only layout where the walk-up can
    happen: the checkout has no nested `.git`, so `git rev-parse` run inside it resolves to the
    repository above. Before the fix this returned that commit -- a revision from an unrelated
    project presented as the model's own.
    """
    from workbench.scripts import evaluate_models

    enclosing = _enclosing_git_repo(tmp_path / "enclosing")
    source = tmp_path / "enclosing" / "third_party" / "NOPIN"
    source.mkdir(parents=True)
    assert not (source / ".git").exists(), "the fixture must be a VENDORED checkout"

    model = {"model_id": "nopin", "upstream_commit_sha": "deadbeef" * 5}
    # With a model, the authoritative path answers.
    assert evaluate_models.pinned_revision(source, model) is None
    # Without a model there is no record to read, and the enclosing commit must NOT be borrowed.
    assert evaluate_models.pinned_revision(source) is None, (
        "the pathless overload returned the enclosing repository's revision for a vendored checkout "
        f"({enclosing})")
    # Control: the walk-up WOULD have reached that commit, so the assertion above is meaningful.
    assert enclosing == subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                                       capture_output=True, text=True).stdout.strip()


def test_a_corrupt_vendored_record_does_not_borrow_the_enclosing_commit(tmp_path):
    """A corrupt manifest is an unavailable state, not a licence to read an unrelated repository."""
    from workbench.scripts import evaluate_models

    _enclosing_git_repo(tmp_path / "enclosing")
    source = tmp_path / "enclosing" / "third_party" / "BADPIN"
    source.mkdir(parents=True)
    (source / "provenance.json").write_text("{ this is not json")

    model = {"model_id": "badpin", "upstream_commit_sha": "deadbeef" * 5}
    assert evaluate_models.pinned_revision(source, model) is None
    assert evaluate_models.pinned_revision(source) is None


def test_a_checkout_with_its_own_git_still_answers_from_git(tmp_path):
    """The guard is narrow: a genuine nested checkout's revision IS its own."""
    import subprocess

    from workbench.scripts import evaluate_models

    _enclosing_git_repo(tmp_path / "enclosing")
    source = tmp_path / "enclosing" / "third_party" / "NESTED"
    source.mkdir(parents=True)
    run = lambda *args: subprocess.run(["git", *args], cwd=source, check=True,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run("init", "-q")
    run("config", "user.email", "t@t.invalid")
    run("config", "user.name", "T")
    (source / ".keep").write_text("")
    run("add", ".keep")
    run("commit", "-q", "-m", "upstream")
    own = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, check=True,
                         capture_output=True, text=True).stdout.strip()

    assert evaluate_models.pinned_revision(source) == own
def test_a_per_query_artifact_that_disagrees_with_the_aggregate_is_refused(monkeypatch, tmp_path: Path) -> None:
    """The published artifact must reproduce the aggregate THIS run recorded, checked in the parent.

    `cross_check_with_native` cannot establish this: it runs inside the observer and compares the
    observation against native `test()` calls made in that SAME process, on the same model object.
    Native `test()` batches a category's queries and reduces them on the GPU, so two passes compute
    features whose last bits differ. On a real ENCODER run the two passes agreed on 99.8% of queries
    -- enough for the in-process check to pass, and not enough for the published ranks to reproduce
    the aggregate the report was certified from (30 hit-counts apart across the six Recall values,
    with both sides recording INSTRUMENTATION_PARITY_VERIFIED). Recall@k is an exact hit count over
    the same population, so the parent compares counts and refuses on any difference.

    The artifact here is self-consistent -- each target really is at its recorded rank -- so nothing
    the observer or the publisher could have checked would have caught it.
    """
    module = load_module()
    run_id = "20260101T000000000000Z_limn_ckpt_deadbeef"
    # All three queries have their target inside the top 10, so the artifact's reconstructed
    # `dress_r10` is 100% while the aggregate below says 2%: a genuine disagreement. Without the check
    # the parent publishes this artifact as the run's evidence.
    queries = [
        {"category": "dress", "target_rank": 1, "top_results": [{"rank": 1, "image_id": "a", "score": 1.0}]},
        {"category": "dress", "target_rank": 5, "top_results": [{"rank": 5, "image_id": "b", "score": 0.5}]},
        {"category": "dress", "target_rank": 7, "top_results": [{"rank": 7, "image_id": "c", "score": 0.1}]},
    ]

    def fake_invoke(plan, config, run_id_arg, root):
        directory = tmp_path / "logs" / f"{plan.model_id}__{plan.checkpoint_id}__{run_id_arg}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "stdout.log").write_text('{"dress_r1": 1.0, "dress_r10": 2.0, "dress_r50": 3.0}')
        (directory / "stderr.log").write_text("")
        return 0, directory, {"orchestrator_python": "3", "platform": "test"}

    def observer_run(argv, **kwargs):
        if "export" in " ".join(argv):
            directory = tmp_path / "logs"
            directory.mkdir(parents=True, exist_ok=True)
            staged = directory / "payload.json"
            staged.write_text("{}")

            class Ok:
                returncode = 0
                stdout = json.dumps({"payload": str(staged)})
                stderr = ""
            return Ok()
        artifact = tmp_path / "results" / "run.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps({
            "run": {"run_id": run_id, "model_id": "limn", "protocol_id": "fashioniq_val_split",
                    "instrumentation_parity": "INSTRUMENTATION_PARITY_VERIFIED"},
            "queries": queries,
        }))

        class Ok2:
            returncode = 0
            stdout = json.dumps({"published": str(artifact)})
            stderr = ""
        return Ok2()

    monkeypatch.setattr(module, "invoke_evaluator", fake_invoke)
    monkeypatch.setattr(module.subprocess, "run", observer_run)
    monkeypatch.setattr(module, "extract_aggregate_metrics",
                        lambda *a, **k: type("E", (), {
                            "extraction_status": "EXTRACTED_VERIFIED", "parser_id": "p",
                            "parser_version": 1,
                            "observed_metrics": {"r10": 1.0, "r50": 2.0, "mean": 1.5},
                            "category_metrics": {"dress": {"r1": 1.0, "r10": 2.0, "r50": 3.0}},
                            "parity_status": "PARITY_NOT_EVALUATED",
                            "metric_source": "stdout"})())
    plan = module.EvaluationPlan(
        model_id="limn", checkpoint_id="base_iter0_all_categories", protocol="fashioniq_val_split",
        dataset_root=tmp_path, source=tmp_path, checkpoint=tmp_path / "ckpt", cwd=tmp_path,
        command=[sys.executable, "x"], pin="a" * 40, actual_source_commit="a" * 40,
        paper_metrics=None)
    assert module.execute(plan, config(tmp_path), run_id=run_id) == 0

    report = json.loads(sorted(tmp_path.rglob("*_aggregate.json"))[-1].read_text())
    reason = report.get("per_query_export_reason") or ""
    assert "does not reproduce" in reason, (
        f"a disagreeing artifact must be refused by the aggregate agreement check; reason={reason!r}")
    assert report["per_query_export_available"] is False
    assert report.get("per_query_artifact_digest") is None


def test_evaluator_environment_resolves_hf_home_from_workbench_env(monkeypatch, tmp_path: Path):
    module = load_module()
    configured = tmp_path / "model-cache"
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr("workbench.backend.model_settings.file_settings",
                        lambda: {"HF_HOME": str(configured)})

    child_env = module._evaluator_environment({}, "tme")
    assert child_env["HF_HOME"] == str(configured)
    result = subprocess.run(
        [sys.executable, "-c", "import os; print(os.environ['HF_HOME'])"],
        env=child_env, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == str(configured)
    assert "HF_HOME" not in __import__("os").environ
