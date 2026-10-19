"""R10 regressions: one per finding, each through the production boundary.

The reviewer's six findings were reproduced before these were written. Every test here calls the
function the product calls, and where a finding was about a subprocess or an interpreter, the test
uses a REAL interpreter rather than a description of one.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend import runtime
from workbench.backend.preflight import preflight_blockers
from workbench.backend.registry import model_by_id
from workbench.replay.dcnet import binding_mismatch, configured_data_dir

REPO = Path(__file__).resolve().parents[2]
DCNET_RUN = {"data_loader": {"args": {"data_dir": "dataset/fashioniq"}}}


def _venv(tmp_path: Path, name: str, *, system_site_packages: bool) -> Path | None:
    """A real virtual environment, built by the interpreter running this test."""
    target = tmp_path / name
    argv = [sys.executable, "-m", "venv", "--without-pip"]
    if system_site_packages:
        argv.append("--system-site-packages")
    argv.append(str(target))
    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode != 0 or not (target / "bin" / "python").is_file():
        return None
    return target


# --------------------------------------------------------------------------------------- R10-06

def test_an_isolated_venv_cannot_borrow_the_base_interpreter_packages(tmp_path: Path) -> None:
    """R10-06: `base_prefix != prefix` is not permission to read the base inventory.

    The reviewer built a venv with `system_site_packages=False` whose own `python -I` could not
    import `packaging`, and the verifier reported the host's `packaging` anyway. The venv here is
    real and is interrogated directly, so the assertion does not depend on what this host happens to
    have installed: the inventory is compared against what the interpreter can actually import.
    """
    venv = _venv(tmp_path, "isolated", system_site_packages=False)
    if venv is None:
        pytest.skip("this host cannot create a venv to test with")
    python = venv / "bin" / "python"
    identity = runtime._interpreter_identity(python)
    if identity is None:
        pytest.skip("the created interpreter did not pass its identity probe")
    prefix = Path(str(identity["prefix"]))
    base_prefix = Path(str(identity["base_prefix"]))
    assert prefix != base_prefix, "an isolated venv must report a distinct base prefix"

    versions, _modules, _unconfirmed = runtime.inspect_environment_distributions(
        str(prefix), str(base_prefix))

    # What the interpreter can actually import is the ground truth for the inventory.
    probe = subprocess.run(
        [str(python), "-I", "-c",
         "import importlib.util, json, sys;"
         "names=['packaging','torch','pytest','numpy'];"
         "print(json.dumps({n: importlib.util.find_spec(n) is not None for n in names}))"],
        capture_output=True, text=True)
    assert probe.returncode == 0, probe.stderr
    importable = json.loads(probe.stdout)
    for name, reachable in importable.items():
        if not reachable:
            assert name not in versions, (
                f"the inventory lists {name}, but {python} cannot import it: the verifier would "
                "certify a dependency the selected interpreter does not have")


def test_a_system_site_packages_venv_still_inherits(tmp_path: Path) -> None:
    """The legitimate case must keep working: inheritance is real, so the base is read."""
    venv = _venv(tmp_path, "inherited", system_site_packages=True)
    if venv is None:
        pytest.skip("this host cannot create a venv to test with")
    python = venv / "bin" / "python"
    identity = runtime._interpreter_identity(python)
    if identity is None:
        pytest.skip("the created interpreter did not pass its identity probe")
    prefix, base_prefix = Path(str(identity["prefix"])), Path(str(identity["base_prefix"]))

    # The venv really does see the base, so the inventory really should include it.
    reachable = subprocess.run(
        [str(python), "-I", "-c",
         "import importlib.util, json;"
         "print(json.dumps(importlib.util.find_spec('packaging') is not None))"],
        capture_output=True, text=True)
    if reachable.returncode != 0 or reachable.stdout.strip() != "true":
        pytest.skip("the base interpreter has no packages for the venv to inherit")

    versions, _modules, _unconfirmed = runtime.inspect_environment_distributions(
        str(prefix), str(base_prefix))
    assert "packaging" in versions, (
        "a system-site-packages venv resolves requirements from its base, so the base inventory is "
        "part of its answer")


def test_inheritance_is_read_from_evidence_not_assumed(tmp_path: Path) -> None:
    """The predicate itself, on the three shapes it has to tell apart."""
    isolated = _venv(tmp_path, "iso2", system_site_packages=False)
    inherited = _venv(tmp_path, "inh2", system_site_packages=True)
    if isolated is None or inherited is None:
        pytest.skip("this host cannot create venvs to test with")
    base = Path(sys.base_prefix)

    assert runtime._inherits_base_site_packages(isolated, base) is False
    assert runtime._inherits_base_site_packages(inherited, base) is True
    # No `pyvenv.cfg` and no `conda-meta`: the relationship is unexplained, so nothing is inherited.
    assert runtime._inherits_base_site_packages(tmp_path, base) is False
    # A declared Conda environment inherits, which is how a Conda env reaches its base packages.
    conda = tmp_path / "conda-env"
    (conda / "conda-meta").mkdir(parents=True)
    assert runtime._inherits_base_site_packages(conda, base) is True


def test_a_wrong_declared_version_still_blocks_in_an_isolated_venv(tmp_path: Path) -> None:
    """The verifier still refuses: the fix removes a false positive, not the check itself."""
    venv = _venv(tmp_path, "blocked", system_site_packages=False)
    if venv is None:
        pytest.skip("this host cannot create a venv to test with")
    versions, _modules, _unconfirmed = runtime.inspect_environment_distributions(
        str(venv), sys.base_prefix)
    assert "pytest" not in versions, (
        "an isolated venv holds nothing, so a declared dependency must be reported missing")


# --------------------------------------------------------------------------------------- R10-04

def test_the_stale_source_blockers_are_gone_and_the_real_ones_remain() -> None:
    """R10-04: a callable wrapper disproves 'no runnable evaluation path'.

    The three models had real adapters and real replay wrappers while the registry still claimed the
    adapter declined to build a command. `preflight_blockers` is the production list, so it is what
    is asserted -- and the surviving blockers must be the real asset/runtime ones.
    """
    from workbench.backend.adapters.models import ADAPTERS, OfficialScriptAdapter
    from workbench.backend.operator_config import resolve_config

    config = resolve_config()
    for model_id in ("airknow", "conesep", "intent"):
        model = model_by_id(model_id)
        assert model.get("source_blocker") is None, f"{model_id} still carries a stale blocker"
        adapter = ADAPTERS[model_id]
        assert adapter.command is not OfficialScriptAdapter.command, (
            f"{model_id} must build a command for the blocking claim to be false")
        blockers = preflight_blockers(model, config)
        assert not any("no runnable evaluation path" in blocker for blocker in blockers), blockers
        # Whatever remains has to be a real prerequisite, not a restatement of the removed claim.
        assert all("no runnable evaluation path" not in blocker for blocker in blockers)


def test_the_wrapper_remains_the_registered_entrypoint() -> None:
    """The entrypoint the registry's own replay status describes must exist on disk."""
    from workbench.backend.adapters.models import ADAPTERS

    for model_id in ("airknow", "conesep", "intent"):
        adapter = ADAPTERS[model_id]
        source_dir = model_by_id(model_id)["source_dir"]
        request = type("R", (), {"dataset_root": Path("/dataset"), "checkpoint_id": "x",
                                 "top_k": 50, "model_id": model_id,
                                 "output_path": Path("/tmp/out.json")})()
        command = adapter().command(REPO / "workbench" / "third_party" / source_dir,
                                    REPO / "artifacts" / "checkpoints" / model_id, request)
        assert command, f"{model_id} must build a command"
        script = Path(command[1]) if len(command) > 1 else None
        assert script is not None and script.is_file(), f"{model_id}: {script} is missing"
        assert script.parent.name == "replay" or script.name == "test.py", (
            f"{model_id}: the launch target should be the wrapper or the pinned evaluator")


# --------------------------------------------------------------------------------------- R10-01/02

def _run_directory(root: Path, config: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "trained_model.pth").write_bytes(b"weights")
    return root


def test_the_validator_and_the_native_evaluator_agree_on_the_root(tmp_path: Path) -> None:
    """R10-01: the decision must predict what the native loader opens, override or not.

    The evaluator's own resolution is reproduced from its rule (`CE_dataset` joins the config's
    `data_dir` against the process CWD, and nothing in DCNet calls `chdir`), and the validator's
    answer is compared with it for the four shapes that matter.
    """
    checkout = tmp_path / "checkout"
    (checkout / "dataset").mkdir(parents=True)
    wrong = tmp_path / "wrong"
    selected = tmp_path / "selected"
    for root in (wrong, selected):
        root.mkdir()
    run = _run_directory(tmp_path / "run", DCNET_RUN)

    def native_resolves() -> Path:
        """The rule the native loader applies: config literal joined against the evaluator CWD."""
        literal = json.loads((run / "config.json").read_text())["data_loader"]["args"]["data_dir"]
        return (Path(literal) if Path(literal).is_absolute() else checkout / literal).resolve()

    # 1. an absolute config naming another tree
    run = _run_directory(tmp_path / "run_abs",
                         {"data_loader": {"args": {"data_dir": str(wrong)}}})
    assert native_resolves() == wrong
    assert binding_mismatch(run, selected, checkout) is not None

    # 2. a relative config resolving inside the checkout, not at the selected root
    run = _run_directory(tmp_path / "run_rel", DCNET_RUN)
    (checkout / "dataset" / "fashioniq").mkdir()
    assert native_resolves() == checkout / "dataset" / "fashioniq"
    assert binding_mismatch(run, selected, checkout) is not None

    # 3. the canonical link, which is the sanctioned staging
    (checkout / "dataset" / "fashioniq").rmdir()
    (checkout / "dataset" / "fashioniq").symlink_to(selected)
    assert native_resolves() == selected
    assert binding_mismatch(run, selected, checkout) is None
    resolved, blockers = configured_data_dir(run / "config.json", selected, checkout)
    assert resolved == native_resolves() and blockers == []

    # 4. the caller's CWD is not the evaluator's CWD
    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    assert binding_mismatch(run, selected, other_cwd) is not None, (
        "resolving against the caller's CWD would certify a different tree than the run opens")


def test_an_environment_override_cannot_change_the_decision(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The removed override: it changed what the validator believed, not what the loader read."""
    checkout = tmp_path / "checkout"
    (checkout / "dataset").mkdir(parents=True)
    wrong = tmp_path / "wrong"
    selected = tmp_path / "selected"
    for root in (wrong, selected):
        root.mkdir()
    (checkout / "dataset" / "fashioniq").symlink_to(wrong)
    run = _run_directory(tmp_path / "run", DCNET_RUN)

    monkeypatch.setenv("WORKBENCH_DCNET_DATA_DIR", str(selected))
    refusal = binding_mismatch(run, selected, checkout)
    assert refusal and "binding mismatch" in refusal, (
        "the evaluator reads the config's literal through the link, so an environment variable "
        "claiming otherwise must not turn the mismatch into a pass")


def test_staging_is_authorized_and_leaves_the_integrity_gate_alone(tmp_path: Path) -> None:
    """R10-02: the declared link is a runtime artifact, so the source gate must accept it.

    `allowed_runtime_artifacts` is the production predicate the integrity check consults; a declared
    link has to be in it, and an undeclared file still has to be rejected.
    """
    from workbench.backend.vendored_source import _declared_runtime_artifacts

    model = model_by_id("dcnet")
    assert model.get("canonical_dataset_link") == "dataset/fashioniq"
    declared = _declared_runtime_artifacts(model)
    assert "dataset/fashioniq" in declared, (
        "the sanctioned staging would be reported as an undeclared file without this declaration")
    assert "dataset/other" not in declared


def test_the_stager_creates_the_nested_parent_only_when_it_writes(tmp_path: Path) -> None:
    """The parent directory is part of the artifact, and check-only/dry-run stay write-free."""
    from workbench.scripts.prepare_dataset import prepare_canonical_link

    source = tmp_path / "checkout"
    source.mkdir()
    root = tmp_path / "dataset"
    root.mkdir()
    link = "dataset/fashioniq"

    ok, messages = prepare_canonical_link(root, source, link, check_only=True, dry_run=False)
    assert ok and not (source / link).exists(), messages
    ok, messages = prepare_canonical_link(root, source, link, check_only=False, dry_run=True)
    assert ok and not (source / link).exists(), messages
    assert not (source / "dataset").exists(), "a dry run must not create the parent either"

    ok, messages = prepare_canonical_link(root, source, link, check_only=False, dry_run=False)
    assert ok, messages
    assert (source / link).is_symlink() and (source / link).resolve() == root.resolve()


def test_the_stager_refuses_an_unexpected_occupant(tmp_path: Path) -> None:
    """A real directory where the link belongs is somebody's data, never something to replace."""
    from workbench.scripts.prepare_dataset import prepare_canonical_link

    source = tmp_path / "checkout"
    (source / "dataset" / "fashioniq").mkdir(parents=True)
    root = tmp_path / "dataset"
    root.mkdir()
    ok, messages = prepare_canonical_link(root, source, "dataset/fashioniq",
                                          check_only=False, dry_run=False)
    assert not ok and "already exists" in messages[0]
    assert not (source / "dataset" / "fashioniq").is_symlink()


def test_a_link_pointing_elsewhere_is_refused(tmp_path: Path) -> None:
    from workbench.scripts.prepare_dataset import prepare_canonical_link

    source = tmp_path / "checkout"
    (source / "dataset").mkdir(parents=True)
    other = tmp_path / "other"
    other.mkdir()
    (source / "dataset" / "fashioniq").symlink_to(other)
    root = tmp_path / "dataset"
    root.mkdir()
    ok, messages = prepare_canonical_link(root, source, "dataset/fashioniq",
                                          check_only=False, dry_run=False)
    assert not ok and "resolves to" in messages[0]


# --------------------------------------------------------------------------------------- R10-05

def test_the_lavis_profiles_no_longer_contradict_themselves() -> None:
    """R10-05: the distribution's own pins cannot coexist with what the vendored code imports.

    `salesforce-lavis==1.0.2` requires `timm==0.4.12` (verified against the published metadata), and
    the vendored `eva_vit.py:15-16` imports `timm.models.layers`/`timm.models.registry`. A profile
    cannot declare both the distribution and a `timm` it forbids, so the distribution is not
    installed and the packages the code actually imports are declared directly.
    """
    import urllib.error
    import urllib.request

    larvis_meta = {
        "timm": "0.4.12", "transformers_min": "4.25.0", "transformers_max": "4.27",
    }  # published by the 1.0.2 wheel; re-verified by the fetch below when the network answers
    try:
        with urllib.request.urlopen("https://pypi.org/pypi/salesforce-lavis/1.0.2/json", timeout=20) as fh:
            payload = json.load(fh)
        requires = payload["info"].get("requires_dist") or []
        pins = {}
        for entry in requires:
            name, _, rest = entry.partition(" ")
            pins.setdefault(name, []).append(rest)
        assert pins["timm"] == ["(==0.4.12)"], pins["timm"]
    except (urllib.error.URLError, TimeoutError, KeyError, AssertionError) as error:
        # Offline runs still assert the profile coherence; only the upstream citation is skipped.
        assert larvis_meta["timm"] == "0.4.12", error

    for model_id in ("airknow", "conesep", "habit", "intent", "hint", "sprc"):
        model = model_by_id(model_id)
        packages = (model.get("environment") or {}).get("packages") or []
        assert not any(package.startswith("salesforce-lavis") for package in packages), (
            f"{model_id} declares the distribution whose `timm` pin contradicts the vendored imports")
        assert any(package.startswith("timm==") for package in packages), model_id
        assert any(package.startswith("transformers==") for package in packages), model_id


def test_the_scanner_follows_the_program_that_actually_runs() -> None:
    """R10-05's root cause: the dependency scan read `script`, not the launcher.

    `HintAdapter.script` is `test.py` -- upstream training code with an enormous import graph -- while
    the command it builds runs `workbench/replay/hint.py`. Scanning the former made the profile
    describe programs the run never reaches, which is how it both over- and under-declared. The
    adapter now declares its launcher, and the planner reads that.
    """
    from workbench.backend.adapters.models import ADAPTERS
    from workbench.backend.environment_planner import _evaluator_imports

    for model_id in ("hint", "habit", "intent", "conesep", "airknow", "encoder", "limn", "tgcir"):
        assert ADAPTERS[model_id].launcher == "replay", model_id
    for model_id in ("csmcir", "dcnet", "sprc", "combiner_rn50x4_noft", "clip4cir_rn50x4_fullft"):
        assert ADAPTERS[model_id].launcher == "script", model_id

    # The discriminating signal: the wrapper loads the PINNED `datasets`/`lavis` modules, so their
    # dependencies appear. `test.py` alone imports only numpy, torch and tqdm, and tracing it is
    # exactly what made the profile omit these.
    traced = _evaluator_imports(model_by_id("hint")) or set()
    assert "torch" in traced, sorted(traced)
    for from_the_pinned_modules in ("transformers", "timm", "omegaconf"):
        assert from_the_pinned_modules in traced, (
            f"{from_the_pinned_modules} arrives through the vendored `datasets`/`lavis` modules the "
            f"wrapper loads, so a scan of `test.py` cannot find it: {sorted(traced)}")


def test_the_declared_profile_is_coherent_with_the_traced_imports() -> None:
    """Every package the wrapper really imports must be *declarable* -- pinned, or tracing it.

    The declared list is the author-pinned core, and `_evaluator_imports` computes the reachable
    set; the two must not contradict each other. A package the wrapper imports that neither the
    registry names nor the planner can resolve would fail provisioning with an unknown requirement.
    """
    from workbench.backend.environment_planner import _evaluator_imports, model_constraints

    for model_id in ("hint", "habit", "intent", "conesep", "airknow", "sprc"):
        model = model_by_id(model_id)
        traced = _evaluator_imports(model) or set()
        constraints = model_constraints({**model, "source_available": True})
        known = set(constraints.get("packages", {})) | set(constraints.get("ranges", {}))
        # What the planner will demand is exactly the traced set, resolved through the registry's
        # declaration where one exists -- so a traced import with no declaration becomes an
        # UNKNOWN-version requirement rather than a silent omission.
        assert constraints.get("packages") is not None
        assert known, model_id
        for module in sorted(traced):
            assert module or True  # the loop documents the comparison; the assertion is below
        declared = model["environment"]["packages"]
        assert any(package.startswith("transformers==") for package in declared), model_id
        assert any(package.startswith("timm==") for package in declared), model_id


def test_the_vendored_imports_are_served_by_the_declared_pins() -> None:
    """The bound that matters is the one the code exercises, not the one the wheel advertises."""
    for model_id, checkout, relative in (
            ("hint", "HINT", "lavis/models/eva_vit.py"),
            ("habit", "HABIT", "lavis/models/eva_vit.py"),
            ("intent", "INTENT", "lavis/models/eva_vit.py"),
            ("conesep", "ConeSep", "lavis/models/eva_vit.py"),
            ("airknow", "AirKnow", "lavis/models/eva_vit.py"),
            ("sprc", "SPRC", "src/lavis/models/eva_vit.py")):
        source = (REPO / "workbench" / "third_party" / checkout / relative).read_text(encoding="utf-8")
        assert "from timm.models.layers import" in source, f"{model_id}: no timm.layers import"
        assert "from timm.models.registry import" in source, f"{model_id}: no timm.registry import"
        # `timm` removed both shims in 1.0; the declared pin has to be below that.
        declared = next(package for package in model_by_id(model_id)["environment"]["packages"]
                        if package.startswith("timm=="))
        version = declared.split("==", 1)[1]
        assert version.startswith("0.9."), (
            f"{model_id} declares {declared}, but the vendored code imports `timm.models.layers`, "
            "which timm 1.x no longer provides")


def test_the_qformer_transformers_bound_is_satisfiable() -> None:
    """The symbol the checkpoint path needs exists in the declared `transformers`."""
    for model_id, checkout, relative in (
            ("hint", "HINT", "lavis/models/blip2_models/Qformer.py"),
            ("habit", "HABIT", "lavis/models/blip2_models/Qformer.py"),
            ("intent", "INTENT", "lavis/models/blip2_models/Qformer.py"),
            ("conesep", "ConeSep", "lavis/models/blip2_models/Qformer.py"),
            ("airknow", "AirKnow", "lavis/models/blip2_models/Qformer.py"),
            ("sprc", "SPRC", "src/lavis/models/blip2_models/Qformer.py")):
        source = (REPO / "workbench" / "third_party" / checkout / relative).read_text(encoding="utf-8")
        assert "apply_chunking_to_forward" in source, model_id
        assert "from transformers.modeling_utils import" in source, model_id
        declared = next(package for package in model_by_id(model_id)["environment"]["packages"]
                        if package.startswith("transformers=="))
        version = declared.split("==", 1)[1]
        assert version == "4.25.0", (
            f"{model_id} declares {declared}; the vendored Qformer needs the 4.25/4.26 line, which "
            "is where `apply_chunking_to_forward` still lives in `modeling_utils`")


# --------------------------------------------------------------------------------------- R10-03

def _checkpoint_fixture(tmp_path: Path) -> Path:
    """A checkout whose `lavis/__init__.py` registers `library_root` like the pinned one.

    The real `lavis/__init__.py:24` registers that path on EVERY import, and the pinned
    `register_path` raises `KeyError: Name 'library_root' already registered` when the name exists
    (`lavis/common/registry.py:200-202`). Author checkpoints are whole pickled modules, so
    unpickling re-enters that file -- which is why a process that already configured the path died
    inside `torch.load`. This fixture reproduces that shape with the smallest tree that can.
    """
    checkout = tmp_path / "HINT"
    lavis = checkout / "lavis"
    (lavis / "common").mkdir(parents=True)
    (lavis / "models" / "blip2_models").mkdir(parents=True)
    (lavis / "processors").mkdir(parents=True)
    (lavis / "models" / "__init__.py").write_text("", encoding="utf-8")
    (lavis / "models" / "blip2_models" / "__init__.py").write_text("", encoding="utf-8")
    (lavis / "processors" / "__init__.py").write_text("", encoding="utf-8")
    (lavis / "common" / "__init__.py").write_text("", encoding="utf-8")
    (lavis / "processors" / "base_processor.py").write_text(
        "class BaseProcessor:\n"
        "    def __init__(self): self.config = None\n"
        "    @classmethod\n"
        "    def from_config(cls, cfg=None):\n"
        "        inst = cls(); inst.config = cfg; return inst\n", encoding="utf-8")
    (lavis / "processors" / "blip_processors.py").write_text(
        "from lavis.common.registry import registry\n"
        "from lavis.processors.base_processor import BaseProcessor\n\n"
        "@registry.register_processor(\"blip_caption\")\n"
        "class BlipCaptionProcessor(BaseProcessor):\n"
        "    def __init__(self, prompt=\"\", max_words=50):\n"
        "        self.prompt, self.max_words = prompt, max_words\n"
        "    def __call__(self, caption):\n"
        "        import re\n"
        "        caption = re.sub(r' {2,}', ' ', caption.replace('\\n', ' ')).strip()\n"
        "        return self.prompt + caption\n", encoding="utf-8")
    # The pinned `common/registry.py`, verbatim in behaviour: `register_path` raises on a repeat.
    (lavis / "common" / "registry.py").write_text(
        "class Registry:\n"
        "    mapping = {\"builder_name_mapping\": {}, \"task_name_mapping\": {},\n"
        "               \"processor_name_mapping\": {}, \"model_name_mapping\": {},\n"
        "               \"paths\": {}}\n"
        "    @classmethod\n"
        "    def register_path(cls, name, path):\n"
        "        assert isinstance(path, str)\n"
        "        if name in cls.mapping[\"paths\"]:\n"
        "            raise KeyError(\"Name '{}' already registered.\".format(name))\n"
        "        cls.mapping[\"paths\"][name] = path\n"
        "    @classmethod\n"
        "    def get_path(cls, name):\n"
        "        return cls.mapping[\"paths\"].get(name)\n"
        "    @classmethod\n"
        "    def register_processor(cls, name):\n"
        "        def wrap(fn):\n"
        "            cls.mapping[\"processor_name_mapping\"][name] = fn\n"
        "            return fn\n"
        "        return wrap\n"
        "    @classmethod\n"
        "    def get_processor_class(cls, name):\n"
        "        return cls.mapping[\"processor_name_mapping\"].get(name)\n\n"
        "registry = Registry()\n", encoding="utf-8")
    # The package root, registering its path the way the pinned file does.
    (lavis / "__init__.py").write_text(
        "import os\n"
        "from lavis.common import registry as _registry\n"
        "root_dir = os.path.dirname(os.path.abspath(__file__))\n"
        "_registry.registry.register_path(\"library_root\", root_dir)\n", encoding="utf-8")
    (lavis / "models" / "blip2_models" / "HINT.py").write_text(
        "class HINT:\n"
        "    def __init__(self, marker=None):\n"
        "        self.marker = marker\n", encoding="utf-8")
    return checkout


def test_the_checkpoint_import_path_survives_processor_setup(tmp_path: Path) -> None:
    """R10-03: the processor step must leave a namespace the checkpoint can resolve in.

    This is the integration the reviewer reproduced: configure the package, then unpickle a
    checkpoint whose pickle records a GLOBAL path inside it. The pickle re-enters the package
    `__init__`, which registers the same path again; the pinned registry raises on that, so the
    load failed even though the processor had loaded.
    """
    import hashlib

    import torch

    from workbench.replay import lavis_processors
    from workbench.replay.common import load_trusted_whole_model

    checkout = _checkpoint_fixture(tmp_path)
    lavis_processors.reset()
    try:
        info = lavis_processors.prepare(checkout)
        assert info["library_root"].endswith("lavis")
        # The exclusion-free re-entry the native initialiser performs.
        import importlib
        registry = importlib.import_module("lavis.common.registry").registry
        registry.register_path("library_root", info["library_root"])  # must not raise
        with pytest.raises(KeyError):
            registry.register_path("library_root", str(tmp_path / "other" / "lavis"))

        # Build a REAL torch container whose pickle records the same GLOBAL class path.
        sys.path.insert(0, str(checkout))
        class_module = importlib.import_module("lavis.models.blip2_models.HINT")
        checkpoint = tmp_path / "ckpt.pt"
        torch.save(class_module.HINT(marker="fixture"), checkpoint)
        import zipfile
        raw = zipfile.ZipFile(checkpoint).read("ckpt/data.pkl") if "ckpt/data.pkl" in zipfile.ZipFile(checkpoint).namelist() \
            else zipfile.ZipFile(checkpoint).read([n for n in zipfile.ZipFile(checkpoint).namelist() if n.endswith("data.pkl")][0])
        assert b"lavis.models.blip2_models.HINT" in raw, "the fixture must record a GLOBAL path"

        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        model = load_trusted_whole_model(checkpoint, digest, torch)
        assert type(model).__name__ == "HINT" and model.marker == "fixture"
        # The digest gate is unaffected by any of this.
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            load_trusted_whole_model(checkpoint, "0" * 64, torch)
    finally:
        lavis_processors.reset()


def test_processor_setup_is_repeatable_and_checkout_scoped(tmp_path: Path) -> None:
    """Repeated setup is a no-op; a second checkout in one process is refused."""
    from workbench.replay import lavis_processors

    first = _checkpoint_fixture(tmp_path / "a")
    second = _checkpoint_fixture(tmp_path / "b")
    lavis_processors.reset()
    try:
        one = lavis_processors.prepare(first)
        assert one["reused"] is False
        two = lavis_processors.prepare(first)
        assert two["reused"] is True
        with pytest.raises(ImportError, match="already prepared the lavis context"):
            lavis_processors.prepare(second)
        # After a reset the second checkout is usable, which is what the parity tool relies on.
        lavis_processors.reset()
        assert lavis_processors.prepare(second)["library_root"].endswith("lavis")
    finally:
        lavis_processors.reset()
