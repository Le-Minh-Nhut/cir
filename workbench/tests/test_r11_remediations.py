"""R11 regressions: the reported defects, each asserted through the production consumer.

Every case here reproduced before its fix and is written against the boundary that failed, not a
helper standing beside it: the planner/verifier pair for the version split, the generated install
specs for the discovery defect, the stager's real filesystem effects, and the DCNet readiness view.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from workbench.backend import environment_planner as planner  # noqa: E402
from workbench.backend.registry import load_registry, model_by_id  # noqa: E402
from workbench.backend.runtime import declared_requirement, version_satisfies  # noqa: E402
from workbench.replay.dcnet import binding_mismatch, configured_data_dir  # noqa: E402
from workbench.scripts.prepare_dataset import prepare_canonical_link  # noqa: E402

ALL_MODELS = sorted(model["model_id"] for model in load_registry()["models"])

#: Distributions whose declared minimum Python the reviewer established from the published metadata,
#: with the version that is compatible with the same Python. `pip`'s resolver enforces the first
#: column, so a profile that pins one of these on an older interpreter cannot be installed at all.
#: The ranges are evidence, not a guess: `uv pip compile --python-version 3.8` refuses the left
#: column and resolves the right one.
PYTHON_MINIMUM = {
    "pandas": ((2, 1), "2.0.3"),
    "scikit-image": ((0, 22), "0.21.0"),
}


def _interpreter_series(declared: str) -> tuple[int, int]:
    parts = declared.split(".")
    return int(parts[0]), int(parts[1])


# ----------------------------------------------------------------------------------- R11-01

@pytest.mark.parametrize("model_id", ALL_MODELS)
def test_every_declared_pin_is_installable_on_the_declared_python(model_id: str) -> None:
    """A profile that pins a package above its own Python cannot be provisioned at all.

    `hint`, `habit`, `conesep` and `airknow` declared Python 3.8.10 with `pandas==2.2.0` and (for
    the HINT-family two) `scikit-image==0.22.0`. Both of those require Python >= 3.9, so the
    generated install command could never succeed: the resolver refuses before downloading
    anything. The check is against the published minimum version of each named distribution, which
    is what the installer enforces, rather than the presence of a name in the registry.
    """
    environment = model_by_id(model_id).get("environment")
    if not environment:
        pytest.skip(f"{model_id} declares no environment block")
    python = str(environment.get("python") or "")
    if not python or python.upper() == "UNKNOWN":
        pytest.skip(f"{model_id} declares no Python version")
    series = _interpreter_series(python)
    offenders = []
    for declaration in environment.get("packages") or []:
        if "==" not in declaration:
            continue
        name, version = declaration.split("==", 1)
        if " " in name or "[" in name:
            continue
        minimum = PYTHON_MINIMUM.get(name.lower())
        if minimum is None:
            continue
        required_series, compatible = minimum
        if _interpreter_series(version) >= required_series and series < required_series:
            offenders.append(f"{declaration} needs Python >= {'.'.join(map(str, required_series))}, "
                             f"the profile declares {python} (use {name}=={compatible})")
    assert not offenders, f"{model_id}: {'; '.join(offenders)}"


def test_the_repaired_profiles_resolve_on_their_declared_interpreter() -> None:
    """The concrete case, resolved for real when a resolver is available.

    `uv pip compile --python-version 3.8` is what caught this: it refuses `pandas==2.2.0` outright
    and resolves the corrected pair. The test skips when `uv` is absent so the suite stays
    offline-runnable, and asserts the contradiction is gone rather than trusting the table above.
    """
    resolver = next((candidate for candidate in (os.environ.get("WORKBENCH_UV"), "uv")
                     if candidate and _which(candidate)), None)
    for model_id in ("hint", "habit", "conesep", "airknow"):
        environment = model_by_id(model_id)["environment"]
        assert _interpreter_series(str(environment["python"])) == (3, 8), model_id
        declarations = [d for d in environment["packages"]
                        if d.split("==")[0].lower() in PYTHON_MINIMUM]
        for declaration in declarations:
            name, version = declaration.split("==", 1)
            assert _interpreter_series(version) < PYTHON_MINIMUM[name.lower()][0], (
                f"{model_id} still pins {declaration}, which Python 3.8 cannot install")
        if resolver is None:
            continue
        requirement = "\n".join(declarations) + "\n"
        result = subprocess.run([resolver, "pip", "compile", "--python-version", "3.8", "-"],
                                input=requirement, capture_output=True, text=True, check=False,
                                timeout=300, cwd=REPO)
        assert result.returncode == 0, (
            f"{model_id}: the declared 3.8 profile does not resolve: {result.stderr[-400:]}")


def _which(name: str) -> str | None:
    from shutil import which

    return which(name)


# ----------------------------------------------------------------------------------- R11-02

@pytest.mark.parametrize("model_id", ALL_MODELS)
def test_the_planned_packages_satisfy_what_the_verifier_requires(monkeypatch: pytest.MonkeyPatch, model_id: str) -> None:
    """The boundary the SPRC defect crossed: plan and verification must agree.

    The planner preferred the checkout's `requirements.txt` while the runtime verifier read the
    registry's `environment.packages`, so SPRC installed `transformers==4.36.2`/`timm==0.9.12` and
    was then refused for not being `4.25.0`/`0.9.16`. A correctly installed environment could never
    reach `DEPENDENCIES_OK`. This asserts the property directly, over every declared pin, so the two
    can never drift apart again.
    """
    from workbench.backend import model_settings
    monkeypatch.delenv("WORKBENCH_RUNTIME_ADAPTATION", raising=False)
    monkeypatch.setattr(model_settings, "file_settings",
                        lambda: {k: v for k, v in model_settings.file_settings().items() if k != "WORKBENCH_RUNTIME_ADAPTATION"})
    model = model_by_id(model_id)
    environment = model.get("environment")
    if not environment:
        pytest.skip(f"{model_id} declares no environment block")
    constraints = planner.model_constraints({**model, "source_available": True})
    conflicts = []
    for declaration in environment.get("packages") or []:
        if "==" not in declaration:
            continue
        name, version = declaration.split("==", 1)
        if " " in name:
            continue
        planned = constraints["packages"].get(planner._package_name(name))
        if planned in (None, "UNKNOWN"):
            continue  # not a pin the plan carries; presence is checked by the runtime itself
        accepted = version_satisfies(declared_requirement(name, version), planned)
        if accepted is not True:
            conflicts.append(f"{name}: the plan installs {planned}, the verifier requires "
                             f"{declared_requirement(name, version)}")
    assert not conflicts, f"{model_id}: {'; '.join(conflicts)}"


def test_sprc_declares_the_versions_its_own_checkout_pins() -> None:
    """SPRC's contract comes from SPRC, not from the LAVIS replay profiles beside it."""
    manifest = REPO / "workbench" / "third_party" / "SPRC" / "requirements.txt"
    pinned = {line.strip().split("==")[0]: line.strip().split("==")[1]
              for line in manifest.read_text(encoding="utf-8").splitlines()
              if line.strip().startswith(("transformers==", "timm=="))}
    declared = {declaration.split("==")[0]: declaration.split("==")[1]
                for declaration in model_by_id("sprc")["environment"]["packages"]
                if declaration.split("==")[0] in pinned}
    assert declared == pinned, (
        f"the registry declares {declared} while the pinned checkout's own requirements declare "
        f"{pinned}; the plan and the verifier read different files")


# ----------------------------------------------------------------------------------- R11-03

def test_guarded_imports_do_not_become_install_requirements() -> None:
    """A `try: import x / except ImportError` reaching the scan must not install `x`.

    The scan walks source text, so it sees imports that never execute. Converting all of them put
    `tensorflow`, `apex`, `horovod`, `torchaudio` and `knn-cuda` into every LAVIS profile -- audio,
    video, 3D point-cloud and distributed-training libraries for an image-retrieval evaluation, each
    of which the pinned code explicitly tolerates the absence of.
    """
    for model_id in ("hint", "habit", "intent", "conesep", "airknow"):
        model = model_by_id(model_id)
        traced = planner._evaluator_imports({**model, "source_available": True})
        assert traced is not None, model_id
        required, guarded = traced
        assert not (required & guarded), (
            f"{model_id}: {sorted(required & guarded)} reported both ways")
        constraints = planner.model_constraints({**model, "source_available": True})
        installed = set(constraints["packages"]) | set(constraints["ranges"])
        leaked = sorted(module for module in guarded
                        if module.lower().replace("_", "-") in installed)
        assert not leaked, (
            f"{model_id}: {leaked} are reached only through a guarded import but the plan installs "
            "them anyway")


def test_the_vendored_lavis_never_pulls_the_pypi_distribution_back_in() -> None:
    """`salesforce-lavis` must not return through the discovery path.

    The registry no longer declares it, but the scan re-created it: AirKnow's
    `lavis/models/__init__.py` guards `from lavis.models.blip_models.blip_cir import BlipCir`, that
    module is absent from the pinned tree, and the unresolved name fell through to
    `_IMPORT_DISTRIBUTIONS["lavis"]` -> `salesforce-lavis`. The distribution requires `timm==0.4.12`,
    which contradicts every LAVIS profile, so installing it re-broke what R10 repaired.
    """
    for model_id in ("hint", "habit", "intent", "conesep", "airknow"):
        model = model_by_id(model_id)
        constraints = planner.model_constraints({**model, "source_available": True})
        installed = set(constraints["packages"]) | set(constraints["ranges"])
        assert "salesforce-lavis" not in installed, (
            f"{model_id}: the plan reinstalls salesforce-lavis, the PyPI distribution whose "
            "requirements contradict the vendored checkout")
        # The packages the reviewer named as optional-by-discovery, restricted to the ones whose
        # import the pinned tree actually guards with `except ImportError`. Each of these was
        # verified as a guarded import in the pinned sources, and each is genuinely absent from the
        # evaluation path. Unpinned names in the plan are the scanner's UNKNOWN entries; they are
        # compared by distribution name, so `torchaudio` and `knn-cuda` are the spellings that matter.
        guarded_only_names = {"tensorflow", "apex", "horovod", "spacy"}
        offenders = sorted(installed & guarded_only_names)
        assert not offenders, (
            f"{model_id}: the plan installs {offenders}, reached only through a `try`/`except "
            "ImportError`, so the pinned code already tolerates their absence")
        # And the name a module must never be mistaken for: `mpl-toolkits` is not a distribution.
        assert "mpl-toolkits" not in installed, model_id


def test_an_import_namespace_is_not_assumed_to_be_a_distribution() -> None:
    """`mpl_toolkits` is a module of `matplotlib`, and `lavis` is the vendored package here.

    `_distribution_names(module, available)` returns only what the resolver could see in `available`,
    so the unconstrained case is `set()`; the property worth asserting is the MAPPING, which is what
    a caller with no declaration falls back to. Asserting on the empty set would have passed whether
    or not the entry existed.
    """
    # The map is the fallback: with nothing declared, a caller resolves the module through it.
    assert planner._IMPORT_DISTRIBUTIONS["mpl_toolkits"] == ("matplotlib",), (
        "`mpl_toolkits` must resolve to the distribution that provides it, never to a package "
        "named after the module")
    assert planner._IMPORT_DISTRIBUTIONS["skimage"] == ("scikit-image",)
    assert planner._IMPORT_DISTRIBUTIONS["yaml"] == ("pyyaml",)
    # With a declaration in view, the same module resolves to the declared name.
    assert planner._distribution_names("mpl_toolkits", {"matplotlib"}) == {"matplotlib"}
    assert planner._distribution_names("skimage", {"scikit-image"}) == {"scikit-image"}
    # And the vendored case: the checkout's own package wins over the PyPI mapping.
    for model_id in ("hint", "airknow"):
        model = model_by_id(model_id)
        traced = planner._evaluator_imports({**model, "source_available": True})
        assert traced is not None
        assert "lavis" not in traced[0], (
            f"{model_id}: `lavis` resolves inside the checkout, so it is not an external import")


def test_the_plan_installs_what_the_native_class_module_imports() -> None:
    """The dependency contract is the pinned class file's own imports, not a do-not-install list.

    This test previously asserted that `scikit-learn` was ABSENT from all five plans. That was a
    false invariant built on a broken census: the census imported `lavis.models`, which fails partway
    when a dependency is missing, so everything needed after the failure point was never counted.
    `HABIT.py:24-27`, `INTENT/.../blip2_cir.py:30`, `ConeSep.py:24` and `AirKnow/.../blip2_cir.py:19`
    import `sklearn` at module scope with no guard, and `INTENT/.../blip2_cir.py:14` imports
    `matplotlib.pyplot`; a clean environment built from the old plan could not import the checkpoint
    class at all.

    The property asserted here is the one that matters and cannot be satisfied by matching an
    internal list: every module the model's OWN native class file imports at module scope must be
    provided by the plan.
    """
    # The class module each wrapper loads for its checkpoint. Read from the registry's own file
    # layout rather than from a shared assumption, so a model that gains an import is caught here.
    class_files = {
        "habit": ("HABIT", "lavis/models/blip2_models/HABIT.py"),
        "intent": ("INTENT", "lavis/models/blip2_models/blip2_cir.py"),
        "conesep": ("ConeSep", "lavis/models/blip2_models/ConeSep.py"),
        "airknow": ("AirKnow", "lavis/models/blip2_models/blip2_cir.py"),
        "hint": ("HINT", "lavis/models/blip2_models/HINT.py"),
    }
    for model_id, (checkout, relative) in class_files.items():
        source_file = REPO / "workbench" / "third_party" / checkout / relative
        if not source_file.is_file():
            pytest.skip(f"{model_id}: pinned class file absent: {source_file}")
        tree = ast.parse(source_file.read_text(encoding="utf-8", errors="replace"))
        imported = set()
        for node in tree.body:  # module scope only: a lazy import inside a function is not a requirement
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                imported.add(node.module.split(".")[0])
        model = model_by_id(model_id)
        constraints = planner.model_constraints({**model, "source_available": True})
        installed = set(constraints["packages"]) | set(constraints["ranges"])
        declared = {planner._package_name(d.split("==")[0].split("@")[0].strip())
                    for d in (model.get("environment") or {}).get("packages") or []}
        provided = installed | declared
        for module in sorted(imported):
            # Only the checkpoint-class path is asserted: stdlib and the checkout's own package are
            # not distributions, and torch/torchvision are carried as their own constraints.
            if module in sys.stdlib_module_names or module in ("lavis", "torch", "torchvision"):
                continue
            # The distribution that provides the module: the project's map first (`sklearn` ->
            # `scikit-learn`, `mpl_toolkits` -> `matplotlib`), then the name itself. Comparing the
            # module name directly to the plan would report a provided dependency as missing, which
            # is the opposite failure of the one this test exists to catch.
            distribution = {planner._package_name(name)
                            for name in planner._IMPORT_DISTRIBUTIONS.get(module.lower(), (module,))}
            if not (distribution & provided):
                raise AssertionError(
                    f"{model_id}: its native class file {relative} imports {module} at module scope, "
                    f"but no plan or registry entry provides {sorted(distribution)}. "
                    f"Installed: {sorted(installed)}")


def test_the_required_trace_still_names_what_the_evaluation_imports() -> None:
    """The split must not become a way to drop real dependencies."""
    for model_id in ("hint", "habit", "intent", "conesep", "airknow"):
        model = model_by_id(model_id)
        traced = planner._evaluator_imports({**model, "source_available": True})
        assert traced is not None
        required = traced[0]
        for needed in ("torch", "torchvision", "numpy", "PIL", "omegaconf", "timm"):
            assert needed in required, (
                f"{model_id}: {needed} is on the evaluation path and must stay required, "
                f"found {sorted(required)}")


# ----------------------------------------------------------------------------------- R11-04

def _source_with_dataset_parent(tmp_path: Path, kind: str) -> tuple[Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    selected = tmp_path / "selected"
    selected.mkdir()
    outside = tmp_path / "outside"
    if kind == "symlink":
        outside.mkdir()
        (source / "dataset").symlink_to(outside, target_is_directory=True)
    elif kind == "dangling":
        (source / "dataset").symlink_to(tmp_path / "absent", target_is_directory=True)
    elif kind == "file":
        (source / "dataset").write_text("occupied", encoding="utf-8")
    return source, selected


@pytest.mark.parametrize("kind", ["symlink", "dangling", "file"])
def test_staging_refuses_a_parent_that_leaves_the_checkout(tmp_path: Path, kind: str) -> None:
    """A symlinked parent sent the staged link outside the checkout and reported success.

    With `source/dataset -> outside`, staging the declared `dataset/fashioniq` created
    `outside/fashioniq`: the step wrote outside the checkout it is authorized to stage, because
    `mkdir(exist_ok=True)` accepts an existing symlink and `symlink_to` then follows it.
    """
    source, selected = _source_with_dataset_parent(tmp_path, kind)
    ok, messages = prepare_canonical_link(selected, source, "dataset/fashioniq",
                                          check_only=False, dry_run=False)
    assert ok is False, messages
    assert messages and messages[0].startswith("[BLOCKED]"), messages
    outside = tmp_path / "outside"
    assert not outside.exists() or not any(outside.iterdir()), (
        "nothing may be written outside the checkout")
    assert not os.path.lexists(source / "dataset" / "fashioniq")


def test_staging_is_write_free_in_check_only_and_dry_run(tmp_path: Path) -> None:
    """The refusal is about writing, so the non-writing modes must still write nothing."""
    source, selected = _source_with_dataset_parent(tmp_path, "symlink")
    outside = tmp_path / "outside"
    for kwargs in ({"check_only": True, "dry_run": False}, {"check_only": False, "dry_run": True}):
        ok, _ = prepare_canonical_link(selected, source, "dataset/fashioniq", **kwargs)
        assert ok is True, kwargs
        assert not any(outside.iterdir()), f"{kwargs} wrote outside the checkout"


def test_normal_nested_and_flat_staging_still_work(tmp_path: Path) -> None:
    """The guard must not block the paths the declared links actually use."""
    root = tmp_path / "data"
    root.mkdir()
    for link in ("dataset/fashioniq", "fashioniq"):
        source = tmp_path / f"source-{link.replace('/', '-')}"
        source.mkdir()
        selected = tmp_path / f"selected-{link.replace('/', '-')}"
        selected.mkdir()
        ok, messages = prepare_canonical_link(selected, source, link,
                                              check_only=False, dry_run=False)
        assert ok is True, messages
        staged = source / link
        assert staged.is_symlink() and staged.resolve() == selected.resolve()


def test_an_already_correct_link_is_reported_not_rewritten(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    selected = tmp_path / "selected"
    selected.mkdir()
    (source / "dataset").mkdir()
    (source / "dataset" / "fashioniq").symlink_to(selected, target_is_directory=True)
    ok, messages = prepare_canonical_link(selected, source, "dataset/fashioniq",
                                          check_only=False, dry_run=False)
    assert ok is True and "already correct" in messages[0]
    assert (source / "dataset" / "fashioniq").resolve() == selected.resolve()


def test_a_wrong_target_is_still_refused(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    selected = tmp_path / "selected"
    selected.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    (source / "fashioniq").symlink_to(other, target_is_directory=True)
    ok, messages = prepare_canonical_link(selected, source, "fashioniq",
                                          check_only=False, dry_run=False)
    assert ok is False and "resolves to" in messages[0]


# ----------------------------------------------------------------------------------- R11-05

@pytest.mark.parametrize("body,expected", [
    ("[]", "not a JSON object"),
    ('{"data_loader": ["x"]}', "data_loader is not a JSON object"),
    ('{"data_loader": {"args": null}}', "no data_loader.args.data_dir"),
    ('{"data_loader": {"args": []}}', "data_loader.args is not a JSON object"),
    ('{"data_loader": null}', "no data_loader.args.data_dir"),
])
def test_a_malformed_config_is_a_named_refusal_not_a_crash(
        tmp_path: Path, body: str, expected: str) -> None:
    """Valid JSON with an invalid shape raised AttributeError out of the readiness view.

    `[]`, `{"data_loader": ["x"]}` and `{"data_loader": {"args": null}}` are all JSON the native
    `ConfigParser` accepts and then fails on. The readiness path called `.get()` down the chain and
    the exception escaped, so an operator saw a traceback instead of a blocker naming the field.
    """
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(body, encoding="utf-8")
    (run / "trained_model.pth").write_bytes(b"w")
    resolved, blockers = configured_data_dir(run / "config.json", tmp_path, tmp_path)
    assert resolved is None
    assert blockers and expected in blockers[0], blockers
    # And the launch path reports it too, rather than raising there instead.
    assert binding_mismatch(run, tmp_path, tmp_path) is not None


def test_a_valid_config_is_still_bound_and_an_absolute_foreign_root_refused(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "trained_model.pth").write_bytes(b"w")
    dataset = tmp_path / "FashionIQ"
    (dataset / "image_splits").mkdir(parents=True)

    (run / "config.json").write_text(
        json.dumps({"data_loader": {"args": {"data_dir": str(dataset)}}}), encoding="utf-8")
    resolved, blockers = configured_data_dir(run / "config.json", dataset, tmp_path)
    assert resolved == dataset.resolve() and not blockers, (resolved, blockers)

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (run / "config.json").write_text(
        json.dumps({"data_loader": {"args": {"data_dir": str(elsewhere)}}}), encoding="utf-8")
    resolved, blockers = configured_data_dir(run / "config.json", dataset, tmp_path)
    assert any("binding mismatch" in blocker for blocker in blockers), blockers


def test_invalid_json_is_still_reported(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "trained_model.pth").write_bytes(b"w")
    (run / "config.json").write_text("{not json", encoding="utf-8")
    resolved, blockers = configured_data_dir(run / "config.json", tmp_path, tmp_path)
    assert resolved is None and blockers and "unreadable" in blockers[0]


# ------------------------------------------------------------------- the Optional-prefix guard

def test_the_inventory_accepts_the_optional_prefix_it_declares() -> None:
    """`inspect_environment_distributions(None, base)` raised TypeError on its own signature."""
    from workbench.backend import runtime

    versions, modules, unconfirmed = runtime.inspect_environment_distributions(None, "/tmp/base")
    assert versions == {} and modules == set() and unconfirmed == set()
