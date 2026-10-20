"""R12 regressions: the three defects the deep recheck found, asserted at the boundary that failed.

Each case reproduced before its fix. The properties here are the ones that cannot be satisfied by
matching an internal table: the plan must provide what the NATIVE CLASS FILE imports, the installer
must receive the SOURCE the registry declared, and the documented runbook must name entrypoints that
exist.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from workbench.backend import environment_planner as planner  # noqa: E402
from workbench.backend.registry import model_by_id  # noqa: E402

CLIP_REFERENCE = ("clip @ git+https://github.com/openai/CLIP.git@"
                  "a1d071733d7111c9c014f024669f959182114e33")

#: The module each LAVIS wrapper loads to unpickle its checkpoint, with the checkout that holds it.
CLASS_MODULES = {
    "habit": ("HABIT", "lavis/models/blip2_models/HABIT.py"),
    "intent": ("INTENT", "lavis/models/blip2_models/blip2_cir.py"),
    "conesep": ("ConeSep", "lavis/models/blip2_models/ConeSep.py"),
    "airknow": ("AirKnow", "lavis/models/blip2_models/blip2_cir.py"),
    "hint": ("HINT", "lavis/models/blip2_models/HINT.py"),
}


def _module_scope_imports(path: Path) -> set[str]:
    """Top-level-module names the file imports at MODULE scope.

    Module scope is the requirement: an import inside a function is reached only if that function is
    called, while a module-scope import decides whether the class can be unpickled at all.
    """
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            imported.add(node.module.split(".")[0])
    return imported


# ----------------------------------------------------------------------------------- R12-01

@pytest.mark.parametrize("model_id", sorted(CLASS_MODULES))
def test_the_plan_provides_every_module_the_native_class_imports(model_id: str) -> None:
    """The dependency contract is the pinned class file, not a do-not-install list.

    An earlier version asserted `scikit-learn` was ABSENT from all five LAVIS plans. That invariant
    was false and had been derived from a broken census: the census called
    `importlib.import_module("lavis.models")`, which fails partway when a dependency is missing, so
    everything needed after the failure point was never counted. `HABIT.py:24-27`,
    `INTENT/.../blip2_cir.py:30`, `ConeSep.py:24` and `AirKnow/.../blip2_cir.py:19` import `sklearn`
    at module scope with no guard, and `INTENT/.../blip2_cir.py:14` imports `matplotlib.pyplot`. A
    clean environment built from that plan could not import the checkpoint class at all.
    """
    checkout, relative = CLASS_MODULES[model_id]
    source_file = REPO / "workbench" / "third_party" / checkout / relative
    if not source_file.is_file():
        pytest.skip(f"{model_id}: pinned class file absent: {source_file}")

    model = model_by_id(model_id)
    constraints = planner.model_constraints({**model, "source_available": True})
    provided = set(constraints["packages"]) | set(constraints["ranges"]) | {
        planner._package_name(d.split("==")[0].split("@")[0].strip())
        for d in (model.get("environment") or {}).get("packages") or []}

    missing = []
    for module in sorted(_module_scope_imports(source_file)):
        if module in sys.stdlib_module_names or module in ("lavis", "torch", "torchvision"):
            continue
        # The distribution that provides the module: the project's own map first (`sklearn` ->
        # `scikit-learn`, `mpl_toolkits` -> `matplotlib`), then the name itself.
        distributions = {planner._package_name(name)
                         for name in planner._IMPORT_DISTRIBUTIONS.get(module.lower(), (module,))}
        if not (distributions & provided):
            missing.append(f"{module} (-> {sorted(distributions)})")
    assert not missing, (
        f"{model_id}: its native class {relative} imports {missing} at module scope, but nothing in "
        f"the plan or the registry provides them")


def test_the_native_class_files_that_need_scikit_learn_are_named() -> None:
    """Only the models whose OWN class file imports it: HINT does not, and must not be assumed to."""
    expected = {"habit": True, "intent": True, "conesep": True, "airknow": True, "hint": False}
    for model_id, needs_sklearn in expected.items():
        checkout, relative = CLASS_MODULES[model_id]
        source_file = REPO / "workbench" / "third_party" / checkout / relative
        if not source_file.is_file():
            pytest.skip(f"{model_id}: pinned class file absent")
        imports = _module_scope_imports(source_file)
        assert ("sklearn" in imports) is needs_sklearn, (
            f"{model_id}: {'imports' if 'sklearn' in imports else 'does not import'} sklearn at "
            f"module scope in {relative}; the registry declaration must follow the source, and one "
            "model's census is not another's contract")


def test_guarded_and_unreachable_packages_stay_out_of_the_plan() -> None:
    """The correction must not become a licence to install everything the text mentions.

    These are reached only through a `try`/`except ImportError` or through audio/point-cloud code the
    FashionIQ evaluation never loads. The set is decided by the pinned sources, not by a blacklist
    that overrides them: what changed is that `sklearn`/`matplotlib` left it because they are really
    imported.
    """
    # `diffusers` and `easydict` are NOT here: they are reached through `lavis.models` once its
    # dependencies exist, which was verified by running the import, so they are requirements.
    still_excluded = {"torchaudio", "knn-cuda", "termcolor", "tensorflow", "apex", "horovod",
                      "spacy", "moviepy", "nltk", "h5py"}
    for model_id in CLASS_MODULES:
        constraints = planner.model_constraints({**model_by_id(model_id), "source_available": True})
        installed = set(constraints["packages"]) | set(constraints["ranges"])
        offenders = sorted(installed & still_excluded)
        assert not offenders, f"{model_id}: the plan installs {offenders}, which the run never loads"


# ----------------------------------------------------------------------------------- R12-02

def test_a_direct_reference_survives_parsing() -> None:
    """`name @ url@rev` is a SOURCE. Collapsing it to UNKNOWN substituted a PyPI stranger."""
    name, spec = planner._parse_requirement(CLIP_REFERENCE)
    assert name == "clip"
    assert planner._is_reference(spec), "the reference must not be reduced to a version placeholder"
    assert planner._reference_of(spec) == CLIP_REFERENCE, (
        "the URL and revision must survive byte-for-byte: the revision is what pins the API the "
        "evaluator was validated against")


def test_a_direct_reference_reaches_the_install_command_and_the_group_merge() -> None:
    """End to end through the machinery the installer actually consumes."""
    name, spec = planner._parse_requirement(CLIP_REFERENCE)
    spec_set = {"python": "3.9", "torch": "2.0.1", "torchvision": "0.15.2", "cuda": "UNKNOWN",
                "packages": {name: spec}, "ranges": {}, "pip_only": [name], "evidence": []}
    combined, error = planner._constraints([spec_set])
    assert error is None, error
    assert planner._is_reference(combined["packages"][name]), combined["packages"]
    specs = planner._pip_specs_for_group({"constraints": combined, "members": ["sprc"]}, {})
    assert CLIP_REFERENCE in specs, (
        f"the installer must be handed the declared source, not a bare name; got {specs}")


def test_two_different_sources_for_one_name_cannot_share_an_environment() -> None:
    """Grouping must refuse, not pick whichever it saw last."""
    name, spec = planner._parse_requirement(CLIP_REFERENCE)
    other = spec.replace("a1d071", "deadbee")
    first = {"python": "3.9", "torch": "2.0.1", "torchvision": "0.15.2", "cuda": "UNKNOWN",
             "packages": {name: spec}, "ranges": {}, "pip_only": [name], "evidence": []}
    second = {**first, "packages": {name: other}}
    combined, error = planner._constraints([first, second])
    assert combined is None and error and "different sources" in error, (combined, error)

    combined, error = planner._constraints([first, first])
    assert error is None and planner._reference_of(combined["packages"][name]) == CLIP_REFERENCE


def test_a_reference_and_a_version_conflict_in_either_order() -> None:
    """The conflict must be reported whichever one arrives first.

    An independent audit of this change found both halves missing: with the reference first, the
    version branch called `.extend()` on a string and raised `AttributeError`; with the version
    first, a list was not recognised as a version constraint and the reference silently overwrote
    it, so two models that cannot share an installation were grouped together. A conflict that is
    only detected in one order is not detected.
    """
    name, spec = planner._parse_requirement(CLIP_REFERENCE)
    base = {"python": "3.9", "torch": "UNKNOWN", "torchvision": "UNKNOWN", "cuda": "UNKNOWN",
            "ranges": {}, "pip_only": [name], "evidence": []}
    reference_spec = {**base, "packages": {name: spec}}
    version_spec = {**base, "packages": {name: "==0.2.0"}}

    for order, pair in (("reference first", [reference_spec, version_spec]),
                        ("version first", [version_spec, reference_spec])):
        try:
            combined, error = planner._constraints(pair)
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"{order}: raised {type(exc).__name__} instead of refusing: {exc}")
        assert combined is None and error, f"{order}: the conflict was accepted: {combined}"
        assert "cannot describe the same installation" in error, f"{order}: {error}"


def test_the_conda_solve_builder_skips_a_source_requirement() -> None:
    """`_solver_solve` must not concatenate a reference with version terms.

    `_conda_constraints` removes references before the solver sees them, so this only bites a caller
    that hands the raw constraints straight through -- which is exactly the kind of latent break the
    audit looked for and found. The guard belongs in the builder, not only in its usual caller.
    """
    name, spec = planner._parse_requirement(
        "custom-pkg @ git+https://example.invalid/repo.git@1.0.0")
    constraints = {"python": [("==", "3.9")], "packages": {name: spec, "pyyaml": [("==", "6.0")]},
                   "ranges": {}}
    try:
        # A binary that does not exist: the point is that it must not raise while BUILDING the spec
        # list, before any process is started.
        planner._solver_solve("definitely-not-a-real-solver", constraints)
    except TypeError as exc:
        raise AssertionError(f"the solve builder mishandled the reference: {exc}")
    except Exception:  # noqa: BLE001
        pass  # a missing solver binary is expected and irrelevant here


def test_the_pin_scan_does_not_parse_a_reference_as_a_version() -> None:
    """`_terms` would read the URL as a version, so a source is diverted before it is scanned."""
    model = {"model_id": "fixture", "source_available": True,
             "environment": {"python": "3.9",
                             "packages": [CLIP_REFERENCE, "timm==0.9.16"]}}
    constraints = planner.model_constraints(model)
    assert planner._is_reference(constraints["packages"].get("clip")), constraints["packages"]
    assert "clip" in constraints["pip_only"], constraints["pip_only"]
    assert not planner._is_reference(constraints["packages"].get("timm")), constraints["packages"]


def test_source_identity_requires_the_repository_and_the_revision() -> None:
    """Asserted as a decision, not as the presence of a message in the source.

    The previous version asserted that the string "the source repository differs" appeared in the
    verifier's source. That is a source-text assertion: it cannot fail when the comparison is wrong,
    and it fails when the comparison is rewritten even though the behaviour improved.
    """
    from workbench.backend import package_source

    identity = package_source.identity_of_declaration(
        "clip @ git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33")
    assert package_source.mismatch(
        identity, {"url": "https://github.com/openai/CLIP.git",
                   "vcs_info": {"commit_id": "a1d071733d7111c9c014f024669f959182114e33"}}) is None
    assert package_source.mismatch(
        identity, {"url": "https://github.com/other/CLIP.git",
                   "vcs_info": {"commit_id": "a1d071733d7111c9c014f024669f959182114e33"}}) is not None


# ----------------------------------------------------------------------------------- R12-03

def test_the_pc_runbook_commands_are_checkable_and_valid() -> None:
    """Syntax and entrypoint checks need no GPU, so the spec is audited rather than exempted."""
    from workbench.tests.test_handoff_commands import _spec_commands

    commands = _spec_commands()
    assert commands, "the PC spec must contain runnable commands"
    problems = []
    for command in commands:
        for target in __import__("re").findall(r"python[0-9.]* -m (workbench[A-Za-z0-9_.]*)", command):
            path = REPO / Path(*target.split("."))
            if not (path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()):
                problems.append(f"-m {target} does not exist")
        for script in __import__("re").findall(r"(workbench/[A-Za-z0-9_./-]+\.py)", command):
            if not (REPO / script).is_file():
                problems.append(f"{script} does not exist")
    assert not problems, f"the PC runbook names things that are not there: {sorted(set(problems))}"


def test_the_documented_wrapper_flags_match_the_real_parser() -> None:
    """The example must pass the flags the wrapper actually defines.

    `hint.py` requires `--checkpoint-root`, `--dataset-root` and `--source-root`; a documented
    invocation missing one fails at argparse with exit code 2, which is exactly what an operator
    following the runbook would hit.
    """
    spec = (REPO / "workbench" / "docs" / "MODEL_STATUS_SPEC.md").read_text(encoding="utf-8")
    for flag in ("--checkpoint-root", "--dataset-root", "--source-root"):
        assert flag in spec, f"the runbook must show {flag}: it is required by the wrapper"
    # The wrapper's own declaration is the authority for the flag set.
    wrapper = REPO / "workbench" / "replay" / "hint.py"
    text = wrapper.read_text(encoding="utf-8")
    for flag in ("--checkpoint-root", "--dataset-root", "--source-root"):
        assert f'"{flag}"' in text or f"'{flag}'" in text, (
            f"the runbook documents {flag} but the wrapper does not define it")
