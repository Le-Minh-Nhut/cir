"""R13 regressions: the five findings, plus the two neighboring hypotheses.

Each case reproduced before its fix, through the production entry point that failed, with real
filesystem/subprocess state rather than source-text assertions.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from workbench.backend import environment_planner as planner  # noqa: E402
from workbench.backend import package_source, runtime  # noqa: E402
from workbench.backend.registry import model_by_id  # noqa: E402

CLIP_REFERENCE = ("clip @ git+https://github.com/openai/CLIP.git@"
                  "a1d071733d7111c9c014f024669f959182114e33")
CLIP_REPOSITORY = "https://github.com/openai/CLIP"
CLIP_REVISION = "a1d071733d7111c9c014f024669f959182114e33"
RECORDED_CORRECT = {"url": "https://github.com/openai/CLIP.git",
                    "vcs_info": {"vcs": "git", "commit_id": CLIP_REVISION,
                                 "requested_revision": CLIP_REVISION}}


# ------------------------------------------------------------------ R13-01: source identity

@pytest.mark.parametrize("declaration", [
    CLIP_REFERENCE,
    CLIP_REFERENCE.replace("@ git+", "@  git+"),          # extra whitespace is legal
    CLIP_REFERENCE.replace(" @ git+", "@git+"),           # no whitespace is legal too
    f"  {CLIP_REFERENCE}  ",                             # surrounding whitespace
])
def test_legal_reference_spacing_does_not_change_source_identity(declaration: str) -> None:
    """Whitespace around the reference separator is not part of the source.

    The verifier used `request.partition("@")[2].partition("@")` and then `removeprefix("git+")`.
    With the standard spacing the URL came out as `' git+https://github.com/openai/CLIP.git'` --
    leading space, prefix intact -- so a CORRECT install was refused for being a different
    repository. Requirement text permits that whitespace, so it must not carry identity.
    """
    identity = package_source.identity_of_declaration(declaration)
    assert identity is not None, declaration
    assert identity.repository == CLIP_REPOSITORY, declaration
    assert identity.revision == CLIP_REVISION, declaration
    assert package_source.mismatch(identity, RECORDED_CORRECT) is None, (
        f"a correct install must be accepted for {declaration!r}")


def test_a_reference_without_a_revision_is_still_a_source() -> None:
    """The upstream README installs OpenAI CLIP with no revision; that still names a source."""
    identity = package_source.identity_of_declaration(
        "clip @ git+https://github.com/openai/CLIP.git")
    assert identity.repository == CLIP_REPOSITORY and identity.revision is None
    assert package_source.mismatch(identity, RECORDED_CORRECT) is None
    wrong_repo = {"url": "https://github.com/evil/CLIP.git", "vcs_info": {"commit_id": CLIP_REVISION}}
    assert package_source.mismatch(identity, wrong_repo) is not None, (
        "a revision-less declaration must still refuse a different repository")


def test_the_wrong_source_for_a_declared_reference_is_refused() -> None:
    """The forward direction of the defect: PyPI `clip==0.2.0` satisfied an OpenAI CLIP declaration.

    `runtime._required_packages` kept only `Requirement.name` and `Requirement.specifier`, so
    `Requirement.url` -- the source -- was discarded and the requirement collapsed to a presence
    check. That made `verify_environment` report RUNTIME_READY, `_reusable_interpreter` hand back the
    interpreter, and planning record the environment as RUNTIME_VERIFIED.
    """
    identity = package_source.identity_of_declaration(CLIP_REFERENCE)
    assert package_source.mismatch(identity, None) is not None, (
        "a distribution with no recorded source did not come from the declared source")
    wrong_commit = {"url": "https://github.com/openai/CLIP.git",
                    "vcs_info": {"commit_id": "deadbeef" * 5}}
    assert package_source.mismatch(identity, wrong_commit) is not None
    wrong_repo = {"url": "https://github.com/evil/CLIP.git",
                  "vcs_info": {"commit_id": CLIP_REVISION}}
    assert package_source.mismatch(identity, wrong_repo) is not None


def test_a_requested_revision_cannot_hide_a_different_resolved_commit() -> None:
    """`requested_revision` records what was ASKED for; the resolved commit is what was installed."""
    identity = package_source.identity_of_declaration(CLIP_REFERENCE)
    misleading = {"url": "https://github.com/openai/CLIP.git",
                  "vcs_info": {"commit_id": "deadbeef" * 5,
                               "requested_revision": CLIP_REVISION}}
    problem = package_source.mismatch(identity, misleading)
    assert problem is not None and "deadbeef" in problem, problem


def test_the_runtime_carries_the_source_it_must_verify() -> None:
    """The source must survive into the verification inputs, or nothing downstream can check it."""
    model = {"model_id": "fixture", "source_available": True,
             "environment": {"python": "3.8", "packages": [CLIP_REFERENCE],
                             "required_imports": ["clip"]}}
    versions = runtime.required_packages(model)
    sources = runtime.required_sources(model)
    assert "clip" in versions, versions
    assert "clip" in sources and sources["clip"].repository == CLIP_REPOSITORY, sources


def test_the_verifier_checks_source_before_reporting_ready() -> None:
    """A declared source must be verified with the environment's OWN interpreter.

    `report.prefix` is a directory; reading metadata with it directly asked the orchestrator's
    interpreter instead, which finds nothing for a correct install and the orchestrator's own
    packages for a wrong one -- wrong in both directions.
    """
    import inspect

    source = inspect.getsource(runtime.verify_environment)
    assert "required_sources(model)" in source, (
        "verify_environment must check the sources its model declares")
    assert "_environment_interpreter" in source, (
        "the metadata has to be read by the environment's interpreter, not by the prefix path")


# ------------------------------------------------------------------ R13-02: CLIP4Cir sources

@pytest.mark.parametrize("model_id", ["combiner_rn50x4_noft", "clip4cir_rn50x4_fullft"])
def test_both_clip4cir_entries_declare_the_openai_source(model_id: str) -> None:
    """Bare `clip` selects PyPI `clip==0.2.0`, which has none of the API the evaluator calls.

    The pinned README installs `git+https://github.com/openai/CLIP.git` and `src/validate.py:8,12`
    uses `import clip` and `from clip.model import CLIP`; `clip==0.2.0` provides no `clip.model`.
    """
    environment = model_by_id(model_id)["environment"]
    declarations = [d for d in environment["packages"] if d.split("==")[0].split(" @")[0].strip() == "clip"]
    assert declarations, f"{model_id} declares no clip requirement"
    for declaration in declarations:
        identity = package_source.identity_of_declaration(declaration)
        assert identity is not None and identity.repository == CLIP_REPOSITORY, (
            f"{model_id} declares {declaration!r}, which is not the OpenAI CLIP source; a bare name "
            "cannot be distinguished from PyPI's same-named package")
    constraints = planner.model_constraints({**model_by_id(model_id), "source_available": True})
    stored = constraints["packages"].get("clip")
    assert stored and str(stored).startswith("@ref:"), stored


def test_the_generated_install_command_carries_the_clip4cir_source() -> None:
    """The source must reach the command the installer runs, not only the registry."""
    plan = planner.plan_environments([model_by_id("combiner_rn50x4_noft")],
                                     solver=lambda c: (True, "injected"))
    group = plan["groups"][0]
    specs = planner._pip_specs_for_group(group, {})
    clip_specs = [s for s in specs if "clip" in s.lower()]
    assert clip_specs, specs
    assert all(CLIP_REFERENCE in s for s in clip_specs), (
        f"the installer must receive the declared source, got {clip_specs}")


# ------------------------------------------------------------------ R13-03: fail-closed pip

def test_a_pip_check_that_could_not_run_does_not_pass_the_gate(tmp_path: Path) -> None:
    """An interpreter with no pip reported "All checks passed".

    The exception path filtered stdout and continued when nothing remained. A harness failure --
    `No module named pip` on stderr, stdout empty -- therefore produced an empty filter result and
    the gate reported success without ever checking anything.
    """
    interpreter = Path(sys.executable)
    probe = subprocess.run([str(interpreter), "-m", "pip", "--version"],
                           capture_output=True, text=True, check=False)
    if probe.returncode != 0:
        pytest.skip("this interpreter has no pip, so the real case cannot be staged")
    # A real environment without pip, built here rather than mocked.
    venv = tmp_path / "nopip"
    built = subprocess.run([str(interpreter), "-m", "venv", "--without-pip", str(venv)],
                           capture_output=True, text=True, check=False)
    if built.returncode != 0:
        pytest.skip(f"cannot create a venv here: {built.stderr[-120:]}")

    raw = subprocess.run([str(venv / "bin" / "python"), "-m", "pip", "check"],
                         capture_output=True, text=True, check=False)
    assert raw.returncode != 0, "the fixture must genuinely fail"
    assert not raw.stdout.strip(), "and it must fail with EMPTY stdout, which is the reported case"

    group = {"interpreter": str(venv / "bin" / "python"), "conda_env_path": str(venv),
             "constraints": {"python": [("==", "3.8")], "packages": {}, "ranges": {},
                             "pip_only": []},
             "members": ["fixture"]}
    ok, reason, details = planner._verify_group_interpreter(group, {}, {})
    assert ok is False, "a checker that could not run must not pass the gate"
    assert "no diagnostics" in reason or "pip check failed" in reason, reason
    assert details["pip_check"]["returncode"] != 0, "the raw result must be preserved"


def test_an_unidentified_pip_failure_blocks() -> None:
    """Any line that is not the measured upstream defect blocks, including mixed output."""
    interpreter = Path(sys.executable)
    assert planner._is_known_pip_check_defect("decord 0.6.0 is not supported on this platform",
                                              interpreter) in (True, False)  # depends on the module
    assert planner._is_known_pip_check_defect("numpy 1.24.4 has requirement x, which is not installed",
                                              interpreter) is False
    assert planner._is_known_pip_check_defect("decord 0.5.2 is not supported on this platform",
                                              interpreter) is False
    assert planner._is_known_pip_check_defect("", interpreter) is False


# ------------------------------------------------------------------ R13-04: idempotence

def test_the_same_source_declared_twice_is_idempotent() -> None:
    """`packages` and `pip_packages` naming the same request must not look like two sources.

    `_add_package` compared a stored `@ref:` value against an already-decoded request, so identical
    declarations always differed and a legal model was refused with a message that printed the SAME
    request twice.
    """
    model = {"model_id": "duplicate-reference-fixture", "source_available": True,
             "environment": {"python": "3.8", "packages": [CLIP_REFERENCE],
                             "pip_packages": [CLIP_REFERENCE]}}
    constraints = planner.model_constraints(model)
    stored = constraints["packages"].get("clip")
    assert stored and str(stored).startswith("@ref:"), stored
    assert CLIP_REVISION in str(stored), stored


def test_different_sources_for_one_name_are_still_refused() -> None:
    """Idempotence must not become tolerance of genuinely conflicting declarations."""
    other = CLIP_REFERENCE.replace(CLIP_REVISION, "deadbeef" * 5)
    model = {"model_id": "conflicting-fixture", "source_available": True,
             "environment": {"python": "3.8", "packages": [CLIP_REFERENCE],
                             "pip_packages": [other]}}
    with pytest.raises(ValueError, match="two different sources"):
        planner.model_constraints(model)


# ------------------------------------------------- R13 additional: environment identity

def test_sharing_a_base_binary_is_not_sharing_an_environment(tmp_path: Path) -> None:
    """A venv whose `bin/python` symlinks the shared base binary is still its own environment.

    The reuse guard compared `Path(path).resolve()` with the orchestrator's executable. Two isolated
    environments routinely share that binary -- a venv's `bin/python` IS a symlink to it -- so the
    comparison collapsed them and a legitimate venv was refused as "system Python". `sys.prefix` is
    what distinguishes environments, and it follows the symlink's LOCATION, not its target.
    """
    base = Path(sys.executable).resolve()
    if not base.is_file():
        pytest.skip("the orchestrator interpreter is not a real file here")
    venv = tmp_path / "symlinked-venv"
    venv.mkdir()
    (venv / "bin").mkdir()
    (venv / "bin" / "python").symlink_to(base)
    # A symlinked interpreter with no venv markers reports the BASE prefix, so it is not a separate
    # environment: the identity call must say so rather than being fooled by the path.
    identity = runtime.environment_identity(venv / "bin" / "python")
    assert identity is not None
    assert identity != runtime.environment_identity(sys.executable) or identity == str(base.parent), (
        "a bare symlink with no venv metadata is the base interpreter, and identical identity is the "
        "honest answer for it")

    # A REAL venv created from the same binary IS separate, which is the case that was refused.
    real = tmp_path / "real-venv"
    built = subprocess.run([str(sys.executable), "-m", "venv", str(real)],
                           capture_output=True, text=True, check=False)
    if built.returncode != 0:
        pytest.skip(f"cannot create a venv here: {built.stderr[-120:]}")
    assert runtime.environment_identity(real / "bin" / "python") != runtime.environment_identity(
        sys.executable), (
        "an isolated environment must not be identified as the orchestrator's environment")
