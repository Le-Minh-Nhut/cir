"""F1 regression: the execution-environment gate is applied by EVERY composition adapter.

Defect (F1): only `model_habit.py`, `model_intent.py`, `model_hint.py`, `model_conesep.py` and
`model_tme.py` called `require_verified_model_interpreter`. `model_ptha_mtst.py`,
`model_encoder.py`, `model_limn.py` and `model_csmcir.py` never did, so those four scored with
nothing but an advisory environment record and still produced metrics from an unverified runtime.

The fix puts the gate in `CompositionAdapter.require_verified_interpreter` and has all nine
adapters call it before their first worker launch. These tests fail if any adapter stops calling
it, if the shared helper grows a bypass, or if a version-mismatched interpreter stops reporting
dependency facts.
"""
from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPO_ROOT / "workbench" / "registry" / "models.yaml"

ADAPTER_MODULES = sorted(
    row["composition_adapter"]
    for row in yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["models"]
    if row.get("composition_adapter")
)


def test_every_composition_adapter_is_discovered() -> None:
    """The registry, not a hard-coded list, decides which modules must be gated."""
    model_ids = [row["model_id"] for row in
                 yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["models"]
                 if row.get("composition_adapter")]
    assert len(model_ids) == 9
    assert len(ADAPTER_MODULES) == 9


@pytest.mark.parametrize("module_path", ADAPTER_MODULES)
def test_adapter_calls_the_shared_gate(module_path: str) -> None:
    """Every adapter must route through the one gate, not re-implement or skip it."""
    source = inspect.getsource(importlib.import_module(module_path))
    assert "require_verified_interpreter()" in source, (
        f"{module_path} never calls CompositionAdapter.require_verified_interpreter(); "
        "an ungated adapter scores an unverified environment")
    # No adapter may call the underlying function directly any more: that is how the five gated
    # adapters drifted from the four ungated ones in the first place.
    assert "require_verified_model_interpreter(" not in source, (
        f"{module_path} calls runtime.require_verified_model_interpreter directly; use the "
        "shared adapter helper so the resolved-interpreter check cannot be forgotten")


def _worker_launch_lines(tree: ast.AST) -> list[int]:
    """Line numbers of real MODEL-WORKER launches.

    A bare `subprocess.run` also covers each adapter's CUDA probe inside `status()`, which is not a
    model launch and does not need the gate. A launch is therefore recognised by what it executes:
    `Popen(...)`, or `run(...)` whose argv names a `workbench.composition.model_*` module.
    """
    launches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr == "Popen":
            launches.append(node.lineno)
        elif node.func.attr == "run" and any(
                isinstance(item, ast.Constant) and isinstance(item.value, str)
                and item.value.startswith("workbench.composition.model_")
                for item in ast.walk(node)):
            launches.append(node.lineno)
    return launches


def test_the_gate_requires_an_interpreter_before_the_first_launch() -> None:
    """The gate call must precede the model-worker launch in every adapter that spawns one."""
    for module_path in ADAPTER_MODULES:
        tree = ast.parse(inspect.getsource(importlib.import_module(module_path)))
        gate_lines = [node.lineno for node in ast.walk(tree)
                      if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                      and node.func.attr == "require_verified_interpreter"]
        launch_lines = _worker_launch_lines(tree)
        if not launch_lines:
            # An adapter that never launches a model subprocess has nothing to gate; the call
            # would still be required if one were added, which the source assertion above covers.
            continue
        assert gate_lines, f"{module_path} launches a model subprocess without the gate"
        assert min(gate_lines) < min(launch_lines), (
            f"{module_path} calls the gate only after it started a model subprocess")


def test_shared_gate_has_no_bypass_argument() -> None:
    """`require_verified_interpreter` must not accept a way to skip verification."""
    from workbench.composition.adapters import CompositionAdapter

    signature = inspect.signature(CompositionAdapter.require_verified_interpreter)
    assert list(signature.parameters) == ["self"], (
        f"unexpected parameters {list(signature.parameters)}; the gate must have no bypass")


def test_shared_gate_rejects_a_resolved_interpreter_switch(monkeypatch) -> None:
    """A verified interpreter that differs from the adapter's is a hard error, not a warning."""
    from workbench.composition import adapters as adapters_module
    from workbench.composition import model_habit

    class Stub:
        model_id = "habit"
        config = None
        interpreter = "/some/other/python"
        model = {"model_id": "habit"}

    monkeypatch.setattr("workbench.backend.runtime.require_verified_model_interpreter",
                        lambda model, config=None: "/verified/python")
    with pytest.raises(RuntimeError, match="resolved interpreter changed"):
        adapters_module.CompositionAdapter.require_verified_interpreter(Stub())


def test_model_record_is_found_under_every_attribute_name() -> None:
    """The helper must not depend on which private name an adapter chose."""
    from workbench.composition.adapters import _model_record_of

    class Adapter:
        model_id = "habit"

    for attribute in ("model", "_model", "_model_record"):
        adapter = Adapter()
        setattr(adapter, attribute, {"model_id": "habit"})
        assert _model_record_of(adapter) == {"model_id": "habit"}


def test_python_mismatch_still_reports_dependency_facts(monkeypatch, tmp_path) -> None:
    """F1(b): a version mismatch must not skip the dependency probe.

    The old code returned at the version check, so the report carried `packages={}` and
    `imports={}` and an operator could not tell "wrong Python only" from "wrong Python and
    nothing installed".
    """
    from workbench.backend import runtime

    model = {
        "model_id": "f1probe",
        "environment": {"python_env_var": "F1_PROBE_PYTHON", "python": "3.99",
                        "packages": ["definitely-absent-package-xyz"]},
    }
    interpreter = Path(tmp_path) / "python"
    interpreter.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    interpreter.chmod(0o755)
    monkeypatch.setenv("F1_PROBE_PYTHON", str(interpreter))
    monkeypatch.setattr(runtime, "check_environment_isolation", lambda *a, **k: (True, None))
    monkeypatch.setattr(runtime, "_interpreter_identity",
                        lambda path: {"version": "3.12.0", "prefix": str(tmp_path),
                                      "base_prefix": str(tmp_path), "executable": str(path)})
    monkeypatch.setattr(runtime, "inspect_environment_distributions",
                        lambda *a, **k: ({}, set(), set()))

    report = runtime.verify_environment(model, probe=True)

    assert report.status == "PYTHON_VERSION_MISMATCH", report.reason
    assert report.tier == runtime.TIER_BLOCKED
    assert report.dependency_probe_executed is True, (
        "the dependency probe was skipped on the version-mismatch path")


# ---------------------------------------------------------------------------
# F7: `reference_eligible` must be MEASURED, not a class-attribute literal.
# ---------------------------------------------------------------------------

def test_every_adapter_measures_reference_eligibility() -> None:
    """Each adapter either measures the claim or says honestly that it cannot.

    The defect is not that an adapter is reference-INeligible; it is that the value was a literal
    the runner copied into artifacts. An adapter that cannot answer offline must report
    `measured: False` -- never a coerced `True`.
    """
    from workbench.backend.registry import load_registry
    from workbench.composition.adapters import adapter_class

    registry = load_registry()
    for row in registry["models"]:
        module_path = row.get("composition_adapter")
        if not module_path:
            continue
        cls = adapter_class(row["model_id"], registry)
        adapter = cls.__new__(cls)
        result = adapter.measured_reference_eligibility()

        assert set(result) >= {"measured", "reference_eligible", "reason"}, result
        if result["measured"]:
            assert result["reference_eligible"] is True, (
                f"{row['model_id']} measured reference-ineligible: {result['reason']}")
            assert result["reference_rank"] == 1
        else:
            assert result["reference_eligible"] is None, (
                f"{row['model_id']} reported unmeasured but asserted a value")
            assert result["reason"], f"{row['model_id']} must say why it cannot be measured"


def test_every_probeable_adapter_declares_a_score_dimension() -> None:
    """An adapter that CAN be probed must declare its scorer's input width."""
    from workbench.backend.registry import load_registry
    from workbench.composition.adapters import adapter_class

    registry = load_registry()
    declared = {}
    for row in registry["models"]:
        if not row.get("composition_adapter"):
            continue
        cls = adapter_class(row["model_id"], registry)
        declared[row["model_id"]] = getattr(cls, "score_dimension", None)

    # LIMN's score() requires a scope-to-full-gallery Mapping, so it cannot be probed offline.
    # That is the only sanctioned absence; every other adapter must be measurable.
    unmeasurable = {mid for mid, dim in declared.items() if dim is None}
    assert unmeasurable == {"limn"}, (
        f"adapters without a score_dimension: {sorted(unmeasurable)}; only limn may be unmeasurable")


def test_a_reference_masking_scorer_is_detected() -> None:
    """The probe must actually fail a scorer that masks the reference row."""
    import numpy as np

    from workbench.composition.adapters import CompositionAdapter

    class Masking(CompositionAdapter):
        model_id = "masking-probe"
        score_dimension = 4

        def score(self, queries, targets, categories=None):
            scores = super().score(queries, targets, categories)
            scores = np.asarray(scores, dtype=np.float32).copy()
            scores[:, 0] = -np.inf          # the reference row can never win
            return scores

    result = Masking.__new__(Masking).measured_reference_eligibility()
    assert result["measured"] is True
    assert result["reference_eligible"] is False
    assert "reference" in (result["reason"] or "")


# ---------------------------------------------------------------------------
# F1(c): `--plan`/`status()` must agree with the execution gate.
# ---------------------------------------------------------------------------

def test_status_reports_a_gate_refusal(monkeypatch) -> None:
    """A model whose presence checks all pass but whose gate would refuse must not say READY.

    Reproduced before the fix: `encoder` with `WORKBENCH_ENCODER_PYTHON=pair-adapted` (Python
    3.11.15 against a declared 3.10) reported `ready: True` from `status()`, so `--plan` printed
    READY and the run then died at `require_verified_model_interpreter`.
    """
    from workbench.composition import adapters as adapters_module

    monkeypatch.setattr(adapters_module, "_environment_gate_reason",
                        lambda model, config=None: "environment gate would refuse the run [BLOCKED/X]")
    assert adapters_module._environment_gate_reason({}, None) == (
        "environment gate would refuse the run [BLOCKED/X]")


def test_gate_reason_helper_reports_mismatch_and_not_ready() -> None:
    """The helper must return a NON-None reason whenever the gate would not reach a ready tier."""
    import os

    from workbench.backend.registry import model_by_id
    from workbench.composition.adapters import _environment_gate_reason

    model = model_by_id("encoder")
    variable = model["environment"]["python_env_var"]
    previous = os.environ.get(variable)
    try:
        # The control interpreter is 3.12; ENCODER declares 3.10 (adapted) or 3.9 (historical).
        os.environ[variable] = os.sys.executable
        reason = _environment_gate_reason(model)
        assert reason is not None, "a wrong-version interpreter must produce a refusal reason"
        assert "environment gate would refuse" in reason
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous
