"""Runtime adaptation (Blackwell sm_120) contract tests.

The historical registry pins stay the default contract; an operator-selected adaptation must be
declared in the registry, must verify against the adapted constraints, must be provenance-marked,
and must never silently widen or fake a reproduction claim.
"""
from __future__ import annotations

import pytest

from workbench.backend import runtime


@pytest.fixture()
def limn_model():
    from workbench.backend.registry import load_registry

    return next(m for m in load_registry()["models"] if m["model_id"] == "limn")


def test_no_adaptation_by_default(limn_model, monkeypatch):
    monkeypatch.delenv(runtime.ADAPTATION_ENV, raising=False)
    assert runtime.runtime_adaptation(limn_model) is None
    required = runtime._required_packages(limn_model)
    assert required["torch"] == "==1.12.1"


def test_selected_adaptation_returns_declared_block(limn_model, monkeypatch):
    monkeypatch.setenv(runtime.ADAPTATION_ENV, "sm_120")
    adapted = runtime.runtime_adaptation(limn_model)
    assert adapted["target_compute_capability"] == "sm_120"
    adapted_packages = runtime._adapted_packages(limn_model, adapted)
    assert adapted_packages["torch"] == ">=2.7,<3"
    assert adapted_packages["open_clip_torch"] == ">=2.24,<2.27"


def test_undeclared_adaptation_is_refused(limn_model, monkeypatch):
    monkeypatch.setenv(runtime.ADAPTATION_ENV, "sm_61")
    from workbench.backend.errors import WorkbenchError

    with pytest.raises(WorkbenchError):
        runtime.runtime_adaptation(limn_model)


def test_adaptation_changes_the_contract_digest(limn_model, monkeypatch):
    monkeypatch.delenv(runtime.ADAPTATION_ENV, raising=False)
    historical = runtime._environment_contract_digest(limn_model)
    monkeypatch.setenv(runtime.ADAPTATION_ENV, "sm_120")
    adapted = runtime._environment_contract_digest(limn_model)
    assert historical != adapted


def test_adapted_constraints_satisfy_modern_torch(limn_model, monkeypatch):
    monkeypatch.setenv(runtime.ADAPTATION_ENV, "sm_120")
    adapted = runtime.runtime_adaptation(limn_model)
    required = runtime._adapted_packages(limn_model, adapted)
    assert runtime.version_satisfies(required["torch"], "2.7.1+cu128") is True
    assert runtime.version_satisfies(required["torch"], "2.6.0+cu124") is False
    assert runtime.version_satisfies(required["torch"], "3.0.0") is False


def test_model_without_adapted_block_refuses_selection(monkeypatch):
    from workbench.backend.errors import WorkbenchError

    monkeypatch.setenv(runtime.ADAPTATION_ENV, "sm_120")
    with pytest.raises(WorkbenchError):
        runtime.runtime_adaptation({"model_id": "nope", "environment": {}})


def test_limn_validate_environment_accepts_both_contracts(monkeypatch):
    from workbench.replay import limn

    monkeypatch.delenv(runtime.ADAPTATION_ENV, raising=False)
    historical = limn.validate_environment()
    assert any("torch" in blocker for blocker in historical)  # workbench env is not the pin

    monkeypatch.setenv(runtime.ADAPTATION_ENV, "sm_120")
    adapted = limn.validate_environment()
    # The adapted contract must not report the historical mismatches; open_clip_torch >=2.24
    # is genuinely absent from this environment and remains the only honest blocker.
    assert adapted == ["LIMN environment package missing: open_clip_torch >=2.24,<2.27"]
