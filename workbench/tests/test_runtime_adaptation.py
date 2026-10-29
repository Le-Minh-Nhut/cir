"""Runtime adaptation (Blackwell sm_120) contract tests.

The historical registry pins stay the default contract; an operator-selected adaptation must be
declared in the registry, must verify against the adapted constraints, must be provenance-marked,
and must never silently widen or fake a reproduction claim.
"""
from __future__ import annotations

import pytest

from workbench.backend import runtime


@pytest.fixture(autouse=True)
def isolate_file_settings(monkeypatch):
    monkeypatch.setattr("workbench.backend.model_settings.file_settings", lambda: {})


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
    assert adapted_packages["torch"] == "==2.7.1"
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


@pytest.mark.parametrize("model_id", [
    "csmcir", "conesep", "ptha_mtst", "habit", "intent", "hint", "encoder", "limn", "tme",
])
def test_adapted_constraints_satisfy_modern_torch(model_id, monkeypatch):
    from workbench.backend.registry import model_by_id
    from workbench.backend.environment_planner import model_constraints

    # TME keeps its already-provisioned 2.8.0 / 0.23.0 cu128 env accepted; the other eight pin the
    # Blackwell baseline exactly.
    expectations = {
        "tme": {
            "torch": (">=2.7.1,<2.9", {"2.7.1+cu128": True, "2.8.0+cu128": True,
                                       "2.9.0+cu128": False, "2.6.0+cu124": False}),
            "torchvision": (">=0.22.1,<0.24", {"0.22.1+cu128": True, "0.23.0+cu128": True,
                                               "0.24.0+cu128": False, "0.21.0+cu128": False}),
        },
    }.get(model_id, {
        "torch": ("==2.7.1", {"2.7.1+cu128": True, "2.6.0+cu124": False,
                              "2.9.0+cu128": False, "2.8.0+cu128": False}),
        "torchvision": ("==0.22.1", {"0.22.1+cu128": True}),
    })

    model = model_by_id(model_id)
    monkeypatch.setenv(runtime.ADAPTATION_ENV, "sm_120")
    adapted = runtime.runtime_adaptation(model)
    required = runtime._adapted_packages(model, adapted)
    for package, (constraint, probes) in expectations.items():
        assert required[package] == constraint
        for version, expected in probes.items():
            assert runtime.version_satisfies(required[package], version) is expected
    constraints = model_constraints(model)
    for package, (constraint, _) in expectations.items():
        assert constraints[package] == constraint
    assert constraints["pip_index_url"] == "https://download.pytorch.org/whl/cu128"
    assert adapted["cuda"] == "12.8"

def test_model_without_adapted_block_refuses_selection(monkeypatch):
    from workbench.backend.errors import WorkbenchError

    monkeypatch.setenv(runtime.ADAPTATION_ENV, "sm_120")
    with pytest.raises(WorkbenchError):
        runtime.runtime_adaptation({"model_id": "nope", "environment": {}})


def test_limn_validate_environment_accepts_both_contracts(monkeypatch):
    from workbench.replay import limn

    versions = {"torch": "2.7.1+cpu", "torchvision": "0.22.1+cpu"}

    def package_version(package):
        if package not in versions:
            raise limn.importlib.metadata.PackageNotFoundError(package)
        return versions[package]

    monkeypatch.setattr(limn.importlib.metadata, "version", package_version)
    monkeypatch.delenv(runtime.ADAPTATION_ENV, raising=False)
    historical = limn.validate_environment()
    assert historical == [
        "LIMN environment version mismatch: expected torch ==1.12.1, found 2.7.1+cpu",
        "LIMN environment version mismatch: expected torchvision ==0.13.1, found 0.22.1+cpu",
        "LIMN environment package missing: open_clip_torch ==2.20.0",
    ]

    monkeypatch.setenv(runtime.ADAPTATION_ENV, "sm_120")
    adapted = limn.validate_environment()
    assert adapted == [
        "LIMN environment package missing: open_clip_torch >=2.24,<2.27",
    ]

@pytest.mark.parametrize("model_id", ["conesep", "habit", "hint"])
def test_adapted_runtime_for_ilearn_family(model_id, monkeypatch):
    from workbench.backend.registry import model_by_id
    from workbench.backend.environment_planner import model_constraints
    from workbench.backend.errors import WorkbenchError

    m = model_by_id(model_id)
    # 1. Historical contract when unset
    monkeypatch.delenv(runtime.ADAPTATION_ENV, raising=False)
    assert runtime.runtime_adaptation(m) is None
    historical_env = runtime.resolved_runtime_environment(m)
    assert historical_env["python"] == "3.8.10"
    assert historical_env["pytorch"] == "2.1.0"

    # 2. Adapted contract when sm_120 is selected
    monkeypatch.setenv(runtime.ADAPTATION_ENV, "sm_120")
    adapted = runtime.runtime_adaptation(m)
    assert adapted["target_compute_capability"] == "sm_120"
    assert adapted["python"] == "3.9"
    assert adapted["pytorch"] == "==2.7.1"
    assert adapted["torchvision"] == "==0.22.1"
    assert adapted["cuda"] == "12.8"
    assert adapted["packages"] == ['transformers==4.25.0', 'timm==0.9.16', 'fairscale==0.4.13', 'numpy<2']

    resolved = runtime.resolved_runtime_environment(m)
    assert resolved["python"] == "3.9"
    assert resolved["pytorch"] == "==2.7.1"

    # 3. Planner constraints match resolved runtime
    constraints = model_constraints(m)
    assert constraints["python"] == "3.9"
    assert constraints["torch"] == "==2.7.1"
    assert constraints["runtime_adaptation"] == "sm_120"
    assert constraints["packages"]["transformers"] == "4.25.0"
    assert constraints["packages"]["timm"] == "0.9.16"
    assert constraints["ranges"]["numpy"] == "<2"

    # 4. Unknown adaptation refused
    monkeypatch.setenv(runtime.ADAPTATION_ENV, "sm_999")
    with pytest.raises(WorkbenchError):
        runtime.runtime_adaptation(m)
