"""A GPU capability describes a host, not only an environment.

A capability recorded on one machine must not authorize a run somewhere else, and the record must
say what it covers so an operator can tell evidence from assumption.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from workbench.backend import runtime


def _model(variable: str = "CAP_HOST_PYTHON") -> dict:
    return {"model_id": "cap_host", "environment_required": True,
            "environment": {"python_env_var": variable, "pytorch": "1.12.1", "cuda": "12.4"}}


def test_host_identity_records_what_the_claim_depends_on() -> None:
    """The identity must name the machine, the architecture and the visible-device selection."""
    identity = runtime._host_identity()
    assert identity.get("node"), identity
    assert identity.get("machine"), identity
    assert "cuda_visible_devices" in identity


def test_changing_the_visible_devices_changes_the_capability_key(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A capability for device 2 must not be reused by a run restricted to another device set."""
    interpreter = tmp_path / "python"
    interpreter.write_text("#!/bin/sh\n")
    interpreter.chmod(0o755)
    monkeypatch.setenv("CAP_HOST_PYTHON", str(interpreter))

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    first = runtime._gpu_capability_key(_model())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    second = runtime._gpu_capability_key(_model())
    assert first != second


def test_changing_the_host_machine_changes_the_capability_key(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The same environment on a different host is a different claim."""
    interpreter = tmp_path / "python"
    interpreter.write_text("#!/bin/sh\n")
    interpreter.chmod(0o755)
    monkeypatch.setenv("CAP_HOST_PYTHON", str(interpreter))

    real_uname = runtime.os.uname
    monkeypatch.setattr(runtime.os, "uname",
                        lambda: type("U", (), {"nodename": "host-a", "machine": "x86_64"})())
    first = runtime._gpu_capability_key(_model())
    monkeypatch.setattr(runtime.os, "uname",
                        lambda: type("U", (), {"nodename": "host-b", "machine": "x86_64"})())
    second = runtime._gpu_capability_key(_model())
    assert first != second
    assert real_uname is not None


def test_the_record_states_its_revalidation_rule(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An operator reading the record must see that it is evidence, and when it expires."""
    payload = runtime.record_gpu_capability(
        _model(), {"capability_verified": True, "cuda_available": True}, None)
    assert payload["host_identity"]
    assert "re-probe" in payload["revalidation"]
    assert "driver change" in payload["expires_on"]

    # The rule is documented rather than implied.
    assert "host" in runtime._host_identity.__doc__


def test_a_capability_does_not_transfer_to_another_interpreter(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The existing environment binding must survive the host binding."""
    first_env = tmp_path / "env_a" / "bin"
    second_env = tmp_path / "env_b" / "bin"
    for directory in (first_env, second_env):
        directory.mkdir(parents=True)
        interpreter = directory / "python"
        interpreter.write_text("#!/bin/sh\n")
        interpreter.chmod(0o755)

    monkeypatch.setenv("CAP_HOST_PYTHON", str(first_env / "python"))
    first = runtime._gpu_capability_key(_model())
    monkeypatch.setenv("CAP_HOST_PYTHON", str(second_env / "python"))
    second = runtime._gpu_capability_key(_model())
    assert first != second
