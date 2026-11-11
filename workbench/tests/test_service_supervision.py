"""The launcher supervises its children: unexpected death is restarted, logged, and rate-limited.

Every test here drives real processes (a stub `sleep` and a stub crasher) so the restart path is
exercised end to end, but nothing binds a port or touches the operator's served instance.
"""
from __future__ import annotations

import importlib.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def load_script(name: str = "serve_workbench"):
    path = Path("workbench/scripts") / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stub_services(serve, tmp_path: Path, script: str):
    command = serve.Command("stub", [sys.executable, "-c", script], tmp_path)
    supervisor = serve.Supervisor(commands=[command])
    supervisor.services[0].environment = dict(os.environ)
    return supervisor


def drive(supervisor, predicate, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        supervisor.check(time.monotonic())
        if predicate(supervisor.services[0]):
            return
        time.sleep(0.05)
    raise AssertionError(f"condition not reached in {timeout}s")


def kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def test_child_death_is_restarted_and_logged(tmp_path: Path, capsys) -> None:
    serve = load_script()
    supervisor = stub_services(serve, tmp_path, "import time; time.sleep(120)")
    try:
        supervisor.start(supervisor.services[0])
        first = supervisor.services[0].process.pid
        kill_group(first)
        drive(supervisor, lambda service: service.process is not None and service.process.pid != first)
        second = supervisor.services[0].process.pid
        assert second != first and supervisor.services[0].restarts == 1
        log = capsys.readouterr().err
        assert "[SUPERVISOR]" in log
        assert f"stub PID {first} died at" in log and "signal SIGKILL" in log
        assert f"stub started PID {second}" in log
    finally:
        supervisor.stop()


def test_backoff_is_exponential_capped_and_bounded(tmp_path: Path) -> None:
    serve = load_script()
    delays = [serve.restart_delay(attempt) for attempt in range(1, 12)]
    assert delays[0] == serve.BACKOFF_BASE
    assert delays == sorted(delays)
    assert delays[-1] == serve.BACKOFF_CAP

    # A crash loop cannot spin: the stub dies immediately, and the supervisor's own pacing means
    # far fewer restarts than the 0.2 s poll interval alone would allow.
    supervisor = stub_services(serve, tmp_path, "raise SystemExit(3)")
    try:
        begin = time.monotonic()
        while time.monotonic() - begin < 1.5:
            supervisor.check(time.monotonic())
            time.sleep(0.05)
        # Unthrottled, 1.5 s of a dying child would be dozens of restarts; the backoff keeps the
        # crash loop to a handful.
        assert 1 <= supervisor.services[0].restarts <= 6
    finally:
        supervisor.stop()


def test_log_rotation_keeps_previous_segment(tmp_path: Path) -> None:
    destination = tmp_path / "serve.log"
    driver = "\n".join([
        "import importlib.util, os, pathlib, sys",
        "spec = importlib.util.spec_from_file_location('serve', 'workbench/scripts/serve_workbench.py')",
        "module = importlib.util.module_from_spec(spec)",
        "spec.loader.exec_module(module)",
        f"handle = os.open({str(destination)!r}, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)",
        "os.dup2(handle, 1)",
        "os.dup2(handle, 2)",
        "print('first segment ' + 'x' * 64, flush=True)",
        "rotated = module.rotate_logs(16)",
        "print('second segment', flush=True)",
        f"pathlib.Path({str(destination.with_suffix('.rotated'))!r}).write_text(str(rotated))",
    ])
    result = subprocess.run([sys.executable, "-c", driver], cwd=Path.cwd(), capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    rotated = Path((tmp_path / "serve.rotated").read_text())
    assert rotated != destination and rotated.exists()
    assert "first segment" in destination.read_text()
    assert "first segment" not in rotated.read_text()
    assert "second segment" in rotated.read_text()
    assert rotated.name.startswith("serve.log.")


def test_backoff_ladder_is_exact() -> None:
    serve = load_script()
    assert [serve.restart_delay(n) for n in range(1, 8)] == [
        0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 8.0]
    # failures <= 0 is the first step, not a negative or absent delay.
    assert serve.restart_delay(0) == serve.BACKOFF_BASE == 0.2
    assert serve.restart_delay(-5) == serve.BACKOFF_BASE


def test_backoff_saturates_without_giant_int_arithmetic() -> None:
    serve = load_script()
    # The regression: `BACKOFF_BASE * 2 ** (failures - 1)` built 2**1024 and then raised
    # OverflowError: int too large to convert to float, so a long crash loop crashed the
    # supervisor instead of pacing it.
    begin = time.monotonic()
    for failures in (7, 8, 1025, 10 ** 6, 10 ** 12):
        assert serve.restart_delay(failures) == serve.BACKOFF_CAP == 8.0
    assert time.monotonic() - begin < 1.0, "clamping must happen before the multiply"
    # The exponent is clamped at the smallest one that already saturates the cap.
    assert serve.BACKOFF_BASE * 2 ** serve.BACKOFF_SATURATING_EXPONENT >= serve.BACKOFF_CAP
    assert serve.BACKOFF_BASE * 2 ** (serve.BACKOFF_SATURATING_EXPONENT - 1) < serve.BACKOFF_CAP


def test_backoff_is_monotone_over_a_long_crash_loop() -> None:
    serve = load_script()
    delays = [serve.restart_delay(n) for n in range(1, 2001)]
    assert delays == sorted(delays)
    assert delays[-1] == serve.BACKOFF_CAP
