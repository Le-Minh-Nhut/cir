"""The frontend unit's stop contract: the whole group dies on the FIRST signal.

Measured defect (2026-11-13, isolated instance on a spare port, same argv/cwd/Environment as the
unit):

    KillMode=mixed          npm exits 22 ms, node survives 32.5 s, port held until the cgroup
                            SIGKILL at TimeoutStopSec=10 -> every normal stop burns the full budget
                            and any same-port restart inside that window fails with EADDRINUSE
    KillMode=control-group  npm exits 22 ms, node exits 24 ms, port free at 51 ms, same-port
                            restart succeeds immediately

`Type=simple` makes npm MainPID, but the LISTENER is `node`, a grandchild. `mixed` signals MainPID
only, so it cannot stop the listener; `control-group` signals every member at once. The backend
deliberately keeps `mixed` (signalling a GPU job's child directly would skip the queue's truthful
"failed" bookkeeping); the frontend has no such state, so it takes the mode that actually stops.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from workbench.systemd import render_units

TEMPLATE = Path(__file__).resolve().parents[1] / "systemd" / "cir-workbench-frontend.service.in"
BACKEND_TEMPLATE = Path(__file__).resolve().parents[1] / "systemd" / "cir-workbench-backend.service.in"
TIMEOUT_STOP = re.compile(r"^TimeoutStopSec=(\d+)$", re.M)
KILL_MODE = re.compile(r"^KillMode=(\S+)$", re.M)

def _directive(text: str, pattern: re.Pattern, name: str) -> str:
    """The single uncommented value of one directive, refusing a missing or duplicated setting."""
    values = pattern.findall(text)
    assert len(values) == 1, f"expected exactly one `{name}` directive, got {values}"
    return values[0]

def _active_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")]

def test_the_frontend_signals_the_whole_group_not_only_npm() -> None:
    """`mixed` cannot stop a grandchild listener; that is the measured 32.5 s defect."""
    mode = _directive(TEMPLATE.read_text(encoding="utf-8"), KILL_MODE, "KillMode")
    assert mode == "control-group", (
        f"frontend KillMode is {mode!r}: with `mixed` the `node` grandchild outlives npm and holds "
        "the port for the whole TimeoutStopSec (measured 32.5 s vs 24 ms)")

def test_the_frontend_timeout_is_proportionate_to_the_measured_stop() -> None:
    """A stop that finishes in 51 ms must not be bounded by a 10 s budget."""
    seconds = int(_directive(TEMPLATE.read_text(encoding="utf-8"), TIMEOUT_STOP, "TimeoutStopSec"))
    # The measured worst case under control-group is 51 ms; 1 s would be ~20x. Anything above the
    # old 10 s is the defect returning, and anything below 1 s risks a slow container host.
    assert 1 <= seconds <= 5, f"TimeoutStopSec={seconds} is not proportionate to a ~51 ms stop"
    assert seconds < 10, "the 10 s budget existed only to cover the cgroup-SIGKILL wait `mixed` needed"

def test_the_backend_keeps_mixed_because_its_children_carry_queue_state() -> None:
    """The two units must NOT be homogenised: this is the negative control for the change.

    `jobs.py` records a terminal status per job; SIGTERM-ing a job's process group directly skips
    that bookkeeping, which is exactly why the backend's comment chooses `mixed`. A future edit that
    "simplifies" both units to the same KillMode breaks one of them.
    """
    mode = _directive(BACKEND_TEMPLATE.read_text(encoding="utf-8"), KILL_MODE, "KillMode")
    assert mode == "mixed", f"backend KillMode is {mode!r}; its queue bookkeeping needs MainPID-first"

def test_the_frontend_template_still_renders_a_startable_unit(tmp_path: Path) -> None:
    """The new directives must survive rendering -- a template the renderer drops is not a fix."""
    from workbench.tests.test_systemd_user_services import fixture_config
    rendered = render_units.render(
        fixture_config(tmp_path), host="127.0.0.1", backend_port=8401, frontend_port=5401,
        backend_unit="cir-wb-backend-test", frontend_unit="cir-wb-frontend-test",
        python="/usr/bin/python3", npm="/usr/bin/npm", production_frontend=True)
    text = rendered["cir-wb-frontend-test.service"]
    active = _active_lines(text)
    assert "KillMode=control-group" in active
    assert "TimeoutStopSec=5" in active
    assert "Type=simple" in active
    # npm is MainPID and the listener is its grandchild: the reason control-group is needed.
    assert any(line.startswith("ExecStart=/usr/bin/npm run preview") for line in active)
    assert not any(line.startswith("ExecStop=") for line in active), (
        "the fix is the KillMode, not a hand-rolled ExecStop that has to re-discover the listener")
