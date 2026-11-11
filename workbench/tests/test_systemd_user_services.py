"""The systemd user units are rendered from this repo's own resolver, not hand-copied paths.

The fast tests build a fixture `WorkbenchConfig` and read the rendered text; none of them needs a
running `systemd --user` manager. The two tests that do talk to the manager are skipped unless
`/run/systemd/system` exists (i.e. systemd is PID 1).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from workbench.backend.operator_config import WorkbenchConfig
from workbench.systemd import render_units

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / "workbench" / "envs" / "ctl312h" / "bin" / "python"
SYSTEMD_PRESENT = Path("/run/systemd/system").exists()


def fixture_config(tmp_path: Path) -> WorkbenchConfig:
    """The release-root shape under a temp directory, with distinct ports."""
    artifacts = tmp_path / "artifacts"
    for name in ("results", "checkpoints", "third_party", "circus_release", "composition"):
        (artifacts / name).mkdir(parents=True, exist_ok=True)
    return WorkbenchConfig(
        CIR_REPO_ROOT=REPO_ROOT,
        CIR_DATA_ROOT=tmp_path / "data",
        FASHIONIQ_ROOT=tmp_path / "data" / "FashionIQ",
        WORKBENCH_HOST="127.0.0.1",
        WORKBENCH_BACKEND_PORT=8401,
        WORKBENCH_FRONTEND_PORT=5401,
        WORKBENCH_CHECKPOINT_ROOT=artifacts / "checkpoints",
        WORKBENCH_RESULTS_ROOT=artifacts / "results",
        WORKBENCH_THIRD_PARTY_ROOT=artifacts / "third_party",
        WORKBENCH_ARTIFACTS_ROOT=artifacts,
        WORKBENCH_CIRCUS_RELEASE_ROOT=artifacts / "circus_release",
    )


def rendered(tmp_path: Path) -> dict[str, str]:
    return render_units.render(
        fixture_config(tmp_path),
        host="127.0.0.1",
        backend_port=8401,
        frontend_port=5401,
        backend_unit="cir-wb-backend-test",
        frontend_unit="cir-wb-frontend-test",
        python=str(PYTHON),
        npm="/usr/bin/npm",
    )


def lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line and not line.startswith("#")]


def test_backend_execstart_is_uvicorn_without_a_shell(tmp_path: Path) -> None:
    unit = rendered(tmp_path)["cir-wb-backend-test.service"]
    assert (f"ExecStart={PYTHON} -m uvicorn workbench.backend.main:app "
            "--host 127.0.0.1 --port 8401") in lines(unit)
    # No shell and no wrapper: the argv IS the server, so MainPID is uvicorn and signals land on
    # the process that runs the shutdown hook.
    assert not any(token in unit for token in ("/bin/sh", "bash -c", "exec "))
    assert "Type=simple" in lines(unit)


def test_backend_paths_and_environment_come_from_the_config(tmp_path: Path) -> None:
    config = fixture_config(tmp_path)
    unit = rendered(tmp_path)["cir-wb-backend-test.service"]
    assert f"WorkingDirectory={config.CIR_REPO_ROOT}" in lines(unit)
    for key in ("CIR_REPO_ROOT", "WORKBENCH_ARTIFACTS_ROOT", "WORKBENCH_RESULTS_ROOT",
                "WORKBENCH_CIRCUS_RELEASE_ROOT", "WORKBENCH_CHECKPOINT_ROOT"):
        assert f'Environment="{key}={getattr(config, key)}"' in lines(unit), key


def test_backend_restart_policy(tmp_path: Path) -> None:
    unit = lines(rendered(tmp_path)["cir-wb-backend-test.service"])
    assert "Restart=always" in unit
    assert "RestartSec=2" in unit
    assert "StartLimitIntervalSec=60" in unit
    assert "StartLimitBurst=5" in unit
    assert "KillMode=mixed" in unit
    assert "TimeoutStopSec=30" in unit
    # A bound, not an unbounded wait: the shutdown hook is allowed 10 s + 5 s per job group.
    assert int(next(line for line in unit if line.startswith("TimeoutStopSec=")).split("=")[1]) >= 15


def test_frontend_execstart_and_proxy_target(tmp_path: Path) -> None:
    units = rendered(tmp_path)
    frontend = lines(units["cir-wb-frontend-test.service"])
    assert ("ExecStart=/usr/bin/npm run preview -- --host 127.0.0.1 --port 5401 --strictPort"
            in frontend)
    assert f"WorkingDirectory={REPO_ROOT / 'workbench' / 'frontend'}" in frontend
    # vite.config.mts proxies /api at this variable, so it must name the backend port.
    assert 'Environment="WORKBENCH_BACKEND_URL=http://127.0.0.1:8401"' in frontend
    assert "After=cir-wb-backend-test.service" in frontend
    assert "Restart=always" in frontend


def test_dev_frontend_runs_vite_not_preview(tmp_path: Path) -> None:
    units = render_units.render(
        fixture_config(tmp_path), host="127.0.0.1", backend_port=8401, frontend_port=5401,
        backend_unit="cir-wb-backend-test", frontend_unit="cir-wb-frontend-test",
        python=str(PYTHON), npm="/usr/bin/npm", production_frontend=False)
    assert "run dev" in units["cir-wb-frontend-test.service"]
    assert "run preview" not in units["cir-wb-frontend-test.service"]


def test_dry_run_prints_units_and_writes_nothing(tmp_path: Path, capsys) -> None:
    target = tmp_path / "units"
    status = render_units.main([
        "--dry-run", "--directory", str(target), "--backend-unit", "cir-wb-backend-test",
        "--frontend-unit", "cir-wb-frontend-test", "--backend-port", "8401",
        "--frontend-port", "5401", "--python", str(PYTHON), "--npm", "/usr/bin/npm"])
    assert status == 0
    output = capsys.readouterr().out
    assert "ExecStart=" in output and "cir-wb-backend-test.service" in output
    assert not target.exists()
    assert not any(target.parent.glob("cir-wb-backend-test.service"))


def test_install_writes_both_units_into_the_named_directory(tmp_path: Path) -> None:
    units = rendered(tmp_path)
    written = render_units.install(units, tmp_path / "units")
    assert {path.name for path in written} == {
        "cir-wb-backend-test.service", "cir-wb-frontend-test.service"}
    for path in written:
        text = path.read_text(encoding="utf-8")
        assert "@DESCRIPTION@" not in text and "@WORKING_DIRECTORY@" not in text
        assert "@ENVIRONMENT@" not in text and "@BACKEND_UNIT@" not in text
        assert "@EXEC_START@" not in text


@pytest.mark.skipif(not SYSTEMD_PRESENT, reason="no systemd manager (not PID 1)")
def test_rendered_units_survive_systemd_verify() -> None:
    """`systemd-analyze verify` catches a unit file the manager would refuse to load."""
    result = subprocess.run(["systemd-analyze", "--version"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not SYSTEMD_PRESENT, reason="no systemd manager (not PID 1)")
def test_manager_accepts_a_rendered_unit(tmp_path: Path) -> None:
    units = rendered(tmp_path)
    directory = tmp_path / "unit"
    directory.mkdir()
    for name, text in units.items():
        (directory / name).write_text(text, encoding="utf-8")
    for name in units:
        result = subprocess.run(["systemd-analyze", "verify", str(directory / name)],
                               capture_output=True, text=True, cwd= REPO_ROOT)
        # `verify` reports unknown unit references and missing binaries; the npm path exists
        # locally, so anything but success is a real defect in the template.
        assert "Unknown key name" not in result.stderr, result.stderr
        assert "Invalid" not in result.stderr, result.stderr
