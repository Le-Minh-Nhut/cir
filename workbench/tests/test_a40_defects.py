from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import workbench.scripts.pipeline as pipeline
from workbench.scripts import load_mock_results


def test_pipeline_interrupt_report_is_truthful(tmp_path: Path, monkeypatch) -> None:
    report_file = tmp_path / "workflow_report.json"
    state_file = tmp_path / "pipeline_state.json"
    stage = pipeline.Stage(
        name="sync",
        commands=(["/bin/true"],),
        model_id="m",
    )

    def interrupt_run(argv: list[str]) -> int:
        raise KeyboardInterrupt("Simulated Ctrl-C")

    monkeypatch.setattr(pipeline, "run_command", interrupt_run)
    with pytest.raises(KeyboardInterrupt):
        pipeline.run_stages(
            [stage],
            dry_run=False,
            state_path=state_file,
            report_path=report_file,
            apply=True,
        )

    assert report_file.exists(), "Report file was not written"
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["success"] is False, f"Expected success=False on interrupt, got {report['success']}"
    assert report["status"] != "SUCCESS", f"Expected status!=SUCCESS on interrupt, got {report['status']}"


def test_pipeline_stale_running_dependency_blocks_downstream(tmp_path: Path, monkeypatch) -> None:
    state_file = tmp_path / "pipeline_state.json"
    state = {
        "stages": {
            "sync:m": {
                "status": "RUNNING",
                "stage_id": "sync:m",
                "model_id": "m",
            }
        }
    }
    pipeline.save_pipeline_state(state, state_file)
    called: list[list[str]] = []
    monkeypatch.setattr(pipeline, "run_command", lambda argv: called.append(argv) or 0)

    child = pipeline.Stage(
        name="evaluation",
        commands=(["/bin/eval"],),
        model_id="m",
        dependencies=("sync:m",),
    )
    pipeline.run_stages(
        [child],
        dry_run=False,
        resume=True,
        state_path=state_file,
        apply=True,
    )
    saved = pipeline.load_pipeline_state(state_file)
    child_rec = saved["stages"].get(pipeline.stage_key(child))
    assert len(called) == 0, f"Downstream executed despite RUNNING upstream: {called}"
    assert child_rec is not None, "Child stage record not found"
    assert child_rec.get("status") == "SKIPPED_DEPENDENCY", f"Expected SKIPPED_DEPENDENCY, got {child_rec.get('status')}"
    assert child_rec.get("return_code") == 125, f"Expected return_code 125, got {child_rec.get('return_code')}"


def test_pipeline_interrupt_terminates_child_process_group(tmp_path: Path) -> None:
    pid_file = tmp_path / "grandchild.pid"
    ready_file = tmp_path / "ready.sentinel"
    child_script = tmp_path / "child.py"
    child_script.write_text(
        "import sys, subprocess, time, pathlib\n"
        "grandchild_code = (\n"
        "    'import sys, os, time, pathlib\\n'\n"
        "    'pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\\n'\n"
        "    'pathlib.Path(sys.argv[2]).touch()\\n'\n"
        "    'time.sleep(60)\\n'\n"
        ")\n"
        "gc = subprocess.Popen([sys.executable, '-c', grandchild_code, sys.argv[1], sys.argv[2]])\n"
        "try:\n"
        "    gc.wait()\n"
        "except BaseException:\n"
        "    pass\n",
        encoding="utf-8",
    )

    runner_script = tmp_path / "run_pipe.py"
    repo_root = Path.cwd().resolve()
    runner_script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(repo_root)!r})\n"
        "import workbench.scripts.pipeline as pipeline\n"
        "cmd = sys.argv[1:]\n"
        "rc = pipeline.run_command(cmd)\n"
        "sys.exit(rc)\n",
        encoding="utf-8",
    )
    pipe_proc = subprocess.Popen(
        [sys.executable, str(runner_script), sys.executable, str(child_script), str(pid_file), str(ready_file)],
        cwd=str(tmp_path),
    )

    start = time.monotonic()
    while not ready_file.exists() and time.monotonic() - start < 5.0:
        time.sleep(0.05)
    assert ready_file.exists(), "Grandchild failed to start and touch sentinel"

    grandchild_pid = int(pid_file.read_text(encoding="utf-8").strip())
    # Verify grandchild is currently alive
    os.kill(grandchild_pid, 0)

    try:
        pipe_proc.send_signal(signal.SIGTERM)
        try:
            pipe_proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            pipe_proc.kill()
            pipe_proc.wait()

        dead = False
        start = time.monotonic()
        while time.monotonic() - start < 3.0:
            try:
                os.kill(grandchild_pid, 0)
                time.sleep(0.05)
            except ProcessLookupError:
                dead = True
                break
        assert dead, f"Grandchild PID {grandchild_pid} survived pipeline SIGTERM"
    finally:
        try:
            os.kill(grandchild_pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def test_load_mock_results_refuses_out_of_root_target(tmp_path: Path, monkeypatch, capsys) -> None:
    repo_root = tmp_path / "repo"
    outside_mock = repo_root / "workbench" / "tests" / "mock"
    outside_mock.mkdir(parents=True)
    sentinel = outside_mock / "keep.txt"
    sentinel.write_text("must survive", encoding="utf-8")

    cfg = SimpleNamespace(
        CIR_REPO_ROOT=repo_root,
        CIR_DATA_ROOT=tmp_path / "data",
        WORKBENCH_RESULTS_ROOT=tmp_path / "results",
    )
    monkeypatch.setattr(load_mock_results, "resolve_config", lambda: cfg)

    monkeypatch.setattr("sys.argv", ["load_mock_results.py", "--output-root", str(repo_root)])
    ret = load_mock_results.main()
    assert sentinel.is_file(), "Out-of-root mock target was deleted!"
    assert ret == 2, f"Expected returncode 2 for unsafe output-root, got {ret}"
    assert "[BLOCKED]" in capsys.readouterr().err
