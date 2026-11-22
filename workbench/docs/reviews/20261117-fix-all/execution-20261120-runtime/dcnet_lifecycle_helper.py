#!/usr/bin/env python3
"""Owned diagnostic: real DCNet adapter through isolated production queue and ASGI API."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
QUERY = REPO / "data/FashionIQ/resized_image/dress/B005X4PL1G.jpg"
TARGETS = [QUERY, REPO / "data/FashionIQ/resized_image/dress/B005GMTBPC.jpg"]
TEXT = "Is shiny and silver with shorter sleeves and fit and flare."
CHECKPOINT = "fashioniq_run_directory"


def _worker(args: argparse.Namespace) -> int:
    from workbench.backend.operator_config import resolve_config
    from workbench.composition.model_dcnet import Adapter
    import numpy as np

    adapter = Adapter(resolve_config(), checkpoint_id=CHECKPOINT)
    worker_root = Path(args.run_root) / "workers" / args.run_id
    worker_root.mkdir(parents=True, exist_ok=False)
    adapter.work_root = worker_root
    try:
        identity = adapter.prepare_identity()
        queries = adapter.encode_queries([QUERY], [TEXT], [None])
        targets = adapter.encode_targets(TARGETS, [None, None])
        scores = adapter.score(queries, targets)
        if scores.shape != (1, 2) or not np.isfinite(scores).all():
            raise RuntimeError(f"invalid adapter score output: shape={scores.shape}")
        worker_identity = adapter.worker_identity
        if not worker_identity or worker_identity.get("device") != "cuda":
            raise RuntimeError(f"expected real CUDA forward, got {worker_identity}")
        receipt = {"run_id": args.run_id, "model_id": "dcnet", "checkpoint_id": CHECKPOINT,
                   "query_id": "7:1", "query_text": TEXT, "query_image": str(QUERY),
                   "target_images": [str(path) for path in TARGETS],
                   "feature_shapes": {"queries": list(queries.shape), "targets": list(targets.shape)},
                   "scores": scores.tolist(), "finite": True,
                   "target1_rank": int(1 + np.count_nonzero(scores[0] > scores[0, 1])
                                        + np.count_nonzero(scores[0, :1] == scores[0, 1])),
                   "worker_identity": adapter.worker_identity,
                   "checkpoint_resources": identity["checkpoint_resources"]}
        worker = adapter._worker
        if worker is None or worker.poll() is not None:
            raise RuntimeError("DCNet adapter worker is not alive after forward")
        receipt["worker_pid"] = worker.pid
        marker = Path(args.run_root) / f"{args.run_id}.forward.json"
        marker.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
        worker = adapter._worker
        if worker is None or worker.poll() is not None:
            raise RuntimeError("DCNet adapter worker is not alive after forward")
        print(json.dumps({"forward_complete": True, "marker": str(marker),
                          "worker_pid": worker.pid, "worker_identity": worker_identity}), flush=True)
        if args.phase == "cancel":
            threading.Event().wait()
        result = Path(args.run_root) / f"{args.run_id}.receipt.json"
        tmp = result.with_suffix(".tmp")
        tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(result)
        print(json.dumps({"receipt": str(result), "target1_rank": receipt["target1_rank"]}), flush=True)
        return 0
    finally:
        adapter.close()


def _request(port: int, method: str, path: str) -> tuple[int, Any]:
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def _cancel_actor(args: argparse.Namespace) -> int:
    root = args.run_root
    ready = time.monotonic() + args.timeout
    while time.monotonic() < ready:
        try:
            code, _ = _request(args.port, "GET", "/api/jobs")
            if code == 200:
                break
        except (OSError, urllib.error.URLError):
            pass
        if (root / "actor.stop").exists():
            return 0
        time.sleep(.1)
    else:
        raise TimeoutError("cancel actor could not reach isolated jobs API")
    (root / "cancel-actor.ready").write_text(json.dumps({"pid": os.getpid(), "pgid": os.getpgid(0)}) + "\n", encoding="utf-8")
    trigger = root / "cancel-actor.request.json"
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline and not trigger.is_file():
        if (root / "actor.stop").exists():
            return 0
        time.sleep(.05)
    if not trigger.is_file():
        raise TimeoutError("cancel actor trigger did not arrive")
    request = json.loads(trigger.read_text(encoding="utf-8"))
    code, body = _request(args.port, "POST", request["path"])
    result = {"job_id": request["job_id"], "path": request["path"], "http_status": code,
              "body": body, "client_pid": os.getpid(), "client_pgid": os.getpgid(0)}
    tmp = root / "cancel-actor.response.tmp"
    tmp.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(root / "cancel-actor.response.json")
    print(json.dumps(result), flush=True)
    return 0

def _state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _process_group(pgid: int) -> list[str]:
    output = subprocess.run(["ps", "-e", "-o", "pid=,ppid=,pgid=,stat=,cmd="], check=True,
                            text=True, capture_output=True).stdout
    return [line.strip() for line in output.splitlines()
            if len(line.split()) > 2 and line.split()[2] == str(pgid)]


def _assert_guard() -> dict:
    relative = Path("/proc/self/cgroup").read_text().splitlines()
    cgroup = next(line.split(":", 2)[2] for line in relative if line.startswith("0::"))
    base = Path("/sys/fs/cgroup") / cgroup.lstrip("/")
    values = {name: (base / name).read_text().strip()
              for name in ("memory.max", "memory.swap.max", "memory.oom.group")}
    if values != {"memory.max": str(12 * 1024**3), "memory.swap.max": "0", "memory.oom.group": "1"}:
        raise RuntimeError(f"unexpected systemd memory guard: {values}")
    return {"cgroup": str(base), **values}


def _serve(args: argparse.Namespace) -> int:
    run_root = args.run_root.resolve()
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"run root must be fresh and empty: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    if not QUERY.is_file() or not TARGETS[1].is_file():
        raise FileNotFoundError("required immutable FashionIQ images missing")
    guard = _assert_guard()
    artifacts = run_root / "artifacts"
    artifacts.mkdir()
    (artifacts / "assets").symlink_to(REPO / "workbench/artifacts/assets", target_is_directory=True)
    os.environ["WORKBENCH_ARTIFACTS_ROOT"] = str(artifacts)

    from workbench.backend.jobs import SingleGpuQueue
    from workbench.backend import main
    from workbench.backend.registry import model_by_id
    from workbench.backend.operator_config import resolve_config
    from workbench.backend.runtime import resolve_model_interpreter
    import uvicorn

    interpreter = resolve_model_interpreter(model_by_id("dcnet"), resolve_config())
    if not interpreter or not Path(interpreter).is_file():
        raise RuntimeError(f"DCNet interpreter unresolved: {interpreter}")
    main.JOBS.shutdown()
    queue = SingleGpuQueue()
    main.JOBS = queue
    server = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=args.port,
                                           log_level="warning", lifespan="off"))
    server_thread = threading.Thread(target=server.run, name="isolated-asgi", daemon=True)
    server_thread.start()
    phases = []
    actor = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "cancel-actor",
                              "--run-root", str(run_root), "--port", str(args.port),
                              "--timeout", str(args.timeout)], cwd=REPO, start_new_session=True,
                             stdout=(run_root / "cancel-actor.stdout.log").open("w", encoding="utf-8"),
                             stderr=(run_root / "cancel-actor.stderr.log").open("w", encoding="utf-8"))
    actor_ready = time.monotonic() + 20
    while time.monotonic() < actor_ready and not (run_root / "cancel-actor.ready").is_file():
        if actor.poll() is not None:
            raise RuntimeError(f"independent cancel actor exited before ready: {actor.returncode}")
        time.sleep(.05)
    ready_path = run_root / "cancel-actor.ready"
    if not ready_path.is_file():
        actor.terminate()
        actor.wait(timeout=10)
        raise TimeoutError("independent cancel actor did not become ready")
    actor_identity = json.loads(ready_path.read_text(encoding="utf-8"))
    if actor_identity["pgid"] == os.getpgid(0):
        actor.terminate()
        actor.wait(timeout=10)
        raise RuntimeError(f"cancel actor shares launcher process group: {actor_identity}")
    try:
        deadline = time.monotonic() + 20
        while not server.started and time.monotonic() < deadline:
            time.sleep(.05)
        if not server.started:
            raise TimeoutError("isolated ASGI server failed to start")
        for phase in ("cancel", "retry"):
            phase_root = run_root / "queue_jobs" / phase
            phase_root.mkdir(parents=True)
            command = [sys.executable, str(Path(__file__).resolve()), "worker", "--phase", phase,
                       "--run-id", phase, "--run-root", str(run_root)]
            job = queue.enqueue("dcnet", CHECKPOINT, "diagnostic_q1_t2", command, REPO,
                                phase_root, {"diagnostic": "DCNet Q1/T2 real forward, queue lifecycle only"})
            phases.append({"phase": phase, "job": job, "command": command})
            job_id = job["job_id"]
            state_path = phase_root / "queue_state.json"
            marker = run_root / f"{phase}.forward.json"
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline and not marker.is_file():
                state = _state(state_path) if state_path.exists() else {}
                if state.get("status") in {"failed", "cancelled", "completed"}:
                    raise RuntimeError(f"{phase} job became {state['status']} before forward marker; error={state.get('error')}; log_directory={phase_root}")
                time.sleep(.1)
            if not marker.is_file():
                raise TimeoutError(f"{phase} forward marker absent after {args.timeout}s; state={_state(state_path)}; log_directory={phase_root}")
            code, jobs = _request(args.port, "GET", "/api/jobs")
            if code != 200 or not any(j.get("job_id") == job_id and j.get("status") == "running" for j in jobs):
                raise RuntimeError(f"API does not report running job: {code} {jobs}")
            forward = json.loads(marker.read_text(encoding="utf-8"))
            worker_pid = int(forward["worker_pid"])
            queue_pid = queue._processes[job_id].pid
            pgid = os.getpgid(worker_pid)
            inventory = {"queue_pid": queue_pid, "worker_pid": worker_pid, "pgid": pgid,
                         "group": _process_group(pgid),
                         "gpu": subprocess.run(["nvidia-smi"], check=True, text=True,
                                                capture_output=True).stdout}
            print(json.dumps({"phase": phase, "process_inventory": inventory}), flush=True)
            if os.getpgid(queue_pid) != pgid:
                raise RuntimeError(f"adapter process escaped queue group: queue={queue_pid}, worker={worker_pid}, pgid={pgid}")
            if actor_identity["pid"] in {queue_pid, worker_pid} or actor_identity["pgid"] == pgid:
                raise RuntimeError(f"cancel actor overlaps queue worker process group: actor={actor_identity}, queue={queue_pid}, worker={worker_pid}, pgid={pgid}")
            if phase == "cancel":
                trigger = run_root / "cancel-actor.request.json"
                trigger.write_text(json.dumps({"job_id": job_id,
                    "path": f"/api/jobs/{job_id}/cancel"}, sort_keys=True) + "\n", encoding="utf-8")
                print(json.dumps({"external_cancel_requested": True, "trigger": str(trigger),
                                  "job_id": job_id}), flush=True)
                actor_deadline = time.monotonic() + 30
                while time.monotonic() < actor_deadline and not (run_root / "cancel-actor.response.json").is_file():
                    if actor.poll() is not None:
                        raise RuntimeError(f"external cancel actor exited early: {actor.returncode}")
                    time.sleep(.05)
                response_path = run_root / "cancel-actor.response.json"
                if not response_path.is_file():
                    raise TimeoutError("external HTTP cancel actor did not persist response")
                response = json.loads(response_path.read_text(encoding="utf-8"))
                if response.get("http_status") != 200 or response.get("body", {}).get("status") != "cancelled":
                    raise RuntimeError(f"external HTTP cancel was refused: {response}")
                phases[-1]["cancel_http"] = response
                while time.monotonic() < deadline:
                    state = _state(state_path)
                    if state.get("status") == "cancelled":
                        break
                    if state.get("status") in {"failed", "completed"}:
                        raise RuntimeError(f"cancel phase ended unexpectedly: {state}")
                    time.sleep(.1)
                state = _state(state_path)
                if state.get("status") != "cancelled":
                    raise TimeoutError(f"external queue cancellation timed out: {state}")
                code, jobs = _request(args.port, "GET", "/api/jobs")
                if code != 200 or not any(j.get("job_id") == job_id and j.get("status") == "cancelled" for j in jobs):
                    raise RuntimeError(f"API does not report cancelled job: {code} {jobs}")
                survivors = _process_group(pgid)
                if survivors:
                    raise RuntimeError(f"queue cancel left process group: {survivors}")
                phases[-1]["cancel_state"] = state
            else:
                deadline = time.monotonic() + args.timeout
                while time.monotonic() < deadline:
                    state = _state(state_path)
                    if state.get("status") in {"completed", "failed"}:
                        break
                    time.sleep(.1)
                state = _state(state_path)
                receipt_path = run_root / "retry.receipt.json"
                if state.get("status") != "completed" or not receipt_path.is_file():
                    raise RuntimeError(f"retry failed to complete: {state}")
                phases[-1]["completed_state"] = state
                phases[-1]["receipt"] = json.loads(receipt_path.read_text(encoding="utf-8"))
        code, jobs = _request(args.port, "GET", "/api/jobs")
        if code != 200:
            raise RuntimeError(f"final API job query failed: {code} {jobs}")
        output = run_root / "lifecycle-report.json"
        (run_root / "actor.stop").write_text("done\n", encoding="utf-8")
        actor.wait(timeout=10)
        report = {"guard": guard, "admission": "direct SingleGpuQueue.enqueue; canonical POST /api/jobs not used",
                  "database": "none; queue_state.json filesystem persistence only",
                  "scope": "DCNet real Q1/T2 adapter lifecycle; not parity or full-gallery metric",
                  "phases": phases, "api_jobs": jobs, "queue_worker_alive": queue._worker.is_alive()}
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"report": str(output), "queue_worker_alive": queue._worker.is_alive()}), flush=True)
        return 0
    finally:
        (run_root / "actor.stop").write_text("stop\n", encoding="utf-8")
        if actor.poll() is None:
            actor.wait(timeout=10)
        server.should_exit = True
        server_thread.join(timeout=10)
        queue.shutdown()


def _worker_cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("worker")
    parser.add_argument("--phase", choices=("cancel", "retry"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    return _worker(parser.parse_args())


def main_cli() -> int:
    if sys.argv[1:2] == ["worker"]:
        return _worker_cli()
    if sys.argv[1:2] == ["cancel-actor"]:
        parser = argparse.ArgumentParser()
        parser.add_argument("cancel-actor")
        parser.add_argument("--run-root", type=Path, required=True)
        parser.add_argument("--port", type=int, required=True)
        parser.add_argument("--timeout", type=int, default=900)
        return _cancel_actor(parser.parse_args())
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=900)
    return _serve(parser.parse_args())


if __name__ == "__main__":
    try:
        raise SystemExit(main_cli())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
