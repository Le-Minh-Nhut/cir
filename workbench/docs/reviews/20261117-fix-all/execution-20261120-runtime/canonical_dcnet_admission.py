#!/usr/bin/env python3
"""Isolated canonical DCNet API admission/cancel harness."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.error
import shutil
import urllib.request

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
MODEL = "dcnet"
CHECKPOINT = "fashioniq_run_directory"
PROTOCOL = "fashioniq_full_gallery_ref_excluded"
PAYLOAD = {"model_id": MODEL, "checkpoint_id": CHECKPOINT,
           "protocol_id": PROTOCOL, "top_k": 200}


def _request(port: int, method: str, path: str, body: dict | None = None,
             timeout: float = 30) -> tuple[int, object]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                     method=method,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def _api_job(port: int, job_id: str) -> dict:
    status, jobs = _request(port, "GET", "/api/jobs")
    if status != 200 or not isinstance(jobs, list):
        raise RuntimeError(f"queue API GET failed: {status} {jobs}")
    job = next((item for item in jobs if item.get("job_id") == job_id), None)
    if job is None:
        raise RuntimeError(f"job {job_id} missing from GET /api/jobs")
    return job


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _group_members(pgid: int) -> list[str]:
    output = subprocess.run(["ps", "-e", "-o", "pid=,ppid=,pgid=,stat=,cmd="],
                            check=True, text=True, capture_output=True).stdout
    return [line.strip() for line in output.splitlines()
            if len(line.split()) > 2 and line.split()[2] == str(pgid)]


def _gpu_apps() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        check=True, text=True, capture_output=True)
    return result.stdout.strip()


def _pid_in_group(pid: int, pgid: int) -> bool:
    try:
        return os.getpgid(pid) == pgid
    except ProcessLookupError:
        return False


def _guard() -> dict[str, str]:
    cgroup = next(line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines()
                  if line.startswith("0::"))
    root = Path("/sys/fs/cgroup") / cgroup.lstrip("/")
    values = {key: (root / key).read_text().strip()
              for key in ("memory.max", "memory.swap.max", "memory.oom.group")}
    expected = {"memory.max": str(12 * 1024**3), "memory.swap.max": "0",
                "memory.oom.group": "1"}
    if values != expected:
        raise RuntimeError(f"OOM guard mismatch: {values}")
    return {"cgroup": str(root), **values}


def _actor(root: Path, port: int, timeout: float) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            status, _ = _request(port, "GET", "/api/jobs")
            if status == 200:
                break
        except OSError:
            pass
        if (root / "actor.stop").exists():
            return 0
        time.sleep(.1)
    else:
        raise TimeoutError("cancel actor could not reach isolated API")
    (root / "cancel-actor.ready").write_text(
        json.dumps({"pid": os.getpid(), "pgid": os.getpgid(0)}) + "\n")
    request_path = root / "cancel-request.json"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not request_path.is_file():
        if (root / "actor.stop").exists():
            return 0
        time.sleep(.05)
    if not request_path.is_file():
        raise TimeoutError("cancel actor request not received")
    payload = json.loads(request_path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("path"), str) or not isinstance(payload.get("job_id"), str):
        raise ValueError(f"invalid cancel actor request: {payload!r}")
    status, body = _request(port, "POST", payload["path"])
    response = {"http_status": status, "body": body, "path": payload["path"],
                "job_id": payload["job_id"], "client_pid": os.getpid(),
                "client_pgid": os.getpgid(0)}
    temp = root / "cancel-response.tmp"
    temp.write_text(json.dumps(response, indent=2, sort_keys=True) + "\n")
    temp.replace(root / "cancel-response.json")
    return 0


def _run(args: argparse.Namespace) -> int:
    root = args.run_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"run root must be fresh and empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    progress = {"scope": "canonical POST /api/jobs with HTTP cancel; full official evaluator retry", "run_root": str(root), "events": []}
    def record(event: str, **values) -> None:
        progress["events"].append({"event": event, **values})
        temp = root / "canonical-api-progress.tmp"
        temp.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
        temp.replace(root / "canonical-api-progress.json")
    try:
        guard = _guard()
        record("oom_guard_verified", guard=guard)
    except Exception as error:
        record("preflight_failed", error=f"{type(error).__name__}: {error}")
        raise
    checkpoint = Path("/home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints/dcnet")
    protected_files = [checkpoint / "config.json", checkpoint / "trained_model.pth", checkpoint / "val" / "att_result.pt"]
    checkpoint_baseline = {str(path): _sha256(path) for path in protected_files}
    record("protected_checkpoint_baseline", hashes=checkpoint_baseline)
    (root / "checkpoint-baseline.json").write_text(json.dumps(checkpoint_baseline, indent=2, sort_keys=True) + "\n")
    source_root = REPO / "workbench/third_party/DCNet"
    source_files = ("test.py", "parse_config.py", "data_loader/CE_dataset.py",
                    "data_loader/data_loaders.py", "trainer/trainer_joint.py",
                    "model/model.py", "utils/util.py")
    source_baseline = {str(source_root / name): _sha256(source_root / name)
                       for name in source_files}
    record("protected_source_baseline", source_hashes=source_baseline)
    (root / "source-baseline.json").write_text(json.dumps(
        source_baseline, indent=2, sort_keys=True) + "\n")
    artifacts = root / "artifacts"
    artifacts.mkdir()
    (artifacts / "assets").symlink_to(REPO / "workbench/artifacts/assets", target_is_directory=True)
    checkpoint_root = artifacts / "checkpoint_selection/20261106T002617+0700/cohort/downloads"
    # Registered checkpoint metadata resolves through ../../checkpoints; its parent must exist.
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    staged_checkpoint = artifacts / "checkpoint_selection/20261106T002617+0700/checkpoints/dcnet"
    staged_checkpoint.mkdir(parents=True)
    for name in ("config.json", "trained_model.pth"):
        shutil.copyfile(checkpoint / name, staged_checkpoint / name)
        if _sha256(checkpoint / name) != _sha256(staged_checkpoint / name):
            raise RuntimeError(f"staged DCNet checkpoint digest mismatch: {name}")
    staged_att = staged_checkpoint / "val" / "att_result.pt"
    staged_att.parent.mkdir()
    shutil.copyfile(checkpoint / "val" / "att_result.pt", staged_att)
    os.environ.update({
        "CIR_REPO_ROOT": str(REPO), "CIR_DATA_ROOT": str(REPO / "data"),
        "FASHIONIQ_ROOT": str(REPO / "data/FashionIQ"),
        "WORKBENCH_ARTIFACTS_ROOT": str(artifacts),
        "WORKBENCH_CHECKPOINT_ROOT": str(checkpoint_root),
        "WORKBENCH_RESULTS_ROOT": str(artifacts / "results"),
        "WORKBENCH_THIRD_PARTY_ROOT": str(REPO / "workbench/third_party"),
        "WORKBENCH_CIRCUS_RELEASE_ROOT": str(REPO / "workbench/artifacts/circus_release"),
    })

    probe = subprocess.run([sys.executable, str(REPO / "workbench/scripts/evaluate_models.py"),
                            "--model", MODEL, "--checkpoint", CHECKPOINT,
                            "--verify-gpu-only"], cwd=REPO, text=True, capture_output=True)
    (root / "gpu-probe.stdout.log").write_text(probe.stdout)
    (root / "gpu-probe.stderr.log").write_text(probe.stderr)
    if probe.returncode:
        raise RuntimeError(f"authorized GPU capability probe failed: exit={probe.returncode}; see {root}")

    from workbench.backend import main
    from workbench.backend.jobs import SingleGpuQueue
    import uvicorn

    main.JOBS.shutdown()
    queue = SingleGpuQueue()
    main.JOBS = queue
    server = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=args.port,
                                           log_level="warning", lifespan="off"))
    server_thread = threading.Thread(target=server.run, name="isolated-api", daemon=True)
    server_thread.start()
    actor = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--actor",
                              "--run-root", str(root), "--port", str(args.port),
                              "--timeout", str(args.timeout)], cwd=REPO,
                             start_new_session=True,
                             stdout=(root / "cancel-actor.stdout.log").open("w"),
                             stderr=(root / "cancel-actor.stderr.log").open("w"))
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not (root / "cancel-actor.ready").is_file():
            if actor.poll() is not None:
                raise RuntimeError(f"cancel actor exited before ready: {actor.returncode}")
            time.sleep(.1)
        ready_path = root / "cancel-actor.ready"
        if not ready_path.is_file():
            raise TimeoutError("independent cancellation actor not ready")
        actor_identity = json.loads(ready_path.read_text())
        if actor_identity["pgid"] == os.getpgid(0):
            raise RuntimeError(f"cancel actor shares controller group: {actor_identity}")
        deadline = time.monotonic() + 30
        while not server.started and time.monotonic() < deadline:
            time.sleep(.1)
        if not server.started:
            raise TimeoutError("isolated ASGI server did not start")

        request_started = time.monotonic()
        (root / "http-request.json").write_text(json.dumps(PAYLOAD, indent=2) + "\n")
        try:
            status, response = _request(args.port, "POST", "/api/jobs", PAYLOAD,
                                        timeout=args.timeout)
        except Exception as error:
            (root / "http-request-timing.json").write_text(json.dumps(
                {"timeout_seconds": args.timeout,
                 "elapsed_seconds": time.monotonic() - request_started,
                 "error": f"{type(error).__name__}: {error}"}, indent=2) + "\n")
            raise
        (root / "http-response.json").write_text(
            json.dumps({"status": status, "body": response}, indent=2, sort_keys=True) + "\n")
        (root / "http-request-timing.json").write_text(json.dumps(
            {"timeout_seconds": args.timeout,
             "elapsed_seconds": time.monotonic() - request_started,
             "status": status}, indent=2) + "\n")
        if status != 200 or not isinstance(response, dict) or not response.get("job_id"):
            raise RuntimeError(f"canonical POST refused: {status} {response}")
        job_id = str(response["job_id"])
        job_root = Path(response["log_directory"])
        state_path = job_root / "queue_state.json"
        run_id = str(response["provenance"]["run_id"])
        run_dir = artifacts / "logs" / f"dcnet__{CHECKPOINT}__{run_id}"
        child_marker = run_dir / "dcnet-native-child.json"
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline and not child_marker.is_file():
            state = json.loads(state_path.read_text()) if state_path.is_file() else {}
            if state.get("status") in {"failed", "completed", "cancelled"}:
                raise RuntimeError(f"job terminal before native child: {state}")
            time.sleep(.2)
        if not child_marker.is_file():
            raise TimeoutError(f"native child did not start; queue={job_root}")
        child = json.loads(child_marker.read_text())
        child_pid = int(child["pid"])
        queue_process = queue._processes[job_id]
        pgid = os.getpgid(child_pid)
        if os.getpgid(queue_process.pid) != pgid:
            raise RuntimeError(f"native child escaped queue group: child={child_pid}, queue={queue_process.pid}")
        if actor_identity["pgid"] == pgid or actor_identity["pid"] in {child_pid, queue_process.pid}:
            raise RuntimeError(f"cancel actor overlaps job group: {actor_identity}")
        (root / "process-inventory.json").write_text(json.dumps(
            {"queue_pid": queue_process.pid, "native_pid": child_pid, "job_pgid": pgid,
             "cancel_actor": actor_identity}, indent=2, sort_keys=True) + "\n")
        gpu_deadline = time.monotonic() + args.timeout
        gpu_apps = ""
        while time.monotonic() < gpu_deadline:
            gpu_apps = _gpu_apps()
            gpu_pids = [int(line.split(",", 1)[0].strip()) for line in gpu_apps.splitlines()
                        if line.split(",", 1)[0].strip().isdigit()]
            if any(_pid_in_group(pid, pgid) for pid in gpu_pids):
                break
            current = json.loads(state_path.read_text())
            if current.get("status") in {"failed", "completed", "cancelled"}:
                raise RuntimeError(f"job ended before owned GPU activity was observed: {current}")
            time.sleep(.5)
        else:
            raise TimeoutError(f"DCNet process never appeared in nvidia-smi: {gpu_apps}")
        (root / "gpu-process-observed.txt").write_text(gpu_apps + "\n")
        (root / "cancel-request.json").write_text(json.dumps(
            {"job_id": job_id, "path": f"/api/jobs/{job_id}/cancel"}) + "\n")
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not (root / "cancel-response.json").is_file():
            if actor.poll() is not None:
                raise RuntimeError(f"cancel actor exited: {actor.returncode}")
            time.sleep(.1)
        if not (root / "cancel-response.json").is_file():
            raise TimeoutError("HTTP cancel response not persisted")
        cancel = json.loads((root / "cancel-response.json").read_text())
        if cancel["http_status"] != 200 or cancel["body"].get("status") != "cancelled":
            raise RuntimeError(f"HTTP cancellation refused: {cancel}")
        state = {}
        while time.monotonic() < deadline:
            state = json.loads(state_path.read_text())
            if state.get("status") in {"cancelled", "failed", "completed"}:
                break
            time.sleep(.2)
        if state.get("status") != "cancelled":
            raise RuntimeError(f"job did not persist CANCELLED: {state}")
        api_state = _api_job(args.port, job_id)
        if api_state.get("status") != "cancelled" or api_state.get("status") != state.get("status"):
            raise RuntimeError(f"GET /api/jobs terminal state disagrees with queue backing file: {api_state} / {state}")
        record("cancel_terminal_verified", api_state=api_state, backing_state=state,
               queue_state_path=str(state_path), process_group=pgid)
        survivors = _group_members(pgid)
        if survivors:
            raise RuntimeError(f"owned process group survived HTTP cancellation: {survivors}")
        checkpoint = Path("/home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints/dcnet")
        protected_files = [checkpoint / "config.json", checkpoint / "trained_model.pth",
                           checkpoint / "val" / "att_result.pt"]
        staged_config = run_dir / "dcnet-native-resume" / "config.json"
        stage_weight = run_dir / "dcnet-native-resume" / "trained_model.pth"
        staged_digests = {str(path): _sha256(path) for path in (staged_config, stage_weight)}
        if staged_digests[str(staged_config)] != checkpoint_baseline[str(checkpoint / "config.json")]:
            raise RuntimeError("staged DCNet config differs from protected source")
        if staged_digests[str(stage_weight)] != checkpoint_baseline[str(checkpoint / "trained_model.pth")]:
            raise RuntimeError("staged DCNet weights differ from protected source")
        if stage_weight.is_file():
            stage_weight.unlink()
        final_checkpoint_hashes = {str(path): _sha256(path) for path in protected_files}
        if final_checkpoint_hashes != checkpoint_baseline:
            raise RuntimeError(f"protected registered DCNet files changed: {final_checkpoint_hashes}")
        final_source_hashes = {str(source_root / name): _sha256(source_root / name)
                               for name in source_files}
        if final_source_hashes != source_baseline:
            raise RuntimeError(f"protected DCNet evaluator sources changed: {final_source_hashes}")
        staging_manifest = run_dir / "dcnet-native-staging.json"
        if not staged_config.is_file() or not staging_manifest.is_file():
            raise RuntimeError("native staging manifest/config missing after cancel cleanup")
        record("cancel_cleanup_verified", staged_config=str(staged_config),
               staging_manifest=json.loads(staging_manifest.read_text()),
               full_evaluator_output_observed=(run_dir / "dcnet-native-resume" / "val" / "att_result.pt").is_file())
        # Submit a fresh canonical API retry after cancellation cleanup; observe its persisted and API state.
        retry_status, retry_response = _request(args.port, "POST", "/api/jobs", PAYLOAD)
        (root / "retry-http-response.json").write_text(json.dumps(
            {"status": retry_status, "body": retry_response}, indent=2, sort_keys=True) + "\n")
        if retry_status != 200 or not isinstance(retry_response, dict) or not retry_response.get("job_id"):
            raise RuntimeError(f"canonical retry POST refused: {retry_status} {retry_response}")
        retry_id = str(retry_response["job_id"])
        retry_state_path = Path(retry_response["log_directory"]) / "queue_state.json"
        record("canonical_retry_admitted", job_id=retry_id, http_status=retry_status,
               queue_state_path=str(retry_state_path))
        retry_state = {}
        retry_deadline = time.monotonic() + args.timeout
        while time.monotonic() < retry_deadline:
            retry_api = _api_job(args.port, retry_id)
            retry_state = json.loads(retry_state_path.read_text()) if retry_state_path.is_file() else {}
            if retry_state.get("status") in {"completed", "failed", "blocked", "cancelled"}:
                break
            time.sleep(1)
        else:
            raise TimeoutError(f"canonical retry did not reach terminal state: {retry_state}")
        if retry_api.get("status") != retry_state.get("status"):
            raise RuntimeError(f"retry API/backing state mismatch: {retry_api} / {retry_state}")
        retry_pgid = queue._process_groups.get(retry_id)
        retry_survivors = _group_members(retry_pgid) if retry_pgid else []
        if retry_survivors:
            raise RuntimeError(f"canonical retry left process-group members: {retry_survivors}")
        record("canonical_retry_terminal", api_state=retry_api, backing_state=retry_state,
               queue_state_path=str(retry_state_path), process_group=retry_pgid,
               process_group_members=retry_survivors)
        report = {"scope": "canonical POST API admission, HTTP cancel, queue-state reconciliation, canonical retry terminal state", "guard": guard, "probe_exit": probe.returncode, "probe_stdout": probe.stdout,
                  "request": PAYLOAD, "admission": {"http_status": status, "body": response},
                  "cancel": cancel, "terminal_state": state, "queue_state": str(state_path),
                  "process_group": {"pgid": pgid, "members_after_cancel": survivors},
                  "retry": {"http_status": retry_status, "api_state": retry_api,
                            "backing_state": retry_state, "queue_state": str(retry_state_path)},
                  "protected_original_files": {str(path): final_checkpoint_hashes[str(path)]
                                                for path in protected_files},
                  "staged_weight": {"path": str(stage_weight),
                                    "sha256_before_reclaim": staged_digests[str(stage_weight)],
                                    "reclaimed_after_reap": not stage_weight.exists()},
                  "run_directory": str(run_dir)}
        target = root / "canonical-api-lifecycle-report.json"
        report["retry"]["http_status"] = retry_status
        report["retry"]["api_state"] = retry_api
        report["retry"]["backing_state"] = retry_state
        report["retry"]["queue_state"] = str(retry_state_path)
        report["retry"]["process_group"] = retry_pgid
        report["retry"]["process_group_members"] = retry_survivors
        target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        record("lifecycle_report_written", report=str(target))
        return 0
    except Exception as error:
        record("controller_failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (root / "actor.stop").write_text("done\n")
        if actor.poll() is None:
            actor.wait(timeout=10)
        server.should_exit = True
        server_thread.join(timeout=10)
        queue.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=19187)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--actor", action="store_true")
    args = parser.parse_args()
    if args.actor:
        return _actor(args.run_root, args.port, args.timeout)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
