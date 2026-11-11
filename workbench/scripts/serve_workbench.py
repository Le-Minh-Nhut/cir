#!/usr/bin/env python3
"""Launch workbench backend and frontend without installing dependencies."""
from __future__ import annotations

import argparse
import os
import shlex
import signal
import socket
import stat
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.operator_config import WorkbenchConfig, resolve_config

from typing import NamedTuple

WORKBENCH_ROOT = Path(__file__).resolve().parents[1]


class Command(NamedTuple):
    name: str
    argv: list[str]
    cwd: Path


def port_number(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def build_commands(
    root: Path,
    host: str,
    backend_port: int,
    frontend_port: int,
    backend: bool,
    frontend: bool,
    production_frontend: bool,
) -> list[Command]:
    commands: list[Command] = []
    if backend:
        # `python -m uvicorn`, not bare `uvicorn`: the workbench interpreter is the venv's, and a
        # child spawned from it cannot rely on the venv's bin directory being on PATH.
        commands.append(Command("backend", [sys.executable, "-m", "uvicorn", "workbench.backend.main:app", "--host", host, "--port", str(backend_port)], root.parent))
    if frontend:
        script = "preview" if production_frontend else "dev"
        commands.append(Command("frontend", ["npm", "run", script, "--", "--host", host, "--port", str(frontend_port), "--strictPort"], root / "frontend"))
    return commands


def command_port(command: Command) -> int:
    return int(command.argv[command.argv.index("--port") + 1])


def port_is_available(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind((host, port))
        except OSError:
            return False
    return True


def require_frontend_dependencies(root: Path, production_frontend: bool) -> str | None:
    frontend = root / "frontend"
    if not (frontend / "node_modules").is_dir():
        return f"frontend dependencies missing: {frontend / 'node_modules'} (run npm install yourself)"
    if production_frontend and not (frontend / "dist").is_dir():
        return f"production frontend missing: {frontend / 'dist'} (run npm run build yourself)"
    return None


def stop_children(children: list[subprocess.Popen[object]]) -> None:
    running = [child for child in children if child.poll() is None]
    for child in running:
        try:
            os.killpg(os.getpgid(child.pid), signal.SIGTERM)
        except (AttributeError, OSError):
            child.terminate()
    deadline = time.monotonic() + 5
    for child in running:
        try:
            child.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(child.pid), signal.SIGKILL)
            except (AttributeError, OSError):
                child.kill()
            child.wait()


#: A service that survives this long is considered healthy again and its backoff resets.
STABILITY_WINDOW = 30.0
BACKOFF_BASE = 0.2
BACKOFF_CAP = 8.0
#: The launcher appends to whatever file the operator redirected; rotate into a new segment.
LOG_ROTATE_BYTES = 16 * 1024 * 1024

class Service:
    def __init__(self, command: Command) -> None:
        self.command = command
        self.environment: dict[str, str] = {}
        self.process: subprocess.Popen[object] | None = None
        self.started = 0.0
        self.failures = 0
        self.restarts = 0
        self.restart_at = 0.0

def describe_exit(status: int) -> str:
    if status >= 0:
        return f"exit status {status}"
    try:
        return f"signal {signal.Signals(-status).name}"
    except ValueError:
        return f"signal {-status}"

def restart_delay(failures: int) -> float:
    """Capped exponential backoff: a service that dies on startup cannot spin."""
    return min(BACKOFF_BASE * 2 ** max(0, failures - 1), BACKOFF_CAP)

def regular_log_target(fd: int) -> str | None:
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return None
        return os.readlink(f"/proc/self/fd/{fd}")
    except OSError:
        return None

def rotation_path(path: Path, stamp: str) -> Path:
    candidate = path.with_name(f"{path.name}.{stamp}")
    index = 1
    while candidate.exists():
        index += 1
        candidate = path.with_name(f"{path.name}.{stamp}.{index}")
    return candidate

def rotate_logs(limit: int = LOG_ROTATE_BYTES) -> Path | None:
    """Start a new append-only segment once the inherited log file outgrows `limit`.

    The previous segment is never truncated or rewritten -- a rotation only changes where the
    supervisor and the children spawned afterwards write next.
    """
    target = regular_log_target(2)
    if target is None:
        return None
    try:
        if os.stat(target).st_size < limit:
            return None
    except OSError:
        return None
    path = rotation_path(Path(target), time.strftime("%Y%m%dT%H%M%S", time.localtime()))
    handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        for fd in (1, 2):
            if regular_log_target(fd) == target:
                os.dup2(handle, fd)
    finally:
        os.close(handle)
    return path

class Supervisor:
    """Restart children that exit unexpectedly, with capped exponential backoff.

    Each restart is logged with its timestamp, the exit signal and both PIDs; the children share
    the launcher's stdout/stderr, so their logs accumulate in the same append-only file across
    restarts.
    """

    def __init__(self, commands: list[Command],
                 stability_window: float = STABILITY_WINDOW, log_limit: int = LOG_ROTATE_BYTES) -> None:
        self.commands = commands
        self.stability_window = stability_window
        self.log_limit = log_limit
        self.services = [Service(command) for command in commands]
        self.stopping = False

    def log(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        print(f"[SUPERVISOR] {stamp} {message}", file=sys.stderr, flush=True)

    def start(self, service: Service) -> None:
        rotated = rotate_logs(self.log_limit)
        if rotated is not None:
            self.log(f"log rotated; appending to {rotated}")
        service.process = subprocess.Popen(
            service.command.argv,
            cwd=service.command.cwd,
            env=service.environment,
            start_new_session=True,
        )
        service.started = time.monotonic()
        self.log(f"{service.command.name} started PID {service.process.pid}")

    def check(self, now: float) -> None:
        """One supervision pass.

        A restart is never gated on the port being free: if something else holds it, the
        replacement exits at once, is logged, and the backoff keeps the retry from spinning.
        """
        for service in self.services:
            process = service.process
            if process is None:
                if now >= service.restart_at:
                    self.start(service)
                continue
            status = process.poll()
            if status is None:
                if service.failures and now - service.started >= self.stability_window:
                    self.log(f"{service.command.name} PID {process.pid} stable {now - service.started:.0f}s; backoff reset")
                    service.failures = 0
                continue
            self.reap(process)
            service.process = None
            service.failures += 1
            service.restarts += 1
            delay = restart_delay(service.failures)
            service.restart_at = now + delay
            self.log(
                f"{service.command.name} PID {process.pid} died at {time.strftime('%Y-%m-%dT%H:%M:%S%z')} "
                f"({describe_exit(status)}); restart {service.restarts} in {delay:.2f}s"
            )

    def reap(self, process: subprocess.Popen[object]) -> None:
        # `start_new_session=True` makes the child's pid its process-group id, so leftover group
        # members (npm's shell, vite, an orphaned uvicorn worker) are cleared out of the way
        # before the replacement binds the port.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass

    def stop(self) -> None:
        self.stopping = True
        stop_children([service.process for service in self.services if service.process is not None])

def parser(config: WorkbenchConfig) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--backend-only", action="store_true")
    mode.add_argument("--frontend-only", action="store_true")
    result.add_argument("--config", type=Path, default=None,
                        help="autopilot config YAML; its resolved paths override workbench.env "
                             "for the served backend, so the web, autopilot and readiness "
                             "reports describe the same artifacts")
    result.add_argument("--host", default=config.WORKBENCH_HOST)
    result.add_argument("--backend-port", type=port_number, default=config.WORKBENCH_BACKEND_PORT)
    result.add_argument("--frontend-port", type=port_number, default=config.WORKBENCH_FRONTEND_PORT)
    result.add_argument("--open-browser", action="store_true")
    result.add_argument("--production-frontend", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def main(argv: list[str] | None = None, root: Path = WORKBENCH_ROOT) -> int:
    args = parser(resolve_config()).parse_args(argv)
    backend = not args.frontend_only
    frontend = not args.backend_only
    if args.production_frontend and not frontend:
        print("[BLOCKED] --production-frontend requires frontend operation", file=sys.stderr)
        return 2
    if backend and frontend and args.backend_port == args.frontend_port:
        print("[BLOCKED] backend and frontend ports must differ", file=sys.stderr)
        return 2
    if frontend and not args.dry_run:
        problem = require_frontend_dependencies(root, args.production_frontend)
        if problem:
            print(f"[BLOCKED] {problem}", file=sys.stderr)
            return 2

    commands = build_commands(root, args.host, args.backend_port, args.frontend_port, backend, frontend, args.production_frontend)
    if not args.dry_run:
        for command in commands:
            port = command_port(command)
            if not port_is_available(args.host, port):
                print(f"[BLOCKED] {command.name} port unavailable: {args.host}:{port}", file=sys.stderr)
                return 2
    for command in commands:
        print(f"[RUN] ({command.name}) {shlex.join(command.argv)}")
    if args.dry_run:
        print("[SKIP] dry-run; no processes started")
        return 0

    backend_environment = dict(os.environ)
    if args.config is not None:
        # One resolved configuration: the served backend sees exactly the paths the
        # autopilot config names, so readiness reports, queued jobs and the web UI
        # cannot disagree about where artifacts live.
        from workbench.backend.autopilot import _workbench_config, load_config
        autopilot_config = load_config(args.config)
        resolved = _workbench_config(autopilot_config)
        backend_environment |= {
            "CIR_REPO_ROOT": str(resolved.CIR_REPO_ROOT),
            "FASHIONIQ_ROOT": str(resolved.FASHIONIQ_ROOT),
            "WORKBENCH_ARTIFACTS_ROOT": str(resolved.WORKBENCH_ARTIFACTS_ROOT),
            "WORKBENCH_CHECKPOINT_ROOT": str(resolved.WORKBENCH_CHECKPOINT_ROOT),
            "WORKBENCH_RESULTS_ROOT": str(resolved.WORKBENCH_RESULTS_ROOT),
            "WORKBENCH_THIRD_PARTY_ROOT": str(resolved.WORKBENCH_THIRD_PARTY_ROOT),
            "WORKBENCH_REGISTRY_PATH": str(autopilot_config.paths["registry"]),
        }
    frontend_environment = os.environ | {"WORKBENCH_BACKEND_URL": f"http://127.0.0.1:{args.backend_port}"}
    supervisor = Supervisor(commands=commands)
    for service in supervisor.services:
        service.environment = frontend_environment if service.command.name == "frontend" else backend_environment

    # `docker stop` sends SIGTERM to *this* launcher, not to the children: each child leads its
    # own session, so nothing would run the backend's shutdown handshake and a queued evaluator
    # would keep its GPU. Forward the stop as an ordinary SIGTERM to the backend, which then
    # reaps every live job's process group, then raise SIGINT so the loop below stops the
    # children as usual. `stopping` keeps the supervisor from restarting what it just stopped.
    def forward_stop(signum: int, _frame: object) -> None:
        supervisor.stopping = True
        for service in supervisor.services:
            process = service.process
            if process is not None and process.poll() is None:
                try:
                    os.kill(process.pid, signal.SIGTERM)
                except OSError:
                    pass
        signal.raise_signal(signal.SIGINT)

    for signum in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(signum, forward_stop)
        except (ValueError, OSError):
            pass
    try:
        for service in supervisor.services:
            supervisor.start(service)
        if args.open_browser:
            url = f"http://{args.host}:{args.frontend_port if frontend else args.backend_port}"
            webbrowser.open(url)
            print(f"[OK] opened {url}")
        while not supervisor.stopping:
            supervisor.check(time.monotonic())
            time.sleep(0.2)
        return 0
    except KeyboardInterrupt:
        print("[SKIP] interrupted; stopping children")
        return 130
    finally:
        supervisor.stop()


if __name__ == "__main__":
    raise SystemExit(main())
