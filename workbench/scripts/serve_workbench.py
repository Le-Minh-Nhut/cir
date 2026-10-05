#!/usr/bin/env python3
"""Launch workbench backend and frontend without installing dependencies."""
from __future__ import annotations

import argparse
import os
import shlex
import signal
import socket
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
        commands.append(Command("backend", ["uvicorn", "workbench.backend.main:app", "--host", host, "--port", str(backend_port)], root.parent))
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


def parser(config: WorkbenchConfig) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--backend-only", action="store_true")
    mode.add_argument("--frontend-only", action="store_true")
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

    children: list[subprocess.Popen[object]] = []
    try:
        for command in commands:
            environment = os.environ | ({"WORKBENCH_BACKEND_URL": f"http://127.0.0.1:{args.backend_port}"} if command.name == "frontend" else {})
            children.append(subprocess.Popen(command.argv, cwd=command.cwd, env=environment, start_new_session=True))
            print(f"[OK] started {command.name}")
        if args.open_browser:
            url = f"http://{args.host}:{args.frontend_port if frontend else args.backend_port}"
            webbrowser.open(url)
            print(f"[OK] opened {url}")
        while True:
            for child in children:
                status = child.poll()
                if status is not None:
                    print(f"[WARN] {child.args[0]} exited with status {status}", file=sys.stderr)
                    return status or 1
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("[SKIP] interrupted; stopping children")
        return 130
    finally:
        stop_children(children)


if __name__ == "__main__":
    raise SystemExit(main())
