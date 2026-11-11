#!/usr/bin/env python3
"""Render, install and inspect the cir-workbench systemd *user* units.

The two units run the same server the supervised launcher runs -- same interpreter, same working
directory, same host/port, same `WORKBENCH_CIRCUS_RELEASE_ROOT` -- but with systemd as the
supervisor. Every path and port is read from this repo's own resolver
(`workbench.backend.operator_config.resolve_config`) and from `build_commands` in
`workbench/scripts/serve_workbench.py`, so nothing here hand-copies a path that the launcher
would derive differently.

Install, dry-run, inspect:

    workbench/envs/ctl312h/bin/python workbench/systemd/render_units.py --dry-run
    workbench/envs/ctl312h/bin/python workbench/systemd/render_units.py --install
    systemctl --user daemon-reload
    systemctl --user enable --now cir-workbench-backend.service cir-workbench-frontend.service
    workbench/envs/ctl312h/bin/python workbench/systemd/render_units.py --verify

`--install` writes into `${XDG_CONFIG_HOME:-~/.config}/systemd/user/` and reloads the manager.
`--dry-run` writes nothing. `--verify` shells out to `systemctl --user show` and `journalctl`.
Ports/host default to the repo config, so a test deployment must pass `--backend-port`,
`--frontend-port` and `--set` overrides; the tests in
`workbench/tests/test_systemd_user_services.py` do exactly that against ports 8401/5401.
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.operator_config import WorkbenchConfig, resolve_config
from workbench.scripts.serve_workbench import build_commands

HERE = Path(__file__).resolve().parent
DEFAULT_BACKEND_UNIT = "cir-workbench-backend"
DEFAULT_FRONTEND_UNIT = "cir-workbench-frontend"

#: Every setting the served backend must see explicitly. The unit inherits no shell environment,
#: so the resolved values are rendered into `Environment=` instead of re-derived inside the child.
BACKEND_ENV_KEYS = (
    "CIR_REPO_ROOT",
    "CIR_DATA_ROOT",
    "FASHIONIQ_ROOT",
    "WORKBENCH_ARTIFACTS_ROOT",
    "WORKBENCH_CHECKPOINT_ROOT",
    "WORKBENCH_RESULTS_ROOT",
    "WORKBENCH_THIRD_PARTY_ROOT",
    "WORKBENCH_CIRCUS_RELEASE_ROOT",
)


def unit_dir() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "systemd" / "user"


def backend_environment(config: WorkbenchConfig, extra: dict[str, str]) -> dict[str, str]:
    values = {key: str(getattr(config, key)) for key in BACKEND_ENV_KEYS}
    values.update(extra)
    return values


def frontend_environment(backend_port: int, npm: str) -> dict[str, str]:
    # No shell: systemd's own PATH does not contain the nvm node that owns `npm`, so the node
    # directory the launcher inherited from the operator's shell is rendered in explicitly.
    node_bin = str(Path(npm).parent)
    return {
        "WORKBENCH_BACKEND_URL": f"http://127.0.0.1:{backend_port}",
        "PATH": f"{node_bin}:/usr/local/bin:/usr/bin:/bin",
    }


def environment_lines(values: dict[str, str]) -> str:
    return "\n".join(f'Environment="{key}={value}"' for key, value in sorted(values.items()))


def render(
    config: WorkbenchConfig,
    *,
    host: str,
    backend_port: int,
    frontend_port: int,
    backend_unit: str = DEFAULT_BACKEND_UNIT,
    frontend_unit: str = DEFAULT_FRONTEND_UNIT,
    python: str | None = None,
    npm: str | None = None,
    extra_env: dict[str, str] | None = None,
    production_frontend: bool = True,
) -> dict[str, str]:
    """Return `{unit_file_name: contents}` for both units. Pure: reads templates, touches nothing."""
    python = python or sys.executable
    npm = npm or shutil.which("npm") or "npm"
    extra_env = dict(extra_env or {})

    root = config.CIR_REPO_ROOT / "workbench"
    commands = {command.name: command for command in build_commands(
        root, host, backend_port, frontend_port, True, True, production_frontend)}

    backend = commands["backend"]
    frontend = commands["frontend"]
    # `build_commands` builds the backend argv around `sys.executable`; the caller may name a
    # different interpreter, and the frontend argv starts with the bare `npm` the launcher relied
    # on finding on PATH.
    backend_argv = [python, *backend.argv[1:]]
    frontend_argv = [npm, *frontend.argv[1:]]

    renders = {
        f"{backend_unit}.service": (
            HERE / "cir-workbench-backend.service.in",
            f"CIR workbench backend (uvicorn on {host}:{backend_port})",
            backend,
            backend_argv,
            backend_environment(config, extra_env),
        ),
        f"{frontend_unit}.service": (
            HERE / "cir-workbench-frontend.service.in",
            f"CIR workbench frontend (vite preview on {host}:{frontend_port})",
            frontend,
            frontend_argv,
            frontend_environment(backend_port, npm),
        ),
    }
    output: dict[str, str] = {}
    for name, (template_path, description, command, argv, environment) in renders.items():
        text = template_path.read_text(encoding="utf-8")
        text = text.replace("@DESCRIPTION@", description)
        text = text.replace("@EXEC_START@", shlex.join(argv))
        text = text.replace("@WORKING_DIRECTORY@", str(command.cwd))
        text = text.replace("@BACKEND_UNIT@", backend_unit)
        text = text.replace("@ENVIRONMENT@", environment_lines(environment))
        output[name] = text
    return output


def install(units: dict[str, str], directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in units.items():
        path = directory / name
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written


def verify(unit_names: list[str]) -> int:
    status = 0
    for name in unit_names:
        for command in (
            ["systemctl", "--user", "show", "-p", "MainPID", "-p", "ControlGroup",
             "-p", "ExecMainStartTimestamp", "-p", "KillMode", "--value", name],
            ["journalctl", "--user", "-u", name, "--no-pager", "-n", "10"],
        ):
            print(f"$ {shlex.join(command)}")
            result = subprocess.run(command, text=True, capture_output=True)
            print(result.stdout.rstrip() or result.stderr.rstrip())
            status = status or result.returncode
    return status


def main(argv: list[str] | None = None) -> int:
    config = resolve_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=config.WORKBENCH_HOST)
    parser.add_argument("--backend-port", type=int, default=config.WORKBENCH_BACKEND_PORT)
    parser.add_argument("--frontend-port", type=int, default=config.WORKBENCH_FRONTEND_PORT)
    parser.add_argument("--backend-unit", default=DEFAULT_BACKEND_UNIT)
    parser.add_argument("--frontend-unit", default=DEFAULT_FRONTEND_UNIT)
    parser.add_argument("--python", default=sys.executable, help="interpreter for uvicorn")
    parser.add_argument("--npm", default=shutil.which("npm"), help="npm executable path")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="extra Environment= for the backend (isolated roots, test overrides)")
    parser.add_argument("--dev-frontend", action="store_true", help="run `vite` instead of `vite preview`")
    parser.add_argument("--directory", type=Path, default=unit_dir())
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="print rendered units, write nothing")
    mode.add_argument("--install", action="store_true", help="write units and daemon-reload")
    mode.add_argument("--verify", action="store_true", help="show MainPID/ControlGroup and journal")
    args = parser.parse_args(argv)

    extra_env: dict[str, str] = {}
    for item in args.set:
        key, separator, value = item.partition("=")
        if not separator:
            parser.error(f"--set expects KEY=VALUE, got {item!r}")
        extra_env[key] = value

    units = render(config, host=args.host, backend_port=args.backend_port,
                   frontend_port=args.frontend_port, backend_unit=args.backend_unit,
                   frontend_unit=args.frontend_unit, python=args.python, npm=args.npm,
                   extra_env=extra_env, production_frontend=not args.dev_frontend)
    names = list(units)

    if args.dry_run:
        for name, text in units.items():
            print(f"### {args.directory / name}")
            print(text, end="" if text.endswith("\n") else "\n")
        return 0
    if args.verify:
        return verify([f"{name}" for name in names])

    written = install(units, args.directory)
    for path in written:
        print(f"[WRITE] {path}")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    print("[NEXT] systemctl --user enable --now " + " ".join(shlex.quote(name) for name in names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
