from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tomllib
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workbench.backend.operator_config import WorkbenchConfig


def _read_requirements(path: Path) -> dict[str, str]:
    requirements: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for sep in (">=", "==", "<=", "~="):
            if sep in line:
                name, spec = line.split(sep, 1)
                requirements[name.strip().lower()] = f"{sep}{spec.strip()}"
                break
        else:
            requirements[line.strip().lower()] = ""
    return requirements


def _has_main_guard(tree: ast.AST) -> bool:
    """Structurally check if AST contains `if __name__ == '__main__':` guard."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
            left, right = test.left, test.comparators[0]
            if (isinstance(left, ast.Name) and left.id == "__name__"
                    and isinstance(right, ast.Constant) and right.value == "__main__"):
                return True
            if (isinstance(right, ast.Name) and right.id == "__name__"
                    and isinstance(left, ast.Constant) and left.value == "__main__"):
                return True
    return False


def _scripts_third_party_imports() -> set[str]:
    """Derive direct third-party imports from all workbench/scripts/*.py via AST."""
    scripts_dir = ROOT / "workbench" / "scripts"
    stdlib = sys.stdlib_module_names
    imported = set()
    for p in scripts_dir.glob("*.py"):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    imported.add(node.module.split(".")[0])
    return {m for m in imported if m not in stdlib and m not in {"workbench", "src"}}


def test_controller_manifest_declares_direct_dependencies():
    """Manifest declaration check: requirements.txt and pyproject.toml declare required direct dependencies."""
    req_path = ROOT / "workbench" / "backend" / "requirements.txt"
    assert req_path.is_file(), f"missing requirements manifest: {req_path}"
    reqs = _read_requirements(req_path)

    # Controller manifest must carry packaging and huggingface-hub for CLI and acquisition
    assert "packaging" in reqs, "packaging must be declared in workbench/backend/requirements.txt"
    assert "huggingface-hub" in reqs or "huggingface_hub" in reqs, (
        "huggingface-hub must be declared in workbench/backend/requirements.txt for controller acquisition"
    )

    pyproject_path = ROOT / "pyproject.toml"
    assert pyproject_path.is_file(), f"missing pyproject.toml: {pyproject_path}"
    parsed_pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    raw_deps = parsed_pyproject.get("project", {}).get("dependencies", [])
    project_deps = {dep.split()[0].lower() for dep in raw_deps}

    assert "packaging" in project_deps, "packaging must be declared in root pyproject.toml dependencies"
    for root_dep in ("torch", "torchvision", "numpy", "pillow"):
        assert root_dep in project_deps, f"{root_dep} must be declared in root pyproject.toml dependencies"


def test_doctor_workbench_checks_surface_all_required_imports(tmp_path: Path, monkeypatch):
    """Unit test of reporting path: doctor reports missing modules by name, and covers scripts dependencies."""
    spec = importlib.util.spec_from_file_location("doctor", ROOT / "workbench" / "scripts" / "doctor.py")
    assert spec and spec.loader
    doctor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(doctor)

    cfg = WorkbenchConfig(
        ROOT, ROOT / "data", ROOT / "FashionIQ", "127.0.0.1", 8000, 5173,
        tmp_path / "checkpoints", tmp_path / "results", tmp_path / "third_party"
    )

    # Behavior 1: When all specs resolve, all import checks report OK
    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda _: object())
    checks = doctor.workbench_checks(cfg)
    import_checks = {c["name"]: c for c in checks if c["name"].startswith("import:")}
    checked_module_names = {c["name"].split(":", 1)[1] for c in import_checks.values()}

    # Assert doctor checks cover all direct third-party modules imported by workbench/scripts/*.py
    script_deps = _scripts_third_party_imports()
    assert script_deps.issubset(checked_module_names), (
        f"doctor.workbench_checks does not check all third-party dependencies imported by workbench/scripts: "
        f"{script_deps - checked_module_names}"
    )

    expected_modules = {
        "fastapi", "uvicorn", "duckdb", "pydantic", "yaml",
        "packaging", "huggingface_hub", "numpy", "PIL", "torch",
    }
    assert set(import_checks.keys()) == {f"import:{m}" for m in expected_modules}
    assert all(c["status"] == "OK" for c in import_checks.values())

    # Behavior 2: When delayed imports are absent, doctor surfaces them by name as BLOCKED
    def fake_find_spec(name: str):
        if name in {"torch", "huggingface_hub"}:
            return None
        return object()

    monkeypatch.setattr(doctor.importlib.util, "find_spec", fake_find_spec)
    missing_checks = doctor.workbench_checks(cfg)
    missing_map = {c["name"]: c for c in missing_checks if c["name"].startswith("import:")}

    assert missing_map["import:torch"]["status"] == "BLOCKED"
    assert missing_map["import:torch"]["detail"] == "missing"
    assert missing_map["import:torch"]["evidence"]["module"] == "torch"

    assert missing_map["import:huggingface_hub"]["status"] == "BLOCKED"
    assert missing_map["import:huggingface_hub"]["detail"] == "missing"
    assert missing_map["import:huggingface_hub"]["evidence"]["module"] == "huggingface_hub"


def test_pipeline_command_uses_controller_interpreter():
    """pipeline.command() executes scripts under the controller sys.executable."""
    from workbench.scripts import pipeline

    cmd = pipeline.command("download_auxiliary_assets.py", "--model", "tme")
    assert cmd[0] == sys.executable
    assert cmd[1] == str(pipeline.SCRIPTS / "download_auxiliary_assets.py")
    assert cmd[2:] == ["--model", "tme"]


def test_controller_cli_entrypoints_execute_help():
    """Every controller script with a main guard must start cleanly and accept --help under the controller interpreter."""
    scripts_dir = ROOT / "workbench" / "scripts"
    cli_scripts = []
    for p in sorted(scripts_dir.glob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        if _has_main_guard(tree):
            cli_scripts.append(p)

    assert len(cli_scripts) >= 20, f"expected at least 20 CLI scripts, found {len(cli_scripts)}"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    failures: list[str] = []
    for script in cli_scripts:
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            failures.append(f"{script.name} (exit {result.returncode}): {result.stderr.strip()}")

    assert not failures, f"CLI entrypoints failed --help under controller interpreter:\n" + "\n".join(failures)


def test_lavis_processor_parity_cli_rejects_unknown_model_without_traceback():
    script = ROOT / "workbench/scripts/lavis_processor_parity.py"
    result = subprocess.run([sys.executable, str(script), "not-a-model"], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "Valid choices: airknow, conesep, habit, hint, intent" in result.stderr
    assert "Traceback" not in result.stderr


def test_lavis_processor_parity_valid_model_runs_native_comparison():
    script = ROOT / "workbench/scripts/lavis_processor_parity.py"
    result = subprocess.run([sys.executable, str(script), "hint"], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '"status": "LOADED"' in result.stdout
    assert '"loaded_via": "lavis/models/__init__.py:load_preprocess"' in result.stdout
