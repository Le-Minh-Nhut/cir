from __future__ import annotations

import ast
import argparse
import importlib.util
import inspect
import re
import shlex
import sys
from pathlib import Path

import pytest

from workbench.backend.operator_config import WorkbenchConfig
from workbench.backend.registry import load_registry, model_by_id


ROOT = Path(__file__).resolve().parents[2]
DOCS = tuple(ROOT / path for path in (
    "workbench/docs/PC_REPRODUCTION_HANDOFF.md",
    "workbench/docs/AUTOMATION_RUNBOOK.md",
    "workbench/README.md",
))
SCRIPT_DIR = ROOT / "workbench" / "scripts"
FIXTURE = Path("/fixture")


class ParsedArguments(Exception):
    def __init__(self, namespace: argparse.Namespace):
        self.namespace = namespace


def _commands(path: Path):
    text = path.read_text(encoding="utf-8")
    for block in re.finditer(r"```bash\s*\n(.*?)```", text, re.DOTALL):
        pending = ""
        start = 0
        for number, line in enumerate(block.group(1).splitlines(), 1):
            stripped = line.strip()
            if not pending:
                start = text.count("\n", 0, block.start(1)) + number
            continuation = stripped.endswith("\\")
            pending += (stripped[:-1] if continuation else stripped) + " "
            if continuation:
                continue
            lexer = shlex.shlex(pending, posix=True)
            lexer.commenters = "#"
            lexer.whitespace_split = True
            tokens = list(lexer)
            if tokens:
                yield start, shlex.join(tokens)
            pending = ""
        if pending:
            lexer = shlex.shlex(pending, posix=True)
            lexer.commenters = "#"
            lexer.whitespace_split = True
            tokens = list(lexer)
            if tokens:
                yield start, shlex.join(tokens)


def _fixture_tokens(command: str) -> list[str]:
    placeholders = {"FashionIQ": str(FIXTURE / "FashionIQ"), "model": "csmcir",
                    "MODEL_ID": "csmcir", "MODEL": "csmcir", "PATH": str(FIXTURE),
                    "STAGE_NAME": "evaluation", "root": str(FIXTURE)}
    command = re.sub(r"<([^>]+)>", lambda match: placeholders.get(match.group(1), str(FIXTURE)), command)
    command = re.sub(r"\[([^\]]+)\]", r"\1", command)
    command = re.sub(r"\bMODEL_ID\b|\bMODEL\b", "csmcir", command)
    for variable, value in (("$CIR_REPO_ROOT", str(FIXTURE)),
                            ("$FASHIONIQ_ROOT", str(FIXTURE / "FashionIQ")),
                            ("$WORKBENCH_RESULTS_ROOT", str(FIXTURE / "results"))):
        command = command.replace(variable, value)
    return shlex.split(command)


def _load_script(name: str):
    path = SCRIPT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"handoff_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_real_parser(script: str, argv: list[str], monkeypatch) -> argparse.Namespace:
    module = _load_script(script)
    original_parse_args = argparse.ArgumentParser.parse_args

    def capture(parser, args=None, namespace=None):
        raise ParsedArguments(original_parse_args(parser, args, namespace))

    with monkeypatch.context() as patch:
        patch.setattr(argparse.ArgumentParser, "parse_args", capture)
        try:
            with pytest.raises(ParsedArguments) as parsed:
                if "argv" in inspect.signature(module.main).parameters:
                    module.main(argv)
                else:
                    patch.setattr(sys, "argv", [str(SCRIPT_DIR / f"{script}.py"), *argv])
                    module.main()
        except SystemExit as exit_code:
            # `--help` prints and exits 0 before parse_args returns, and a diagnostic like
            # `--status` exits 1 when no state exists yet. Both mean the parser accepted the
            # command; argparse signals a genuine usage error with 2, which must still fail.
            if exit_code.code in (0, 1):
                return None
            raise
    return parsed.value.namespace


def _worst_case_capabilities(node: ast.AST) -> set[str]:
    """Every capability the expression can require, not only the one branch taken at import time.

    A literal set is its own answer. A computed value (for example the evaluation stage, which needs
    preparation authority only when the model's dataset link still has to be created) is resolved by
    calling the real function with the case that requires the most, so the audit cannot pass a
    command that would be refused in the other branch.
    """
    try:
        return set(ast.literal_eval(node))
    except (ValueError, SyntaxError):
        pass
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_evaluation_capabilities":
        from workbench.scripts import pipeline as pipeline_module

        return set(pipeline_module._evaluation_capabilities(None, _fixture_config()))
    raise AssertionError(f"cannot determine declared capabilities from {ast.dump(node)[:120]}")


def _declared_stage_capabilities(name: str) -> set[str]:
    tree = ast.parse((SCRIPT_DIR / "pipeline.py").read_text(encoding="utf-8"))
    required: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "run_stage":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant) or node.args[0].value != name:
            continue
        keyword = next((item for item in node.keywords if item.arg == "required_capabilities"), None)
        if keyword:
            required.update(_worst_case_capabilities(keyword.value))
    return required


def _fixture_config() -> WorkbenchConfig:
    return WorkbenchConfig(FIXTURE, FIXTURE / "data", FIXTURE / "FashionIQ", "127.0.0.1", 8000, 5173,
                           FIXTURE / "checkpoints", FIXTURE / "results", FIXTURE / "third_party")


def test_every_documented_command_parses_and_pipeline_has_required_authorization(monkeypatch) -> None:
    registry = load_registry()
    model_ids = {model["model_id"] for model in registry["models"]}
    checked = 0
    non_cli = {"export", "tail", "cat"}

    for document in DOCS:
        for line, command in _commands(document):
            tokens = _fixture_tokens(command)
            location = f"{document.relative_to(ROOT)}:{line}: {command}"
            if not tokens:
                continue
            if tokens[0] in non_cli:
                continue
            if tokens[:2] == ["python", "-m"]:
                assert len(tokens) > 2 and tokens[2] in {"pytest", "compileall"}, location
                assert importlib.util.find_spec(tokens[2]) is not None, location
                continue
            if tokens[:2] == ["python", "-c"]:
                assert len(tokens) > 2, location
                continue
            script_at = next((index for index, token in enumerate(tokens)
                              if token.startswith("workbench/scripts/") and token.endswith(".py")), None)
            if script_at is not None:
                script = Path(tokens[script_at]).stem
                arguments = tokens[script_at + 1:]
            else:
                # `workbench/run.sh` is a dispatcher: `run.sh autopilot ...` runs scripts/autopilot.py
                # and anything else runs scripts/pipeline.py. The documented one-command workflow
                # goes through it, so resolve the verb to the script it execs and parse that script's
                # real parser rather than skipping the command.
                runner = next((index for index, token in enumerate(tokens)
                               if token.endswith("workbench/run.sh") or token == "workbench/run.sh"), None)
                assert runner is not None, f"not classified as CLI or non-command: {location}"
                verb = tokens[runner + 1] if len(tokens) > runner + 1 else ""
                script = "autopilot" if verb == "autopilot" else "pipeline"
                arguments = tokens[runner + 1:]
                if script == "pipeline":
                    arguments = tokens[runner + 1:]
                else:
                    arguments = tokens[runner + 2:]
            assert (SCRIPT_DIR / f"{script}.py").is_file(), location
            namespace = _parse_real_parser(script, arguments, monkeypatch)
            checked += 1
            if namespace is None:
                continue  # --help or a diagnostic that exits before returning a namespace

            if getattr(namespace, "model", None):
                assert namespace.model in model_ids, location
                model_by_id(namespace.model, registry)
            if script == "pipeline" and namespace.apply and namespace.mode == "reproduce":
                required = _declared_stage_capabilities("evaluation")
                assert required, "pipeline evaluation stage must declare its required capabilities"
                missing = sorted(cap for cap in required if not getattr(namespace, cap, False))
                assert not missing, f"{location} missing --{' --'.join(cap.replace('_', '-') for cap in missing)}"
    assert checked > 0
