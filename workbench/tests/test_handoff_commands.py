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
    # The two documents the operator is actually handed for the RTX 5070 Ti. They were outside this
    # audit, and a command that had never been runnable survived in both: `--verify-only` was
    # documented in remedies for work it does not do, `verify_run.py` was shown without three of
    # its four required flags, and `limn_export` was shown with a flag it does not accept and
    # without the publisher stage that makes its output an artifact. None of that is visible
    # without executing the commands.
    "workbench/docs/PC_PILOT_HANDOFF.md",
    "workbench/docs/PC_ONE_COMMAND_GUIDE.vi.md",
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


def test_every_document_in_the_handoff_set_has_run_commands_or_says_why_not() -> None:
    """The audited set must be the set the operator is handed, not a subset chosen by hand.

    This test exists because the two documents the RTX 5070 Ti operator actually receives were
    outside the audit, and both contained commands that could not run. A curated `DOCS` tuple is
    exactly the shape of that defect: adding a document is silent. Everything under
    `workbench/docs/` that carries a bash block, plus the repository README, must therefore either
    be audited or be named in `DOCS_EXEMPT` with a reason.
    """
    exempt = {
        "AUTOMATED_DATA_PREPARATION.md", "AUTOPILOT_GUIDE.md", "AUXILIARY_ASSET_AUDIT.md",
        "COT_OURS2_AUDIT.md", "ENGINEERING_FINAL_ACCEPTANCE.md", "FASHIONIQ_PROTOCOL_PARITY.md",
        "FINAL_PC_READINESS.md", "FIRST_GPU_EXPERIMENT.md", "LAPTOP_RELEASE_ACCEPTANCE.md",
        "LEGACY_CLEANUP_AUDIT.md", "LEGACY_REPRODUCTION_RUNBOOK.md", "LEGACY_SOURCE_AUDIT.md",
        "PC_DOCKER_QUICKSTART.md", "PC_TRANSFER_CHECKLIST.md", "PROTOCOL_AUDIT.md",
        "READINESS_MATRIX.md", "RESULT_SCHEMA.md", "UPSTREAM_SOURCE_AUDIT.md",
    }
    audited = {path.name for path in DOCS}
    unclassified = []
    for document in sorted((ROOT / "workbench" / "docs").glob("*.md")):
        if document.name in audited or document.name in exempt:
            continue
        text = document.read_text(encoding="utf-8")
        if "```bash" in text or "```sh" in text:
            unclassified.append(document.name)
    assert not unclassified, (
        "these documents contain shell blocks but are neither audited nor explicitly exempt: "
        f"{unclassified}. Add them to DOCS so their commands are executed, or to the exempt set "
        "with a reason.")


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
            # `docker exec cir-workbench python3 <script>` is a real command of this deployment,
            # and pc.sh uses the same form. The container is the only place the ledger key, the
            # model environments and the canonical results root exist, so a documented verification
            # has to go through it -- refusing the form would push the documentation back to
            # commands that cannot work.
            if tokens[0] == "docker" and "exec" in tokens[:3]:
                exec_at = tokens.index("exec")
                inner = tokens[exec_at + 1:]
                while inner and inner[0].startswith("-") or (inner and inner[0] == "cir-workbench"):
                    inner = inner[1:]
                if inner and inner[0].endswith("python3"):
                    tokens = ["python", *inner[1:]]
                else:
                    continue
            # `curl` and `docker <verb>` are documented operator actions with no Python parser; what
            # must hold is that the endpoint or subcommand they name is the one the code serves.
            if tokens[0] == "curl":
                assert "http://127.0.0.1:8000/api/" in " ".join(tokens), location
                continue
            if tokens[0] == "docker" and "exec" not in tokens[:3]:
                continue
            if tokens[0] in {"git", "sudo", "ls", "cd", "workbench/docker/run.sh"}:
                continue
            if tokens[0] == "./pc.sh":
                # The launcher's own flags are checked by the launcher suite, against the real
                # script; here only that the flag the document names exists at all.
                flags = [token for token in tokens[1:] if token.startswith("--")]
                help_text = (ROOT / "pc.sh").read_text(encoding="utf-8")
                for flag in flags:
                    assert re.search(rf"^\s*{re.escape(flag)}\)", help_text, re.M), \
                        f"{location}: pc.sh does not accept {flag}"
                continue
            if tokens[:2] == ["python", "-m"]:
                assert len(tokens) > 2, location
                module_path = tokens[2]
                if module_path in {"pytest", "compileall"}:
                    assert importlib.util.find_spec(module_path) is not None, location
                    continue
                # A documented `python -m workbench.<module>` must name a module that exists and
                # whose flags the module's own parser accepts -- the same check the `workbench/
                # scripts/*.py` form gets. Skipping it is how `verify_run --run-id` (three required
                # flags missing) stayed documented.
                assert module_path.startswith("workbench."), location
                script = Path(module_path.replace(".", "/") + ".py")
                assert (ROOT / script).is_file(), f"{location}: no such module {module_path}"
                namespace = _parse_real_parser(script.stem, tokens[3:], monkeypatch)
                checked += 1
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
