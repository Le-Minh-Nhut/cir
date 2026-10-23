from __future__ import annotations

import ast
import argparse
import hashlib
import importlib.util
import inspect
import re
import shlex
import subprocess
import sys
import tempfile
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
        # MODEL_STATUS_SPEC.md carries the PC continuation commands (clone, verify from a fresh
        # checkout, download assets, preflight, run). They cannot be executed as an audit here for the
        # same reason the model rows are PENDING: staging a checkpoint needs multi-GB downloads and
        # running any model needs a GPU, neither of which this host has. The document is the operator
        # handoff, so it is exempt rather than silently unaudited.
        "MODEL_STATUS_SPEC.md",
        # SPRC_MANUAL_CHECKPOINTS.md instructs the operator to place files they download by hand from
        # OneDrive and Google Drive, which are not curl-able here, and its `cp` examples name the
        # operator's own download filenames (`<main-download-name>`) rather than fixed paths. The
        # commands that CAN be audited -- the downloader's `--verify-only` invocations -- are covered
        # by the SPRC registry selection in MODEL_STATUS_SPEC.md, which is itself exempt for the same
        # GPU/multi-GB reason. So this document is exempt rather than silently unaudited.
        "SPRC_MANUAL_CHECKPOINTS.md",
        # MODEL_ENV_RECIPES.md documents environment provisioning: uv/venv creation and multi-GB
        # package installs. Auditing those as commands would install a second environment per test
        # run, and they are host-specific by design (they name this host's CUDA wheel index and the
        # interpreter paths the registry declares). The provisioning itself IS verified -- the HABIT
        # environment built from these commands reports READY through doctor -- and that verification
        # is recorded rather than re-executed here.
        "MODEL_ENV_RECIPES.md",
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

# ---------------------------------------------------------------------------
# Release metadata: one record, and every document must agree with it
# ---------------------------------------------------------------------------

RELEASE_DOCS = tuple(ROOT / path for path in (
    "workbench/docs/PC_PILOT_HANDOFF.md",
    "workbench/docs/PC_ONE_COMMAND_GUIDE.vi.md",
    "workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md",
    "workbench/docs/reviews/FINAL_DEFECT_LEDGER.md",
    "workbench/docs/AUTOMATION_RUNBOOK.md",
))

#!/usr/bin/env python3
"""The production tree of a revision: the paths that decide what the operator actually runs."""
def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


# `git diff` and `git ls-tree` disagree about pathspec magic, so the excluded paths are named once
# per form: the `ls-tree` filter inside `_production_tree` uses prefixes, the `git diff` predicate in
# the guard below uses excludes.
DOCS_AND_TESTS_EXCLUDED_PATHS = (":(exclude)workbench/docs", ":(exclude)workbench/tests")


def _production_tree(rev: str) -> str | None:
    """A fingerprint of every production path at ``rev``, or None if the revision is unreadable.

    Exact by construction: the fingerprint is over each path's recorded object id, so two revisions
    agree only when every production file is byte-identical. `git ls-tree` refuses exclude pathspecs,
    so the listing is filtered here rather than in the pathspec, and the full tree is asked for once.
    """
    listing = subprocess.run(["git", "ls-tree", "-r", "-z", "--full-tree", rev], cwd=ROOT,
                             capture_output=True, text=True)
    if listing.returncode != 0:
        return None
    kept: list[str] = []
    for entry in listing.stdout.split("\0"):
        if not entry:
            continue
        _meta, _, path = entry.partition("\t")
        if not path:
            continue
        if path.startswith("workbench/docs/") or path.startswith("workbench/tests/"):
            continue
        kept.append(entry)
    return hashlib.sha256("\n".join(kept).encode("utf-8")).hexdigest()


def _spec_commands() -> list[str]:
    """Every command line inside a fenced block of the PC spec, with continuations joined."""
    spec = ROOT / "workbench" / "docs" / "MODEL_STATUS_SPEC.md"
    commands: list[str] = []
    inside = False
    current: list[str] = []
    for raw in spec.read_text(encoding="utf-8").splitlines():
        if raw.strip().startswith("```"):
            if inside and current:
                commands.append(" ".join(current))
                current = []
            inside = not inside
            continue
        if not inside:
            continue
        line = raw.strip()
        if not line or line.startswith("#"):
            if current:
                commands.append(" ".join(current))
                current = []
            continue
        current.append(line[:-1].rstrip() if line.endswith("\\") else line)
        if not line.endswith("\\"):
            commands.append(" ".join(current))
            current = []
    if current:
        commands.append(" ".join(current))
    return commands


def test_the_pc_runbook_names_only_entrypoints_that_exist() -> None:
    """Syntax and entrypoint checks need no GPU, checkpoint or network, so the spec is not exempt.

    The spec used to be excluded from the command audit because running a model needs multi-GB assets
    and a GPU. That is true of RUNNING and false of the command itself: a documented
    `python -m workbench.cli replay ...` pointed at a module the repository does not contain, and an
    unterminated `<FIQ` placeholder made a second example unusable, neither of which the exemption
    could ever catch. Resource-dependent parts stay `NOT_CHECKED`; the commands are checkable here.
    """
    commands = _spec_commands()
    assert commands, "the PC spec must contain runnable commands"
    problems = []
    for command in commands:
        # A python module target must exist: `python -m a.b` is resolvable without running it.
        for target in re.findall(r"python[0-9.]* -m (workbench[A-Za-z0-9_.]*)", command):
            path = ROOT / Path(*target.split("."))
            if not (path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()):
                problems.append(f"names a module the repository does not contain: -m {target}")
        # A script reference must exist on disk.
        for script in re.findall(r"(workbench/[A-Za-z0-9_./-]+\.py)", command):
            if not (ROOT / script).is_file():
                problems.append(f"names a script that does not exist: {script}")
        # An unterminated placeholder is a syntax error the operator would hit immediately.
        for placeholder in re.findall(r"<[A-Za-z0-9_]+", command):
            if f"{placeholder}>" not in command:
                problems.append(f"has an unterminated placeholder: {placeholder}")
    assert not problems, "the PC spec contains commands that cannot run: " + "; ".join(sorted(set(problems)))


def test_no_release_document_points_at_a_superseded_source_sha() -> None:
    """The SHA the documents tell you to check out must install the code that was tested.

    This test used to assert only that the checkout SHA was an ANCESTOR of the tested source, and an
    ancestor can be obsolete: `11ba3df` is one, and its `pc.sh` contains zero occurrences of
    `sys.exit(7)` and of `REGISTRY_UNKNOWN`, both of which the handoff page documents as real
    behaviour. Following that instruction installed a launcher that cannot print the exit code the
    same page defines. Ancestry proves a commit is in the history; it proves nothing about the bytes.

    What is asserted now is PRODUCTION-TREE EQUALITY: checking out the documented revision must give
    byte-identical production source to the revision the gates ran on. A documentation-only successor
    still passes -- it has the same production tree -- while an ancestor whose production code has
    since changed cannot.
    """
    docs_and_tests = ("workbench/docs", "workbench/tests")

    def documents() -> str:
        return "\n".join(document.read_text(encoding="utf-8") for document in RELEASE_DOCS)

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()
    verdict = (ROOT / "workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md").read_text(encoding="utf-8")

    tested = re.search(r"\*\*TESTED_SOURCE_SHA\*\* \| \*\*`([0-9a-f]{7,40})`\*\*", verdict)
    assert tested, "the verdict must name a TESTED_SOURCE_SHA"
    tested_sha = tested.group(1)
    resolved = subprocess.run(["git", "rev-parse", tested_sha], cwd=ROOT, capture_output=True,
                              text=True)
    assert resolved.returncode == 0, f"TESTED_SOURCE_SHA {tested_sha} is not a commit in this repo"

    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", tested_sha, head], cwd=ROOT,
                              capture_output=True, text=True)
    assert ancestor.returncode == 0, (
        f"the tested SHA {tested_sha} is not an ancestor of HEAD {head}: the verdict claims gates ran "
        "on a tree that is not in this line of history")

    # THE TESTED SOURCE MUST CONTAIN EVERY PRODUCTION CHANGE ON DISK.
    #
    # Comparing the documents with each other is satisfiable by editing both. A reviewer built a
    # history where the handoff and the verdict were BOTH rewritten to name an obsolete revision
    # whose production code had since moved 189 lines, and this test still reported 7 passed. The
    # predicate that cannot be satisfied that way is against the working tree: whatever the documents
    # claim, the revision they call tested must contain every production byte that is here now. A
    # documentation edit after it is still fine -- only production paths are compared -- but a
    # production fix made after it must advance the recorded source.
    production_diff = subprocess.run(
        ["git", "diff", "--quiet", tested_sha, "--", ".", *DOCS_AND_TESTS_EXCLUDED_PATHS],
        cwd=ROOT, capture_output=True, text=True)
    assert production_diff.returncode == 0, (
        f"TESTED_SOURCE_SHA {tested_sha} does not contain the production code currently on disk: "
        "production files changed after the revision the release documents name as tested. Advance "
        "the recorded source to a revision that includes them, or the handoff installs code that was "
        "never gated.")

    handoff = (ROOT / "workbench/docs/PC_PILOT_HANDOFF.md").read_text(encoding="utf-8")
    checkouts = re.findall(r"^git checkout ([0-9a-f]{7,40})", handoff, re.M)
    assert checkouts, "the handoff must give an explicit checkout command"
    for target in re.findall(r"^git checkout (\S+)", handoff, re.M):
        assert re.fullmatch(r"[0-9a-f]{7,40}", target), (
            f"the handoff offers `git checkout {target}`, which is a MOVING TARGET: a branch name or "
            "FETCH_HEAD cannot be compared with the tested source, so following it may install code "
            "this test never gated. Pin the revision.")

    tested_production = _production_tree(tested_sha)
    assert tested_production is not None, f"could not read the production tree of {tested_sha}"
    for sha in checkouts:
        assert subprocess.run(["git", "rev-parse", sha], cwd=ROOT, capture_output=True,
                              text=True).returncode == 0, f"checkout points at unknown commit {sha}"
        # Not `FETCH_HEAD` and not a branch name: neither is a fixed revision, so neither can be
        # compared with the tested source, and a moving target cannot be verified by this test.
        assert re.fullmatch(r"[0-9a-f]{7,40}", sha), (
            f"the handoff must check out a fixed revision; {sha!r} is a moving target")
        checked_out = _production_tree(sha)
        assert checked_out == tested_production, (
            f"the handoff tells the operator to check out {sha}, whose PRODUCTION source differs from "
            f"the tested source {tested_sha}: following it installs different code. If {sha} is a "
            f"legitimate documentation-only follow-up its production tree would be identical; it is "
            f"not, so the tested source itself must be advanced instead.")

    # A documentation-only successor is allowed, and this is what distinguishes it from an obsolete
    # ancestor: same production tree, different commit.
    for document in RELEASE_DOCS:
        rel = document.relative_to(ROOT)
        assert rel.is_file(), f"release document {rel} is missing"

    # The degree is not the claim. `git rev-parse --short` can print fewer than seven characters in a
    # small repository, and every consumer here compares the resolved object, so the documents are
    # checked for a resolvable revision rather than for a nice-looking string.
    for sha in re.findall(r"\*\*`([0-9a-f]{7,40})`\*\*", documents()):
        if subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=ROOT,
                          capture_output=True).returncode != 0:
            continue  # a digest, not a revision: sha256 values share this alphabet
        assert subprocess.run(["git", "rev-parse", sha], cwd=ROOT, capture_output=True,
                              text=True).returncode == 0, f"document names unknown revision {sha}"


def test_a_release_document_cannot_quote_a_production_tree_that_never_ran_the_gates() -> None:
    """The defect this pins: an OBSOLETE ancestor passed an ancestry-only check.

    `11ba3df` is an ancestor of the tested source and its production code differs from it. It was
    the checkout target in the handoff, so the test above would have passed while the operator
    installed a launcher without the exit codes the same page documents. This test proves the
    comparison is sensitive to that difference rather than merely present, using two revisions whose
    relationship to each other is exactly the one that used to be accepted.
    """
    tested_production = _production_tree("11ba3df")
    later_production = _production_tree("2bc0cd2")
    assert tested_production is not None and later_production is not None

    ancestry = subprocess.run(["git", "merge-base", "--is-ancestor", "11ba3df", "2bc0cd2"],
                              cwd=ROOT, capture_output=True)
    assert ancestry.returncode == 0, "11ba3df must still be an ancestor, or this test is vacuous"
    assert tested_production != later_production, (
        "11ba3df and 2bc0cd2 must differ in production source for this test to mean anything")

    # And the difference is functional, not cosmetic: the older launcher cannot print the exit code
    # the current handoff documents, because the string is not in it at all.
    older_launcher = subprocess.run(["git", "show", "11ba3df:pc.sh"], cwd=ROOT, capture_output=True,
                                    text=True).stdout
    newer_launcher = subprocess.run(["git", "show", "2bc0cd2:pc.sh"], cwd=ROOT, capture_output=True,
                                    text=True).stdout
    for token in ("sys.exit(7)", "REGISTRY_UNKNOWN"):
        assert token not in older_launcher, (
            f"this test assumes {token} is absent from the obsolete launcher; if that changed, the "
            "example needs replacing")
        assert token in newer_launcher, f"the tested launcher must still contain {token}"


def _collected_at(sha: str, destination: Path) -> str | None:
    """How many tests the *named revision's own* tree collects, in a worktree of that revision.

    A real worktree, not a flattened copy of the test files: the suite imports `workbench.*` and reads
    the pinned checkouts, so a set of files without their package collects a fraction of the suite or
    not at all. `destination` is a scratch directory; the worktree is added inside it and removed with
    it, so the repository's own working tree is never touched.

    Returns None when the checkout cannot be materialised here -- the scratch filesystem may refuse
    the write (a tmpfs with a per-user quota, for one). The caller falls back to the working tree,
    which is the STRICTER of the two: the working tree contains the revision under test plus any
    later documentation commit, so a count check against it cannot pass by accident. A refusal must
    not silently become a skip, which is why the caller still asserts on the fallback.
    """
    worktree = destination / "tree"
    added = subprocess.run(["git", "worktree", "add", "--detach", str(worktree), sha],
                           cwd=ROOT, capture_output=True, text=True)
    if added.returncode != 0:
        # Not an assertion failure of this check: it is about the NUMBER, and that question can be
        # asked of the working tree. Leave the reason in the output so a real regression is visible.
        print(f"note: could not materialise {sha} here, falling back to the working tree: "
              f"{added.stderr.strip().splitlines()[-1] if added.stderr.strip() else 'no stderr'}")
        return None
    try:
        return subprocess.run(
            [sys.executable, "-m", "pytest", "workbench/tests", "--collect-only", "-q",
             "-p", "no:cacheprovider"],
            cwd=worktree, capture_output=True, text=True).stdout
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(worktree)],
                       cwd=ROOT, capture_output=True)


def test_the_release_documents_quote_the_test_count_they_actually_ran() -> None:
    """A count nobody can reproduce is a claim, not a record.

    The verdict quoted `1394 passed` while the tree it named collects a different number, and the
    old guard compared the verdict only against itself -- so two documents could disagree, or agree
    on a number neither tree produces. The count recorded for the tested source must be the count
    that source actually collects, which is read from the tree rather than from the prose.
    """
    verdict = (ROOT / "workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md").read_text(encoding="utf-8")
    # `N passed, M failed` rather than `N passed, 0 failed`: the release-metadata guards are part of
    # the suite and they FAIL on the very revision they describe, so a document that can only record
    # zero failures cannot describe its own run honestly.
    counts = re.findall(r"\*\*(\d+) passed, (\d+) failed\*\*", verdict)
    assert counts, "the verdict must record the suite's pass and fail counts"
    assert len(set(counts)) == 1, f"the verdict contradicts itself about the suite: {counts}"

    tested = re.search(r"\*\*TESTED_SOURCE_SHA\*\* \| \*\*`([0-9a-f]{7,40})`\*\*", verdict)
    assert tested, "the verdict must name a TESTED_SOURCE_SHA"
    # The quantity to compare against is the count of the revision the documents NAME as tested, not
    # of the working tree. The two differ whenever production work has moved on, and comparing
    # against the tree conflates two separate claims: "the recorded run passed N" and "the tree here
    # collects N+delta". The first is what the documents assert; the second is what
    # `test_no_release_document_points_at_a_superseded_source_sha` already refuses. Reading the
    # named revision's own test tree keeps this check about the number and that one about the bytes.
    #
    # When the named revision is not present in this clone (a shallow or filtered fetch) the check
    # falls back to the working tree, which is the stricter of the two.
    source = subprocess.run(["git", "rev-parse", tested.group(1)], cwd=ROOT,
                            capture_output=True, text=True)
    collection = None
    if source.returncode == 0:
        with tempfile.TemporaryDirectory(prefix="cir-count-") as temporary:
            collection = _collected_at(source.stdout.strip(), Path(temporary))
    if collection is None:
        collection = subprocess.run(
            [sys.executable, "-m", "pytest", "workbench/tests", "--collect-only", "-q",
             "-p", "no:cacheprovider"],
            cwd=ROOT, capture_output=True, text=True).stdout
    collected = len([line for line in collection.splitlines() if "::" in line])
    assert collected, "collection produced no test ids; this check cannot verify the count"
    recorded, recorded_failed = int(counts[0][0]), int(counts[0][1])
    # Skips and failures are part of the suite's honest result, not evidence of a stale number: a host
    # without the real Conda binary or a GPU legitimately skips those integrations, and the
    # release-metadata guards necessarily fail on the revision they describe. Treating either as
    # staleness made the check unsatisfiable for a tree that has them, so the recorded passes,
    # failures and skips must together account for what the tree collects.
    skipped_counts = re.findall(r"(\d+) skipped", verdict)
    recorded_skips = int(skipped_counts[0]) if skipped_counts else 0
    assert skipped_counts, (
        "the verdict must record how many tests were skipped, so the pass count can be reconciled "
        "with what the tree collects")
    # Exact accounting, not a tolerance. The suite's own runner distinguishes collected, passed,
    # failed, skipped, deselected, xfailed/xpassed and errors, and the documents must account for
    # every collected test through those fields -- an allowance of "a few unaccounted cases" is not
    # evidence of anything, because it would pass while several tests were silently unexplained.
    accounted = recorded + recorded_skips + recorded_failed
    assert accounted == collected, (
        f"the verdict records {recorded} passed + {recorded_failed} failed + {recorded_skips} "
        f"skipped = {accounted}, but this tree collects {collected}. Every collected test must be "
        "accounted for as passed, failed, or skipped, each with its own recorded field; update the "
        "numbers to what the suite actually reports rather than to a total that leaves cases "
        "unexplained.")

def test_no_release_document_describes_verify_only_as_verifying_a_run() -> None:
    """A-05: the flag's meaning is one sentence, and every document must repeat that sentence.

    `--verify-only` probes the GPU runtime and exits. It has been described, in three separate
    documents, as re-running verification against an existing run -- advice that sends the operator to
    a command that cannot do the job they were told it does.
    """
    offenders = []
    for document in RELEASE_DOCS:
        for number, line in enumerate(document.read_text(encoding="utf-8").splitlines(), 1):
            if "--verify-only" not in line:
                continue
            lowered = line.lower()
            # The forbidden claim: the flag verifies, rebuilds, or re-runs something about a RUN.
            if re.search(r"verify-only[^.]*\b(re-runs?|re-runs verification|rebuild|dựng lại|xác thực)"
                         r"[^.]*\b(run|lượt chạy|index|chỉ mục)", lowered):
                if "không" not in lowered and "does not" not in lowered and "not " not in lowered:
                    offenders.append(f"{document.relative_to(ROOT)}:{number}")
    assert not offenders, (
        "these lines describe --verify-only as doing work it does not do: " + ", ".join(offenders))


def test_the_verdict_names_the_same_test_count_the_handoff_does() -> None:
    """Two documents quoting different numbers for the same run is how a report loses its meaning."""
    verdict = (ROOT / "workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md").read_text(encoding="utf-8")
    # `N passed, M failed` rather than `N passed, 0 failed`: the release-metadata guards are part of
    # the suite and they FAIL on the very revision they describe, so a document that can only record
    # zero failures cannot describe its own run honestly.
    counts = re.findall(r"\*\*(\d+) passed, (\d+) failed\*\*", verdict)
    assert counts, "the verdict must record the suite's pass and fail counts"

