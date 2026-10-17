"""The trust boundary between the operator and the untrusted official evaluator.

The evaluator is research code from a third party. It must receive what it needs to run and
nothing else: no ledger key, no operator secrets, and no unrelated credentials that happened to
be in the operator's shell. These tests assert on the environment a real child process actually
received, by having that child print it.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import pytest

from workbench.backend.operator_config import WorkbenchConfig
from workbench.scripts import evaluate_models as em

SECRET_NAMES = (
    "WORKBENCH_LEDGER_KEY",
    "WORKBENCH_SECRET_UNUSED",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "WANDB_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GITHUB_TOKEN",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
)


def _config(tmp_path: Path) -> WorkbenchConfig:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    return WorkbenchConfig(
        CIR_REPO_ROOT=repo, CIR_DATA_ROOT=tmp_path, FASHIONIQ_ROOT=tmp_path,
        WORKBENCH_HOST="127.0.0.1", WORKBENCH_BACKEND_PORT=8000,
        WORKBENCH_FRONTEND_PORT=5173, WORKBENCH_CHECKPOINT_ROOT=tmp_path / "ckpts",
        WORKBENCH_RESULTS_ROOT=tmp_path / "results",
        WORKBENCH_THIRD_PARTY_ROOT=tmp_path / "third_party")


_RUN_SEQUENCE = itertools.count()

def _run_evaluator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                   *, passthrough: str | None = None) -> dict:
    """Run a real child that reports its whole environment, and return the report."""
    reporter = tmp_path / "report_env.py"
    reporter.write_text("import json, os\n"
                        "print(json.dumps({k: 'set' for k in os.environ}))\n")
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_bytes(b"x")
    plan = em.EvaluationPlan(model_id="m", checkpoint_id="c", protocol="p",
                             dataset_root=tmp_path, source=tmp_path, checkpoint=checkpoint,
                             cwd=tmp_path, command=[sys.executable, str(reporter)],
                             pin="p" * 40, actual_source_commit="a" * 40)
    if passthrough is not None:
        monkeypatch.setenv("WORKBENCH_EVALUATOR_ENV_PASSTHROUGH", passthrough)
    # A fresh run id per invocation: the run directory is claimed exclusively and an existing one
    # is refused, so looping over names in ONE test must not collide with itself.
    code, directory, _ = em.invoke_evaluator(plan, _config(tmp_path),
                                             f"sec-run-{next(_RUN_SEQUENCE)}", tmp_path / "out")
    assert code == 0
    return json.loads((directory / "stdout.log").read_text().strip().splitlines()[-1])


@pytest.mark.parametrize("name", SECRET_NAMES)
def test_no_credential_reaches_the_evaluator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                             name: str) -> None:
    """Each credential is absent from the child, whatever its value or origin."""
    monkeypatch.setenv(name, "sentinel-value-must-not-appear")
    seen = _run_evaluator(tmp_path, monkeypatch)
    assert name not in seen, f"{name} leaked into the evaluator"


def test_the_allowlist_still_delivers_what_a_run_needs(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Filtering must not break the run: identity, CUDA visibility and paths survive."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("FASHIONIQ_ROOT", str(tmp_path / "FashionIQ"))
    seen = _run_evaluator(tmp_path, monkeypatch)

    assert seen.get("PATH")
    assert seen.get("CUDA_VISIBLE_DEVICES") == "set", "device visibility is an operator decision"
    assert seen.get("evaluation_run_id") == "set"
    assert seen.get("FASHIONIQ_ROOT") == "set"


def test_a_declared_passthrough_name_is_forwarded(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A model may need a specific extra variable; the declaration is explicit and reviewable."""
    monkeypatch.setenv("WORKBENCH_TEST_LIBRARY_ROOT", "/opt/model/lib")
    seen = _run_evaluator(tmp_path, monkeypatch, passthrough="WORKBENCH_TEST_LIBRARY_ROOT")
    assert seen.get("WORKBENCH_TEST_LIBRARY_ROOT") == "set"


def test_an_undeclared_name_is_not_forwarded(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The same variable without a declaration is withheld, so the boundary is not a formality."""
    monkeypatch.setenv("WORKBENCH_TEST_LIBRARY_ROOT", "/opt/model/lib")
    seen = _run_evaluator(tmp_path, monkeypatch)
    assert "WORKBENCH_TEST_LIBRARY_ROOT" not in seen


def test_a_denied_prefix_cannot_be_whitelisted_by_declaration(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Declaring a credential for passthrough must not defeat the deny rule.

    Otherwise a typo in a passthrough list would silently reopen the boundary it exists to close.
    """
    monkeypatch.setenv("HF_TOKEN", "must-not-leak")
    seen = _run_evaluator(tmp_path, monkeypatch, passthrough="HF_TOKEN")
    assert "HF_TOKEN" not in seen


def test_the_operator_keeps_the_key_it_needs_to_sign(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """The boundary must not break the operator: it still authenticates its own ledger entry."""
    monkeypatch.setenv("WORKBENCH_LEDGER_KEY", "operator-signing-key")
    config = _config(tmp_path)
    key = em.ledger_hmac_key(config)
    assert key == b"operator-signing-key"

    # And the key is absent from the evaluator's environment.
    seen = _run_evaluator(tmp_path, monkeypatch)
    assert "WORKBENCH_LEDGER_KEY" not in seen


def test_no_credential_appears_in_the_queued_operator_record(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The queue's own record of a job must not carry a credential either."""
    from workbench.backend.jobs import SingleGpuQueue

    monkeypatch.setenv("HF_TOKEN", "must-not-appear-anywhere")
    monkeypatch.setenv("WORKBENCH_LEDGER_KEY", "operator-key")
    monkeypatch.setenv("WORKBENCH_SECRET_UNUSED", "must-not-appear-anywhere")

    queue = SingleGpuQueue()
    directory = tmp_path / "queue_jobs" / "sec"
    job = queue.enqueue(
        "m", "c", "p",
        [sys.executable, "-c",
         "import os; print('TOKENS', sorted(k for k in os.environ if 'TOKEN' in k));"
         " print('LEDGER', 'WORKBENCH_LEDGER_KEY' in os.environ)"],
        tmp_path, directory, {})
    import time
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if next(i for i in queue.list() if i["job_id"] == job["job_id"])["status"] == "completed":
            break
        time.sleep(0.02)
    child_view = (directory / "queue_stdout.log").read_text()
    record = (directory / "queue_command.json").read_text()
    assert "HF_TOKEN" not in child_view
    assert "must-not-appear-anywhere" not in record
    assert "WORKBENCH_SECRET_UNUSED" not in child_view
    # The operator hop keeps the ledger key, because the operator is the signer: verified through
    # the child's own view rather than assumed.
    assert "LEDGER True" in child_view


def test_an_operator_passthrough_cannot_re_admit_the_ledger_key_path(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`WORKBENCH_EVALUATOR_ENV_PASSTHROUGH` may add a name; it may not defeat the deny prefixes.

    The ledger key is resolved from XDG_CONFIG_HOME, and from `Path.home()` when that is unset, so a
    child that receives HOME can re-derive the operator's key directory and read the HMAC key -- the
    one secret that would let the untrusted evaluator forge its own ledger entry. An independent
    audit found this: the allowlist withheld XDG_CONFIG_HOME but not HOME, and an explicit
    passthrough of `HOME` put it back. Denying by prefix is what makes the declaration a no-op
    rather than an override.
    """
    for declared in ("HOME", "XDG_CONFIG_HOME", "HOME:XDG_CONFIG_HOME"):
        seen = _run_evaluator(tmp_path, monkeypatch, passthrough=declared)
        assert "HOME" not in seen, f"HOME reached the evaluator via passthrough={declared!r}"
        assert "XDG_CONFIG_HOME" not in seen, \
            f"XDG_CONFIG_HOME reached the evaluator via passthrough={declared!r}"


def test_no_name_a_key_path_is_derived_from_reaches_the_evaluator(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The deny-by-prefix rule covers the whole key-path surface, not the two names seen so far."""
    for name in ("HOME", "USERPROFILE", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_RUNTIME_DIR"):
        monkeypatch.setenv(name, "/tmp/must-not-be-forwarded")
        seen = _run_evaluator(tmp_path, monkeypatch, passthrough=name)
        assert name not in seen, f"{name} leaked into the evaluator"


def test_the_evaluator_cannot_read_the_repository_root_ledger_key(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The legacy key location must not be readable by the evaluator child.

    The key is honoured from `<repo_root>/.workbench_ledger_key` when that is the only one, and
    `CIR_REPO_ROOT` IS on the evaluator allowlist (the model needs it to find its inputs). So a
    deployment still carrying a repository-root key would hand the untrusted evaluator the HMAC key
    by a second route, after `HOME` was closed. An independent audit found this.

    The child is not given a redacted CIR_REPO_ROOT, because that would break every model that needs
    it; what changes is WHERE a fresh key lives, and this pins the second half: whether or not a
    legacy key exists, the child must not reach a key file through the variable it does receive.
    """
    repo = tmp_path / "repo"
    (repo / "workbench" / "artifacts").mkdir(parents=True)
    monkeypatch.setenv("CIR_REPO_ROOT", str(repo))
    monkeypatch.delenv("WORKBENCH_LEDGER_KEY", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

    from workbench.scripts import evaluate_models as em

    key = em.ledger_hmac_key(_config(tmp_path))
    assert key, "a key must resolve"

    # The path the resolver chose is one the child's environment can reach ONLY if it is under the
    # repo root. A fresh key must therefore not be created there.
    legacy = repo / ".workbench_ledger_key"
    assert not legacy.exists(), (
        "a fresh ledger key was written under the repo root, where the evaluator child "
        "(which is given CIR_REPO_ROOT) can read it")

    # And the child environment really does carry the variable that made this reachable.
    assert str(repo) in em._evaluator_environment({}).get("CIR_REPO_ROOT", "")
