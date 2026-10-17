from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from workbench.scripts.validate_results import validate_file
from workbench.tests.mock_data import build_mock_runs


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_contracts", SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_doctor_selected_model_json_contract(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    doctor = load_script("doctor")
    report = {
        "overall": "BLOCKED",
        "checks": [{"name": "runtime:encoder", "status": "BLOCKED", "detail": "official command blocked", "evidence": {"checkpoints": []}}],
    }
    selected: dict[str, object] = {}

    def collect(*, model_id: str | None = None, all_models: bool = False, scope: str = "workbench",
                serving: bool = False, allow_gpu_pending: bool = False) -> dict:
        selected.update(model_id=model_id, all_models=all_models, scope=scope, serving=serving,
                        allow_gpu_pending=allow_gpu_pending)
        return report

    monkeypatch.setattr(doctor, "collect", collect)

    assert doctor.main(["--model", "encoder", "--json"]) == 2
    assert selected == {"model_id": "encoder", "all_models": False, "scope": "workbench",
                        "serving": False, "allow_gpu_pending": False}
    assert json.loads(capsys.readouterr().out) == report


def test_doctor_reports_csmcir_and_encoder_runtime_blockers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    doctor = load_script("doctor")
    from workbench.backend.operator_config import WorkbenchConfig
    from workbench.backend.registry import load_registry

    config = WorkbenchConfig(tmp_path, tmp_path / "data", tmp_path / "FashionIQ", "127.0.0.1", 8000, 5173, tmp_path / "checkpoints", tmp_path / "results", tmp_path / "third_party")
    models = {model["model_id"]: model for model in load_registry()["models"]}
    monkeypatch.setattr(doctor, "git", lambda *_: None)

    csmcir = doctor.model_checks(models["csmcir"], config)[1]["evidence"]["checkpoints"][0]
    encoder = doctor.model_checks(models["encoder"], config)[1]["evidence"]["checkpoints"][0]

    assert csmcir["command_ready"] is False
    # The missing link is reported with the wording every entry point now shares.
    assert any(blocker.startswith("canonical dataset link missing:") for blocker in csmcir["runtime_blockers"])
    assert encoder["command_ready"] is False
    assert f"ENCODER asset missing: {tmp_path / 'third_party' / 'ENCODER' / 'open_clip_pytorch_model.bin'}" in encoder["runtime_blockers"]
    assert f"ENCODER evaluator import missing: {tmp_path / 'third_party' / 'ENCODER' / 'datasets1.py'}" in encoder["runtime_blockers"]


def test_strict_real_validation_rejects_missing_provenance(tmp_path: Path) -> None:
    """`--strict-real` refuses an experiment whose own verification inputs are absent.

    The message now comes from the ONE admission policy (`schemas.results.admission`), so the
    validator and the index cannot disagree about this artifact -- which was F-13. The subject of
    the test is unchanged: a declared experiment missing its provenance is refused.
    """
    artifact = tmp_path / "real.json"
    payload = build_mock_runs()[0].model_dump(mode="json")
    payload["run"]["data_kind"] = "experiment"
    payload["run"]["run_id"] = "20260101T000000000000Z_mock_csmcir_fiq_a_deadbeef"
    payload["run"]["instrumentation_parity"] = "INSTRUMENTATION_PARITY_VERIFIED"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    output: list[str] = []

    assert not validate_file(artifact, strict_real=True, output=output.append)
    joined = " ".join(output)
    assert "command and environment digests" in joined, joined


def test_a_complete_experiment_is_admitted_but_only_the_verifier_promotes_it(tmp_path: Path) -> None:
    """The positive path, stated honestly: a complete artifact is ADMITTED, and its promotion to
    verified evidence is the authoritative verifier's decision alone.

    A release gate that refuses everything is useless, so this asserts the admitted case. It also
    asserts that `--strict-real` still refuses an UNVERIFIED experiment -- because an artifact whose
    numbers nobody corroborated must not pass a release gate -- and that a verifier verdict is what
    changes that. That pairing is the whole point of F-07.
    """
    from workbench.backend.schemas.results import (
        FASHIONIQ_CATEGORIES, PER_QUERY_UNVERIFIED, PER_QUERY_VERIFIED, admission, promote,
    )

    base = build_mock_runs()[0]
    payload = base.model_dump(mode="json")
    payload["run"].update({
        "data_kind": "experiment",
        "run_id": "20260101T000000000000Z_mock_fiq_a_deadbeef",
        "command_digest": "c" * 64, "environment_digest": "d" * 64,
        "bundle_manifest_digest": "e" * 64,
        "instrumentation_parity": "INSTRUMENTATION_PARITY_VERIFIED",
    })
    queries = []
    for index, category in enumerate(FASHIONIQ_CATEGORIES):
        reference, target = f"ref-{category}", f"target-{category}"
        queries.append({
            "query_id": f"{category}:{index}:{reference}:{target}", "category": category,
            "annotation_index": index, "reference_id": reference, "target_id": target,
            "raw_captions": ["one", "two"], "model_input_text": "one and two",
            "target_rank": 1,
            "top_results": [{"rank": rank, "image_id": target if rank == 1 else f"d-{rank}",
                             "score": 1.0 - rank / 100} for rank in range(1, 201)],
        })
    payload["queries"] = queries
    artifact = tmp_path / "complete.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    from workbench.backend.schemas.results import ResultRun

    result = ResultRun.model_validate(payload)
    state, reasons = admission(result)
    # The artifact's own content cannot promote it: no provenance problem remains, but a verifier
    # has not spoken, so the honest state is UNVERIFIED.
    assert state == PER_QUERY_UNVERIFIED, (state, reasons)
    assert not any("missing" in reason or "reuse" in reason for reason in reasons), reasons

    # The verifier's verdict is the only thing that promotes it -- and only for the artifact it
    # actually named. The receipt must bind THIS artifact's digest AS READ FROM DISK; a digest the
    # artifact writes about itself is not evidence (a reviewer found that fallback reopened the
    # self-certification hole this mechanism exists to close).
    digest = "a" * 64
    verdict = {"certified": True, "is_research_evidence": True,
               "per_query_artifact_digest": digest, "model_id": result.run.model_id,
               "checkpoint_id": result.run.checkpoint_id}
    assert promote(result, state, verdict, artifact_digest=digest)[0] == PER_QUERY_VERIFIED
    # A refused verdict leaves it unverified.
    assert promote(result, state, {"certified": False, "is_research_evidence": False},
                   artifact_digest=digest)[0] == PER_QUERY_UNVERIFIED
    # A verdict about a DIFFERENT artifact must not promote this one.
    assert promote(result, state, {**verdict, "per_query_artifact_digest": "b" * 64},
                   artifact_digest=digest)[0] == PER_QUERY_UNVERIFIED
    # Nor may a verdict about another model's run.
    assert promote(result, state, {**verdict, "model_id": "airknow"},
                   artifact_digest=digest)[0] == PER_QUERY_UNVERIFIED
    # And a digest the artifact claims about itself is NOT accepted in place of the reader's.
    self_nominated = result.model_copy(update={"run": result.run.model_copy(
        update={"per_query_artifact_digest": "c" * 64})})
    assert promote(self_nominated, state, {**verdict, "per_query_artifact_digest": "c" * 64})[0] \
        == PER_QUERY_UNVERIFIED, "an artifact must not nominate its own digest"

    # `--strict-real` asks whether the artifact RECORDS what its verification needs. A complete
    # artifact does, so it passes -- with an explicit note that the verifier has not yet spoken.
    # (An earlier revision blocked every valid experiment here; a reviewer caught that this made the
    # gate useless.) It must still refuse an artifact whose recorded provenance is INCOMPLETE, which
    # the test above covers.
    output: list[str] = []
    assert validate_file(artifact, strict_real=True, output=output.append), output
    joined = " ".join(output)
    assert "awaiting verifier corroboration" in joined, output
    # And the artifact is still NOT promoted to verified evidence by validation alone.
    assert admission(result)[0] == PER_QUERY_UNVERIFIED


def test_sync_dry_run_and_verify_only_output_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sync = load_script("sync_upstreams")
    model = {"model_id": "demo", "source_available": True, "source_dir": "Demo", "upstream_repo_url": "https://example.invalid/demo.git", "upstream_commit_sha": "a" * 40}
    monkeypatch.setattr(sync, "load_registry", lambda: {"models": [model]})
    monkeypatch.setattr(sync.subprocess, "run", lambda *args, **kwargs: pytest.fail("git must not run"))

    monkeypatch.setattr(sys, "argv", ["sync_upstreams.py", "--model", "demo", "--dry-run", "--output-root", str(tmp_path)])
    sync.main()
    assert capsys.readouterr().out.splitlines() == [
        f"demo: url=https://example.invalid/demo.git source_dir=Demo pin={'a' * 40} local=missing",
        f"[RUN] demo: would clone and checkout at {'a' * 40}",
    ]

    monkeypatch.setattr(sys, "argv", ["sync_upstreams.py", "--model", "demo", "--verify-only", "--output-root", str(tmp_path)])
    with pytest.raises(SystemExit, match="1"):
        sync.main()
    assert capsys.readouterr().out.splitlines() == [
        f"demo: url=https://example.invalid/demo.git source_dir=Demo pin={'a' * 40} local=missing",
        "[BLOCKED] demo: pinned SHA unavailable locally",
    ]


@pytest.mark.parametrize("arguments, expected", [(["--list", "--dry-run"], 0), (["--all-runnable", "--dry-run"], 1)])
def test_evaluation_listing_and_dry_run_never_execute_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, arguments: list[str], expected: int) -> None:
    evaluate = load_script("evaluate_models")
    monkeypatch.setattr(evaluate, "pinned_revision", lambda _: None)
    import workbench.backend.runtime as runtime

    monkeypatch.setattr(runtime, "source_clean_and_pinned", lambda model, _: (True, model["upstream_commit_sha"], None))
    monkeypatch.setattr(evaluate.subprocess, "run", lambda *args, **kwargs: pytest.fail("model command must not run"))

    assert evaluate.main(arguments) == expected


@pytest.mark.parametrize("relative", ["artifacts/checkpoints", "data/FashionIQ", "third_party/upstream"])
def test_cleanup_refuses_protected_checkpoint_dataset_and_source_targets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], relative: str) -> None:
    cleanup = load_script("clean_generated")
    protected = tmp_path / relative
    protected.mkdir(parents=True)
    (protected / "keep").write_text("protected", encoding="utf-8")
    monkeypatch.setattr(cleanup, "cleanup_targets", lambda *_: [protected])

    assert cleanup.clean(tmp_path, True, False, False, False) == 2
    assert protected.exists() and (protected / "keep").is_file()
    assert "[BLOCKED] refusing unsafe path:" in capsys.readouterr().err
