from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import workbench.scripts.pipeline as pipeline


def test_resume_after_failed_subprocess_reexecutes(tmp_path: Path, monkeypatch) -> None:
    """A stage previously recorded as FAILED must be re-executed on resume and transition to COMPLETE on success."""
    state_file = tmp_path / "pipeline_state.json"
    report_file = tmp_path / "workflow_report.json"
    cfg = SimpleNamespace(
        WORKBENCH_ARTIFACTS_ROOT=tmp_path / "artifacts",
        CIR_REPO_ROOT=tmp_path / "repo",
        CIR_DATA_ROOT=tmp_path / "data",
        FASHIONIQ_ROOT=tmp_path / "FashionIQ",
        WORKBENCH_THIRD_PARTY_ROOT=tmp_path / "third_party",
    )
    monkeypatch.setattr(pipeline, "resolve_config", lambda: cfg)

    stage = pipeline.Stage(
        name="sync",
        commands=(["/bin/sync"],),
        model_id="m",
    )
    skey = pipeline.stage_key(stage)
    state = {
        "stages": {
            skey: {
                "stage_id": skey,
                "name": stage.name,
                "model_id": stage.model_id,
                "status": "FAILED",
                "return_code": 1,
                "fingerprint": "stale-failure-fingerprint",
            }
        }
    }
    pipeline.save_pipeline_state(state, state_file)

    called: list[list[str]] = []
    monkeypatch.setattr(pipeline, "run_command", lambda argv: called.append(argv) or 0)

    rc = pipeline.run_stages(
        [stage],
        dry_run=False,
        resume=True,
        state_path=state_file,
        report_path=report_file,
        config=cfg,
        apply=True,
    )

    assert rc == 0
    assert len(called) == 1, f"Expected stage to re-execute on resume, but called was {called}"
    saved = pipeline.load_pipeline_state(state_file)
    rec = saved["stages"][skey]
    assert rec["status"] == "COMPLETE", f"Expected COMPLETE status, got {rec['status']}"
    assert rec["return_code"] == 0, f"Expected return_code 0, got {rec['return_code']}"


def test_interruption_between_subprocess_exit_and_receipt_publication(tmp_path: Path, monkeypatch) -> None:
    """An interruption between subprocess exit 0 and receipt publication leaves outputs unvalidated so the stage re-runs on resume."""
    state_file = tmp_path / "pipeline_state.json"
    report_file = tmp_path / "workflow_report.json"
    cfg = SimpleNamespace(
        WORKBENCH_ARTIFACTS_ROOT=tmp_path / "artifacts",
        CIR_REPO_ROOT=tmp_path / "repo",
        CIR_DATA_ROOT=tmp_path / "data",
        FASHIONIQ_ROOT=tmp_path / "FashionIQ",
        WORKBENCH_THIRD_PARTY_ROOT=tmp_path / "third_party",
    )
    monkeypatch.setattr(pipeline, "resolve_config", lambda: cfg)

    receipt_stage = pipeline.Stage(
        name="provision",
        commands=(["/bin/provision"],),
        receipt=True,
        output_paths=(),
    )
    skey = pipeline.stage_key(receipt_stage)
    receipt_file = pipeline.stage_receipt_path(cfg, receipt_stage)

    called: list[list[str]] = []
    monkeypatch.setattr(pipeline, "run_command", lambda argv: called.append(argv) or 0)

    orig_write_receipt = pipeline.write_stage_receipt

    def interrupt_receipt(*args, **kwargs):
        raise KeyboardInterrupt("Simulated Ctrl-C before receipt publication")

    monkeypatch.setattr(pipeline, "write_stage_receipt", interrupt_receipt)

    with pytest.raises(KeyboardInterrupt):
        pipeline.run_stages(
            [receipt_stage],
            dry_run=False,
            resume=False,
            state_path=state_file,
            report_path=report_file,
            config=cfg,
            apply=True,
        )

    # Invariants after interrupt: command executed once, receipt file does not exist, validation fails
    assert len(called) == 1, "Command must have executed once before interruption"
    assert not receipt_file.exists(), f"Receipt file must not exist after interrupt: {receipt_file}"
    interrupted_state = pipeline.load_pipeline_state(state_file)
    stage_rec = interrupted_state["stages"].get(skey)
    assert pipeline.validate_stage_outputs(receipt_stage, config=cfg, record=stage_rec) is False

    # Restore real receipt publication and resume
    monkeypatch.setattr(pipeline, "write_stage_receipt", orig_write_receipt)
    rc = pipeline.run_stages(
        [receipt_stage],
        dry_run=False,
        resume=True,
        state_path=state_file,
        report_path=report_file,
        config=cfg,
        apply=True,
    )

    assert rc == 0
    assert len(called) == 2, f"Command must execute twice (re-run on resume), got {len(called)}"
    assert receipt_file.is_file(), "Receipt file must exist after successful completion"
    final_state = pipeline.load_pipeline_state(state_file)
    final_rec = final_state["stages"].get(skey)
    assert final_rec["status"] == "COMPLETE"
    assert pipeline.validate_stage_outputs(receipt_stage, config=cfg, record=final_rec) is True


def test_resume_with_changed_inputs_invalidates_stage_and_downstream(tmp_path: Path, monkeypatch) -> None:
    """Modifying an input file invalidates the stage's fingerprint and cascades invalidation to all downstream dependent stages on resume."""
    state_file = tmp_path / "pipeline_state.json"
    report_file = tmp_path / "workflow_report.json"
    cfg = SimpleNamespace(
        WORKBENCH_ARTIFACTS_ROOT=tmp_path / "artifacts",
        CIR_REPO_ROOT=tmp_path / "repo",
        CIR_DATA_ROOT=tmp_path / "data",
        FASHIONIQ_ROOT=tmp_path / "FashionIQ",
        WORKBENCH_THIRD_PARTY_ROOT=tmp_path / "third_party",
    )
    monkeypatch.setattr(pipeline, "resolve_config", lambda: cfg)

    input_file = tmp_path / "source_input.json"
    input_file.write_text('{"v": 1}', encoding="utf-8")

    out_file = tmp_path / "upstream_output.json"
    out_file.write_text('{"out": 1}', encoding="utf-8")

    upstream = pipeline.Stage(
        name="upstream",
        commands=(["/bin/upstream"],),
        input_paths=(input_file,),
        output_paths=(out_file,),
    )
    downstream = pipeline.Stage(
        name="downstream",
        commands=(["/bin/downstream"],),
        dependencies=("upstream",),
    )

    called: list[list[str]] = []
    monkeypatch.setattr(pipeline, "run_command", lambda argv: called.append(argv) or 0)

    # Initial run: both execute and complete
    rc1 = pipeline.run_stages(
        [upstream, downstream],
        dry_run=False,
        resume=True,
        state_path=state_file,
        report_path=report_file,
        config=cfg,
        apply=True,
    )
    assert rc1 == 0
    assert len(called) == 2, f"Initial run expected 2 commands, got {len(called)}"

    # Second run with unchanged inputs: both skipped
    called.clear()
    rc2 = pipeline.run_stages(
        [upstream, downstream],
        dry_run=False,
        resume=True,
        state_path=state_file,
        report_path=report_file,
        config=cfg,
        apply=True,
    )
    assert rc2 == 0
    assert len(called) == 0, f"Unchanged resume expected 0 commands, got {len(called)}"

    # Mutate the input file
    input_file.write_text('{"v": 2}', encoding="utf-8")

    # Third run with changed input: upstream and downstream must both re-run
    called.clear()
    rc3 = pipeline.run_stages(
        [upstream, downstream],
        dry_run=False,
        resume=True,
        state_path=state_file,
        report_path=report_file,
        config=cfg,
        apply=True,
    )
    assert rc3 == 0
    assert len(called) == 2, f"Expected both upstream and downstream to re-run on input change, got {len(called)}"
    assert called[0] == ["/bin/upstream"]
    assert called[1] == ["/bin/downstream"]
