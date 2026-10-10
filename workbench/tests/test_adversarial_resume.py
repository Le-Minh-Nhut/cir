"""Adversarial resume tests (R05-R11) for output validation, interruption, and state durability."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import workbench.scripts.pipeline as pipeline


def _stage(name="eval", out=None, inputs=(), deps=()):
    return pipeline.Stage(name=name, commands=([sys.executable, "-c", "pass"],), input_paths=tuple(inputs), output_paths=tuple(out or ()), dependencies=deps)


# R05: Output deletion invalidates resume
def test_r05_output_deletion_invalidates(tmp_path: Path):
    out = tmp_path / "o.json"
    out.write_text('{"x":1}')
    stage = _stage(out=(out,))
    state = tmp_path / "s.json"
    pipeline.run_stages([stage], dry_run=False, resume=True, state_path=state)
    rec = pipeline.load_pipeline_state(state)["stages"]["eval"]
    assert pipeline.validate_stage_outputs(stage, rec.get("output_fingerprints"))
    out.unlink()
    assert not pipeline.validate_stage_outputs(stage, rec.get("output_fingerprints"))


# R06: Output corruption invalidates resume
def test_r06_output_corruption_invalidates(tmp_path: Path):
    out = tmp_path / "o.json"
    out.write_text('{"x":1}')
    stage = _stage(out=(out,))
    state = tmp_path / "s.json"
    pipeline.run_stages([stage], dry_run=False, resume=True, state_path=state)
    rec = pipeline.load_pipeline_state(state)["stages"]["eval"]
    out.write_text("{corrupt")
    assert not pipeline.validate_stage_outputs(stage, rec.get("output_fingerprints"))


# R07: Interrupted (RUNNING) stage is not skipped
def test_r07_interrupted_not_skipped(tmp_path: Path, monkeypatch):
    state = tmp_path / "s.json"
    called = []
    monkeypatch.setattr(pipeline, "run_command", lambda argv: called.append(argv) or 0)
    stage = _stage()
    pipeline.run_stages([stage], dry_run=False, resume=True, state_path=state)
    data = pipeline.load_pipeline_state(state)
    data["stages"]["eval"]["status"] = "RUNNING"  # simulate interruption
    pipeline.save_pipeline_state(data, state)
    called.clear()
    pipeline.run_stages([stage], dry_run=False, resume=True, state_path=state)
    assert len(called) == 1, "Interrupted RUNNING stage was incorrectly skipped"


# R08: Forced stage invalidates dependents
def test_r08_force_invalidates_dependents(tmp_path: Path, monkeypatch):
    state = tmp_path / "s.json"
    called = []
    monkeypatch.setattr(pipeline, "run_command", lambda argv: called.append(argv) or 0)
    dep = pipeline.Stage(name="sync", commands=(["/sync"],), model_id="m")
    child = pipeline.Stage(name="evaluation", commands=(["/eval"],), model_id="m", dependencies=("sync:m",))
    pipeline.run_stages([dep, child], dry_run=False, resume=True, state_path=state)
    called.clear()
    # Force sync -> child must rerun
    pipeline.run_stages([dep, child], dry_run=False, resume=True, force_stages={"sync"}, state_path=state)
    assert any("/eval" in str(a) for a in called), "Dependent stage not rerun after forced dependency"


# R09: Unknown force-stage name must fail before any execution
def test_r09_unknown_force_stage_fails(tmp_path: Path, monkeypatch):
    called: list[list[str]] = []
    monkeypatch.setattr(pipeline, "run_command", lambda argv: called.append(argv) or 0)
    stages = [pipeline.Stage(name="doctor", commands=(["/d"],))]

    import pytest
    with pytest.raises(pipeline.WorkflowLockError, match="unknown --force-stage"):
        pipeline.run_stages(stages, dry_run=False, force_stages={"nonexistent_stage"},
                            state_path=tmp_path / "s.json")
    assert called == [], "unknown force-stage must not launch any subprocess"
    assert not (tmp_path / "s.json").exists(), "unknown force-stage must not mutate workflow state"

    # The CLI must reject it too, with a nonzero exit and no execution.
    import pytest as _pytest
    with _pytest.raises(SystemExit) as excinfo:
        pipeline.main(["all", "--model", "csmcir", "--force-stage", "nope"])
    assert excinfo.value.code == 2


# R10: Independent model stages remain cached
def test_r10_independent_stages_cached(tmp_path: Path, monkeypatch):
    state = tmp_path / "s.json"
    called = []
    monkeypatch.setattr(pipeline, "run_command", lambda argv: called.append(argv) or 0)
    a = pipeline.Stage(name="evaluation", commands=(["/eval_a"],), model_id="a")
    b = pipeline.Stage(name="evaluation", commands=(["/eval_b"],), model_id="b")
    pipeline.run_stages([a, b], dry_run=False, resume=True, state_path=state)
    called.clear()
    pipeline.run_stages([a, b], dry_run=False, resume=True, state_path=state)
    assert called == [], "Independent unchanged model stages were not cached"


# R11: Concurrent workflows do not silently corrupt state (atomic write leaves valid JSON)
def test_r11_state_atomic_valid_json(tmp_path: Path):
    state = tmp_path / "s.json"
    stage = _stage()
    pipeline.run_stages([stage], dry_run=False, resume=True, state_path=state)
    # State must always be valid JSON after writes
    data = json.loads(state.read_text())
    assert "stages" in data
    assert data["stages"]["eval"]["status"] == "COMPLETE"
    # No leftover temp files
    leftovers = [p for p in tmp_path.iterdir() if ".tmp." in p.name]
    assert leftovers == []


# R12: Side-effecting stage with no output contract cannot be silently trusted
def test_r12_side_effecting_without_output_proof_not_trusted():
    stage = pipeline.Stage(name="checkpoint", commands=(["/download"],), required_capabilities=frozenset({"allow_large_downloads"}))
    assert pipeline.validate_stage_outputs(stage) is False
    read_only = pipeline.Stage(name="doctor", commands=(["/doctor"],))
    assert pipeline.validate_stage_outputs(read_only) is True
