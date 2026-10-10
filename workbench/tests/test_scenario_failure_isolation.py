"""Synthetic multi-model failure isolation E2E (Scenario H)."""
from __future__ import annotations

from pathlib import Path

import workbench.scripts.pipeline as pipeline


def test_scenario_h_blocked_model_does_not_block_independent_model(tmp_path: Path, monkeypatch):
    """A blocked model's dependent stages are skipped, while an independent ready model executes."""
    state = tmp_path / "s.json"
    called = []

    def run(argv):
        called.append(argv)
        # model_a sync fails (blocked); everything else succeeds
        return 1 if "model_a" in str(argv) else 0

    monkeypatch.setattr(pipeline, "run_command", run)

    stages = [
        pipeline.Stage(name="sync", commands=(["/sync_model_a"],), model_id="model_a"),
        pipeline.Stage(name="runtime-preflight", commands=(["/preflight_model_a"],), model_id="model_a", dependencies=("sync:model_a",)),
        pipeline.Stage(name="evaluation", commands=(["/eval_model_a"],), model_id="model_a", dependencies=("runtime-preflight:model_a",)),
        pipeline.Stage(name="sync", commands=(["/sync_model_b"],), model_id="model_b"),
        pipeline.Stage(name="runtime-preflight", commands=(["/preflight_model_b"],), model_id="model_b", dependencies=("sync:model_b",)),
        pipeline.Stage(name="evaluation", commands=(["/eval_model_b"],), model_id="model_b", dependencies=("runtime-preflight:model_b",)),
    ]

    rc = pipeline.run_stages(stages, dry_run=False, continue_on_error=True, state_path=state)

    # model_b executed fully
    assert any("eval_model_b" in str(a) for a in called)
    # model_a dependents skipped
    assert not any("preflight_model_a" in str(a) for a in called)
    assert not any("eval_model_a" in str(a) for a in called)

    # Final status is PARTIAL
    data = pipeline.load_pipeline_state(state)
    assert data["stages"]["sync:model_a"]["status"] == "FAILED"
    assert data["stages"]["runtime-preflight:model_a"]["status"] == "SKIPPED_DEPENDENCY"
    assert data["stages"]["evaluation:model_b"]["status"] == "COMPLETE"
    assert rc == 1
