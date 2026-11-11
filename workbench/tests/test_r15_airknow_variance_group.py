"""R15: an Air-Know artifact must not claim a variance it never measured.

The defect: `evaluate_categories` stamped `run_to_run_delta_macro_r10 = 0.4744` on every native-mode
run and `0.0` on every eval-mode run. Neither number came from the run being described -- 0.4744 is
the difference between two OTHER draws, and 0.0 was never measured at all. Each of the four committed
`raw/airknow_run3cat_gpu_run*.json` also carried the same four-run mean/sd/spread while claiming
`repeat_count: 1`-`4` for a single draw.

The fix: a statistic belongs to a NAMED GROUP of runs with one configuration, computed at emit time
from the artifacts beside it. One draw yields `n: 1`, `sd: null`, `insufficient_draws: true`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from workbench.replay.airknow import (  # noqa: E402
    HISTORICAL_CAMPAIGN_DELTA,
    PROTOCOL,
    TRANSFORM_DESCRIPTOR,
    VARIANCE_METRIC,
    assemble_variance_group,
    run_identity,
    variance_group_key,
)

IDENTITY = run_identity(variant_id="fiq_n05", eval_mode=False, protocol=PROTOCOL,
                        batch_size=64, transform=TRANSFORM_DESCRIPTOR)

def _artifact(path: Path, macro_r10: float, *, identity: dict | None = None,
              timestamp: str | None = None, transform: str | None = None) -> Path:
    """Write a sibling run in the shape the wrapper emits, and return its path."""
    identity = identity or IDENTITY
    document = {
        "model_id": "airknow",
        "protocol": identity["protocol"],
        "aggregate": {VARIANCE_METRIC: macro_r10, "macro_r50": 70.0, "complete": True},
        "timestamp": timestamp,
        "runtime": {
            "variant": identity["checkpoint_id"],
            "eval_mode": identity["eval_mode"],
            "batch_size": identity["batch_size"],
            "transform": transform if transform is not None else identity["transform"],
        },
    }
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path

# --------------------------------------------------------------- (a) one draw borrows nothing

def test_a_single_draw_reports_no_borrowed_sd(tmp_path: Path) -> None:
    """The committed artifacts' defect: an n=1 run claiming a spread measured elsewhere."""
    block = assemble_variance_group(tmp_path, IDENTITY, 52.9, timestamp="2026-11-11T00:00:00+00:00")

    assert block["n"] == 1, "an empty directory plus this run is one draw"
    assert block["sd"] is None, "sd was borrowed from another population"
    assert block["insufficient_draws"] is True
    assert block["mean"] == pytest.approx(52.9), "n=1 mean is this run's own value"
    assert block["spread"] == 0.0, "one draw cannot spread"
    assert block["value_of_this_run"] == pytest.approx(52.9)
    assert block["repeat_count"] == 1 and block["repeat_index"] == 1

def test_the_historical_campaign_figure_is_labelled_not_applicable() -> None:
    """A reader must not be able to mistake 0.4744 for this artifact's own delta."""
    assert HISTORICAL_CAMPAIGN_DELTA["applies_to_this_run"] is False
    assert HISTORICAL_CAMPAIGN_DELTA["value"] == pytest.approx(0.4744)
    assert "provenance" in HISTORICAL_CAMPAIGN_DELTA

# ------------------------------------------------------------- (b) two draws compute real stats

def test_two_draws_produce_the_computed_mean_sd_and_spread(tmp_path: Path) -> None:
    """The statistic exists only when the group really has two comparable runs."""
    _artifact(tmp_path / "run1.json", 52.924086629607395, timestamp="2026-11-10T13:45:47+00:00")
    _artifact(tmp_path / "run2.json", 52.44973964180556, timestamp="2026-11-10T13:53:18+00:00")

    block = assemble_variance_group(tmp_path, IDENTITY, 52.51441404945451,
                                    timestamp="2026-11-10T14:09:15+00:00")

    values = [52.924086629607395, 52.44973964180556, 52.51441404945451]
    assert block["n"] == 3
    assert block["mean"] == pytest.approx(sum(values) / 3)
    assert block["sd"] == pytest.approx(0.2572351280996118, abs=1e-12)
    assert block["spread"] == pytest.approx(max(values) - min(values))
    assert block["insufficient_draws"] is False
    assert block["repeat_count"] == 3 and block["repeat_index"] == 3, (
        "this run was emitted last, so it is the third draw of its group")
    assert len(block["group_members"]) == 3

def test_a_sibling_without_a_recorded_identity_is_excluded_not_pooled(tmp_path: Path) -> None:
    """An old artifact that never recorded its batch size cannot prove it measured the same thing."""
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"model_id": "airknow", "protocol": PROTOCOL,
                                  "aggregate": {VARIANCE_METRIC: 40.0},
                                  "runtime": {"variant": "fiq_n05", "eval_mode": False}}),
                      encoding="utf-8")

    block = assemble_variance_group(tmp_path, IDENTITY, 52.9, timestamp="2026-11-11T00:00:00+00:00")

    assert block["n"] == 1
    assert block["sd"] is None
    reasons = [entry["reason"] for entry in block["group_excluded"]]
    assert any("identity not recorded" in reason for reason in reasons), reasons

def test_a_different_config_is_a_different_group(tmp_path: Path) -> None:
    _artifact(tmp_path / "variant.json", 60.0,
              identity=run_identity(variant_id="fiq_n08", eval_mode=False, protocol=PROTOCOL,
                                    batch_size=64, transform=TRANSFORM_DESCRIPTOR))

    block = assemble_variance_group(tmp_path, IDENTITY, 52.9, timestamp="2026-11-11T00:00:00+00:00")

    assert block["n"] == 1, "a different checkpoint must not contribute to this run's sd"
    assert any("checkpoint_id" in entry["reason"] for entry in block["group_excluded"])

def test_the_group_key_names_the_configuration_not_the_moment() -> None:
    same = run_identity(variant_id="fiq_n05", eval_mode=False, protocol=PROTOCOL,
                        batch_size=64, transform=TRANSFORM_DESCRIPTOR)
    other = run_identity(variant_id="fiq_n05", eval_mode=True, protocol=PROTOCOL,
                         batch_size=64, transform=TRANSFORM_DESCRIPTOR)
    assert variance_group_key(same) == variance_group_key(dict(reversed(list(same.items()))))
    assert variance_group_key(same) != variance_group_key(other)

# ------------------------------------------------------------------- (c) mixed modes refuse

def test_a_mixed_mode_group_raises(tmp_path: Path) -> None:
    """Train-mode has live dropout; eval-mode does not. A pooled sd would describe two experiments."""
    _artifact(tmp_path / "evalmode.json", 52.9,
              identity=run_identity(variant_id="fiq_n05", eval_mode=True, protocol=PROTOCOL,
                                    batch_size=64, transform=TRANSFORM_DESCRIPTOR))

    with pytest.raises(ValueError, match="mixes eval modes"):
        assemble_variance_group(tmp_path, IDENTITY, 52.9, timestamp="2026-11-11T00:00:00+00:00")

def test_a_rewritten_output_file_is_not_its_own_draw(tmp_path: Path) -> None:
    """Writing the artifact under the same name must not make it contribute to its own group."""
    own = _artifact(tmp_path / "run.json", 99.0, timestamp="2026-11-10T00:00:00+00:00")

    block = assemble_variance_group(tmp_path, IDENTITY, 52.9,
                                    timestamp="2026-11-11T00:00:00+00:00", exclude_path=own)

    assert block["n"] == 1
    assert any("own output file" in entry["reason"] for entry in block["group_excluded"])

# ------------------------------------------------------------ (d) no unmeasured field survives

def test_the_wrapper_source_emits_no_unmeasured_delta() -> None:
    """Absence assertion, so a future edit cannot quietly reintroduce the field."""
    source = (REPO / "workbench" / "replay" / "airknow.py").read_text(encoding="utf-8")

    assert "MEASURED_RUN_TO_RUN_DELTA_MACRO_R10" not in source
    assert '"run_to_run_delta_macro_r10"' not in source, (
        "a per-run delta field is back; the only permitted figure is the labelled historical note")
    assert "0.0 if eval_mode else" not in source and "0.0 if args.eval_mode else" not in source
    assert "assemble_variance_group(" in source

# ------------------------------------------------- (d') the emitted ARTIFACT, not just source

def _written_artifact(tmp_path: Path, monkeypatch, evaluator, name: str = "airknow_run.json") -> dict:
    """Run `main` with the model launch stubbed out; return the artifact it wrote."""
    from workbench.replay import airknow

    monkeypatch.setattr(airknow, "preflight", lambda *a, **k: [])
    monkeypatch.setattr(airknow, "evaluate_categories", evaluator)
    output = tmp_path / "out" / name
    output.parent.mkdir(parents=True, exist_ok=True)
    code = airknow.main(["--checkpoint-root", str(tmp_path), "--dataset-root", str(tmp_path),
                         "--source-root", str(tmp_path), "--output", str(output)])
    assert code == 0
    return json.loads(output.read_text(encoding="utf-8"))

def _stub_evaluator(macro_r10: float):
    from workbench.replay.common import ReplayResult, macro_aggregate

    def evaluate(checkpoint_root, dataset_root, source_root, **kwargs):
        results = [ReplayResult(model_id="airknow", checkpoint_id="fiq_n05", category=category,
                                protocol=PROTOCOL, r1=1.0, r10=macro_r10, r50=2.0, complete=True,
                                provenance={"determinism": {"seed": None}})
                   for category in ("dress", "shirt", "toptee")]
        return results, macro_aggregate(results)

    return evaluate

def test_a_lone_artifact_declares_insufficient_draws(tmp_path: Path, monkeypatch) -> None:
    document = _written_artifact(tmp_path, monkeypatch, _stub_evaluator(52.9))

    determinism = document["determinism"]
    assert "run_to_run_delta_macro_r10" not in determinism
    group = determinism["variance_group"]
    assert group["n"] == 1 and group["sd"] is None and group["insufficient_draws"] is True
    assert group["repeat_index"] == 1 and group["repeat_count"] == 1
    assert determinism["historical_campaign_delta_macro_r10"]["applies_to_this_run"] is False
    # The per-category rows carry the same block: a per-row reader sees the same verdict.
    for row in document["categories"]:
        assert row["provenance"]["determinism"]["variance_group"]["n"] == 1

def test_a_second_artifact_in_the_same_directory_gets_the_group_statistic(tmp_path: Path,
                                                                        monkeypatch) -> None:
    """Two draws of one config: the second artifact's block covers both, computed from disk."""
    first = _written_artifact(tmp_path, monkeypatch, _stub_evaluator(52.924086629607395), "run1.json")
    second = _written_artifact(tmp_path, monkeypatch, _stub_evaluator(52.44973964180556), "run2.json")

    group = second["determinism"]["variance_group"]
    assert first["determinism"]["variance_group"]["n"] == 1
    assert group["n"] == 2, "the sibling artifact was not read"
    assert group["sd"] == pytest.approx(0.33541397171009174, abs=1e-9)
    assert group["repeat_count"] == 2 and group["insufficient_draws"] is False
    assert group["variance_group_key"] == \
        first["determinism"]["variance_group"]["variance_group_key"]
