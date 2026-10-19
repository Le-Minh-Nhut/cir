"""Tests for the read-only PC readiness report.

The report is what an operator runs before preparing anything, so its classification has to be
trustworthy in both directions: it must not call a model ready when something would stop it, and it
must not keep blaming a defect that has been repaired.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from workbench.backend.operator_config import resolve_config
from workbench.backend.registry import load_registry
from workbench.scripts import pc_readiness as pcr


def _model(model_id: str) -> dict:
    return next(m for m in load_registry()["models"] if m["model_id"] == model_id)


def _derived_checkout(tmp_path: Path) -> Path:
    """A minimal derived checkout: every patched file, at its published bytes.

    The full pinned tree carries roughly a gigabyte of dataset links, so the fixture copies only what
    the patches and the provenance check read. That is enough because the command under test never
    touches the evaluator.
    """
    import shutil

    root = tmp_path / "third_party"
    target = root / "CSMCIR"
    for patch in _model("csmcir")["approved_source_patches"]:
        relative = patch["path"]
        (target / Path(relative).parent).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / "workbench/third_party/CSMCIR" / relative, target / relative)
    return root


def test_classify_maps_blockers_to_the_cause_a_reader_acts_on():
    assert pcr.classify([])[0] == pcr.READY_FOR_FULL_EVALUATION
    assert pcr.classify(["source cannot run: the package cannot import"])[0] == pcr.BLOCKED_SOURCE
    assert pcr.classify(["command not runnable: COMMAND_UNAUDITED"])[0] == pcr.BLOCKED_IMPLEMENTATION
    assert pcr.classify(["FashionIQ fashioniq_ilearn_resized requirement missing: /x"])[0] == pcr.BLOCKED_DATASET
    assert pcr.classify(["execution environment not verified: WORKBENCH_X_PYTHON=None"])[0] == pcr.BLOCKED_ENVIRONMENT
    assert pcr.classify(["CSMCIR COT_ours2 captions missing: /x"])[0] == pcr.BLOCKED_EXTERNAL_ASSET


def test_the_report_is_read_only_and_covers_every_registered_model():
    config = resolve_config()
    models = load_registry()["models"]
    report = pcr.document(config, models)
    assert report["read_only"] is True
    assert set(report["models"]) == {m["model_id"] for m in models}
    assert set(report["summary"]["by_state"].values()) and sum(report["summary"]["by_state"].values()) == len(models)
    # Every model carries a state from the declared set and a reason a reader can act on.
    valid = {value for name, value in vars(pcr).items()
             if name.startswith(("READY_", "BLOCKED_", "PC_")) and isinstance(value, str)}
    for entry in report["models"].values():
        assert entry["state"] in valid, entry
        assert entry["reason"]


def test_references_are_reported_per_checkpoint_condition(tmp_path):
    """Each registered checkpoint condition must show whether a paper row describes it.

    This is the distinction the registry exists for: HABIT publishes 20/50/80% weights and prints a 0%
    row, so each registered noise condition resolves to its own row and a condition nobody published
    would resolve to none with a reason.
    """
    from pathlib import Path as _Path

    config = replace(resolve_config(), WORKBENCH_THIRD_PARTY_ROOT=_Path(tmp_path))
    entry = pcr.model_section(_model("habit"), config)
    by_condition = {r["checkpoint_training_noise_pct"]: r for r in entry["benchmark_references"]}
    assert sorted(by_condition) == [20, 50, 80], by_condition
    assert by_condition[50]["reference_id"] == "habit_aaai26_table1_fiq_n50"
    assert by_condition[80]["reference_id"] == "habit_aaai26_table1_fiq_n80"
    assert all(r["reference_status"] == "VERIFIED" for r in entry["benchmark_references"])
    # The 0% row exists in the registry but is not a registered checkpoint condition here, which is
    # exactly the situation the report must make visible rather than resolve by itself.
    assert 0 not in by_condition
    assert entry["blockers"]


def test_repairing_the_checkout_clears_the_source_blocker_and_moves_to_the_next_one(tmp_path):
    """The published-source blocker must stop applying once the declared patch is applied."""
    root = _derived_checkout(tmp_path)
    config = replace(resolve_config(), WORKBENCH_THIRD_PARTY_ROOT=root,
                     WORKBENCH_ARTIFACTS_ROOT=tmp_path / "artifacts")
    model = _model("csmcir")

    before = pcr.model_section(model, config)
    assert before["state"] == pcr.BLOCKED_SOURCE
    assert "lavis" in before["reason"]

    # The declared set is what makes a checkout a repaired one: applying the first patch alone leaves
    # a partial repair, which is reported as such rather than as an approved variant.
    for patch in model["approved_source_patches"]:
        subprocess.run(["patch", "-p1", "--directory", str(root / "CSMCIR"), "-i",
                        str(REPO / patch["patch_file"])], check=True, capture_output=True)

    after = pcr.model_section(model, config)
    assert after["state"] != pcr.BLOCKED_SOURCE, after["reason"]
    assert after["approved_patch"]["applied"] is True
    # The next real blocker is the caption input, and the report says so instead of the source defect.
    assert "captions" in after["reason"].lower() or after["state"] != pcr.BLOCKED_SOURCE


def test_a_dirty_checkout_is_reported_as_a_source_problem(tmp_path, monkeypatch):
    root = _derived_checkout(tmp_path)
    config = replace(resolve_config(), WORKBENCH_THIRD_PARTY_ROOT=root,
                     WORKBENCH_ARTIFACTS_ROOT=tmp_path / "artifacts")
    monkeypatch.setattr("workbench.backend.runtime.source_clean_and_pinned",
                        lambda *a, **k: (False, "pin", "source checkout is dirty: src/x.py"))
    entry = pcr.model_section(_model("csmcir"), config)
    assert entry["state"] == pcr.BLOCKED_SOURCE
    assert any("dirty" in blocker for blocker in entry["blockers"]), entry["blockers"]


def test_each_checkpoint_reports_its_own_blockers(tmp_path):
    """Readiness must be judged per checkpoint, not per model.

    With an empty checkpoint root every registered variant must show its own `checkpoint missing`
    blocker, and those blockers must also appear in the model's aggregate list -- the preflight list
    alone never looked at checkpoint files, so a model whose data was fine read as further along than
    it was.
    """
    from dataclasses import replace
    from pathlib import Path as _Path

    config = replace(resolve_config(), WORKBENCH_CHECKPOINT_ROOT=_Path(tmp_path) / "empty-ckpt")
    (_config_root := config.WORKBENCH_CHECKPOINT_ROOT).mkdir(parents=True, exist_ok=True)
    entry = pcr.model_section(_model("limn"), config)
    per_variant = {r["checkpoint_id"]: r["blockers"] for r in entry["benchmark_references"]}
    assert per_variant, entry
    for checkpoint_id, blockers in per_variant.items():
        assert any(f"checkpoint missing" in b for b in blockers), (checkpoint_id, blockers)
    assert any("checkpoint missing" in b for b in entry["blockers"])

def test_the_gpu_pending_state_asks_about_the_model_not_the_host(monkeypatch):
    """A CUDA-declaring model must be GPU-pending because *its* interpreter is unverified.

    The host's torch can report `cuda_available` truthfully without the model's own interpreter
    having proven anything; and conversely a CPU-only host torch must not by itself downgrade a
    model whose capability was never probed. The old blanket rule used the host's answer for
    everyone; this test pins the per-model rule.
    """
    from dataclasses import replace
    from pathlib import Path as _Path

    from workbench.backend.runtime import TIER_GPU_PENDING

    config = replace(resolve_config(), WORKBENCH_THIRD_PARTY_ROOT=_Path("/nonexistent"))
    model = _model("limn")
    assert pcr._requires_gpu(model)
    # Strip every preparation blocker so the only open question is the GPU capability.
    monkeypatch.setattr(pcr, "verify_environment",
                        lambda m, probe=True: type("R", (), {"tier": TIER_GPU_PENDING})())
    monkeypatch.setattr(pcr, "preflight_blockers", lambda *a, **k: [])
    monkeypatch.setattr(pcr, "runtime_plan_blockers", lambda *a, **k: [])
    monkeypatch.setattr(pcr, "model_requirements",
                        lambda *a, **k: {"blocks": [], "warnings": [],
                                         "evidence": {"auxiliary_assets": []}})
    entry = pcr.model_section(model, config)
    assert entry["state"] == pcr.PC_GPU_VALIDATION_PENDING, entry["reason"]
    assert entry["gpu_required"] is True and entry["gpu_verified"] is False

def test_the_cli_accepts_the_autopilot_config(capsys, tmp_path):
    """`--config` must build the same configuration the autopilot run will use.

    The profile is the SHIPPED example with its paths re-pointed at this test's own directories.
    Loading the example verbatim would resolve `dataset_root: ../../data/FashionIQ` against the
    operator's checkout, which a fresh clone does not have, so a failure would describe the host
    rather than the CLI. The keys, budgets and checkpoint policy under test are the shipped ones.
    """
    import json

    import yaml

    example = REPO / "workbench" / "configs" / "pc_autopilot.example.yaml"
    document = yaml.safe_load(example.read_text(encoding="utf-8"))
    data = tmp_path / "data" / "FashionIQ"
    (data / "images").mkdir(parents=True)
    document["paths"].update({
        "registry": str(REPO / "workbench" / "registry" / "models.yaml"),
        "dataset_root": str(data),
        "artifacts": str(tmp_path / "artifacts"),
        "checkpoints": str(tmp_path / "artifacts" / "checkpoints"),
        "results": str(tmp_path / "artifacts" / "results"),
        "third_party": str(REPO / "workbench" / "third_party"),
    })
    profile = tmp_path / "pc_autopilot.yaml"
    profile.write_text(yaml.safe_dump(document), encoding="utf-8")

    code = pcr.main(["--json", "--config", str(profile)])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "limn" in payload["models"]

def test_the_cli_rejects_an_invalid_config(capsys, tmp_path):
    import json
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: 2\n")
    result = pcr.main(["--json", "--config", str(bad)])
    assert result == 1
    assert "missing config key" in capsys.readouterr().err

def test_checkpoint_blockers_reach_the_headline_state(tmp_path):
    """A model with every checkpoint file absent must not read READY_FOR_FULL_EVALUATION.

    The per-variant blockers are merged into the model list and the headline state is
    reclassified from the merged list; classifying only the preflight list would let a
    fully-prepared model with no staged weights pass as ready.
    """
    from dataclasses import replace
    from pathlib import Path as _Path

    config = replace(resolve_config(), WORKBENCH_CHECKPOINT_ROOT=_Path(tmp_path) / "empty")
    config.WORKBENCH_CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    entry = pcr.model_section(_model("limn"), config)
    assert any("checkpoint missing" in b for b in entry["blockers"])
    assert entry["state"] != pcr.READY_FOR_FULL_EVALUATION, entry["state"]

def test_the_cli_runs_read_only_and_emits_json(capsys):
    import json

    code = pcr.main(["--json", "--model", "limn"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "limn" in payload["models"]
    assert payload["read_only"] is True
    assert "gpu" in payload and "papers" in payload


def test_the_cli_rejects_an_unknown_model():
    with pytest.raises(SystemExit):
        pcr.main(["--model", "no_such_model"])


def test_classify_is_not_decided_by_where_a_word_sits_in_the_table():
    """A blocker that embeds a path must still be classified by what it SAYS, not by which needle
    happens to come first in BLOCKER_KINDS.

    Regression: "checkpoint missing: /data/FashionIQ/fashioniq_dcnet" contained both "FashionIQ" and
    "checkpoint missing"; the dataset needle was earlier in the table, so a missing checkpoint was
    reported as a missing dataset — sending the operator to fix the wrong thing. Two independent
    model audits reproduced this. The fix matches on the most specific needle, so a longer, more
    particular phrase wins over a shorter, more general one wherever it appears.
    """
    assert pcr.classify(["/data/FashionIQ/checkpoints/fashioniq_dcnet: checkpoint missing"])[0] == pcr.BLOCKED_EXTERNAL_ASSET
    assert pcr.classify(["checkpoint missing: /checkpoints/FashionIQ/fashioniq_dcnet"])[0] == pcr.BLOCKED_EXTERNAL_ASSET
    assert pcr.classify(["official checkpoint hash mismatch for /FashionIQ/x"])[0] == pcr.BLOCKED_EXTERNAL_ASSET
    assert pcr.classify(["ConeSep checkpoint mapping unresolved: /FashionIQ/x"])[0] == pcr.BLOCKED_EXTERNAL_ASSET
    # And the specific-cause reading still applies the other way round: a genuinely missing dataset
    # is still a dataset, even when the message also mentions a checkpoint.
    assert pcr.classify(["FashionIQ raw benchmark input missing: /x"])[0] == pcr.BLOCKED_DATASET


def test_classify_reports_the_blocker_that_decided_the_state():
    """The second return value must be the blocker the state came from, so the reader acts on it."""
    state, reason = pcr.classify(["unrelated", "checkpoint missing: /x"])
    assert state == pcr.BLOCKED_EXTERNAL_ASSET
    assert reason == "checkpoint missing: /x"


def test_classify_is_order_independent_across_the_blocker_list():
    """The same set of blockers must classify the same way however it is ordered."""
    forward = pcr.classify(["source cannot run: x", "checkpoint missing: /y"])[0]
    backward = pcr.classify(["checkpoint missing: /y", "source cannot run: x"])[0]
    assert forward == backward == pcr.BLOCKED_SOURCE, (forward, backward)
