"""F1: `--seed` is recorded, never applied -- and must not mint a run id.

The audit found `seed` hashed into `identity` (`runner.py`) while no seeding call existed anywhere in
the composition path. A field that changes `run_id` and changes nothing else is the worst kind of
provenance: it looks like a reproducibility control and is decorative.

The honest fix is to stop pretending. Nothing in this path draws randomness at encode time, so
seeding it would be a claim about a variable that does not exist. Verified by reading:

* every encode-time transform is deterministic -- `targetpad_transform` is
  `TargetPad/Resize/CenterCrop/Normalize`, ENCODER uses open_clip `preprocess_val`, LIMN uses
  `open_clip.image_transform(..., is_train=False)`;
* every adapter calls `.eval()` and encodes under `torch.no_grad()`, so `nn.Dropout` and
  open_clip's `PatchDropout` (gated on `self.training`) are inactive;
* no `extract_retrieval_compose` body contains `torch.rand*`/`multinomial`/`bernoulli`/`F.dropout`;
* no `strict=False` load can leave a randomly-initialised parameter reachable: ENCODER, CSMCIR and
  TME refuse on `missing_keys`, and the others load a whole pickled module or use `strict=True`.

What DOES draw randomness is `metrics.paired_bootstrap_gap`, which takes its own explicit `seed`
argument and is never called by the composition runner. That seed is a function parameter, not a run
variable, so it does not belong in the run identity either.

Contract pinned here: changing `--seed` changes the recorded value and does NOT change the run id or
the results, so two runs that differ only in `--seed` share one artifact instead of forking the tree.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

from workbench.composition import runner
from workbench.tests.test_circus_cli import _fixture, _run


def _run_seed(config, release, adapters, artifacts, monkeypatch, capsys, seed: int):
    rc, report, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                             "--models", "fixture3d", "--seed", str(seed))
    assert rc == 0, error
    row = report["models"][0]
    summary = json.loads(Path(row["summaries"]["full"]).read_text(encoding="utf-8"))
    return row, summary


def _ranks(summary, artifacts: Path, attempt: str) -> list[int]:
    path = runner._result_outputs(
        artifacts, summary["protocol_id"], summary["model_id"], summary["checkpoint_id"],
        attempt, "full", summary["conditions"][0], synthetic=summary["is_synthetic"])
    return [json.loads(line)["rank"] for line in path.read_text(encoding="utf-8").splitlines()]


def test_changing_seed_does_not_change_the_run_id(tmp_path, monkeypatch, capsys) -> None:
    """`--seed` is recorded provenance, not a controlled variable: it must not fork the run id.

    One fixture, two runs in the same artifact tree: the identity binds resolved absolute paths, so
    two tmp roots would differ for a reason that has nothing to do with the seed.
    """
    config, release, adapters, artifacts = _fixture(tmp_path)
    first_row, first_summary = _run_seed(config, release, adapters, artifacts,
                                        monkeypatch, capsys, 0)
    second_row, second_summary = _run_seed(config, release, adapters, artifacts,
                                          monkeypatch, capsys, 7)

    assert first_summary["seed"] == 0
    assert second_summary["seed"] == 7, "the flag must still be recorded, it is just not applied"
    assert first_summary["seed_applied"] is False
    assert second_summary["seed_applied"] is False
    assert first_row["run_id"] == second_row["run_id"], (
        "seed is not applied to anything, so it must not mint a second run id for identical work")
    # The second run shares the run id, so its attempt gets the usual `-retry-1` suffix: the work
    # is recognised as the same identity, which is exactly the point.
    assert second_row["attempt"].startswith(first_row["run_id"])


def test_changing_seed_does_not_change_the_results(tmp_path, monkeypatch, capsys) -> None:
    """Pin the other half: a seed hook must not drift in without a declared identity change."""
    config, release, adapters, artifacts = _fixture(tmp_path)
    first_row, first_summary = _run_seed(config, release, adapters, artifacts,
                                        monkeypatch, capsys, 0)
    second_row, second_summary = _run_seed(config, release, adapters, artifacts,
                                          monkeypatch, capsys, 7)

    assert first_summary["metrics"] == second_summary["metrics"]
    assert (_ranks(first_summary, artifacts, first_row["attempt"])
            == _ranks(second_summary, artifacts, second_row["attempt"]))


def test_seed_is_absent_from_the_run_identity() -> None:
    """The identity digest must not contain a field it does not control."""
    source = inspect.getsource(runner.run_composition)
    identity_block = source[source.index("identity = {"):source.index("run_id = _digest(identity)")]
    assert '"seed"' not in identity_block, "seed must not be hashed into identity: nothing applies it"


def test_cli_documents_seed_as_recorded_only() -> None:
    """The help text must not promise determinism the flag cannot deliver."""
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location("cli_help_probe", scripts / "composition_eval.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    action = next(item for item in module._parser()._actions if item.dest == "seed")
    assert action.default == 0
    text = (action.help or "").lower()
    assert "record" in text
    assert "deterministic summaries" not in text, (
        "the old help text implied the seed made the run deterministic; it never did")
