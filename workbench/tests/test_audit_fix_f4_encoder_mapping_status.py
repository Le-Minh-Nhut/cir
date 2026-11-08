"""F4: ENCODER's adapter must READ `checkpoint_mapping_status`, not assert it.

`model_encoder.py` used to construct the checkpoint record handed to `_assert_checkpoint_loaded`
with the literal `"checkpoint_mapping_status": "VERIFIED_METADATA"`. That literal is one of the four
gates in `replay/encoder.py::_unused_checkpoint_keys`, which decides whether the two provably-unread
`binding_decoder.fc.0.*` tensors may be skipped silently. With the literal in place a registry
downgrade could never tighten the loader: the adapter would keep claiming the strongest status no
matter what the registry said.

These tests therefore assert the DOWNGRADE case. A test that only asserted the happy path would pass
with the literal still present, so it would prove nothing.

Both the parent's payload hunk and the worker's read hunk are exercised: the job payload is written
by the real `Adapter._run_job`, and the record is built by the real `_worker_main`. The four line
edits are faked (replay modules, whole-model load, the model object), not the decision under test.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from workbench.composition import model_encoder
from workbench.replay import encoder as replay_encoder


class _FakeStateDict(dict):
    pass


class _FakeReport:
    """A load report whose only unusual entries are the two whitelistable tensors."""

    def __init__(self) -> None:
        self.missing_keys: list[str] = []
        self.unexpected_keys = sorted(replay_encoder._UNUSED_CHECKPOINT_KEYS)


class _FakeModel:
    """Stands in for `model_try2.Encoder`; records that the gate let it through."""

    def __init__(self) -> None:
        self.loaded = False

    def load_state_dict(self, state, strict=False):
        self.loaded = True
        return _FakeReport()

    def cuda(self):
        return self

    def eval(self):
        return self

    def state_dict(self):
        # The pinned module does NOT declare the whitelisted names.
        return {"backbone.something.weight": object()}


def _open_clip_stub():
    """Both `import open_clip` and the returned module must be the same object."""
    return SimpleNamespace(
        create_model_and_transforms=lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("REACHED_PAST_MAPPING_GATE")))


def _fake_replay_modules(model):
    def loader(source_root):
        # `create_model_and_transforms` is the next thing the worker touches after the mapping gate
        # accepts. Raising a sentinel there proves the gate let the load through.
        model_try2 = SimpleNamespace(Encoder=lambda **kwargs: model)
        return object(), _open_clip_stub(), object(), object(), model_try2

    return loader


@pytest.fixture
def encoder_job(tmp_path, monkeypatch):
    """Run the real worker on a real job file, with the four external edges faked."""

    def run(status: str, sentinel: str = "REACHED_PAST_MAPPING_GATE"):
        model = _FakeModel()
        worker_dir = tmp_path / "worker"
        worker_dir.mkdir(exist_ok=True)
        job_path = worker_dir / f"job-{status}.json"
        job_path.write_text(json.dumps({
            "phase": "queries",
            "image_paths": [str(tmp_path / "image.png")],
            "texts": ["edit"],
            "categories": [None],
            "output_path": str(worker_dir / "output.npy"),
            "checkpoint_path": str(tmp_path / "checkpoint.pt"),
            "expected_sha256": replay_encoder._UNUSED_KEYS_CHECKPOINT_SHA,
            # This is what the parent's patched payload now carries. When the parent still
            # hard-coded the status, this key did not exist and the worker asserted the literal.
            "checkpoint_mapping_status": status,
            "source_root": str(tmp_path),
            "dataset_root": str(tmp_path),
            "batch_size": 1,
        }), encoding="utf-8")

        monkeypatch.setattr(replay_encoder, "_load_replay_modules", _fake_replay_modules(model))
        monkeypatch.setattr(replay_encoder, "source_pin_status", lambda *a, **k: (True, None))
        # The worker does a plain `import open_clip` before it builds the checkpoint record, so the
        # name must be importable even though the test never reaches a real transform.
        monkeypatch.setitem(sys.modules, "open_clip", _open_clip_stub())
        monkeypatch.setattr("workbench.replay.common.load_trusted_whole_model",
                            lambda *a, **k: _FakeStateDict({"anything": 1}))
        # No need to fake the thread/decode helpers: both paths return before reaching them --
        # UNVERIFIED refuses at the mapping gate, VERIFIED stops at `create_model_and_transforms`.

        cwd = Path.cwd()
        try:
            model_encoder._worker_main(job_path)
        finally:
            os.chdir(cwd)
        return model

    return run


def _capture(capsys) -> dict:
    out = capsys.readouterr().out
    line = next(line for line in out.splitlines() if line.startswith("CIRCUS_WORKER:"))
    return json.loads(line[len("CIRCUS_WORKER:"):])


def test_downgraded_registry_status_is_obeyed(encoder_job, capsys) -> None:
    """`UNVERIFIED` in the registry must refuse, not get silently upgraded to VERIFIED_METADATA."""
    encoder_job("UNVERIFIED")
    response = _capture(capsys)

    assert response["ok"] is False, (
        "a downgraded mapping status must refuse the load; accepting it proves the literal survives")
    assert "does not match the model" in response["error"]
    assert "unexpected" in response["error"]


def test_absent_registry_status_is_obeyed_and_fails_closed(encoder_job, capsys) -> None:
    """A missing key must not default to the permissive status."""
    encoder_job(None)
    response = _capture(capsys)
    assert response["ok"] is False
    assert "does not match the model" in response["error"]


def test_verified_status_still_accepts_the_two_unused_tensors(encoder_job, capsys) -> None:
    """The happy path is preserved: behaviour today is unchanged by the fix."""
    encoder_job("VERIFIED_METADATA")
    response = _capture(capsys)
    # The gate accepted, so the worker went on to the transform, where the fixture raises.
    assert response["ok"] is False
    assert response["error"] == "REACHED_PAST_MAPPING_GATE", response


def test_adapter_reads_status_from_the_registry_record() -> None:
    """The adapter's own record carries the registry value, so the payload can transmit it."""
    from workbench.backend.registry import checkpoint_by_id, model_by_id

    model = model_by_id("encoder")
    checkpoint_id = model.get("per_query_checkpoint")
    checkpoint = checkpoint_by_id(model, checkpoint_id)

    # The value the patch transmits exists in the registry, and is not a test-only invention.
    assert "checkpoint_mapping_status" in checkpoint
    assert checkpoint["checkpoint_mapping_status"] == "VERIFIED_METADATA"
    assert checkpoint["expected_sha256"] == replay_encoder._UNUSED_KEYS_CHECKPOINT_SHA

    # The literal is gone from both sites that used to assert it.
    source = Path(model_encoder.__file__).read_text(encoding="utf-8")
    assert '"checkpoint_mapping_status": "VERIFIED_METADATA"' not in source, (
        "the worker must read the status, not assert it")
    assert '"checkpoint_mapping_status": self._checkpoint.get("checkpoint_mapping_status")' in source
    assert '"checkpoint_mapping_status": job.get("checkpoint_mapping_status")' in source


def test_gate_accepts_the_shipped_registry_value() -> None:
    """Confirm the shipped status still lets the two unused tensors through, as before."""
    from workbench.backend.registry import checkpoint_by_id, model_by_id

    model = model_by_id("encoder")
    checkpoint = checkpoint_by_id(model, model.get("per_query_checkpoint"))
    permitted = replay_encoder._unused_checkpoint_keys(
        _FakeModel(), checkpoint, Path("/home/heheboiz/data/cir/workbench/third_party/ENCODER"))
    assert set(permitted) == set(replay_encoder._UNUSED_CHECKPOINT_KEYS)
