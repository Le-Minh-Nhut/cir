"""R03: the run identity must cover the relation tensor the features were ACTUALLY computed with.

The defect: `runner.py` built `identity` (including `_relation_identity(adapter.relation_state)`)
immediately after `Adapter.__init__`, where ENCODER's `relation_state` is still `None` because the
worker only fills it in when it replies to an encode job. So the run id, the fingerprint and the
summary all recorded `not_recorded` while the features came from the seeded `learnable_relation`.

These tests drive the REAL `run_composition` with a synthetic adapter that mimics that lifecycle --
NumPy features, no model, no GPU -- and assert the four states the fix must distinguish:

1. the identity hash covers the value `prepare_identity()` produced (fails before the hook existed);
2. a relation that CHANGES during encoding fails the run instead of being written beside the numbers;
3. a relation-bearing adapter that cannot name its draw is REFUSED, never published as
   `not_recorded`;
4. a stale rank payload recording no relation is never reused.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from workbench.composition import runner
from workbench.composition.adapters import CompositionAdapter
from workbench.composition.data import Gallery, Query, ReleaseData

DIGEST = "a" * 64
OTHER_DIGEST = "b" * 64


class _LifecycleAdapter(CompositionAdapter):
    """`relation_state` is None until the handshake, exactly like ENCODER's worker reply."""

    model_id = "deferred_probe"
    #: How the run should behave: "recorded" (the fix), "drift" (changes during encoding),
    #: "cannot_name" (handshake produces nothing).
    mode = "recorded"

    def __init__(self, config, checkpoint_id=None):
        super().__init__(config, checkpoint_id)
        self.relation_state = None
        self.identity_state = None

    def prepare_identity(self):
        if self.mode == "cannot_name":
            return None
        self.relation_state = {"seed": 20261111, "sha256": DIGEST,
                               "expected_sha256": DIGEST, "matches_pinned_digest": True}
        self.identity_state = dict(self.relation_state)
        return self.relation_state

    def encode_targets(self, image_paths, categories):
        _drift(self)
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(image_paths), 1))

    def encode_queries(self, image_paths, texts, categories):
        _drift(self)
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(image_paths), 1))


class _BearingLifecycleAdapter(_LifecycleAdapter):
    relation_bearing = True


def _drift(adapter) -> None:
    if adapter.mode == "drift":
        adapter.relation_state = {"seed": 20261111, "sha256": OTHER_DIGEST,
                                  "expected_sha256": DIGEST, "matches_pinned_digest": False}


def _run(tmp_path: Path, adapter: type[CompositionAdapter], **kwargs):
    image = tmp_path / "reference.png"
    Image.new("RGB", (5, 5), "white").save(image)
    config = SimpleNamespace(
        WORKBENCH_ARTIFACTS_ROOT=tmp_path / "artifacts", FASHIONIQ_ROOT=tmp_path,
        WORKBENCH_THIRD_PARTY_ROOT=tmp_path / "vendor",
        WORKBENCH_CHECKPOINT_ROOT=tmp_path / "checkpoints")
    data = ReleaseData(
        Gallery("probe_protocol", ("images/reference.png", "images/target.png")),
        (Query("q0", 0, "caption", "r", ("t",), None, ("full",)),),
        {"r": "images/reference.png", "t": "images/target.png"},
        {"is_synthetic": True})
    spec = runner.ModelSpec(adapter.model_id, "checkpoint", adapter, {}, "checkpoint_hash")
    with patch.object(runner, "model_interpreter_identity", return_value={"probe": "synthetic"}), \
            patch.object(runner, "_gpu_identity", return_value=None), \
            patch.object(runner, "composition_variant_provenance",
                         return_value={"probe": "synthetic"}):
        return runner.run_composition(
            data, [spec], config, subsets=["full"], batch_size=2,
            conditions=["multimodal"], image_resolver=lambda *_: image,
            synthetic=True, keep_features=True, **kwargs)


def _summaries(tmp_path: Path) -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((tmp_path / "artifacts/composition/summary").rglob("*.json"))]


def _rank_payloads(tmp_path: Path) -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((tmp_path / "artifacts/composition/runs").rglob("ranks/*.json"))]


def test_the_identity_covers_the_state_the_handshake_produced(tmp_path: Path) -> None:
    """Before the hook, `identity['relation_state']` was `not_recorded` and the id proved nothing."""
    adapter = type("Recorded", (_LifecycleAdapter,), {"mode": "recorded"})
    report = _run(tmp_path, adapter)

    assert report["models"][0]["status"] == "COMPLETED", report["models"][0]
    payload = _rank_payloads(tmp_path)[0]
    identity = payload["identity"]
    assert identity["relation_state"]["status"] == "recorded"
    assert identity["relation_state"]["sha256"] == DIGEST
    # The id is a function of THAT identity, so it covers the real tensor rather than the pre-encode
    # `None`: this is the claim the defect made false.
    assert payload["fingerprint"] == runner._digest(identity)
    assert report["models"][0]["run_id"] == runner._digest(identity)[:24]
    for summary in _summaries(tmp_path):
        assert summary["relation_state"]["sha256"] == DIGEST


def test_encoding_with_a_different_relation_fails_the_run(tmp_path: Path) -> None:
    """A run whose features came from another tensor is not a measurement of the id it minted."""
    adapter = type("Drifting", (_LifecycleAdapter,), {"mode": "drift", "relation_bearing": True})
    report = _run(tmp_path, adapter)

    model = report["models"][0]
    assert model["status"] == "FAILED", model
    assert "relation_state_drift" in model["reason"], model
    for summary in _summaries(tmp_path):
        assert summary["execution_status"] == "FAILED"
        # Never `recorded`: the features and the identity disagree, so no digest is claimed.
        assert summary["relation_state"].get("sha256") != DIGEST


def test_a_relation_bearing_adapter_that_cannot_name_its_draw_is_refused(tmp_path: Path) -> None:
    """`not_recorded` for a model that draws one is a refusal, and it is visible as one."""
    adapter = type("Unnamed", (_BearingLifecycleAdapter,), {"mode": "cannot_name"})
    report = _run(tmp_path, adapter)

    model = report["models"][0]
    assert model["status"] == "FAILED", model
    assert "relation_state_unrecorded" in model["reason"], model
    for summary in _summaries(tmp_path):
        assert summary["execution_status"] == "FAILED"
        assert summary["relation_state"]["status"] == "not_recorded"


def test_a_stale_payload_recording_no_relation_is_not_reused(tmp_path: Path) -> None:
    """The reuse path must not hand back ranks an unpinned draw produced."""
    adapter = type("Recorded", (_LifecycleAdapter,), {"mode": "recorded"})
    _run(tmp_path, adapter)
    payloads = list((tmp_path / "artifacts/composition/runs").rglob("ranks/*.json"))
    assert payloads
    payload = json.loads(payloads[0].read_text(encoding="utf-8"))

    recorded = payload["identity"]["relation_state"]
    assert runner._recorded_relation_matches(payload, recorded) is True

    stale = json.loads(payloads[0].read_text(encoding="utf-8"))
    stale["identity"]["relation_state"] = {"status": "not_recorded",
                                           "reason": "this artifact predates the field"}
    assert runner._recorded_relation_matches(stale, recorded) is False

    # A model that draws nothing keeps the old behaviour: fingerprint alone decides.
    assert runner._recorded_relation_matches(
        stale, {"status": "not_recorded", "reason": "no draw"}) is True


def test_the_hook_runs_before_the_identity_is_hashed() -> None:
    """Static: the call site is inside the construction block, above `identity = {`."""
    source = Path(runner.__file__).read_text(encoding="utf-8")
    hook = source.index("adapter.prepare_identity()")
    identity = source.index("            identity = {")
    assert hook < identity, (
        "prepare_identity() must be called before the identity is built, or the hash covers the "
        "pre-encode state (R03)")
    assert source.index("adapter = spec.adapter_factory(config, checkpoint_id)") < hook


@pytest.mark.parametrize("field", ["sha256", "expected_sha256", "matches_pinned_digest"])
def test_the_summary_block_carries_the_identity_fields(field: str) -> None:
    """The summary block is the EFFECTIVE state, and it keeps the fields the gate reads."""
    state = runner._relation_identity({"sha256": DIGEST, "expected_sha256": DIGEST,
                                       "matches_pinned_digest": True})
    assert field in state
