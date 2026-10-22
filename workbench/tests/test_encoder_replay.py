"""Tests for ENCODER replay wrapper and preflight validation."""
from __future__ import annotations

from pathlib import Path
import pytest

from workbench.replay.encoder import (
    AUXILIARY_ASSET_NAME,
    CHECKPOINTS,
    DEFAULT_CHECKPOINT,
    MODEL,
    PROTOCOL,
    _UNUSED_CHECKPOINT_KEYS,
    _assert_checkpoint_loaded,
    _unused_checkpoint_keys,
    preflight,
    validate_auxiliary_assets,
    validate_checkpoint,
    validate_dataset,
    validate_source,
)


def test_encoder_protocol_is_val_split():
    assert PROTOCOL == "fashioniq_val_split"

# --- the released checkpoint's two unused entries ----------------------------------------------
#
# The author artifact carries `backbone.binding_decoder.fc.0.{weight,bias}`; the pinned
# `BindingDecoder` declares no `fc` and no forward path reads one, so those two tensors are weights
# for a layer that does not exist in this revision. The allowance is narrow on purpose: it is bound
# to the exact artifact by digest, to the registry's verified mapping, to the pinned clean source,
# and to the keys being genuinely undeclared by the module. Each test below removes one of those
# conditions and requires the allowance to disappear, so a future edit cannot widen it by accident.

class _DeclaresNoKeys:
    def state_dict(self):
        return {}


class _DeclaresTheWeight:
    def state_dict(self):
        return {"backbone.binding_decoder.fc.0.weight": None}


def _pinned_source() -> Path:
    from workbench.backend.operator_config import resolve_config
    return (resolve_config().WORKBENCH_THIRD_PARTY_ROOT / MODEL["source_dir"]).resolve()

def test_the_released_checkpoint_unused_keys_are_permitted():
    """The positive control: the real artifact, its real registry entry, the real pinned source."""
    permitted = _unused_checkpoint_keys(_DeclaresNoKeys(), DEFAULT_CHECKPOINT, _pinned_source())
    assert permitted == _UNUSED_CHECKPOINT_KEYS
    assert permitted == {"backbone.binding_decoder.fc.0.weight", "backbone.binding_decoder.fc.0.bias"}

def test_a_different_checkpoint_gets_no_allowance():
    """A digest that is not the observed artifact cannot borrow the exception."""
    variant = {**DEFAULT_CHECKPOINT, "expected_sha256": "b" * 64}
    assert _unused_checkpoint_keys(_DeclaresNoKeys(), variant, _pinned_source()) == frozenset()

def test_an_unverified_mapping_gets_no_allowance():
    variant = {**DEFAULT_CHECKPOINT, "checkpoint_mapping_status": "UNVERIFIED"}
    assert _unused_checkpoint_keys(_DeclaresNoKeys(), variant, _pinned_source()) == frozenset()

def test_a_module_that_declares_the_key_gets_no_allowance():
    """If the model ever declares it, the key is a real parameter and must not be dropped."""
    assert _unused_checkpoint_keys(_DeclaresTheWeight(), DEFAULT_CHECKPOINT, _pinned_source()) == frozenset()

def test_an_unpinned_source_gets_no_allowance(tmp_path: Path):
    """The "the model never reads it" claim is about the code that will run, so the pin must hold."""
    assert _unused_checkpoint_keys(_DeclaresNoKeys(), DEFAULT_CHECKPOINT, tmp_path) == frozenset()

def test_the_unused_keys_are_absent_from_the_pinned_source():
    """The claim the allowance rests on, checked against the checkout rather than asserted."""
    source = _pinned_source()
    model_source = (source / "model_try2.py").read_text(encoding="utf-8")
    assert "self.fc" in model_source, "sanity: Backbone does declare an fc of its own"
    # The allowance names `binding_decoder.fc` specifically; nothing in the tree may read that path.
    for path in source.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        assert "binding_decoder.fc" not in text, f"{path} references the permitted-but-unused path"



def test_encoder_checkpoint_variants_registered():
    assert "fashioniq" in CHECKPOINTS


def test_encoder_preflight_fails_on_empty_paths(tmp_path: Path):
    source_dir = tmp_path / "source"
    ckpt_dir = tmp_path / "checkpoints"
    data_dir = tmp_path / "dataset"
    source_dir.mkdir()
    ckpt_dir.mkdir()
    data_dir.mkdir()

    blockers = preflight(ckpt_dir, data_dir, source_dir)
    assert any("source" in b.lower() for b in blockers)
    assert any("checkpoint" in b.lower() for b in blockers)
    assert any("dataset" in b.lower() for b in blockers)
    assert any("auxiliary asset" in b.lower() for b in blockers)


def test_encoder_auxiliary_asset_validation(tmp_path: Path):
    blockers = validate_auxiliary_assets(tmp_path)
    assert len(blockers) == 1
    assert AUXILIARY_ASSET_NAME in blockers[0]

    asset = tmp_path / AUXILIARY_ASSET_NAME
    asset.write_bytes(b"dummy open_clip model")
    assert validate_auxiliary_assets(tmp_path) == []


def test_encoder_test_function_signature():
    encoder_source = Path(__file__).resolve().parents[2] / "workbench" / "third_party" / "ENCODER"
    test_py = encoder_source / "test.py"
    assert test_py.is_file()
    for line in test_py.read_text().splitlines():
        if line.startswith("def test("):
            assert line.strip() == "def test(params, model, testset, category):"
            break
    else:
        pytest.fail("def test(...) not found in ENCODER test.py")

class _FakeReport:
    def __init__(self, missing=(), unexpected=()):
        self.missing_keys = list(missing)
        self.unexpected_keys = list(unexpected)


def test_the_loader_gate_allows_only_the_justified_keys():
    """The gate the evaluator actually calls, not just the helper behind it.

    A regression that only exercised `_unused_checkpoint_keys` would still pass if the loader
    stopped consulting it, so this drives `_assert_checkpoint_loaded` directly: the released
    artifact's two entries are tolerated, and it reports which ones it skipped.
    """
    _assert_checkpoint_loaded(_DeclaresNoKeys(), _FakeReport(
        unexpected=sorted(_UNUSED_CHECKPOINT_KEYS)), DEFAULT_CHECKPOINT, _pinned_source())

def test_the_loader_gate_still_refuses_an_unrelated_unexpected_key():
    with pytest.raises(ValueError, match="unexpected key"):
        _assert_checkpoint_loaded(_DeclaresNoKeys(), _FakeReport(
            unexpected=["backbone.something_else.weight"]), DEFAULT_CHECKPOINT, _pinned_source())

def test_the_loader_gate_still_refuses_a_missing_parameter():
    """Parameter coverage is the check that keeps a wrong-schema checkpoint from scoring."""
    with pytest.raises(ValueError, match="missing parameter"):
        _assert_checkpoint_loaded(_DeclaresNoKeys(), _FakeReport(
            missing=["backbone.clip.positional_embedding"]), DEFAULT_CHECKPOINT, _pinned_source())

def test_the_loader_gate_refuses_when_the_allowance_does_not_apply():
    """Same keys, different artifact: the digest gate must withhold the allowance."""
    with pytest.raises(ValueError, match="unexpected key"):
        _assert_checkpoint_loaded(_DeclaresNoKeys(), _FakeReport(
            unexpected=sorted(_UNUSED_CHECKPOINT_KEYS)),
            {**DEFAULT_CHECKPOINT, "expected_sha256": "c" * 64}, _pinned_source())
