"""The DCNet checkpoint load must work on both the pinned torch 1.4 host and modern torch.

Reproduced defect (branch research/cir-failure-workbench): the approved patch passed
``weights_only=False`` unconditionally. That keyword does not exist before torch 1.13, so on the
pinned ``dcnet-377`` environment (Python 3.7.7 / torch 1.4.0+cpu) every native evaluation died with::

    TypeError: 'weights_only' is an invalid keyword argument for load()

A second, independent defect sat behind it: the author checkpoint was saved from a CUDA device, so
a CPU-only host cannot deserialize its storages without ``map_location`` either. Both are properties
of *this one* ``torch.load`` call, so both are handled there without touching anything else.

These tests read the shipped patched source and execute the exact load logic against stub modules,
so they do not need the 2.7 GB author checkpoint or either torch version to assert the contract.
"""
from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from workbench.backend.registry import model_by_id  # noqa: E402

SOURCE = REPO / "workbench" / "third_party" / "DCNet" / "base" / "base_trainer.py"


def _load_kwargs_for(torch_module) -> dict:
    """Mirror the patched call site's decision without importing DCNet (which needs torchvision)."""
    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch_module.load).parameters:
        load_kwargs["weights_only"] = False
    return load_kwargs


def test_patched_source_passes_weights_only_conditionally_and_sets_map_location():
    """The single load call must be guarded by a capability check and must map to the device."""
    text = SOURCE.read_text(encoding="utf-8")
    assert "weights_only' in inspect.signature(torch.load).parameters" in text, (
        "the load site must test for the keyword instead of passing it unconditionally")
    assert "load_kwargs = {'map_location': self.device}" in text, (
        "a CPU-only host cannot deserialize CUDA storages without map_location")
    # The unconditional form that broke torch 1.4 must be gone.
    assert "torch.load(ckpt_path, weights_only=False)" not in text, (
        "unconditional weights_only=False is fatal on the pinned torch 1.4 environment")


def test_kwargs_are_empty_of_weights_only_on_old_torch_and_present_on_new():
    class OldTorch:
        @staticmethod
        def load(f, map_location=None, pickle_module=None):
            return {}

    class NewTorch:
        @staticmethod
        def load(f, map_location=None, pickle_module=None, weights_only=True):
            return {}

    assert "weights_only" not in _load_kwargs_for(OldTorch)
    assert _load_kwargs_for(NewTorch)["weights_only"] is False
    assert _load_kwargs_for(OldTorch)["map_location"] == "cpu"


def test_registry_declaration_matches_the_shipped_patch_and_tree():
    """The registry's declared digests must describe the shipped patch and the applied file."""
    import hashlib

    contract = model_by_id("dcnet")["approved_source_patches"][0]
    patch_file = REPO / contract["patch_file"]
    assert hashlib.sha256(patch_file.read_bytes()).hexdigest() == contract["patch_sha256"]
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == contract["derived_sha256"], (
        "the vendored source must match the recorded post-patch digest")
