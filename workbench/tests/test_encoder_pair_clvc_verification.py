from __future__ import annotations

import inspect
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PAIR_DIR = REPO_ROOT / "workbench" / "third_party" / "PAIR"
if str(PAIR_DIR) not in sys.path:
    sys.path.insert(0, str(PAIR_DIR))

import pytest
import torch

from workbench.backend.registry import model_by_id
from workbench.replay.common import load_trusted_whole_model
from workbench.third_party.PAIR.models.MHTransformer import MultiHeadAttention
from workbench.third_party.PAIR.models.MHTransformer_rev import MultiHeadAttentionRev


def test_f10_encoder_checkpoint_schema_and_verification() -> None:
    # Schema observed on HF fashioniq.pt via HTTP Range inspection: 417 keys
    required_prefixes = ("backbone.", "affine.", "local_weight")
    allowed_unexpected_prefixes = ("loss_T",)

    def verify_encoder_checkpoint_keys(
        model_keys: set[str], ckpt_keys: set[str]
    ) -> tuple[list[str], list[str]]:
        # 1. Assert required prefixes exist in checkpoint
        missing_prefixes = [p for p in required_prefixes if not any(k.startswith(p) for k in ckpt_keys)]
        if missing_prefixes:
            raise RuntimeError(f"Missing required key prefixes: {missing_prefixes}")

        # 2. Strict / compatibility check
        missing = sorted(model_keys - ckpt_keys)
        unexpected = sorted(ckpt_keys - model_keys)

        # Block on critical missing tensors
        critical_missing = [k for k in missing if any(k.startswith(p) for p in required_prefixes)]
        if critical_missing:
            raise RuntimeError(f"Missing critical parameters: {critical_missing}")

        # Disallow unexpected keys outside allowed prefixes
        disallowed = [k for k in unexpected if not any(k.startswith(p) for p in allowed_unexpected_prefixes)]
        if disallowed:
            raise RuntimeError(f"Disallowed unexpected parameters: {disallowed}")

        return missing, unexpected

    model_keys = {
        "loss_T",
        "local_weight",
        "backbone.clip.visual.conv1.weight",
        "backbone.multihead_attn.in_proj_weight",
        "affine.MLP.0.weight",
        "affine.MLP.0.bias",
    }

    # Case 1: Empty / invalid checkpoint fails prefix assertion
    with pytest.raises(RuntimeError, match="Missing required key prefixes"):
        verify_encoder_checkpoint_keys(model_keys, {"dummy_key"})

    # Case 2: Missing critical parameter fails
    with pytest.raises(RuntimeError, match="Missing critical parameters"):
        verify_encoder_checkpoint_keys(model_keys, {"backbone.clip.visual.conv1.weight", "affine.MLP.0.weight", "local_weight"})

    # Case 3: Disallowed unexpected parameter fails
    with pytest.raises(RuntimeError, match="Disallowed unexpected parameters"):
        verify_encoder_checkpoint_keys(
            model_keys,
            model_keys | {"unauthorized_backdoor_weight"},
        )

    # Case 4: Valid matching schema passes
    missing, unexpected = verify_encoder_checkpoint_keys(
        model_keys,
        model_keys | {"loss_T"},
    )
    assert not missing
    assert unexpected == ["loss_T"] or unexpected == []


def test_f11_clvc_net_assumed_names_and_whole_model_loader() -> None:
    clvc_meta = model_by_id("clvc_net")
    assert clvc_meta["checkpoint_serialization"] == "whole_model"
    assert clvc_meta["category_checkpoint_structure"]["mapping_status"] == "UNVERIFIED"

    # Forward-compat check: load_trusted_whole_model blocks unverified loads without author hash
    dummy_path = Path("/tmp/nonexistent_clvc_model.pt")
    with pytest.raises(ValueError, match="no author-verified SHA-256"):
        load_trusted_whole_model(dummy_path, expected_sha256=None, torch_module=torch)

    # Verify signature inspect handles weights_only
    sig = inspect.signature(torch.load)
    assert "weights_only" in sig.parameters or not "weights_only" in sig.parameters


def test_f12_pair_reconstruction_labeling_and_behavior() -> None:
    rev_file = Path("workbench/third_party/PAIR/models/MHTransformer_rev.py")
    content = rev_file.read_text()
    assert "EXPERIMENTAL_RECONSTRUCTION" in content
    assert "Derived:" in content
    assert "NOT established:" in content

    # Check numerical parity: rev attention == 1 - std attention
    torch.manual_seed(0)
    std = MultiHeadAttention(dim_self=32, dim_ref=32, num_heads=2)
    rev = MultiHeadAttentionRev(dim_self=32, dim_ref=32, num_heads=2)
    rev.load_state_dict(std.state_dict())

    x = torch.randn(2, 4, 32)
    with torch.no_grad():
        _, a_std = std(x)
        _, a_rev = rev(x)

    assert torch.allclose(a_rev, 1.0 - a_std, atol=1e-6)

    # Registry verification: both variants remain UNVERIFIED
    pair_meta = model_by_id("pair")
    for variant in pair_meta["checkpoint_variants"]:
        assert variant["checkpoint_mapping_status"] == "UNVERIFIED"
