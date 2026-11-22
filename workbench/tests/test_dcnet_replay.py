"""Tests for DCNet replay preflight and contract validation."""
from __future__ import annotations

from pathlib import Path
import pytest

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.replay.dcnet import (
    MODEL,
    PROTOCOL,
    REQUIRED_RUN_FILES,
    preflight,
    validate_dataset,
    validate_run_directory,
    validate_source,
)


def test_dcnet_protocol_and_model_id():
    assert MODEL["model_id"] == "dcnet"
    assert PROTOCOL == "fashioniq_full_gallery_ref_excluded"
    assert "config.json" in REQUIRED_RUN_FILES
    assert "trained_model.pth" in REQUIRED_RUN_FILES


def test_dcnet_validate_source_pinned():
    dcnet_source = Path(__file__).resolve().parents[2] / "workbench" / "third_party" / "DCNet"
    blockers = validate_source(dcnet_source)
    assert blockers == []


def test_dcnet_run_dir_fails_on_missing_files(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    blockers = validate_run_directory(run_dir)
    assert any("config.json" in b for b in blockers)
    assert any("trained_model.pth" in b for b in blockers)


def test_dcnet_dataset_fails_on_missing_glove_and_resized_images(tmp_path: Path):
    ds_dir = tmp_path / "dataset"
    ds_dir.mkdir()
    (ds_dir / "captions").mkdir()
    (ds_dir / "image_splits").mkdir()
    for cat in CATEGORIES:
        (ds_dir / "image_splits" / f"split.{cat}.val.json").write_text("[]")

    blockers = validate_dataset(ds_dir)
    assert any("resized_images" in b for b in blockers)
    assert any("glove" in b for b in blockers)

def test_pairwise_scorer_uses_query_conditioned_gallery_and_softmax_fusion():
    import torch
    import torch.nn.functional as F

    from workbench.replay.dcnet import score_pairwise

    class Model:
        modalities = ["image"]
        feat2d = False

        def get_correction(self, source, gallery):
            assert source["image"].shape == (2, 1)
            assert gallery["image"].shape == (3, 1)
            return {"image": gallery["image"].unsqueeze(0).expand(2, -1, -1)}

        def get_score_matrix(self, trg_embds, src_embds, **kwargs):
            assert kwargs == {"subspaces": ["image"], "l2renorm": False,
                              "dist": False, "val": True}
            if trg_embds["image"].ndim == 2:
                return src_embds["image"] @ trg_embds["image"].T
            return (src_embds["image"].unsqueeze(1) * trg_embds["image"]).sum(-1)

    model = Model()
    composed = {"image": torch.tensor([[1.], [2.]])}
    source = {"image": torch.tensor([[4.], [5.]])}
    text = {"image": torch.tensor([[3.], [4.]])}
    gallery = {"image": torch.tensor([[1.], [2.], [3.]])}
    composition, correction, fused = score_pairwise(model, composed, source, text, gallery)
    expected_comp = composed["image"] @ gallery["image"].T
    expected_corr = (text["image"].unsqueeze(1) * gallery["image"].unsqueeze(0)).sum(-1)
    torch.testing.assert_close(composition, expected_comp)
    torch.testing.assert_close(correction, expected_corr)
    torch.testing.assert_close(fused, F.log_softmax(expected_comp, -1) +
                               F.log_softmax(expected_corr, -1))


def test_chunked_global_logsoftmax_matches_full_gallery():
    import torch
    import torch.nn.functional as F

    comp = torch.tensor([[0.1, 0.7, -0.2, 1.3], [2.0, -0.5, 0.6, 0.2]])
    corr = torch.tensor([[1.2, -0.3, 0.4, 0.8], [-1.0, 0.5, 1.5, 0.1]])
    expected = F.log_softmax(comp, -1) + F.log_softmax(corr, -1)
    comp_lse = corr_lse = None
    for start in range(0, comp.shape[1], 2):
        a = torch.logsumexp(comp[:, start:start + 2], -1)
        b = torch.logsumexp(corr[:, start:start + 2], -1)
        comp_lse = a if comp_lse is None else torch.logaddexp(comp_lse, a)
        corr_lse = b if corr_lse is None else torch.logaddexp(corr_lse, b)
    assert comp_lse is not None and corr_lse is not None
    actual = (comp - comp_lse[:, None]) + (corr - corr_lse[:, None])
    torch.testing.assert_close(actual, expected)

def test_score_chunks_then_apply_native_full_gallery_logsoftmax():
    import torch
    import torch.nn.functional as F

    comp_chunks = [torch.tensor([[0.1, 0.7], [-0.5, 1.3]]),
                   torch.tensor([[-0.2, 1.3], [2.0, 0.2]])]
    corr_chunks = [torch.tensor([[1.2, -0.3], [0.5, 0.1]]),
                   torch.tensor([[0.4, 0.8], [-1.0, 1.5]])]
    comp_rows = torch.empty((2, 4))
    corr_rows = torch.empty((2, 4))
    for index, (comp, corr) in enumerate(zip(comp_chunks, corr_chunks)):
        start = index * 2
        comp_rows[:, start:start + 2] = comp
        corr_rows[:, start:start + 2] = corr
    expected_comp = torch.tensor([[0.1, 0.7, -0.2, 1.3], [-0.5, 1.3, 2.0, 0.2]])
    expected_corr = torch.tensor([[1.2, -0.3, 0.4, 0.8], [0.5, 0.1, -1.0, 1.5]])
    actual = F.log_softmax(comp_rows, dim=-1) + F.log_softmax(corr_rows, dim=-1)
    expected = F.log_softmax(expected_comp, dim=-1) + F.log_softmax(expected_corr, dim=-1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
