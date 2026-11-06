"""PTHA target-feature contract: the repair must project once, through trained weights.

Why this test exists: `blip2_qformer_cir_cat.py` defines `extract_target_features` twice and keeps the
second, which returns unprojected Q-Former states `[N,32,768]`, while `inference()` matmuls them
against a 256-wide query. The observed failure on a real FashionIQ batch was

    Expected size for first two dimensions of batch2 tensor to be: [3, 256] but got: [3, 768]

`workbench.replay.ptha_mtst._target_feature_contract` applies the checkpoint's own `vision_proj`
once at that boundary. These are fake-tensor regressions for the boundary itself; they establish the
contract, not model quality. Real GPU evidence lives in the session measurement files.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

torch = pytest.importorskip("torch")

from workbench.replay.ptha_mtst import _target_feature_contract


class _FakeHead:
    """Stand-in for Blip2QformerCirCat's trained projection boundary."""

    def __init__(self, hidden: int = 768, embed: int = 256, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.vision_proj = torch.nn.Linear(hidden, embed)
        with torch.no_grad():
            self.vision_proj.weight.copy_(torch.randn(embed, hidden, generator=generator))
            self.vision_proj.bias.copy_(torch.randn(embed, generator=generator))

    def inference_target(self, target_embeds):
        """The pinned author method, reproduced verbatim from the source."""
        return torch.nn.functional.normalize(self.vision_proj(target_embeds), dim=-1)


def _index_features(batch: int = 2, tokens: int = 4, hidden: int = 768, seed: int = 1):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(batch, tokens, hidden, generator=generator),
            torch.randn(batch, tokens + 1, 1408, generator=generator))


def test_unprojected_targets_cannot_meet_the_query_side():
    """The pre-fix failure mode stays reproducible, so the regression is real."""
    head = _FakeHead()
    unprojected, _raw = _index_features()
    query_side = torch.zeros(2, 1, 1, 256)                       # text_proj output width
    with pytest.raises(RuntimeError):
        torch.matmul(query_side, unprojected.permute(0, 2, 1))


def test_contract_projects_to_the_embeddings_width():
    head = _FakeHead()
    unprojected, raw = _index_features()
    (projected, returned_raw), report = _target_feature_contract(head, (unprojected, raw), torch)
    assert projected.shape == (2, 4, 256)
    assert returned_raw is raw, "the frozen vision embeds must pass through untouched"
    assert report["before_shape"] == [2, 4, 768]
    assert report["after_shape"] == [2, 4, 256]
    assert report["new_parameters_introduced"] == 0
    assert report["boundary_equivalence_max_abs_diff"] == 0.0
    # unit rows, because the author's own method normalises
    assert torch.allclose(projected.norm(dim=-1), torch.ones(2, 4), atol=1e-6)


def test_projected_targets_meet_the_query_side():
    head = _FakeHead()
    unprojected, raw = _index_features()
    (projected, _), _report = _target_feature_contract(head, (unprojected, raw), torch)
    query_side = torch.zeros(2, 1, 1, 256)
    sim = torch.matmul(query_side, projected.permute(0, 2, 1)).squeeze()
    assert sim.shape[-1] == 4


def test_second_projection_is_structurally_impossible():
    """The guard against double projection is the head's own geometry, not a flag."""
    head = _FakeHead()
    unprojected, raw = _index_features()
    (projected, _), report = _target_feature_contract(head, (unprojected, raw), torch)
    assert report["double_projection_possible"] is False
    with pytest.raises(RuntimeError):
        head.inference_target(projected)


def test_unrelated_class_contract_is_not_silently_rewritten():
    """A class that projects INSIDE inference must not be handed projected targets twice.

    `blip2_qformer_tgir_concatembedding_sim.inference` projects target embeds itself (source line 429).
    The repair lives in the FashionIQ transfer path only, so this module exposes no global switch that
    could project on its behalf.
    """
    import workbench.replay.ptha_mtst as wrapper

    assert not hasattr(wrapper, "PROJECT_TARGETS_EVERYWHERE")
    source = Path(wrapper.__file__).read_text()
    assert "tgir_ce_sim" in source, "the sibling class entry must stay declared"
    # the projection is applied at exactly one call site in the module
    assert source.count("_target_feature_contract(") == 2, (
        "expected one definition plus one call site; a second call would double-project")
