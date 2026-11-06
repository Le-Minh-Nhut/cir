"""PTHA target-feature contract: each supported class gets ONE projection, in the right place.

The two supported classes agree on extraction and disagree on projection. Pinned source:

* `blip2_qformer_cir_cat.py` -- `extract_target_features` is defined twice (lines 453 and 483) and the
  second one returns raw Q-Former states with the projection commented out, while `inference()`
  (line 287) multiplies the 256-wide query against whatever targets it is handed. Measured before
  repair on a real FashionIQ batch:
  `Expected size for first two dimensions of batch2 tensor to be: [3, 256] but got: [3, 768]`.
  So the boundary must project.

* `blip2_qformer_tgir_concatembedding_sim.py:429` -- `inference()` runs
  `F.normalize(self.vision_proj(target_embeds), dim=-1)` itself, and `extract_target_features`
  (line 501) also returns raw states. So the boundary must NOT project.

These tests call the production dispatcher `workbench.replay.ptha_mtst._target_feature_contract`
with fixtures shaped like the real classes, and count `vision_proj` applications end-to-end
(boundary + `inference`). They prove the control flow and the projection arithmetic; they do NOT
prove an author checkpoint or a GPU run succeeded. That evidence is separate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

torch = pytest.importorskip("torch")

from workbench.replay.ptha_mtst import TARGET_FEATURE_CONTRACTS, _target_feature_contract

HIDDEN = 768
EMBED = 256
TOKENS = 32


class _CountingProj(torch.nn.Linear):
    """A `vision_proj` that records how many times it is actually applied."""

    def __init__(self, hidden: int = HIDDEN, embed: int = EMBED, seed: int = 0) -> None:
        super().__init__(hidden, embed)
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.weight.copy_(torch.randn(embed, hidden, generator=generator) * 0.05)
            self.bias.copy_(torch.randn(embed, generator=generator) * 0.05)
        self.calls = 0

    def forward(self, inputs):
        self.calls += 1
        return super().forward(inputs)


class _Fixture:
    """Shared weights and behaviour of both pinned classes, without their BLIP-2 towers."""

    def __init__(self, seed: int = 0) -> None:
        self.vision_proj = _CountingProj(seed=seed)
        self.text_proj = torch.nn.Linear(HIDDEN, EMBED)
        with torch.no_grad():
            self.text_proj.weight.zero_()
            self.text_proj.bias.zero_()

    def inference_target(self, target_embeds):
        """Verbatim from both pinned classes (`cir_cat.py:447`, `tgir...:492`)."""
        return torch.nn.functional.normalize(self.vision_proj(target_embeds), dim=-1)


class Blip2QformerCirCat(_Fixture):
    """Projects nothing itself: `inference()` consumes already-projected targets."""

    def inference(self, reference_embeds, target_feats, text):
        # cir_cat.py:332-334 -- matmul only, no vision_proj on targets.
        query = torch.ones(reference_embeds.shape[0], 1, 1, EMBED)
        query = query / query.norm(dim=-1, keepdim=True)
        sim = torch.matmul(query, target_feats.permute(0, 2, 1)).squeeze()
        return sim.max(-1)[0]


class Blip2Qformer4TGIRCESIM(_Fixture):
    """Projects targets itself inside `inference()`: `tgir...:429`."""

    def inference(self, reference_embeds, target_embeds, text):
        target_feats = torch.nn.functional.normalize(self.vision_proj(target_embeds), dim=-1)
        query = torch.ones(reference_embeds.shape[0], 1, 1, EMBED)
        query = query / query.norm(dim=-1, keepdim=True)
        sim = torch.matmul(query, target_feats.permute(0, 2, 1)).squeeze()
        return sim.max(-1)[0]


def _native_index_features(batch: int = 3, tokens: int = TOKENS, seed: int = 1):
    """The exact 2-tuple `utils.extract_index_blip_features` returns: (targets, frozen_vision)."""
    generator = torch.Generator().manual_seed(seed)
    targets = torch.randn(batch, tokens, HIDDEN, generator=generator)
    frozen = torch.randn(batch, tokens + 1, 1408, generator=generator)
    return (targets, frozen)


# --------------------------------------------------------------------------- dispatch table


def test_both_supported_classes_are_declared_with_distinct_contracts():
    assert TARGET_FEATURE_CONTRACTS["Blip2QformerCirCat"] == "project_at_boundary"
    assert TARGET_FEATURE_CONTRACTS["Blip2Qformer4TGIRCESIM"] == "project_inside_inference"
    assert len(set(TARGET_FEATURE_CONTRACTS.values())) == 2


def test_undeclared_class_refuses_instead_of_guessing():
    class SomeOtherModel(_Fixture):
        pass

    with pytest.raises(ValueError, match="undeclared"):
        _target_feature_contract(SomeOtherModel(), _native_index_features(), torch)


def test_unknown_contract_name_refuses():
    with pytest.raises(ValueError, match="unknown PTHA\\+MTST target-feature contract"):
        _target_feature_contract(Blip2QformerCirCat(), _native_index_features(), torch, "invented")


def test_malformed_index_features_refuse():
    model = Blip2QformerCirCat()
    with pytest.raises(ValueError, match="native \\(target_features, raw_vision_embeds\\)"):
        _target_feature_contract(model, _native_index_features()[0], torch)
    with pytest.raises(ValueError, match="native \\(target_features, raw_vision_embeds\\)"):
        _target_feature_contract(model, {"targets": 1, "frozen": 2}, torch)


# --------------------------------------------------------------------------- CirCat branch


def test_circat_boundary_projects_once_and_inference_consumes_projected_targets():
    model = Blip2QformerCirCat()
    targets, frozen = _native_index_features()
    assert model.vision_proj.calls == 0

    (boundary_targets, boundary_frozen), report = _target_feature_contract(model, (targets, frozen), torch)

    # Projected exactly once, at the boundary.
    # One production call, plus one inside the equivalence check that runs on the same weights.
    assert model.vision_proj.calls == 2
    assert report["action"] == "projected_at_boundary_via_inference_target"
    assert report["contract"] == "project_at_boundary"
    assert report["extracted_shape"] == [3, TOKENS, HIDDEN]
    assert report["output_shape"] == [3, TOKENS, EMBED]
    assert report["new_parameters_introduced"] == 0
    # The frozen vision embeddings are the same object, unchanged.
    assert boundary_frozen is frozen
    assert list(boundary_targets.shape) == [3, TOKENS, EMBED]

    # Scoring now works, and adds no further projection: reset the counter to isolate inference.
    model.vision_proj.calls = 0
    reference = torch.randn(frozen.shape[0], frozen.shape[1], HIDDEN)
    scores = model.inference(reference, boundary_targets, ["a caption"])
    assert model.vision_proj.calls == 0, "cir_cat inference must not project targets again"
    # [B, N]: one score per (query, gallery image); the max is over the TOKENS axis.
    assert list(scores.shape) == [3, 3]
    assert torch.isfinite(scores).all()


def test_circat_boundary_output_equals_the_models_own_projection_operation():
    """Boundary result == the author's `F.normalize(vision_proj(x))` with the same weights."""
    model = Blip2QformerCirCat()
    targets, frozen = _native_index_features()
    with torch.no_grad():
        expected = torch.nn.functional.normalize(model.vision_proj(targets.clone()), dim=-1)
    after_calls = model.vision_proj.calls
    (projected, _), report = _target_feature_contract(model, (targets, frozen), torch)
    assert model.vision_proj.calls == after_calls + 2, "one production call plus the equivalence check"
    assert report["equivalence_vs_direct_projection_max_abs_diff"] == 0.0
    assert torch.equal(projected, expected)
    # Unit rows, because the author's own method normalises.
    assert torch.allclose(projected.norm(dim=-1), torch.ones(3, TOKENS), atol=1e-6)


def test_circat_raw_targets_would_not_meet_the_query_side():
    """The pre-fix failure stays reproducible, so the repair is not a no-op."""
    raw, _frozen = _native_index_features()
    query_side = torch.zeros(3, 1, 1, EMBED)
    with pytest.raises(RuntimeError):
        torch.matmul(query_side, raw.permute(0, 2, 1))


# --------------------------------------------------------------------------- TGIR branch


def test_tgir_boundary_passes_raw_targets_and_inference_projects_exactly_once():
    model = Blip2Qformer4TGIRCESIM()
    targets, frozen = _native_index_features()
    assert model.vision_proj.calls == 0

    (boundary_targets, boundary_frozen), report = _target_feature_contract(model, (targets, frozen), torch)

    # NOT projected at the boundary.
    assert model.vision_proj.calls == 0, "tgir must not be projected at the boundary"
    assert report["action"] == "passed_through_unprojected"
    assert report["contract"] == "project_inside_inference"
    assert boundary_targets is targets
    assert boundary_frozen is frozen
    assert report["output_shape"] == [3, TOKENS, HIDDEN]
    assert report["new_parameters_introduced"] == 0

    # `inference()` projects exactly once, and the scores are usable.
    reference = torch.randn(frozen.shape[0], frozen.shape[1], HIDDEN)
    scores = model.inference(reference, boundary_targets, ["a caption"])
    assert model.vision_proj.calls == 1, "exactly one projection total, inside inference"
    assert list(scores.shape) == [3, 3]
    assert torch.isfinite(scores).all()


def test_tgir_scores_are_unchanged_by_the_boundary():
    """A pass-through boundary must be numerically inert for this class."""
    model = Blip2Qformer4TGIRCESIM()
    targets, frozen = _native_index_features()
    reference = torch.randn(frozen.shape[0], frozen.shape[1], HIDDEN)

    direct_scores = model.inference(reference, targets, ["a caption"]).clone()
    model.vision_proj.calls = 0
    (boundary_targets, _), _report = _target_feature_contract(model, (targets, frozen), torch)
    through_scores = model.inference(reference, boundary_targets, ["a caption"])
    assert torch.equal(direct_scores, through_scores)


def test_tgir_double_projection_raises_rather_than_scoring():
    """What the old unconditional dispatch actually did to this branch.

    `vision_proj` maps 768->256, so handing `inference()` an already-projected tensor makes the
    in-inference projection fail: the branch was USABLE-NEVER, not quietly wrong. Recorded as the
    measured failure mode so the fix is not justified by an invented one.
    """
    model = Blip2Qformer4TGIRCESIM()
    targets, _frozen = _native_index_features()
    reference = torch.randn(3, 2, HIDDEN)

    pre_projected = model.inference_target(targets)          # the WRONG extra step
    assert list(pre_projected.shape) == [3, TOKENS, EMBED]
    with pytest.raises(RuntimeError, match="mat1 and mat2 shapes cannot be multiplied"):
        model.inference(reference, pre_projected, ["a caption"])

    # And the correct dispatch does not raise.
    scores = model.inference(reference, targets, ["a caption"])
    assert torch.isfinite(scores).all()


# --------------------------------------------------------------------------- geometry guards


def test_boundary_projection_width_mismatch_is_refused():
    """If extraction stops returning unprojected width, the declared contract no longer holds."""
    model = Blip2QformerCirCat()
    _raw, frozen = _native_index_features()
    generator = torch.Generator().manual_seed(7)
    wrong = torch.randn(3, TOKENS, EMBED, generator=generator)
    with pytest.raises(ValueError, match="expects unprojected targets of width 768"):
        _target_feature_contract(model, (wrong, frozen), torch)
