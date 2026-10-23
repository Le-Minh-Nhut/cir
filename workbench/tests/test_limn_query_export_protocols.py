"""The three FashionIQ protocols must remain three, not collapse into one.

§28 asks for fixtures where all three protocols produce DIFFERENT ranks, so a test cannot pass if
an implementation quietly treats them as interchangeable. The fixture below is built so that the
same query has target rank 3 under Protocol A, 2 under Protocol B and 1 under Protocol C:

  full split:  ["imgA", "imgB", "imgZ", "imgC", "imgD"]   (imgZ is a distractor no annotation uses)
  query:       reference imgA, target imgB
  scores:      imgA 0.9 (the reference), imgZ 0.8 (a stranger), imgB 0.7 (the target)

  Protocol A  reference ELIGIBLE, full gallery -> imgA, imgZ, imgB -> target rank 3
  Protocol B  reference EXCLUDED, full gallery -> imgZ, imgB       -> target rank 2
  Protocol C  reference EXCLUDED, pair-union   -> imgB (imgZ absent) -> target rank 1

A protocol collapse shows up here as two of the three numbers agreeing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from datasets.fashioniq import FashionIQAnnotation, build_pair_union_gallery
from evaluation.fashioniq import (
    build_original_gallery,
    recall_at_k,
    recall_at_k_reference_excluded,
)
from evaluation.fashioniq_encoder import fashioniq_encoder_recall_at_k

SPLIT = ["imgA", "imgB", "imgZ", "imgC", "imgD"]
QUERY_REFERENCE, QUERY_TARGET = "imgA", "imgB"
#: imgA is the query's own reference and its highest-scoring match; imgZ is a distractor that
#: exists in the full split but belongs to no annotation, so it disappears under a pair-union.
SCORES = {"imgA": 0.9, "imgZ": 0.8, "imgB": 0.7, "imgC": 0.4, "imgD": 0.3}


def _annotation() -> FashionIQAnnotation:
    return FashionIQAnnotation(QUERY_REFERENCE, QUERY_TARGET, ("one", "two"), "dress", 0)


def _annotation_root(root: Path) -> Path:
    """The caption directory: `load_fashioniq_annotations` reads `cap.{category}.{split}.json` here."""
    (root / "captions").mkdir(parents=True, exist_ok=True)
    (root / "captions" / "cap.dress.val.json").write_text(json.dumps(
        [{"candidate": QUERY_REFERENCE, "target": QUERY_TARGET, "captions": ["one", "two"]}]))
    return root / "captions"


def _split_root(root: Path) -> Path:
    """The split directory: `load_fashioniq_split_ids` reads `split.{category}.{split}.json` here."""
    (root / "image_splits").mkdir(parents=True, exist_ok=True)
    (root / "image_splits" / "split.dress.val.json").write_text(json.dumps(SPLIT))
    return root / "image_splits"


def _vector(gallery: list[str]) -> torch.Tensor:
    return torch.tensor([[SCORES[image_id] for image_id in gallery]])


def _rank_of_target(scores: torch.Tensor, gallery: list[str], reference_excluded: bool) -> int:
    """The target's 1-based rank under the protocol's own reference policy."""
    row = scores[0]
    order = torch.argsort(row, descending=True)
    reference_column = gallery.index(QUERY_REFERENCE)
    if reference_excluded:
        order = order[order != reference_column]
    return int((order == gallery.index(QUERY_TARGET)).nonzero()[0].item()) + 1


def test_the_three_protocols_produce_different_target_ranks(tmp_path: Path) -> None:
    """The load-bearing assertion: 3, 2, 1. Any collapse makes two of these equal."""
    split_root = _split_root(tmp_path)
    annotations = [_annotation()]

    original = build_original_gallery(split_root, "dress", "val")
    pair_union = build_pair_union_gallery(annotations)

    # The two galleries genuinely differ in size, which is the mechanism behind the divergence.
    assert original == SPLIT
    assert pair_union == ["imgA", "imgB"]
    assert len(original) != len(pair_union)

    rank_a = _rank_of_target(_vector(original), original, reference_excluded=False)
    rank_b = _rank_of_target(_vector(original), original, reference_excluded=True)
    rank_c = _rank_of_target(_vector(pair_union), pair_union, reference_excluded=True)

    assert (rank_a, rank_b, rank_c) == (3, 2, 1), (rank_a, rank_b, rank_c)


def test_protocol_a_and_b_differ_only_in_reference_eligibility(tmp_path: Path) -> None:
    """Same gallery, opposite reference policy — the difference must be observable.

    At k=2 the two disagree: under A the reference occupies rank 1 and imgZ rank 2, so the target
    (rank 3) is still not retrieved; under B the reference is gone, leaving imgZ rank 1 and the
    target rank 2, so it is. At k=1 both are 0 -- imgZ outranks the target under either policy --
    which is exactly why the protocols cannot be distinguished by a single shallow k.
    """
    gallery = build_original_gallery(_split_root(tmp_path), "dress", "val")
    scores = _vector(gallery)

    assert recall_at_k(scores, [QUERY_TARGET], gallery, 1) == 0.0
    assert recall_at_k_reference_excluded(scores, [QUERY_TARGET], [QUERY_REFERENCE], gallery, 1) == 0.0
    assert recall_at_k(scores, [QUERY_TARGET], gallery, 2) == 0.0
    assert recall_at_k_reference_excluded(scores, [QUERY_TARGET], [QUERY_REFERENCE], gallery, 2) == 100.0


def test_protocol_b_and_c_differ_in_the_candidate_universe(tmp_path: Path) -> None:
    """Same reference policy, different gallery — a stranger in the full split changes the rank."""
    annotations = [_annotation()]
    original = build_original_gallery(_split_root(tmp_path), "dress", "val")
    pair_union = build_pair_union_gallery(annotations)

    # imgZ outranks the target in the full split but is not a candidate in the pair-union.
    full = recall_at_k_reference_excluded(_vector(original), [QUERY_TARGET], [QUERY_REFERENCE], original, 1)
    reduced = recall_at_k_reference_excluded(_vector(pair_union), [QUERY_TARGET], [QUERY_REFERENCE], pair_union, 1)

    assert full == 0.0, "under the full gallery imgZ outranks the target at k=1"
    assert reduced == 100.0, "under the pair-union imgZ is not a candidate"


def test_protocol_c_uses_the_pair_union_and_excludes_the_reference(tmp_path: Path) -> None:
    gallery = build_pair_union_gallery([_annotation()])
    scores = _vector(gallery)

    assert fashioniq_encoder_recall_at_k(scores, [QUERY_TARGET], [QUERY_REFERENCE], gallery, 1) == 100.0
    # The reference is excluded, so a gallery containing only the reference cannot hit.
    only_reference = ["imgA"]
    with pytest.raises(ValueError):
        fashioniq_encoder_recall_at_k(_vector(["imgA"]), [QUERY_TARGET], [QUERY_REFERENCE],
                                      only_reference, 1)


@pytest.mark.parametrize("k", [1, 2])
def test_protocol_c_is_stricter_than_a_and_b_at_every_shallow_k(tmp_path: Path, k: int) -> None:
    """The pair-union has no stranger to outrank the target, so C retrieves it and A never does."""
    original = build_original_gallery(_split_root(tmp_path), "dress", "val")
    reduced = build_pair_union_gallery([_annotation()])

    values = {
        "A": recall_at_k(_vector(original), [QUERY_TARGET], original, k),
        "C": fashioniq_encoder_recall_at_k(_vector(reduced), [QUERY_TARGET], [QUERY_REFERENCE], reduced, k),
    }
    while_eligible = recall_at_k_reference_excluded(
        _vector(original), [QUERY_TARGET], [QUERY_REFERENCE], original, k)

    assert values["A"] == 0.0, values      # the reference always occupies rank 1 under A
    assert values["C"] == 100.0, values    # imgZ is not a candidate, so the target is rank 1
    if k == 1:
        # The stranger still shadows the target under B at the shallowest depth.
        assert while_eligible == 0.0
    else:
        assert while_eligible == 100.0


def test_the_protocol_ids_are_exactly_the_registry_names() -> None:
    from workbench.backend.registry import load_registry
    from workbench.backend.schemas.results import PROTOCOL_IDS

    assert PROTOCOL_IDS == frozenset(load_registry()["protocols"])


def test_registry_membership_matches_the_declared_protocols() -> None:
    """Each registered model's native protocol must be declared, or explicitly none."""
    from workbench.backend.registry import load_registry

    # Read from the registry rather than keeping a second copy: `load_registry` already refuses an
    # unknown protocol for any model, so a duplicated set here can only drift out of step with the
    # authority it is meant to check.
    declared = set(load_registry()["protocols"])
    for model in load_registry()["models"]:
        native = model.get("native_protocol")
        assert native is None or native in declared, (model["model_id"], native)
        for supported in model.get("supported_protocols") or []:
            assert supported in declared, (model["model_id"], supported)


def test_protocol_literature_labels_have_exactly_one_owner() -> None:
    """Schema labels match registry declarations; publication imports the schema mapping."""
    import workbench.backend.schemas.results as schema
    import workbench.replay.publication as publication
    from workbench.backend.registry import load_registry

    assert publication.PROTOCOL_LITERATURE_LABELS is schema.PROTOCOL_LITERATURE_LABELS
    assert schema.PROTOCOL_LITERATURE_LABELS == {
        protocol_id: protocol["literature_split_label"]
        for protocol_id, protocol in load_registry()["protocols"].items()
    }


def test_per_query_run_metadata_accepts_every_registry_protocol() -> None:
    from workbench.backend.registry import load_registry
    from workbench.backend.schemas.results import PROTOCOL_IDS, RunMetadata

    for protocol_id, protocol in load_registry()["protocols"].items():
        metadata = RunMetadata.model_validate({
            "run_id": "run", "dataset": "FashionIQ", "split": "val",
            "protocol_id": protocol_id,
            "literature_split_label": protocol["literature_split_label"],
            "evaluation_noise_pct": 0, "model_id": "model", "method_name": "method",
            "checkpoint_id": "checkpoint", "top_k_saved": 1, "gallery_size": 1,
            "timestamp": "2026-10-24T00:00:00Z",
        })
        assert metadata.protocol_id in PROTOCOL_IDS


def test_no_other_module_restates_the_protocol_label_map() -> None:
    """Scan the first-party tree for a second literal copy of the mapping."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    owners = {"workbench/backend/schemas/results.py"}
    offenders = []
    for path in list(root.glob("workbench/**/*.py")) + list(root.glob("src/**/*.py")):
        rel = path.relative_to(root).as_posix()
        if rel in owners or "/tests/" in rel or "__pycache__" in rel:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if re.search(r"fashioniq_full_gallery_ref_excluded\s*:\s*[\"'](?:val|original)[\"']", text):
            offenders.append(rel)
    assert offenders == [], f"protocol label map restated in: {offenders}"
