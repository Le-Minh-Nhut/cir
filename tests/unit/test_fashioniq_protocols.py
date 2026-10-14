"""FashionIQ protocol semantics: the three families must stay operationally distinct.

`fashioniq_original_split` (full validation gallery, reference eligible),
`fashioniq_full_gallery_ref_excluded` (same gallery, reference removed from the ranking) and
`fashioniq_val_split` (pair-union gallery, reference removed) are different cohorts. The
registry advertises all three, so each must actually be implemented and must not be reachable
by accident from another.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from datasets.fashioniq import FashionIQAnnotation, build_pair_union_gallery
from evaluation.fashioniq import (
    build_fashioniq_gallery,
    evaluate_fashioniq_category,
    evaluate_fashioniq_recall,
    macro_average_fashioniq,
    recall_at_k,
    recall_at_k_reference_excluded,
)


def _split_root(tmp_path: Path, ids: list[str]) -> Path:
    """A split root laid out as `load_fashioniq_split_ids` expects: `split.<cat>.<split>.json`."""
    (tmp_path / "split.dress.val.json").write_text(json.dumps(ids))
    return tmp_path


ANNOTATIONS = [
    FashionIQAnnotation("ref-a", "target-a", ("one", "two"), "dress", 0),
    FashionIQAnnotation("ref-b", "target-b", ("three", "four"), "dress", 1),
]
FULL_GALLERY = ["ref-a", "target-a", "ref-b", "target-b"]


# --------------------------------------------------------------------------- gallery families


def test_full_gallery_protocol_uses_the_declared_split_order(tmp_path: Path) -> None:
    root = _split_root(tmp_path, FULL_GALLERY)
    gallery = build_fashioniq_gallery("fashioniq_original", root, "dress", ANNOTATIONS, "val")
    assert gallery == FULL_GALLERY


def test_reference_excluded_protocol_uses_the_same_full_gallery(tmp_path: Path) -> None:
    """The two full-gallery protocols differ in exclusion, not in their candidate universe."""
    root = _split_root(tmp_path, FULL_GALLERY)
    included = build_fashioniq_gallery("fashioniq_original", root, "dress", ANNOTATIONS, "val")
    excluded = build_fashioniq_gallery(
        "fashioniq_full_gallery_ref_excluded", root, "dress", ANNOTATIONS, "val")
    assert excluded == included == FULL_GALLERY
    assert len(excluded) == 4, "not the pair-union subset"


def test_val_protocol_uses_the_pair_union_not_the_full_split(tmp_path: Path) -> None:
    root = _split_root(tmp_path, FULL_GALLERY + ["extra-1", "extra-2"])
    gallery = build_fashioniq_gallery("fashioniq_val", root, "dress", ANNOTATIONS, "val")
    assert gallery == build_pair_union_gallery(annotations=ANNOTATIONS)
    assert "extra-1" not in gallery


def test_an_unknown_protocol_is_refused(tmp_path: Path) -> None:
    root = _split_root(tmp_path, FULL_GALLERY)
    with pytest.raises(ValueError, match="Unsupported FashionIQ protocol"):
        build_fashioniq_gallery("fashioniq_typo", root, "dress", ANNOTATIONS, "val")


# ---------------------------------------------------------------------------- reference rule


def test_reference_eligible_protocol_counts_a_reference_hit() -> None:
    """If the reference is the only image above the target, an eligible run retrieves it."""
    # query 0: reference is rank 1, target rank 2 -> a top-1 retrieval is the reference.
    scores = torch.tensor([[0.9, 0.8, 0.1, 0.0]])
    assert recall_at_k(scores, ["target-a"], FULL_GALLERY, k=1) == 0.0
    # The reference occupies the winning slot, so an exclusion protocol must still find the target.
    assert recall_at_k_reference_excluded(
        scores, ["target-a"], ["ref-a"], FULL_GALLERY, k=1) == 100.0


def test_excluded_protocol_removes_the_reference_rather_than_masking_it() -> None:
    """The target must be reachable even when the reference outranks everything."""
    scores = torch.tensor([[0.9, 0.8, 0.1, 0.0]])
    assert recall_at_k_reference_excluded(scores, ["target-a"], ["ref-a"], FULL_GALLERY, k=1) == 100.0
    # k = 1 with the reference removed: the target is the first remaining image.
    assert recall_at_k_reference_excluded(scores, ["target-a"], ["ref-a"], FULL_GALLERY, k=1) != 0.0


def test_exclusion_does_not_change_the_denominator() -> None:
    """Recall is hit fraction over queries; excluding a reference must not shift the base."""
    # Gallery order is [ref-a, target-a, ref-b, target-b].
    # Query 0: ref-a > target-a, so removing ref-a puts that query's target first (hit at k=1).
    # Query 1: target-a > target-b > ref-b, so removing ref-b still leaves a foreign image
    # first: a miss at k=1, a hit at k=2.
    scores = torch.tensor([
        [0.9, 0.8, 0.1, 0.0],
        [0.1, 0.9, 0.7, 0.8],
    ])
    assert recall_at_k_reference_excluded(
        scores, ["target-a", "target-b"], ["ref-a", "ref-b"], FULL_GALLERY, k=1) == 50.0
    # Widening to k=2 finds the second query too, and the base stays two queries.
    assert recall_at_k_reference_excluded(
        scores, ["target-a", "target-b"], ["ref-a", "ref-b"], FULL_GALLERY, k=2) == 100.0


def test_targets_and_references_must_be_in_the_gallery() -> None:
    scores = torch.tensor([[0.9, 0.8, 0.1, 0.0]])
    with pytest.raises(AssertionError, match="missing from gallery"):
        recall_at_k_reference_excluded(scores, ["nope"], ["ref-a"], FULL_GALLERY, k=1)
    with pytest.raises(AssertionError, match="missing from gallery"):
        recall_at_k_reference_excluded(scores, ["target-a"], ["nope"], FULL_GALLERY, k=1)


def test_reference_ids_switch_the_excluding_rule_on(tmp_path: Path) -> None:
    """The same helper must apply exclusion when references are supplied, and not otherwise.

    A four-image gallery is smaller than k, so the production k values are exercised through
    the category evaluation on a gallery wide enough to accept them.
    """
    scores = torch.tensor([[0.9, 0.8, 0.1, 0.0], [0.1, 0.9, 0.7, 0.8]])
    targets = ["target-a", "target-b"]
    references = ["ref-a", "ref-b"]

    # Both rules are reachable through the same entry points, selected purely by whether
    # reference ids are supplied; a four-image gallery cannot satisfy the production k values,
    # so the k=2 helpers are used to make the rule itself observable here.
    assert recall_at_k(scores, targets, FULL_GALLERY, k=2) == 100.0
    assert recall_at_k_reference_excluded(scores, targets, references, FULL_GALLERY, k=2) == 100.0

    # The entry points forward references, and reject a gallery too small for their k values
    # rather than silently shortening it.
    with pytest.raises(AssertionError):
        evaluate_fashioniq_recall(scores, targets, FULL_GALLERY, references)
    with pytest.raises(AssertionError):
        evaluate_fashioniq_category(scores, targets, FULL_GALLERY, references)


def test_k_larger_than_the_gallery_is_refused() -> None:
    scores = torch.tensor([[0.9, 0.8, 0.1, 0.0]])
    with pytest.raises(AssertionError):
        recall_at_k_reference_excluded(scores, ["target-a"], ["ref-a"], FULL_GALLERY, k=5)
    with pytest.raises(AssertionError):
        recall_at_k_reference_excluded(scores, ["target-a"], ["ref-a"], FULL_GALLERY, k=0)
    # k == gallery_count is legal: every image is ranked.
    assert recall_at_k_reference_excluded(
        scores, ["target-a"], ["ref-a"], FULL_GALLERY, k=4) == 100.0


# ------------------------------------------------------------------------------ aggregation


def test_category_macro_is_an_equal_weight_mean() -> None:
    """The registry's macro is the mean of category recalls, not a query-weighted pool."""
    results = {
        "dress": {"recall_at_10": 60.0, "recall_at_50": 80.0},
        "shirt": {"recall_at_10": 40.0, "recall_at_50": 60.0},
        "toptee": {"recall_at_10": 50.0, "recall_at_50": 70.0},
    }
    average = macro_average_fashioniq(results)
    assert average["recall_at_10"] == pytest.approx(50.0)
    assert average["recall_at_50"] == pytest.approx(70.0)
    assert average["mean_recall"] == pytest.approx(60.0)


def test_duplicate_gallery_ids_are_refused(tmp_path: Path) -> None:
    """A gallery with repeated ids would silently distort every rank."""
    root = _split_root(tmp_path, ["a", "a"])
    with pytest.raises(AssertionError):
        build_fashioniq_gallery("fashioniq_original", root, "dress", ANNOTATIONS, "val")


# --------------------------------------------------------------------- image inventory

def test_an_official_evaluator_extension_mismatch_is_reported(tmp_path: Path) -> None:
    """CSMCIR opens `{id}.png` while the official release ships `.jpg`.

    The evaluator's extension is hard-coded in its source, so a `.jpg` layout can only fail on
    the GPU host as a missing file. The laptop preflight must report it instead.
    """
    from workbench.backend.fashioniq_layout import image_extension_problems

    root = tmp_path / "FashionIQ"
    (root / "images").mkdir(parents=True)
    (root / "image_splits").mkdir(parents=True)
    for category in ("dress", "shirt", "toptee"):
        (root / "image_splits" / f"split.{category}.val.json").write_text(json.dumps(["a", "b"]))
    for name in ("a.jpg", "b.jpg"):
        (root / "images" / name).write_bytes(b"\xff\xd8\xff")

    problems = image_extension_problems(root, ".png")
    assert problems, "a .jpg-only layout must fail a .png requirement"
    assert ".png" in problems[0]

    # Converting the basenames satisfies the declared requirement.
    for name in ("a", "b"):
        (root / "images" / f"{name}.png").write_bytes(b"\x89PNG")
    assert image_extension_problems(root, ".png") == []


def test_an_extension_requirement_is_checked_against_every_validation_id(tmp_path: Path) -> None:
    """A single missing image must be reported, with the offending id named."""
    from workbench.backend.fashioniq_layout import image_extension_problems

    root = tmp_path / "FashionIQ"
    (root / "images").mkdir(parents=True)
    (root / "image_splits").mkdir(parents=True)
    for category in ("dress", "shirt", "toptee"):
        ids = ["present", "absent"] if category == "dress" else ["present"]
        (root / "image_splits" / f"split.{category}.val.json").write_text(json.dumps(ids))
    (root / "images" / "present.png").write_bytes(b"\x89PNG")

    problems = image_extension_problems(root, ".png")
    assert len(problems) == 1, problems
    assert "absent.png" in problems[0]


def test_registry_protocol_ids_and_internal_aliases_build_the_same_cohort(
        tmp_path: Path) -> None:
    """A caller holding a registry protocol id must be able to evaluate that family.

    The registry uses `fashioniq_original_split` / `fashioniq_val_split` while earlier callers
    used the shorter `fashioniq_original` / `fashioniq_val`. Only the short spellings were
    accepted, so the two registry ids raised `Unsupported FashionIQ protocol` — a gap, not a
    naming preference. Each pair must build the identical cohort.
    """
    root = _split_root(tmp_path, FULL_GALLERY + ["extra-1"])
    pairs = [("fashioniq_original", "fashioniq_original_split"),
             ("fashioniq_val", "fashioniq_val_split")]

    for internal, registry_id in pairs:
        from_internal = build_fashioniq_gallery(internal, root, "dress", ANNOTATIONS, "val")
        from_registry = build_fashioniq_gallery(registry_id, root, "dress", ANNOTATIONS, "val")
        assert from_registry == from_internal, registry_id

    # The registry id of the exclusion family is one name, and it is the full gallery.
    excluded = build_fashioniq_gallery(
        "fashioniq_full_gallery_ref_excluded", root, "dress", ANNOTATIONS, "val")
    assert excluded == FULL_GALLERY + ["extra-1"]

    # Every id the registry declares must be buildable.
    from workbench.backend.registry import load_registry
    for protocol_id in load_registry()["protocols"]:
        assert build_fashioniq_gallery(protocol_id, root, "dress", ANNOTATIONS, "val")


def test_the_image_check_looks_where_the_layout_reads(tmp_path: Path) -> None:
    """Each layout reads a different image directory, so the check follows the layout.

    The iLearn family opens `resized_image/<category>/{id}.jpg`, DCNet `resized_images/{id}.jpg`,
    and the standard layout `images/{id}.png`. Checking `images/` for every model both misses a
    real mismatch and reports a false one.
    """
    from workbench.backend.fashioniq_layout import image_extension_problems

    root = tmp_path / "FashionIQ"
    (root / "image_splits").mkdir(parents=True)
    for category in ("dress", "shirt", "toptee"):
        (root / "image_splits" / f"split.{category}.val.json").write_text(json.dumps(["a", "b"]))

    # iLearn: per-category resized JPEGs.
    for category in ("dress", "shirt", "toptee"):
        folder = root / "resized_image" / category
        folder.mkdir(parents=True)
        for name in ("a", "b"):
            (folder / f"{name}.jpg").write_bytes(b"\xff\xd8\xff")
    assert image_extension_problems(root, ".jpg", "fashioniq_ilearn_resized") == []
    # A missing member is reported with the offending id.
    (root / "resized_image" / "dress" / "b.jpg").unlink()
    problems = image_extension_problems(root, ".jpg", "fashioniq_ilearn_resized")
    assert len(problems) == 1 and "b.jpg" in problems[0]

    # DCNet: flat resized JPEGs, a different directory again.
    assert image_extension_problems(root, ".jpg", "fashioniq_dcnet"), \
        "resized_images is not resized_image"
    flat = root / "resized_images"
    flat.mkdir()
    for name in ("a", "b"):
        (flat / f"{name}.jpg").write_bytes(b"\xff\xd8\xff")
    assert image_extension_problems(root, ".jpg", "fashioniq_dcnet") == []

    # An unknown layout must be refused rather than silently checked as the standard one.
    assert "unknown FashionIQ layout" in image_extension_problems(root, ".jpg", "nope")[0]


def test_every_model_with_an_extension_requirement_has_a_known_layout() -> None:
    """The declaration and the layout must agree, or the check would look in the wrong place."""
    from workbench.backend.fashioniq_layout import _LAYOUT_IMAGE_DIRECTORY
    from workbench.backend.registry import load_registry

    for model in load_registry()["models"]:
        if not model.get("fashioniq_image_extensions"):
            continue
        assert model.get("fashioniq_layout") in _LAYOUT_IMAGE_DIRECTORY, model["model_id"]


def test_the_reference_policy_argument_is_enforced() -> None:
    """The policy argument must reject a mismatch, or a forgotten reference is silent.

    `evaluate_fashioniq_recall` can only switch on whether references were supplied, so without this
    guard an exclusion cohort evaluated with no references would be scored as a reference-eligible
    one. The argument exists so that mistake raises instead of quietly changing the ranking rule.
    """
    from evaluation.fashioniq import evaluate_fashioniq_recall

    gallery = ["ref"] + [f"mid{i}" for i in range(60)] + ["target"]
    targets = ["target"]
    scores = torch.arange(float(len(gallery))).unsqueeze(0)

    # Consistent call: excluded policy with references.
    excluded = evaluate_fashioniq_recall(scores, targets, gallery, reference_ids=["ref"],
                                         reference_excluded=True)
    assert excluded["recall_at_10"] in (0.0, 100.0)

    # Consistent call: eligible policy without references.
    eligible = evaluate_fashioniq_recall(scores, targets, gallery, reference_excluded=False)
    assert eligible["recall_at_10"] in (0.0, 100.0)

    # Inconsistent calls must raise rather than silently change the rule.
    with pytest.raises(ValueError, match="reference policy"):
        evaluate_fashioniq_recall(scores, targets, gallery, reference_excluded=True)
    with pytest.raises(ValueError, match="reference policy"):
        evaluate_fashioniq_recall(scores, targets, gallery, reference_ids=["ref"],
                                  reference_excluded=False)
