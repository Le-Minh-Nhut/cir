"""LIMN query-to-checkpoint routing: the 119 unscored CIRCUS/FashionIQ queries.

The recorded explanation for ``coverage 0.98018`` (119 of 6,003 unscored) was that a query whose
reference image appears in two FashionIQ category splits has no determinable category. Only part of
that is true. The *split* membership is ambiguous, but the query text itself is the reference
image's two validation captions joined by ``" and "``, so the caption row it came from names the
category the authors annotated it under. This test pins the routing rule and, importantly, pins
that the two queries whose caption pair is annotated identically under ``shirt`` and ``toptee`` stay
unscored instead of defaulting to one of them.

Split into three layers so a host without the external artifacts still checks the logic:

* pure-function checks on hand-built fixtures (always run);
* the real 119-query classification, skipped when the release or annotations are absent;
* the adapter seam, checked with ``object.__new__`` so no checkpoint or GPU is touched.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from workbench.composition.model_limn import (
    Adapter,
    caption_scope_index,
    normalize_query_text,
    resolve_query_category,
)

SCOPES = Adapter.category_scopes

#: The 119 unscored queries as the recorded run classified them: cause -> qid -> scope resolved
#: from the author caption annotation. ``None`` means the annotation does not determine a category.
RECORDED = {
    "reference in one split, test split of the other": {"7:2179": "shirt", "7:4246": "toptee"},
    "reference in both splits, one annotation row": {"7:2228": "shirt", "7:4066": "toptee"},
    "reference in both splits, identical caption pair filed twice": {"7:3038": None, "7:3790": None},
}

def _row(candidate: str, *captions: str) -> dict[str, object]:
    return {"candidate": candidate, "captions": list(captions)}

def _write_captions(root: Path, category: str, rows: list[dict[str, object]]) -> None:
    (root / "captions").mkdir(parents=True, exist_ok=True)
    (root / "captions" / f"cap.{category}.val.json").write_text(json.dumps(rows), encoding="utf-8")

def _query(text: str, reference_id: str = "7:1", category: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(qid="7:1", query_txt=text, reference_id=reference_id, category=category)

@pytest.fixture()
def fixture_root(tmp_path: Path) -> Path:
    """A garment whose caption pair is duplicated across the two ambiguous categories."""
    _write_captions(tmp_path, "dress", [_row("B000D", "is red", "is short")])
    _write_captions(tmp_path, "shirt", [
        _row("B000AM", "is darker and has a different graphic", "is darker"),
        _row("B000S", "is blue", "is long"),
    ])
    _write_captions(tmp_path, "toptee", [
        _row("B000AM", "is darker and has a different graphic", "is darker"),
        _row("B000T", "is green", "is sleeveless"),
    ])
    return tmp_path

NAMES = {"7:1": "mbeir_images/fashioniq_images/B000S.jpg",
         "7:2": "mbeir_images/fashioniq_images/B000T.jpg",
         "7:3": "mbeir_images/fashioniq_images/B000AM.jpg",
         "7:4": "mbeir_images/fashioniq_images/B000D.jpg",
         "7:5": "mbeir_images/fashioniq_images/B999X.jpg",
         "7:6": "mbeir_images/fashioniq_images/B000D2.jpg"}

def test_normalize_query_text_joins_like_the_release() -> None:
    assert normalize_query_text("Is blue. And   LONG!") == "is blue and long"
    assert normalize_query_text("") == ""

def test_caption_index_keys_on_image_and_text(fixture_root: Path) -> None:
    index = caption_scope_index(fixture_root)
    assert index[("B000S", "is blue and is long")] == frozenset({"shirt"})
    assert index[("B000T", "is green and is sleeveless")] == frozenset({"toptee"})
    # The same caption text filed under both ambiguous categories maps to both: undecidable.
    assert index[("B000AM", "is darker and has a different graphic and is darker")] == (
        frozenset({"shirt", "toptee"}))

def test_caption_index_skips_missing_and_malformed_annotation_files(tmp_path: Path) -> None:
    assert caption_scope_index(tmp_path) == {}
    _write_captions(tmp_path, "shirt", [])
    (tmp_path / "captions" / "cap.toptee.val.json").write_text("not json", encoding="utf-8")
    (tmp_path / "captions" / "cap.dress.val.json").write_text('{"rows": 1}', encoding="utf-8")
    assert caption_scope_index(tmp_path) == {}

def test_loader_assigned_category_is_never_reinterpreted(fixture_root: Path) -> None:
    """The 5,884 queries the release already scoped must keep that scope byte-for-byte.

    Here the annotation would say ``toptee``; the loader said ``shirt``; the loader wins.
    """
    index = caption_scope_index(fixture_root)
    query = _query("Is green and is sleeveless", "7:2", category="shirt")
    assert resolve_query_category(SCOPES, index, query, NAMES) == "shirt"

def test_annotation_resolves_the_gap_the_loader_left(fixture_root: Path) -> None:
    index = caption_scope_index(fixture_root)
    assert resolve_query_category(SCOPES, index, _query("Is blue and is long", "7:1"), NAMES) == "shirt"
    assert resolve_query_category(SCOPES, index, _query("Is green and is sleeveless", "7:2"), NAMES) == "toptee"

def test_text_filed_under_two_categories_stays_unscored(fixture_root: Path) -> None:
    """No default. The same caption pair under shirt and toptee is the release's own ambiguity."""
    index = caption_scope_index(fixture_root)
    query = _query("Is darker and has a different graphic and is darker", "7:3")
    assert resolve_query_category(SCOPES, index, query, NAMES) is None

def test_unknown_reference_or_text_stays_unscored(fixture_root: Path) -> None:
    index = caption_scope_index(fixture_root)
    # Reference id the candidate table does not know: no identity, no evidence, no guess.
    assert resolve_query_category(SCOPES, index, _query("Is blue and is long", "7:9"), NAMES) is None
    # Known image, text that no annotation row produced.
    assert resolve_query_category(SCOPES, index, _query("is something else", "7:1"), NAMES) is None
    # No candidate table at all.
    assert resolve_query_category(SCOPES, index, _query("Is blue and is long", "7:1"), None) is None

def test_text_shared_by_two_images_follows_its_own_reference(fixture_root: Path) -> None:
    """Identical caption text on a dress image and a shirt image must not leak across categories."""
    _write_captions(fixture_root, "dress", [_row("B000D", "is red", "is short"),
                                            _row("B000D2", "is blue", "is long")])
    index = caption_scope_index(fixture_root)
    assert resolve_query_category(SCOPES, index, _query("Is blue and is long", "7:1"), NAMES) == "shirt"
    assert resolve_query_category(SCOPES, index, _query("Is blue and is long", "7:6"), NAMES) == "dress"

def test_scope_outside_declared_scopes_is_refused(fixture_root: Path) -> None:
    """A category the adapter has no checkpoint for must resolve to unscored, not to itself."""
    index = caption_scope_index(fixture_root)
    query = _query("Is blue and is long", "7:1", category="shoes")
    assert resolve_query_category(SCOPES, index, query, NAMES) is None

def test_adapter_seam_is_pure_routing(fixture_root: Path) -> None:
    """The adapter method must answer from the annotation files alone: no worker, no GPU, no model."""
    adapter = object.__new__(Adapter)
    adapter.dataset_root = fixture_root
    adapter._caption_scopes = None
    adapter._candidate_names = dict(NAMES)
    adapter.config = SimpleNamespace(WORKBENCH_CIRCUS_RELEASE_ROOT=fixture_root)
    assert adapter.query_scope(_query("Is blue and is long", "7:1")) == "shirt"
    assert adapter.query_scope(_query("Is green and is sleeveless", "7:2")) == "toptee"
    assert adapter.query_scope(_query("Is darker and has a different graphic and is darker", "7:3")) is None
    assert adapter._caption_scopes is not None, "the index is built once and cached"

RELEASE = Path(__file__).resolve().parents[2] / "workbench/artifacts/circus_release"
FASHIONIQ = Path(__file__).resolve().parents[2] / "data/FashionIQ"

def _require_real_data() -> tuple[object, object]:
    if not (RELEASE / "mbeir_fashioniq_task7_test.jsonl").is_file():
        pytest.skip("CIRCUS release is not present on this host (external artifact)")
    if not (FASHIONIQ / "captions/cap.shirt.val.json").is_file():
        pytest.skip("FashionIQ annotations are not present on this host (external artifact)")
    from workbench.composition import data as release_data

    loaded = release_data.load_release(RELEASE, verify_hashes=False)
    return release_data.assign_garment_categories(loaded, FASHIONIQ), loaded

def test_all_119_unscored_queries_are_classified() -> None:
    """Every one of the 119 gets a decision, and the recorded explanation is checked, not assumed."""
    assigned, _ = _require_real_data()
    index = caption_scope_index(FASHIONIQ)
    ambiguous = [query for query in assigned.queries if query.category is None]
    assert len(ambiguous) == 119, "the recorded run's unscored set changed"

    resolved = {query.qid: resolve_query_category(SCOPES, index, query, assigned.candidate_names)
                for query in ambiguous}
    determinable = {qid: scope for qid, scope in resolved.items() if scope is not None}
    assert len(determinable) == 117, sorted(qid for qid, scope in resolved.items() if scope is None)
    assert {qid for qid, scope in resolved.items() if scope is None} == {"7:3038", "7:3790"}
    assert set(determinable.values()) == {"shirt", "toptee"}

    for cause, examples in RECORDED.items():
        for qid, expected in examples.items():
            assert resolved[qid] == expected, f"{qid} ({cause})"

def test_annotation_never_contradicts_the_loader_on_the_other_5884() -> None:
    """Routing must be extend-only: no currently scored query may change scope."""
    assigned, _ = _require_real_data()
    index = caption_scope_index(FASHIONIQ)
    changed = []
    for query in assigned.queries:
        if query.category is None:
            continue
        routed = resolve_query_category(SCOPES, index, query, assigned.candidate_names)
        if routed != query.category:
            changed.append((query.qid, query.category, routed))
    assert not changed, changed[:10]

def test_rescoring_arithmetic_closes_at_6003() -> None:
    """The accounting the reports need: 5,884 + 117 + 2 == 6,003, with no silent drop."""
    assigned, _ = _require_real_data()
    index = caption_scope_index(FASHIONIQ)
    routed = [resolve_query_category(SCOPES, index, query, assigned.candidate_names)
              for query in assigned.queries]
    scored = sum(scope is not None for scope in routed)
    unscored = len(routed) - scored
    assert (scored, unscored) == (6001, 2)
    assert scored + unscored == len(assigned.queries) == 6003
    assert all(scope in SCOPES for scope in routed if scope is not None)
