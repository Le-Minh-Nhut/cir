"""R05: a gutted annotation file must not read as "this category has no row".

Three probes reproduced on HEAD and pinned here:

* ``toptee_empty`` -- ``cap.toptee.val.json`` emptied to ``[]`` (valid JSON) still routed an
  ambiguous query to ``"shirt"``;
* ``toptee_invalid_row`` -- one wrong-typed row did the same;
* the loader's ``assign_garment_categories`` accepted a split emptied to ``[]`` because
  ``len({...}) == len([])``, so a category silently lost every identity.

Every refusal below is about PROVENANCE, not about the release saying something different: a
``[]`` file is not evidence that a garment is absent from that category, and a clean
``None``-vs-``None`` cannot tell the two apart.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from workbench.composition.data import (
    ANNOTATION_PIN,
    assign_garment_categories,
    inspect_annotation,
    inspect_annotation_set,
)
from workbench.composition.model_limn import (
    Adapter,
    CaptionScopeIndex,
    caption_scope_index,
    resolve_query_category,
    resolve_query_scope,
)

SCOPES = Adapter.category_scopes
ROW = {"candidate": "reference", "captions": ["alpha", "beta"]}
NAMES = {"7:1": "mbeir_images/fashioniq_images/reference.jpg",
         "7:2": "mbeir_images/fashioniq_images/B000D.jpg"}


def _query(text: str = "alpha and beta", category: str | None = None,
           reference_id: str = "7:1") -> SimpleNamespace:
    return SimpleNamespace(qid="7:1", index=0, query_txt=text, reference_id=reference_id,
                           category=category)


def _write(root: Path, category: str, rows: object, *, suffix: str = "val") -> None:
    (root / "captions").mkdir(parents=True, exist_ok=True)
    (root / "captions" / f"cap.{category}.{suffix}.json").write_text(
        json.dumps(rows), encoding="utf-8")


@pytest.fixture()
def fixture_root(tmp_path: Path) -> Path:
    """Synthetic annotations: no pinned bytes, so the pin is explicit and empty.

    ``dress`` holds an unrelated row so the SET is complete; tests that need an incomplete set
    gut or corrupt exactly the file they want to make unusable.
    """
    _write(tmp_path, "dress", [{"candidate": "B000D", "captions": ["is red", "is short"]}])
    _write(tmp_path, "shirt", [ROW])
    _write(tmp_path, "toptee", [ROW])
    return tmp_path


def _index(root: Path) -> CaptionScopeIndex:
    return caption_scope_index(root, {})


# --------------------------------------------------------------- the three probe cases

def test_empty_annotation_file_refuses_routing(fixture_root: Path) -> None:
    """PROBE ``toptee_empty``: ``[]`` is valid JSON, and used to yield a unique wrong category."""
    _write(fixture_root, "toptee", [])
    index = _index(fixture_root)
    assert index.empty == frozenset({"toptee"})
    assert index.incomplete
    assert resolve_query_category(SCOPES, index, _query(), NAMES) is None
    record = resolve_query_scope(index, _query(), NAMES)
    assert record["scope"] is None
    assert record["reason"] == "unresolved:annotation_index_incomplete"
    assert record["provenance_incomplete"] is True


def test_malformed_row_refuses_routing(fixture_root: Path) -> None:
    """PROBE ``toptee_invalid_row``: one wrong-typed row makes the whole file unusable."""
    _write(fixture_root, "toptee", [{"candidate": 123, "captions": ["alpha", "beta"]}])
    index = _index(fixture_root)
    assert list(index.malformed) == ["toptee"]
    assert "row 0" in index.malformed["toptee"]
    assert resolve_query_category(SCOPES, index, _query(), NAMES) is None


def test_a_pair_filed_under_both_categories_stays_unscored(fixture_root: Path) -> None:
    """PROBE ``both_valid``: genuine ambiguity, still ``None`` with a COMPLETE index."""
    index = _index(fixture_root)
    assert index.complete and not index.incomplete
    record = resolve_query_scope(index, _query(), NAMES)
    assert record["scope"] is None
    assert record["reason"] == "unresolved:annotation_names_two_categories"
    assert record["holding_categories"] == ("shirt", "toptee")
    assert resolve_query_category(SCOPES, index, _query(), NAMES) is None


def test_a_single_row_still_resolves(fixture_root: Path) -> None:
    """The positive control: one holder, complete set, still routes as before."""
    _write(fixture_root, "toptee", [{"candidate": "B000T", "captions": ["is green", "is sleeveless"]}])
    index = _index(fixture_root)
    assert resolve_query_category(SCOPES, index, _query(), NAMES) == "shirt"
    record = resolve_query_scope(index, _query(), NAMES)
    assert (record["scope"], record["reason"]) == ("shirt", "annotation:shirt")


# --------------------------------------------------------------- schema, at row granularity

@pytest.mark.parametrize("bad", [
    {"candidate": 123, "captions": ["a", "b"]},
    {"candidate": "x", "captions": "a and b"},
    {"candidate": "x", "captions": ["a", 3]},
    {"candidate": "x"},
    {"captions": ["a", "b"]},
    ["x", "a and b"],
])
def test_any_malformed_row_invalidates_the_file(tmp_path: Path, bad: object) -> None:
    path = tmp_path / "cap.shirt.val.json"
    path.write_text(json.dumps([ROW, bad]), encoding="utf-8")
    report = inspect_annotation(path, pin={})
    assert not report.usable
    assert report.problem == "malformed"
    assert report.detail is not None and report.detail.startswith("row 1")


def test_counts_and_digests_are_enforced_against_the_pin(tmp_path: Path) -> None:
    """An emptied or truncated released file fails LOUDLY, not as a category with no rows."""
    dress = Path("data/FashionIQ/captions/cap.dress.val.json")
    if not dress.is_file():
        pytest.skip("FashionIQ annotations are not present on this host (external artifact)")
    rows = json.loads(dress.read_text(encoding="utf-8"))
    truncated = tmp_path / "cap.dress.val.json"
    truncated.write_text(json.dumps(rows[:-1]), encoding="utf-8")
    report = inspect_annotation(truncated)
    assert report.problem == "count_mismatch"
    assert report.usable is False

    emptied = tmp_path / "cap.shirt.val.json"
    emptied.write_text("[]", encoding="utf-8")
    report = inspect_annotation(emptied)
    assert (report.problem, report.count) == ("empty", 0)

    # Same row count, different bytes: the digest is the only check that catches it.
    edited = tmp_path / "cap.dress.val.json"
    payload = [dict(rows[0], captions=["changed", "text"]), *rows[1:]]
    edited.write_text(json.dumps(payload), encoding="utf-8")
    assert len(payload) == ANNOTATION_PIN["cap.dress.val.json"]["rows"]
    assert inspect_annotation(edited).problem == "digest_mismatch"


def test_a_complete_pin_matching_set_resolves_exactly_as_before() -> None:
    """The good path does not regress: real released files, 6003 keys, 117 of 119 routed."""
    fashioniq = Path(__file__).resolve().parents[2] / "data/FashionIQ"
    if not (fashioniq / "captions/cap.shirt.val.json").is_file():
        pytest.skip("FashionIQ annotations are not present on this host (external artifact)")
    index = caption_scope_index(fashioniq, ANNOTATION_PIN)
    assert index.complete, index.report()
    assert not index.incomplete
    assert index.counts == {"dress": 2017, "shirt": 2038, "toptee": 1961}
    for category, digest in index.digests.items():
        assert digest == ANNOTATION_PIN[f"cap.{category}.val.json"]["sha256"]
    assert len(index) == 6003


# --------------------------------------------------------------- loader side

@dataclass(frozen=True)
class _Query:
    qid: str
    reference_id: str
    category: str | None = None


@dataclass(frozen=True)
class _Data:
    queries: tuple
    candidate_names: dict
    provenance: dict


def _loader_data(*qids: str) -> _Data:
    return _Data(tuple(_Query(qid, "r") for qid in qids),
                 {"r": "images/reference.jpg"}, {"is_synthetic": True})


def _write_loader(root: Path, *, splits: dict[str, list[str]],
                  captions: dict[str, list[dict]] | None = None) -> None:
    (root / "image_splits").mkdir(parents=True, exist_ok=True)
    (root / "captions").mkdir(parents=True, exist_ok=True)
    captions = captions or {}
    for category, values in splits.items():
        (root / "image_splits" / f"split.{category}.val.json").write_text(
            json.dumps(values), encoding="utf-8")
        (root / "captions" / f"cap.{category}.test.json").write_text(
            json.dumps(captions.get(category, [])), encoding="utf-8")


def _loader_caption_row(candidate: str) -> dict:
    return {"candidate": candidate, "captions": ["a caption", "another"]}


def test_loader_emptied_split_makes_assignment_unavailable(tmp_path: Path) -> None:
    """PROBE loader case: a ``[]`` split used to pass validation and delete the identities."""
    captions = {category: [_loader_caption_row("other")]
                for category in ("dress", "shirt", "toptee")}
    # Control: with every split present the reference is genuinely ambiguous (shirt + toptee).
    _write_loader(tmp_path, splits={"dress": [], "shirt": ["reference"], "toptee": ["reference"]},
                  captions=captions)
    assigned = assign_garment_categories(_loader_data("7:1"), tmp_path, annotation_pin={})
    assert assigned.provenance["category_resolution"]["status"] == "unavailable"
    assert assigned.queries[0].category is None

    # The exact defect: gutting ONE split used to leave shirt as the single holder.
    _write_loader(tmp_path, splits={"dress": ["other"], "shirt": ["reference"], "toptee": []},
                  captions=captions)
    assigned = assign_garment_categories(_loader_data("7:1"), tmp_path, annotation_pin={})
    assert assigned.queries[0].category is None, "an emptied split must not resolve to shirt"
    resolution = assigned.provenance["category_resolution"]
    assert resolution["status"] == "unavailable"
    assert resolution["unavailable_reasons"] == {"split.toptee.val.json": "empty"}

    # Restore it: the ambiguity is real and unchanged, which is what the refusal protected.
    _write_loader(tmp_path, splits={"dress": ["other"], "shirt": ["reference"],
                                    "toptee": ["reference"]}, captions=captions)
    assigned = assign_garment_categories(_loader_data("7:1"), tmp_path, annotation_pin={})
    resolution = assigned.provenance["category_resolution"]
    assert resolution["status"] == "ambiguous"
    assert resolution["ambiguous_examples"][0]["categories"] == ("shirt", "toptee")


def test_loader_names_every_unusable_file_and_never_partially_assigns(tmp_path: Path) -> None:
    _write_loader(tmp_path, splits={"dress": ["reference"], "shirt": ["reference"],
                                    "toptee": ["reference"]},
                  captions={"dress": [{"candidate": "x", "captions": ["a"]}],
                            "shirt": [{"candidate": "y", "captions": 7}],
                            "toptee": []})
    assigned = assign_garment_categories(_loader_data("7:1", "7:2"), tmp_path, annotation_pin={})
    assert [query.category for query in assigned.queries] == [None, None]
    resolution = assigned.provenance["category_resolution"]
    assert resolution["status"] == "unavailable"
    assert resolution["resolved_count"] == resolution["ambiguous_count"] == 0
    assert resolution["unresolved_count"] == 2
    problems = {record["file"]: record["problem"]
                for record in resolution["annotation_provenance"]["files"]}
    assert problems["cap.shirt.test.json"] == "malformed"
    assert problems["cap.toptee.test.json"] == "empty"
    assert problems["cap.dress.test.json"] is None


def test_loader_reports_a_dishonest_duplicate_split(tmp_path: Path) -> None:
    """Same row count, repeated identity: still not usable as a unique set of identities."""
    captions = {category: [_loader_caption_row("other")]
                for category in ("dress", "shirt", "toptee")}
    _write_loader(tmp_path, splits={"dress": ["a", "a"], "shirt": ["r"], "toptee": ["r"]},
                  captions=captions)
    assigned = assign_garment_categories(_loader_data("7:1"), tmp_path, annotation_pin={})
    resolution = assigned.provenance["category_resolution"]
    assert resolution["status"] == "unavailable"
    assert resolution["unavailable_reasons"] == {"split.dress.val.json": "duplicate_identity"}


def test_loader_refuses_a_real_set_that_is_off_pin(tmp_path: Path) -> None:
    """Released (non-synthetic) releases are held to ``ANNOTATION_PIN``."""
    fashioniq = Path(__file__).resolve().parents[2] / "data/FashionIQ"
    if not (fashioniq / "captions/cap.shirt.val.json").is_file():
        pytest.skip("FashionIQ annotations are not present on this host (external artifact)")
    import shutil

    copied = tmp_path / "FashionIQ"
    for sub in ("captions", "image_splits"):
        shutil.copytree(fashioniq / sub, copied / sub)
    # The loader's own identity inputs (not the val captions, which routing reads).
    split = copied / "image_splits/split.dress.val.json"
    rows = json.loads(split.read_text(encoding="utf-8"))
    split.write_text(json.dumps(rows[:-1]), encoding="utf-8")
    data = _Data((_Query("7:1", "r"),), {"r": "images/reference.jpg"}, {})  # not synthetic
    assigned = assign_garment_categories(data, copied)
    resolution = assigned.provenance["category_resolution"]
    assert resolution["status"] == "unavailable"
    assert resolution["unavailable_reasons"] == {"split.dress.val.json": "count_mismatch"}
    assert all(query.category is None for query in assigned.queries)


# --------------------------------------------------------------- loader routing, all three states

def test_loader_assigned_category_is_reported_as_unverified_when_the_index_is_incomplete(
        fixture_root: Path) -> None:
    """Requirement 3: the route is kept, but marked ``provenance_incomplete``."""
    _write(fixture_root, "toptee", [])
    index = _index(fixture_root)
    record = resolve_query_scope(index, _query(category="shirt"), NAMES)
    assert (record["scope"], record["reason"]) == ("shirt", "loader")
    assert record["provenance_incomplete"] is True
    # With a complete set the same loader route is verified: the decision itself never changes.
    _write(fixture_root, "toptee", [{"candidate": "B000T", "captions": ["is green"]}])
    complete = resolve_query_scope(_index(fixture_root), _query(category="shirt"), NAMES)
    assert (complete["scope"], complete["reason"]) == ("shirt", "loader")
    assert complete["provenance_incomplete"] is False


def test_a_plain_dict_index_cannot_prove_completeness() -> None:
    """The regression guard for the hole itself: an untyped ``{}`` must not read as complete."""
    assert resolve_query_scope({}, _query(), NAMES)["reason"] == "unresolved:annotation_index_incomplete"
    assert _index_for_plain() is False


def _index_for_plain() -> bool:
    from workbench.composition.model_limn import _index_complete
    return _index_complete({("reference", "alpha and beta"): frozenset({"shirt"})})


# --------------------------------------------------------------- the export (requirement 4)

def _at(qid: str, text: str, *, reference_id: str = "7:1",
        category: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(qid=qid, index=0, query_txt=text, reference_id=reference_id,
                           category=category)


def test_adapter_exports_a_per_query_rationale_map(fixture_root: Path, tmp_path: Path) -> None:
    """Requirement 4: ``qid -> scope`` with ``loader`` / ``annotation:<cat>`` / ``unresolved:<why>``."""
    adapter = object.__new__(Adapter)
    adapter.dataset_root = fixture_root
    adapter._caption_scopes = None
    adapter._candidate_names = dict(NAMES)
    adapter.annotation_pin = {}
    # The runner's real layout: <artifacts>/composition/runs/<run_id>/work/<model>.
    run_root = tmp_path / "composition" / "runs" / "abc123"
    adapter.work_root = run_root / "work" / "limn"
    adapter.work_root.mkdir(parents=True)

    assert adapter.query_scope(_at("7:1", "is red and is short", reference_id="7:2")) == "dress"
    assert adapter.query_scope(_at("7:2", "alpha and beta")) is None  # shirt AND toptee
    assert adapter.query_scope(_at("7:3", "is red and is short", reference_id="7:2",
                                   category="dress")) == "dress"
    assert adapter.query_scope(_at("7:4", "nothing matches", reference_id="7:9")) is None
    resolution = adapter.query_scope_resolution()
    assert resolution["scope_by_qid"] == {"7:1": "dress", "7:2": None, "7:3": "dress",
                                          "7:4": None}
    reasons = {record["qid"]: record["reason"] for record in resolution["decisions"]}
    assert reasons == {"7:1": "annotation:dress", "7:2": "unresolved:annotation_names_two_categories",
                       "7:3": "loader", "7:4": "unresolved:no_reference_identity"}

    path = adapter._write_scope_artifact()
    assert path is not None and path.is_file()
    assert path == run_root / "scope" / "query_scope_limn.json"
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["query_count"] == 4
    assert written["annotation"]["complete"] is True
    assert written["provenance_incomplete_count"] == 0
    assert [record["qid"] for record in written["unresolved"]] == ["7:2", "7:4"]


def test_unresolved_queries_carry_a_per_scope_diagnostic(fixture_root: Path) -> None:
    """Requirement 5: what each candidate file said, its count, and whether it was complete."""
    adapter = object.__new__(Adapter)
    adapter.dataset_root = fixture_root
    adapter._caption_scopes = None
    adapter._candidate_names = dict(NAMES)
    adapter.annotation_pin = {}
    assert adapter.query_scope(_query("alpha and beta")) is None  # filed under shirt AND toptee
    record = adapter._scope_diagnostics(adapter.caption_scope_index(), _query("alpha and beta"),
                                        NAMES)
    assert record["reference_stem"] == "reference"
    assert record["dress"]["holds_pair"] is False
    for category in ("dress", "shirt", "toptee"):
        entry = record[category]
        assert entry["file"] == f"cap.{category}.val.json"
        assert entry["usable"] is True
        assert entry["count"] == 1
    assert record["shirt"]["holds_pair"] is True
    assert record["toptee"]["holds_pair"] is True


def test_real_unresolved_pair_has_its_own_diagnostics() -> None:
    """``7:3038`` / ``7:3790`` stay unscored, each with the reason and the evidence."""
    fashioniq = Path(__file__).resolve().parents[2] / "data/FashionIQ"
    release = Path(__file__).resolve().parents[2] / "workbench/artifacts/circus_release"
    if not (fashioniq / "captions/cap.shirt.val.json").is_file():
        pytest.skip("FashionIQ annotations are not present on this host (external artifact)")
    index = caption_scope_index(fashioniq)
    from workbench.composition.data import load_release

    if not (release / "mbeir_fashioniq_task7_test.jsonl").is_file():
        pytest.skip("CIRCUS release is not present on this host (external artifact)")
    data = load_release(release, verify_hashes=False)
    names = data.candidate_names
    from workbench.composition.data import assign_garment_categories as _assign

    assigned = _assign(data, fashioniq)
    by_qid = {query.qid: query for query in assigned.queries}
    for qid in ("7:3038", "7:3790"):
        record = resolve_query_scope(index, by_qid[qid], names)
        assert record["scope"] is None, qid
        assert record["reason"] == "unresolved:annotation_names_two_categories"
        assert record["holding_categories"] == ("shirt", "toptee")
        assert record["provenance_incomplete"] is False


def test_pin_is_data_owned_and_matches_the_committed_scope_map() -> None:
    """The pin is not remembered: it is checked against the committed M-02 map."""
    scope_map = (Path(__file__).resolve().parents[1]
                 / "docs/reviews/20261111-master-audit/raw/m02/qid_scope_map.json")
    if not scope_map.is_file():
        pytest.skip("committed M-02 scope map is not present on this host")
    committed = json.loads(scope_map.read_text(encoding="utf-8"))["sha256"]
    merged = {**committed["caption_files"], **committed["loader_identity_support_files"]}
    for name, entry in ANNOTATION_PIN.items():
        relative = next(path for path in merged if Path(path).name == name)
        assert merged[relative] == entry["sha256"], name
