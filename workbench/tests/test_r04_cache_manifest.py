"""R04: a reusable cache root must PROVE it describes the requested inputs.

Reproduced failure (bundled probe `cache_and_absence_probe.py`): a cache root that symlinked an OLD
annotations directory and held tensors made by a foreign CLIP transform was handed to the real
`run_cache_root` for a DIFFERENT dataset root. The native constructor took its warm branch on file
presence, loaded the foreign tensors, ran the declared targetpad transform zero times and raised
nothing. `effective_annotation_identity="OLD_ANNOTATIONS"`, `loaded_query_pixels` a foreign tensor.

These tests build that exact shape synthetically -- no model, no GPU, no real dataset -- and assert the
three required verdicts: foreign cache REFUSED with the differing field named, matching manifest
ACCEPTED, absent manifest REFUSED (unless the operator explicitly opts out, which the run then records).
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from workbench.replay.common import (  # noqa: E402
    CACHE_MANIFEST_NAME,
    CACHE_VERIFICATION_OPT_OUT_ENV,
    CACHE_WARM_SENTINEL,
    CacheManifestError,
    cache_requirement,
    cache_tensor_digests,
    compare_cache_manifest,
    describe_cache_inputs,
    gallery_identity,
    isolated_cache_root,
    reusable_cache_root,
    run_cache_root,
    write_cache_manifest,
)

CATEGORIES = ("dress", "shirt", "toptee")

def _dataset(root: Path, identity: str) -> Path:
    """A minimal annotation set: the split lists ARE the gallery order the constructors index."""
    for name in ("captions", "image_splits", "resized_image"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "captions" / "identity.txt").write_text(identity, encoding="utf-8")
    for category in CATEGORIES:
        (root / "captions" / f"cap.{category}.val.json").write_text("[]", encoding="utf-8")
        (root / "image_splits" / f"split.{category}.val.json").write_text(
            json.dumps(["a", "b", "c"]), encoding="utf-8")
        (root / "resized_image" / category).mkdir(exist_ok=True)
    return root

def _warm_cache(cache: Path, source: Path, *, tensors: dict[str, bytes] | None = None) -> Path:
    """A cache root in the shape the native constructor leaves: sentinel + pickles + input links."""
    cache.mkdir(parents=True, exist_ok=True)
    for name in ("captions", "image_splits", "resized_image"):
        link = cache / name
        if not link.exists():
            link.symlink_to(source / name, target_is_directory=True)
    (cache / CACHE_WARM_SENTINEL).write_text("[]", encoding="utf-8")
    for name, payload in (tensors or {}).items():
        (cache / name).write_bytes(payload)
    return cache

def _tensors(marker: str) -> dict[str, bytes]:
    return {f"test_queries_{category}.pkl": pickle.dumps([{"source_img_data": marker}])
            for category in CATEGORIES} | {
        f"test_targets_{category}.pkl": pickle.dumps([{"target_img_data": marker}])
        for category in CATEGORIES}

def _manifest(cache: Path, dataset: Path, *, transform: str = "FOREIGN_TRANSFORM") -> None:
    """The manifest a warm pass would write for this cache/dataset pair."""
    write_cache_manifest(cache, {
        **cache_requirement(dataset, model_id="airknow", transform=None),
        "model_id": "airknow",
        "transform": transform,
        "checkpoint": {"checkpoint_id": "fiq_n05", "expected_sha256": None},
        "state": {"seed": None, "eval_mode": False, "noise_ratio": 0},
        "tensors": cache_tensor_digests(cache),
        "cache_files": sorted(cache_tensor_digests(cache)),
        "written_by": {"script": "test", "version": 1},
    })

# ------------------------------------------------------------ the probe: foreign links + tensors

def test_foreign_annotation_symlink_is_refused_naming_the_field(tmp_path: Path) -> None:
    """The probe's first step: the cache's links point at OLD annotations, not the requested ones."""
    requested = _dataset(tmp_path / "requested", "REQUESTED_ANNOTATIONS")
    old = _dataset(tmp_path / "old", "OLD_ANNOTATIONS")
    cache = _warm_cache(tmp_path / "cache", old, tensors=_tensors("FOREIGN_CLIP_TRANSFORM_TENSOR"))

    with pytest.raises(CacheManifestError) as caught:
        with reusable_cache_root(requested, cache,
                                 requirement=cache_requirement(requested, model_id="airknow")):
            pytest.fail("a cache linking a different annotation root was accepted")

    text = " ".join(caught.value.mismatches)
    assert "annotations.captions.resolved_path" in text
    assert str((requested / "captions").resolve()) in text
    assert str((old / "captions").resolve()) in text

def test_a_manifest_over_foreign_links_is_still_refused(tmp_path: Path) -> None:
    """Even a self-consistent manifest cannot launder a link that points somewhere else."""
    requested = _dataset(tmp_path / "requested", "REQUESTED_ANNOTATIONS")
    old = _dataset(tmp_path / "old", "OLD_ANNOTATIONS")
    cache = _warm_cache(tmp_path / "cache", old, tensors=_tensors("FOREIGN"))
    _manifest(cache, old)                      # manifest describes the OLD inputs, honestly

    with pytest.raises(CacheManifestError, match="resolved_path"):
        with reusable_cache_root(requested, cache,
                                 requirement=cache_requirement(requested, model_id="airknow")):
            pytest.fail("matching manifest over foreign links was accepted")

def test_foreign_tensors_under_matching_links_are_refused_by_digest(tmp_path: Path) -> None:
    """Links right, pixels foreign: the manifest differs on the tensor digests."""
    requested = _dataset(tmp_path / "requested", "REQUESTED_ANNOTATIONS")
    cache = _warm_cache(tmp_path / "cache", requested, tensors=_tensors("FOREIGN"))
    _manifest(cache, requested, transform="FOREIGN_CLIP_TRANSFORM")
    (cache / "test_queries_dress.pkl").write_bytes(b"rewritten-after-the-manifest")

    with pytest.raises(CacheManifestError) as caught:
        with reusable_cache_root(requested, cache,
                                 requirement=cache_requirement(requested, model_id="airknow")):
            pytest.fail("a rewritten tensor was accepted")

    assert any("test_queries_dress.pkl" in field for field in caught.value.mismatches)

def test_a_different_transform_is_refused(tmp_path: Path) -> None:
    """The declared transform must be the transform that produced the pixels."""
    requested = _dataset(tmp_path / "requested", "REQUESTED")
    cache = _warm_cache(tmp_path / "cache", requested, tensors=_tensors("PIXELS"))
    _manifest(cache, requested, transform="FOREIGN_CLIP_TRANSFORM")

    with pytest.raises(CacheManifestError, match="transform"):
        with reusable_cache_root(
                requested, cache,
                requirement=cache_requirement(requested, model_id="airknow",
                                              transform="data_utils.targetpad_transform(1.25, 224)")):
            pytest.fail("a cache built by a different transform was accepted")

# ------------------------------------------------------------------- the two required accept/refuse

def test_a_matching_manifest_is_accepted_and_recorded(tmp_path: Path) -> None:
    """A cache the warm path wrote with matching inputs is reused, and the run says so."""
    dataset = _dataset(tmp_path / "dataset", "REQUESTED")
    cache = _warm_cache(tmp_path / "cache", dataset, tensors=_tensors("PIXELS"))
    _manifest(cache, dataset, transform="data_utils.targetpad_transform(1.25, 224)")
    record: dict = {}

    with reusable_cache_root(
            dataset, cache,
            requirement=cache_requirement(dataset, model_id="airknow",
                                          transform="data_utils.targetpad_transform(1.25, 224)"),
            record=record) as root:
        assert root == cache

    assert record["cache_manifest_status"] == "verified"
    assert record["cache_manifest_path"].endswith(CACHE_MANIFEST_NAME)
    assert record["cache_manifest_digest"] == \
        __import__("workbench.replay.common", fromlist=["sha256_file"]).sha256_file(
            cache / CACHE_MANIFEST_NAME)
    assert record["cache_manifest_opted_out"] is False

def test_a_symlink_that_moved_between_warm_and_reuse_is_refused(tmp_path: Path) -> None:
    """The link name is unchanged; only its TARGET moved. That must not compare equal."""
    first = _dataset(tmp_path / "annotations-v1", "V1")
    second = _dataset(tmp_path / "annotations-v2", "V2")
    cache = _warm_cache(tmp_path / "cache", first, tensors=_tensors("PIXELS"))
    _manifest(cache, first, transform="T")
    (cache / "captions").unlink()
    (cache / "captions").symlink_to(second / "captions", target_is_directory=True)

    with pytest.raises(CacheManifestError, match="annotations.captions.resolved_path"):
        with reusable_cache_root(first, cache, requirement=cache_requirement(first, model_id="airknow")):
            pytest.fail("a moved symlink compared equal to the original target")

# ------------------------------------------------------------------------------ unverified caches

def test_a_cache_with_no_manifest_is_refused(tmp_path: Path) -> None:
    """Presence is not proof. The old behaviour -- reuse whatever files are there -- is the defect."""
    dataset = _dataset(tmp_path / "dataset", "REQUESTED")
    cache = _warm_cache(tmp_path / "cache", dataset, tensors=_tensors("PIXELS"))

    with pytest.raises(CacheManifestError) as caught:
        with reusable_cache_root(dataset, cache,
                                 requirement=cache_requirement(dataset, model_id="airknow")):
            pytest.fail("an unverified cache root was reused silently")

    assert "missing" in " ".join(caught.value.mismatches)

def test_a_schema_version_change_is_refused(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "dataset", "REQUESTED")
    cache = _warm_cache(tmp_path / "cache", dataset, tensors=_tensors("PIXELS"))
    _manifest(cache, dataset, transform="T")
    document = json.loads((cache / CACHE_MANIFEST_NAME).read_text(encoding="utf-8"))
    document["schema_version"] = 0
    (cache / CACHE_MANIFEST_NAME).write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(CacheManifestError, match="schema_version"):
        with reusable_cache_root(dataset, cache, requirement=cache_requirement(dataset, model_id="airknow")):
            pytest.fail("an old-schema manifest was read as if current")

def test_a_gallery_count_change_is_refused(tmp_path: Path) -> None:
    """A gallery that grew or reordered makes every cached target id wrong."""
    dataset = _dataset(tmp_path / "dataset", "REQUESTED")
    cache = _warm_cache(tmp_path / "cache", dataset, tensors=_tensors("PIXELS"))
    _manifest(cache, dataset, transform="T")
    (dataset / "image_splits" / "split.dress.val.json").write_text(
        json.dumps(["a", "b", "c", "d"]), encoding="utf-8")

    with pytest.raises(CacheManifestError, match="gallery.dress.count"):
        with reusable_cache_root(dataset, cache, requirement=cache_requirement(dataset, model_id="airknow")):
            pytest.fail("a cache built against a shorter gallery was reused")

def test_the_operator_opt_out_is_explicit_and_visible(tmp_path: Path, monkeypatch) -> None:
    """Trusting an unverified cache is possible, but the run records that it happened."""
    dataset = _dataset(tmp_path / "dataset", "REQUESTED")
    cache = _warm_cache(tmp_path / "cache", dataset, tensors=_tensors("PIXELS"))
    monkeypatch.setenv(CACHE_VERIFICATION_OPT_OUT_ENV, "1")
    record: dict = {}

    with reusable_cache_root(dataset, cache,
                             requirement=cache_requirement(dataset, model_id="airknow"),
                             record=record) as root:
        assert root == cache

    assert record["cache_manifest_opted_out"] is True
    # The ENTRY was not verified (opt-out), so the status must not read `verified`; a manifest is
    # still written on the way out, describing what this pass produced, so a later run can check it.
    assert record["cache_manifest_status"] == "not_verified_opt_out"
    assert (cache / CACHE_MANIFEST_NAME).is_file()

def test_a_cold_cache_root_reports_rebuilt(tmp_path: Path, monkeypatch) -> None:
    """A root with no tensors has nothing to verify; the run must say it built cold."""
    dataset = _dataset(tmp_path / "dataset", "REQUESTED")
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("WORKBENCH_REPLAY_CACHE_ROOT", str(cache))
    record: dict = {}

    with run_cache_root(dataset, requirement=cache_requirement(dataset, model_id="airknow"),
                        record=record):
        assert not (cache / CACHE_WARM_SENTINEL).exists()

    assert record["cache_manifest_status"] == "rebuilt"

def test_the_isolated_root_records_rebuilt_and_has_no_manifest(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "dataset", "REQUESTED")
    record: dict = {}
    with run_cache_root(dataset, record=record) as root:
        assert not (Path(root) / CACHE_MANIFEST_NAME).exists()
    assert record["cache_manifest_status"] == "rebuilt"
    assert record["cache_manifest_digest"] is None

# --------------------------------------------------------------------------- manifest mechanics

def test_the_manifest_is_written_atomically(tmp_path: Path) -> None:
    """Temp file + os.replace: no partially written manifest can describe a half-written cache."""
    cache = tmp_path / "cache"
    cache.mkdir()
    write_cache_manifest(cache, {"model_id": "airknow"})
    leftovers = [path for path in cache.iterdir() if path.name != CACHE_MANIFEST_NAME]
    assert leftovers == [], f"atomic write left temporary files behind: {leftovers}"
    document = json.loads((cache / CACHE_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert document["written_at"] and document["schema_version"] == 1

def test_the_manifest_records_both_requested_and_resolved_annotation_paths(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "dataset", "REQUESTED")
    link_root = tmp_path / "linked"
    link_root.mkdir()
    for name in ("captions", "image_splits", "resized_image"):
        (link_root / name).symlink_to(dataset / name, target_is_directory=True)

    described = describe_cache_inputs(link_root)
    assert described["captions"]["requested_path"] == str(link_root / "captions")
    assert described["captions"]["resolved_path"] == str((dataset / "captions").resolve())
    assert described["captions"]["digest"] == describe_cache_inputs(dataset)["captions"]["digest"]

def test_gallery_identity_records_order_and_count(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "dataset", "REQUESTED")
    identity = gallery_identity(dataset)
    assert identity["dress"]["count"] == 3
    assert identity["dress"]["first"] == "a" and identity["dress"]["last"] == "c"
    assert identity["dress"]["digest"]

def test_compare_returns_every_differing_field_not_just_the_first(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "dataset", "REQUESTED")
    cache = _warm_cache(tmp_path / "cache", dataset, tensors=_tensors("PIXELS"))
    write_cache_manifest(cache, {"model_id": "other", "transform": "OTHER"})

    mismatches = compare_cache_manifest(cache, cache_requirement(dataset, model_id="airknow",
                                                                 transform="MINE"))
    fields = " ".join(mismatches)
    assert "model_id" in fields and "transform" in fields
