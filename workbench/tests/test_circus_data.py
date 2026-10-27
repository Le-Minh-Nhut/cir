from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from workbench.composition.data import (
    EXPECTED_QUERY_COUNTS,
    EXPECTED_VALIDATED_PART_COUNTS,
    RELEASED_GALLERY_IMAGES,
    assign_garment_categories,
    load_release,
    resolve_release_paths,
    select_queries,
    verify_release_hashes,
)


@pytest.fixture(scope="module")
def release_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("circus-release")
    paths = resolve_release_paths(root)
    for path in (paths.sf, paths.validated_solved, paths.validated_unsolved):
        path.parent.mkdir(parents=True, exist_ok=True)
    paths.aggregated_audit.parent.mkdir(parents=True, exist_ok=True)

    def query_row(index: int) -> dict[str, object]:
        return {"qid": f"7:{index}", "query_txt": f"exact text and not split {index}",
                "query_img_path": "mbeir_images/fashioniq_images/reference.jpg",
                "query_modality": "image,text", "query_src_content": "{}",
                "pos_cand_list": ["7:2"], "neg_cand_list": [], "task_id": 7}

    def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    full = [query_row(index + 1) for index in range(EXPECTED_QUERY_COUNTS["full"])]
    sf = [query_row(index + 1) for index in range(EXPECTED_QUERY_COUNTS["sf"])]
    solved = [query_row(index + 1) for index in range(EXPECTED_VALIDATED_PART_COUNTS["validated_solved"])]
    unsolved = [query_row(index + 369) for index in range(EXPECTED_VALIDATED_PART_COUNTS["validated_unsolved"])]
    write_jsonl(paths.mbeir_query, full)
    write_jsonl(paths.sf, list(reversed(sf)))
    write_jsonl(paths.validated_solved, list(reversed(solved)))
    write_jsonl(paths.validated_unsolved, unsolved)

    candidates = [{"txt": None, "img_path": f"mbeir_images/fashioniq_images/image{index:05}.jpg",
                   "modality": "image", "did": f"7:{index + 1}", "src_content": "{}"}
                  for index in range(RELEASED_GALLERY_IMAGES)]
    candidates[0]["img_path"] = "mbeir_images/fashioniq_images/reference.jpg"
    candidates[1]["img_path"] = "mbeir_images/fashioniq_images/target.jpg"
    write_jsonl(paths.mbeir_candidates, candidates)
    paths.aggregated_audit.write_text(
        json.dumps({"gallery_paths": [f"<path_to_M-BEIR>/{row['img_path']}" for row in candidates]}),
        encoding="utf-8",
    )
    return root


def _copy_release(source_root: Path, target_root: Path) -> Path:
    shutil.copytree(source_root, target_root)
    return target_root


def test_release_membership_and_gallery_are_identity_based_and_stable(release_root: Path) -> None:
    data = load_release(release_root, verify_hashes=False)
    assert len(data.queries) == EXPECTED_QUERY_COUNTS["full"]
    validated = select_queries(data, "validated")
    assert len(select_queries(data, "sf")) == EXPECTED_QUERY_COUNTS["sf"]
    assert len(validated) == EXPECTED_QUERY_COUNTS["validated"]
    assert data.gallery.size == RELEASED_GALLERY_IMAGES
    assert data.gallery.image_ids[:3] == (
        "mbeir_images/fashioniq_images/reference.jpg",
        "mbeir_images/fashioniq_images/target.jpg",
        "mbeir_images/fashioniq_images/image00002.jpg",
    )
    assert data.gallery.image_ids[-1] == f"mbeir_images/fashioniq_images/image{RELEASED_GALLERY_IMAGES - 1:05}.jpg"
    assert data.provenance["gallery_candidate_basename_match"] is True
    assert data.gallery == load_release(release_root, verify_hashes=False).gallery
    gallery = data.gallery
    for subset in ("full", "sf", "validated"):
        select_queries(data, subset)
        assert data.gallery is gallery


def test_validated_parts_are_disjoint_and_identity_subsets(release_root: Path) -> None:
    data = load_release(release_root, verify_hashes=False)
    solved = {f"7:{i}" for i in range(1, 369)}
    unsolved = {f"7:{i}" for i in range(369, 587)}
    validated = {query.qid for query in select_queries(data, "validated")}
    sf = {query.qid for query in select_queries(data, "sf")}
    full = {query.qid for query in data.queries}
    assert not solved & unsolved
    assert len(solved) == 368
    assert len(unsolved) == 218
    assert validated == solved | unsolved
    weighted_mean = (len(solved) * 1.0 + len(unsolved) * 0.0) / len(validated)
    assert weighted_mean == pytest.approx(368 / 586)
    assert weighted_mean != (1.0 + 0.0) / 2
    assert validated <= sf <= full


def test_validated_identity_errors_cannot_be_disabled_in_synthetic_mode(release_root: Path, tmp_path: Path) -> None:
    root = _copy_release(release_root, tmp_path / "overlapping-parts")
    paths = resolve_release_paths(root)
    rows = paths.validated_unsolved.read_text(encoding="utf-8").splitlines()
    rows[0] = paths.validated_solved.read_text(encoding="utf-8").splitlines()[0]
    paths.validated_unsolved.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="validated parts overlap by identity"):
        load_release(root, verify_hashes=False, expect_released_counts=False)


def test_released_counts_and_gallery_size_fail_loudly(release_root: Path, tmp_path: Path) -> None:
    broken = _copy_release(release_root, tmp_path / "bad-count")
    paths = resolve_release_paths(broken)
    paths.mbeir_query.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="full count 1 != expected 6003"):
        load_release(broken, verify_hashes=False)

    broken = _copy_release(release_root, tmp_path / "bad-gallery")
    paths = resolve_release_paths(broken)
    audit = json.loads(paths.aggregated_audit.read_text(encoding="utf-8"))
    audit["gallery_paths"].pop()
    paths.aggregated_audit.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="gallery count 74380 != expected 74381"):
        load_release(broken, verify_hashes=False)

def test_synthetic_gallery_order_is_preserved(release_root: Path, tmp_path: Path) -> None:
    root = _copy_release(release_root, tmp_path / "reordered-gallery")
    paths = resolve_release_paths(root)
    audit = json.loads(paths.aggregated_audit.read_text(encoding="utf-8"))
    audit["gallery_paths"][0], audit["gallery_paths"][1] = audit["gallery_paths"][1], audit["gallery_paths"][0]
    paths.aggregated_audit.write_text(json.dumps(audit), encoding="utf-8")
    data = load_release(root, verify_hashes=False, expect_released_counts=False)
    assert data.gallery.image_ids[:2] == (
        "mbeir_images/fashioniq_images/target.jpg",
        "mbeir_images/fashioniq_images/reference.jpg",
    )


def test_synthetic_fixture_marks_count_failures_and_skips_hashes(tmp_path: Path) -> None:
    root = tmp_path / "synthetic"
    paths = resolve_release_paths(root)
    for path in (paths.mbeir_query, paths.mbeir_candidates, paths.sf,
                 paths.validated_solved, paths.validated_unsolved):
        path.parent.mkdir(parents=True, exist_ok=True)
    paths.mbeir_query.write_text(json.dumps({
        "qid": "7:query", "query_txt": "synthetic edit", "query_img_path": "mbeir_images/reference.jpg",
        "pos_cand_list": ["7:2"],
    }) + "\n", encoding="utf-8")
    paths.sf.write_text("", encoding="utf-8")
    paths.validated_solved.write_text("", encoding="utf-8")
    paths.validated_unsolved.write_text("", encoding="utf-8")
    paths.mbeir_candidates.write_text("\n".join(json.dumps(row) for row in (
        {"did": "7:1", "img_path": "mbeir_images/reference.jpg"},
        {"did": "7:2", "img_path": "mbeir_images/target.jpg"},
    )) + "\n", encoding="utf-8")
    paths.aggregated_audit.parent.mkdir(parents=True, exist_ok=True)
    paths.aggregated_audit.write_text(json.dumps({"gallery_paths": [
        "<path_to_M-BEIR>/mbeir_images/reference.jpg",
        "<path_to_M-BEIR>/mbeir_images/target.jpg",
    ]}), encoding="utf-8")

    data = load_release(root, expect_released_counts=False)
    assert data.provenance["is_synthetic"] is True
    assert data.provenance["hash_verification"] == "skipped"
    assert data.provenance["failed_expectations"]
    with pytest.raises(ValueError, match="candidate count 2 != expected 74381"):
        load_release(root)


def test_namespaced_unresolved_id_names_candidate(release_root: Path, tmp_path: Path) -> None:
    root = _copy_release(release_root, tmp_path / "unresolved")
    paths = resolve_release_paths(root)
    rows = paths.mbeir_query.read_text(encoding="utf-8").splitlines()
    row = json.loads(rows[0])
    row["pos_cand_list"] = ["unknown:target"]
    rows[0] = json.dumps(row)
    paths.mbeir_query.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown:target"):
        load_release(root, verify_hashes=False)


def test_hash_mismatch_is_reported(release_root: Path, tmp_path: Path) -> None:
    root = _copy_release(release_root, tmp_path / "hash-mismatch")
    paths = resolve_release_paths(root)
    paths.sf.write_text(paths.sf.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    reports = verify_release_hashes(paths)
    assert len(reports) == 6
    assert all(report.matches is False for report in reports)
    loaded = load_release(root)
    assert loaded.provenance["hash_verification"] == "mismatch"
    assert all(report.matches is False for report in loaded.provenance["hash_reports"])


def test_flat_download_directory_paths_resolve(tmp_path: Path) -> None:
    root = tmp_path / "downloads"
    for name in (
        "mbeir_query.jsonl", "mbeir_cand_pool.jsonl", "mbeir_fashioniq_task7_test_multimodal.jsonl",
        "validated_solved.jsonl", "validated_unsolved.jsonl",
        "retrieval_data_fashioniq_task7_aggregated_best_rank_mm_any.json",
    ):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).touch()
    paths = resolve_release_paths(root)
    assert paths.mbeir_query == root / "mbeir_query.jsonl"
    assert paths.mbeir_candidates == root / "mbeir_cand_pool.jsonl"
    assert paths.sf == root / "mbeir_fashioniq_task7_test_multimodal.jsonl"
    assert paths.validated_solved == root / "validated_solved.jsonl"
    assert paths.aggregated_audit == root / "retrieval_data_fashioniq_task7_aggregated_best_rank_mm_any.json"

def test_nested_validated_paths_and_override(tmp_path: Path) -> None:
    root = tmp_path / "release"
    nested = root / "final_dataset/query_jsonl/fashioniq"
    for part in ("validated_solved", "validated_unsolved"):
        path = nested / part / f"mbeir_fashioniq_task7_test_{part}.jsonl"
        path.parent.mkdir(parents=True)
        path.touch()
    assert resolve_release_paths(root).validated_solved == (
        nested / "validated_solved/mbeir_fashioniq_task7_test_validated_solved.jsonl"
    )
    override = tmp_path / "validation-override"
    for part in ("validated_solved", "validated_unsolved"):
        path = override / part / f"mbeir_fashioniq_task7_test_{part}.jsonl"
        path.parent.mkdir(parents=True)
        path.touch()
    assert resolve_release_paths(root, validated_root=override).validated_solved == (
        override / "validated_solved/mbeir_fashioniq_task7_test_validated_solved.jsonl"
    )


def test_category_assignment_keeps_ambiguous_queries_unscoped(release_root: Path, tmp_path: Path) -> None:
    data = load_release(release_root, verify_hashes=False)
    root = tmp_path / "FashionIQ"
    (root / "captions").mkdir(parents=True)
    (root / "image_splits").mkdir()
    (root / "captions/cap.dress.test.json").write_text(
        json.dumps([{"candidate": "reference", "captions": ["synthetic"]}]), encoding="utf-8")
    (root / "captions/cap.shirt.test.json").write_text(
        json.dumps([{"candidate": "reference", "captions": ["synthetic"]}]), encoding="utf-8")
    (root / "captions/cap.toptee.test.json").write_text("[]", encoding="utf-8")
    for category in ("dress", "shirt", "toptee"):
        (root / f"image_splits/split.{category}.val.json").write_text("[]", encoding="utf-8")
    assigned = assign_garment_categories(data, root)
    resolution = assigned.provenance["category_resolution"]
    assert assigned.queries[0].category is None
    assert resolution["status"] == "ambiguous"
    assert resolution["resolved_count"] == 0
    assert resolution["ambiguous_count"] == 6003
    assert resolution["unresolved_count"] == 0
    assert resolution["ambiguous_qids"] == tuple(query.qid for query in data.queries)
    assert resolution["ambiguous_examples"][0] == {
        "qid": data.queries[0].qid, "categories": ("dress", "shirt")}
    assert sum(resolution[key] for key in ("resolved_count", "ambiguous_count", "unresolved_count")) == len(data.queries)
