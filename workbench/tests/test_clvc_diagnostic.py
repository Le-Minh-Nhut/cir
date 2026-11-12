"""CLVC-Net's dual-scope diagnostic: the descriptor contract, and the two defects it fixes.

The LIMN suite (`test_limn_diagnostic_record.py`, `test_limn_dual_scope_diagnostic.py`) pins LIMN's
behaviour. These tests pin what is CLVC-specific and what must hold for ANY model on the shared
validation body:

* a missing checkpoint digest is a NAMED defect, never a silent skip (the CLVC artifact has no
  `sha256` per scope; before this fix the reader compared nothing and reported `complete: true`);
* the canonical cohort may come from a run summary when the artifact does not record one, and its
  provenance travels with the payload;
* the third condition is `clvc_image_encoder_only` -- a named ablation -- and `image_only_empty_text`
  is not silently substituted for it;
* rank-1 of that ablation is the query's own reference image, so it is labelled a self-match and
  never counted as a retrieval hit;
* the target is outside the stored window in every shipped cell, and the payload says so instead of
  implying a position inside it.

Run: workbench/envs/ctl312h/bin/python -m pytest workbench/tests/test_clvc_diagnostic.py -q
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend import limn_diagnostic
from workbench.backend.limn_diagnostic import (
    CLVC,
    DIAGNOSTIC_COMPLETE,
    DIAGNOSTIC_EVIDENCE_DEFECT,
    DIAGNOSTIC_NOT_PUBLISHED,
    LIMN,
    MODELS,
    DiagnosticEvidenceError,
    load_candidates,
    load_diagnostic,
    model_by_id,
)

REPO = Path(__file__).resolve().parents[2]
SHIPPED = REPO / "workbench/docs/reviews/20261112-parity-repair/raw/clvc_diag"
LIMN_SHIPPED = REPO / "workbench/docs/reviews/20261109-native15-composition-limn/limn_dual_scope.json"
CONDITIONS = ("multimodal", "text_only_black_image", "clvc_image_encoder_only")
SCOPES = ("shirt", "toptee")
QIDS = ("7:3038", "7:3790")


def _config(root: Path) -> SimpleNamespace:
    return SimpleNamespace(CIR_REPO_ROOT=str(root),
                           FASHIONIQ_ROOT=str(REPO / "data/FashionIQ"))


@pytest.fixture()
def mirrored(tmp_path: Path):
    """A temp repo root holding a copy of the shipped CLVC evidence, plus the real run summary.

    `workbench/artifacts` is symlinked rather than copied so the authoritative cohort is read from
    the bytes the repo actually ships -- the point of the fixture is to vary the ARTIFACT, not to
    fake its cohort source.
    """
    (tmp_path / "workbench").mkdir(parents=True, exist_ok=True)
    (tmp_path / "workbench/artifacts").symlink_to(REPO / "workbench/artifacts")
    target = tmp_path.joinpath(*CLVC.evidence_relative)
    target.mkdir(parents=True)
    for path in SHIPPED.glob("*"):
        shutil.copy(path, target / path.name)
    return tmp_path, target


def _rewrite(target: Path, mutate) -> None:
    path = target / "clvc_dual_scope.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _clear() -> None:
    limn_diagnostic._RECORD_CACHE.clear()


def test_the_descriptor_table_names_both_models() -> None:
    assert sorted(MODELS) == ["clvc", "limn"]
    assert model_by_id("clvc") is CLVC
    assert model_by_id(None) is LIMN, "the pre-descriptor default must stay LIMN"
    assert model_by_id("limn") is LIMN


def test_an_unknown_model_is_refused_not_guessed() -> None:
    with pytest.raises(DiagnosticEvidenceError, match="unknown diagnostic model"):
        model_by_id("dcnet")


def test_clvc_third_condition_is_the_named_ablation() -> None:
    assert CLVC.conditions[2] == "clvc_image_encoder_only"
    assert CLVC.condition_kinds["clvc_image_encoder_only"] == "ablation"
    assert CLVC.condition_kinds["multimodal"] == "protocol"
    # The protocol condition CLVC cannot encode is NOT in the tuple: substituting it would add a cell
    # the model has no numbers for.
    assert "image_only_empty_text" not in CLVC.conditions


@pytest.mark.skipif(not SHIPPED.is_dir(), reason="CLVC dual-scope evidence not shipped on this host")
def test_shipped_clvc_evidence_is_complete_with_no_defects() -> None:
    _clear()
    result = load_diagnostic(model="clvc")
    assert result["available"] is True
    assert result["status"] == DIAGNOSTIC_COMPLETE, result["defects"]
    assert result["complete"] is True
    assert result["defects"] == []
    assert result["model"] == "clvc"
    assert result["diagnostic"] is True and result["canonical"] is False
    assert list(result["conditions"]) == list(CONDITIONS)
    assert set(result["queries"]) == set(QIDS)
    for qid in QIDS:
        assert list(result["queries"][qid]["scopes"]) == list(SCOPES)
        for scope in SCOPES:
            assert set(result["queries"][qid]["scopes"][scope]) == set(CONDITIONS)


@pytest.mark.skipif(not SHIPPED.is_dir(), reason="CLVC dual-scope evidence not shipped on this host")
def test_shipped_clvc_cohort_comes_from_the_run_summary_with_provenance() -> None:
    _clear()
    result = load_diagnostic(model="clvc")
    cohort = result["canonical_cohort"]
    assert cohort["query_count"] == 6003
    assert cohort["scored_count"] == 6001
    assert cohort["unscored_count"] == 2
    assert cohort["query_count_per_subset"] == {"full": 6003, "sf": 4069, "validated": 586}
    assert cohort["subset"] == "full"
    provenance = result["canonical_cohort_provenance"]
    assert provenance["source"] == "run_summary"
    assert provenance["run_id"] == "da0a2162be0e159a1a456cec"
    assert len(provenance["sha256"]) == 64
    # The counts are never invented: they equal the artifact's own canonical_status.
    assert result["canonical_status"]["scored_count"] == cohort["scored_count"]
    assert sorted(result["canonical_status"]["unscored_qids"]) == sorted(QIDS)


@pytest.mark.skipif(not SHIPPED.is_dir(), reason="CLVC dual-scope evidence not shipped on this host")
def test_a_missing_summary_digest_is_a_named_defect_not_a_silent_skip(mirrored) -> None:
    """The CLVC artifact stores no per-scope `sha256`; the digest the comparison needs is
    `local_sha256_recomputed`. Removing it must NOT report `complete: true`."""
    root, target = mirrored

    def drop(payload: dict) -> None:
        for scope in SCOPES:
            payload["checkpoints"][scope].pop("local_sha256_recomputed", None)

    _rewrite(target, drop)
    _clear()
    result = load_diagnostic(config=_config(root), model="clvc")
    assert result["status"] == DIAGNOSTIC_EVIDENCE_DEFECT
    assert result["complete"] is False
    codes = {defect["code"] for defect in result["defects"]}
    assert codes == {"checkpoint_summary_digest_missing"}
    # One per cell: the gap is per combination, not a single global note.
    assert len(result["defects"]) == 12
    sample = result["defects"][0]
    assert "local_sha256_recomputed" in sample["detail"]
    assert "unverified" in sample["detail"]
    # The evidence is still served -- a defect names a gap, it does not delete the measurement.
    window = load_candidates("7:3038", "shirt", "multimodal", depth=100,
                             config=_config(root), model="clvc")
    assert window["depth_served"] == 100
    assert window["complete"] is False


@pytest.mark.skipif(not SHIPPED.is_dir(), reason="CLVC dual-scope evidence not shipped on this host")
def test_a_mismatched_summary_digest_is_still_a_defect(mirrored) -> None:
    root, target = mirrored
    _rewrite(target, lambda p: p["checkpoints"]["shirt"].__setitem__(
        "local_sha256_recomputed", "b" * 64))
    _clear()
    result = load_diagnostic(config=_config(root), model="clvc")
    assert result["complete"] is False
    codes = {defect["code"] for defect in result["defects"]}
    assert codes == {"checkpoint_summary_vs_export"}
    assert {defect["scope"] for defect in result["defects"]} == {"shirt"}


@pytest.mark.skipif(not SHIPPED.is_dir(), reason="CLVC dual-scope evidence not shipped on this host")
def test_missing_cohort_and_missing_run_summary_is_refused(mirrored) -> None:
    """Neither source for the cohort means no cohort; the payload must not be served without one.

    The artifact is present and healthy here -- only the two cohort sources are gone -- so the refusal
    is specifically about the missing population, not about missing evidence.
    """
    root, target = mirrored
    _rewrite(target, lambda p: p.pop("canonical_cohort", None))
    # Replace the symlinked artifacts tree with an empty one, so the run summary is absent while the
    # evidence directory still resolves.
    (root / "workbench/artifacts").unlink()
    (root / "workbench/artifacts").mkdir()
    _clear()
    with pytest.raises(DiagnosticEvidenceError, match="authoritative run summary is missing"):
        load_diagnostic(config=_config(root), model="clvc")


@pytest.mark.skipif(not SHIPPED.is_dir(), reason="CLVC dual-scope evidence not shipped on this host")
def test_rank_one_of_the_ablation_is_labelled_a_reference_self_match() -> None:
    """All four ablation cells rank the reference image 1st at score ~1.0. That is a self-match."""
    _clear()
    for qid in QIDS:
        for scope in SCOPES:
            window = load_candidates(qid, scope, "clvc_image_encoder_only", depth=100,
                                     model="clvc")
            first = window["candidates"][0]
            assert first["rank"] == 1
            assert first["is_reference"] is True, (qid, scope)
            assert first["match_kind"] == "reference_self_match"
            assert first["score"] == pytest.approx(1.0, abs=1e-6)
            assert first["is_target"] is False
            assert window["rank_1_is_reference_self_match"] is True
            # The reference is named, so a reader can check the claim against the release record.
            assert window["reference_asset"] == first["gallery_id"].split("/")[-1]


@pytest.mark.skipif(not SHIPPED.is_dir(), reason="CLVC dual-scope evidence not shipped on this host")
def test_protocol_conditions_do_not_claim_a_reference_self_match_at_rank_one() -> None:
    _clear()
    for qid in QIDS:
        for scope in SCOPES:
            window = load_candidates(qid, scope, "multimodal", depth=100, model="clvc")
            assert window["rank_1_is_reference_self_match"] is (
                window["candidates"][0]["image_stem"] == window["reference_asset"])


@pytest.mark.skipif(not SHIPPED.is_dir(), reason="CLVC dual-scope evidence not shipped on this host")
def test_the_target_is_reported_outside_the_stored_window_in_all_twelve_cells() -> None:
    _clear()
    observed = []
    for qid in QIDS:
        for scope in SCOPES:
            for condition in CONDITIONS:
                window = load_candidates(qid, scope, condition, depth=100, model="clvc")
                assert window["depth_stored"] == 100
                assert window["target"]["in_stored_window"] is False
                assert window["target"]["in_window"] is False
                assert window["target_within_export_depth"] is False
                assert window["target_outside_stored_window"] is True
                assert window["target_rank"] > 100, "a rank inside the window contradicts the export"
                assert sum(1 for c in window["candidates"] if c["is_target"]) == 0
                observed.append(window["target_rank"])
    # The 12 exact ranks the inventory records, in (query, scope, condition) order.
    assert observed == [848, 304, 8702, 1253, 1549, 1267,
                        412, 1466, 5481, 259, 568, 4535]


@pytest.mark.skipif(not SHIPPED.is_dir(), reason="CLVC dual-scope evidence not shipped on this host")
def test_the_ablation_reports_that_no_text_was_encoded() -> None:
    _clear()
    window = load_candidates("7:3038", "shirt", "clvc_image_encoder_only", model="clvc")
    assert window["model_text"] == ""
    assert window["effectively_encoded"]["text_is_empty"] is True
    assert "transports no text" in window["effectively_encoded"]["note"]
    # The released caption is still reported beside it: the empty string is what the model saw, not
    # what the query was.
    assert window["raw_edit_caption"] == "Is darker and has a different graphic and is darker."
    protocol = load_candidates("7:3038", "shirt", "multimodal", model="clvc")
    assert protocol["effectively_encoded"]["text_is_empty"] is False
    assert protocol["model_text"] == protocol["raw_edit_caption"]


@pytest.mark.skipif(not SHIPPED.is_dir(), reason="CLVC dual-scope evidence not shipped on this host")
def test_every_shipped_candidate_resolves_to_a_served_image() -> None:
    """1200/1200 through the real resolver. A path convention resolves only 279, so this pins the
    resolver, not a filename pattern."""
    _clear()
    missing = []
    total = 0
    for qid in QIDS:
        for scope in SCOPES:
            for condition in CONDITIONS:
                window = load_candidates(qid, scope, condition, depth=100, model="clvc")
                for candidate in window["candidates"]:
                    total += 1
                    if candidate["image_url"] is None:
                        missing.append((qid, scope, condition, candidate["rank"]))
    assert total == 1200
    assert missing == []


@pytest.mark.skipif(not SHIPPED.is_dir(), reason="CLVC dual-scope evidence not shipped on this host")
def test_image_only_empty_text_is_absent_from_clvc_conditions_and_refused() -> None:
    _clear()
    result = load_diagnostic(model="clvc")
    assert "image_only_empty_text" not in result["conditions"]
    with pytest.raises(DiagnosticEvidenceError, match="not a supported condition"):
        load_candidates("7:3038", "shirt", "image_only_empty_text", model="clvc")


@pytest.mark.skipif(not SHIPPED.is_dir(), reason="CLVC dual-scope evidence not shipped on this host")
def test_the_summary_and_the_window_describe_the_same_clvc_combination() -> None:
    _clear()
    summary = load_diagnostic(model="clvc")
    for qid in QIDS:
        for scope in SCOPES:
            for condition in CONDITIONS:
                reading = summary["queries"][qid]["scopes"][scope][condition]
                window = load_candidates(qid, scope, condition, depth=100, model="clvc")
                assert window["target_rank"] == reading["target_rank"]
                assert window["target_score"] == reading["target_score"]
                assert window["condition_kind"] == reading["condition_kind"]
                assert window["identity"]["manifest_digest"] == summary["manifest"]["digest"]


@pytest.mark.skipif(not LIMN_SHIPPED.is_file(), reason="LIMN evidence not on this host")
def test_the_limn_payload_keeps_its_exact_shape() -> None:
    """The descriptor must not have added a key to LIMN's payload, nor reordered a list."""
    _clear()
    result = load_diagnostic()
    assert result["conditions"] == ["multimodal", "text_only_black_image", "image_only_empty_text"]
    assert result["scopes"] == ["shirt", "toptee"]
    assert result["label"] == "ambiguous category — diagnostic comparison"
    # The additive CLVC-only keys must be absent, so LIMN's bytes cannot drift.
    for key in ("model", "model_name", "diagnostic", "canonical", "condition_kinds",
                "condition_labels", "canonical_cohort_provenance", "reference_match_note", "models"):
        assert key not in result, key
    window = load_candidates("7:3038", "shirt", "multimodal", depth=100)
    for key in ("model", "diagnostic", "is_reference", "effectively_encoded",
                "rank_1_is_reference_self_match", "target_outside_stored_window"):
        assert key not in window
    assert "is_reference" not in window["candidates"][0]


@pytest.mark.skipif(not SHIPPED.is_dir(), reason="CLVC dual-scope evidence not shipped on this host")
def test_clvc_unpublished_evidence_is_a_named_absence(tmp_path: Path) -> None:
    _clear()
    empty = SimpleNamespace(CIR_REPO_ROOT=str(tmp_path),
                            FASHIONIQ_ROOT=str(REPO / "data/FashionIQ"))
    result = load_diagnostic(config=empty, model="clvc")
    assert result["available"] is False
    assert result["status"] == DIAGNOSTIC_NOT_PUBLISHED
    assert "clvc_dual_scope.json" in result["reason"]


@pytest.mark.skipif(not SHIPPED.is_dir(), reason="CLVC dual-scope evidence not shipped on this host")
def test_the_wrong_artifact_type_is_refused_for_the_right_model(tmp_path: Path) -> None:
    """Pointing CLVC's descriptor at LIMN's file must be refused, not misread."""
    _clear()
    if not LIMN_SHIPPED.is_file():
        pytest.skip("LIMN evidence not on this host")
    target = tmp_path.joinpath(*CLVC.evidence_relative)
    target.mkdir(parents=True)
    shutil.copy(LIMN_SHIPPED, target / "clvc_dual_scope.json")
    for scope in SCOPES:
        for condition in CONDITIONS:
            (target / f"clvc_dual_scope.{scope}.{condition}.jsonl").write_text("{}\n",
                                                                             encoding="utf-8")
    with pytest.raises(DiagnosticEvidenceError, match="unexpected artifact type"):
        load_diagnostic(config=_config(tmp_path), model="clvc")
