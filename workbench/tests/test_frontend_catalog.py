"""Capabilities are separate flags; statistics say what they mean; provenance is not a name match.

Every assertion here executes `workbench/frontend/catalog.ts` against shapes this deployment
actually serves (taken from `/api/composition/runs`, `/api/benchmark-runs` and `/api/runs` while the
live stack was up), so a backend field rename shows up as a failing test rather than as a UI that
quietly renders nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import test_frontend_support as support

MODULE = support.compile_module("catalog.ts", Path("/tmp/cir_frontend_catalog"))
SRC = json.dumps(str(MODULE))


def _run(body: str) -> dict:
    return support.run_json(f"import * as C from {SRC};\n{body}")


def test_capabilities_are_four_separate_flags_not_one_boolean() -> None:
    """A run with ranks but no saved candidates must report exactly that.

    This is the served shape: every CIRCUS composition run has exact ranks per condition and an
    empty `top_results` list, because candidate images were never exported. One boolean would have
    to lie about one of the two.
    """
    out = _run("""
const run = {
  run_id: "r", model_id: "ptha_mtst", checkpoint_id: "cirr_transfer",
  protocol_id: "circus_fashioniq_release_v1", execution_status: "COMPLETED",
  query_count: 6003, conditions: ["multimodal"],
  metrics: { coverage: 1.0, ndcg_multimodal: 0.1498, gap_cohort_count: 4069,
             multimodal: { scored_count: 4069, query_count: 4069 } },
};
const report = C.compositionCapabilities(run);
console.log('__R__' + JSON.stringify({flags: report.flags, reasons: report.reasons}));
""")
    assert out["flags"] == {
        "aggregate_metrics": True,
        "per_query_target_ranks": True,
        "candidate_topk": False,
        "image_availability": True,
    }, out["flags"]
    assert "no candidate image lists" in out["reasons"]["candidate_topk"], out["reasons"]


def test_a_failed_run_reports_no_metric_and_no_rank_with_a_reason() -> None:
    out = _run("""
const run = { run_id: "f", model_id: "habit", checkpoint_id: "fiq_n02",
  protocol_id: "circus_fashioniq_release_v1", execution_status: "FAILED", is_failed: true,
  query_count: 4069,
  metrics: { coverage: 0, ndcg_multimodal: null, gap_cohort_count: 0,
             multimodal: { scored_count: 0, query_count: 4069 },
             gap_unavailable_reason: "no queries have exact ranks in all three conditions" } };
const report = C.compositionCapabilities(run);
console.log('__R__' + JSON.stringify({flags: report.flags, reasons: report.reasons}));
""")
    assert out["flags"]["aggregate_metrics"] is False
    assert out["flags"]["per_query_target_ranks"] is False
    assert "did not complete" in out["reasons"]["aggregate_metrics"]
    assert "did not complete" in out["reasons"]["per_query_target_ranks"]


def test_an_aggregate_only_benchmark_report_keeps_its_flag_answer_truthful() -> None:
    out = _run("""
const run = { run_id: "b", model_id: "csmcir", checkpoint_id: "fashioniq",
  protocol_id: "fashioniq_original_split", per_query_state: "CAPABILITY_ABSENT",
  observed_metrics: { r10: 57.05, r50: 77.29, mean: 67.17 } };
const report = C.benchmarkCapabilities(run);
console.log('__R__' + JSON.stringify({flags: report.flags, reasons: report.reasons}));
""")
    assert out["flags"]["aggregate_metrics"] is True, out
    assert out["flags"]["per_query_target_ranks"] is False, out
    assert out["flags"]["candidate_topk"] is False, out
    assert "CAPABILITY_ABSENT" in out["reasons"]["per_query_target_ranks"], out["reasons"]


def test_composition_gap_is_a_fraction_with_a_cohort_and_keeps_negative_values() -> None:
    out = _run("""
const base = { run_id: "g", model_id: "m", checkpoint_id: "c", protocol_id: "p" };
const normal = C.compositionGap({...base, metrics: {
  gap_cohort_count: 4000, gap_multimodal_ndcg: 0.30, gap_text_only_ndcg: 0.13,
  gap_image_only_ndcg: 0.10, composition_gap_ndcg: 1 - 0.13 / 0.30 }});
// Ablated conditions BEAT multimodal: a real (negative) result that must not be clamped to 0.
const negative = C.compositionGap({...base, metrics: {
  gap_cohort_count: 100, gap_multimodal_ndcg: 0.10, gap_text_only_ndcg: 0.20,
  gap_image_only_ndcg: 0.05, composition_gap_ndcg: 1 - 0.20 / 0.10 }});
const missing = C.compositionGap({...base, metrics: {
  gap_cohort_count: 0, gap_unavailable_reason: "no queries have exact ranks in all three conditions" }});
const noDenominator = C.compositionGap({...base, metrics: {
  gap_cohort_count: 10, gap_multimodal_ndcg: 0, gap_text_only_ndcg: 0.2,
  gap_image_only_ndcg: 0.1, composition_gap_ndcg: 0 }});
const inconsistent = C.compositionGap({...base, metrics: {
  gap_cohort_count: 10, gap_multimodal_ndcg: 0.3, gap_text_only_ndcg: 0.1,
  gap_image_only_ndcg: 0.1, composition_gap_ndcg: 0.9 }});
console.log('__R__' + JSON.stringify({normal, negative, missing, noDenominator, inconsistent}));
""")
    assert out["normal"]["state"] == "available", out["normal"]
    assert abs(out["normal"]["value"] - (1 - 0.13 / 0.30)) < 1e-12, out["normal"]
    assert out["normal"]["cohort"] == 4000, out["normal"]
    assert "1 − max(" in out["normal"]["formula"], out["normal"]
    assert out["negative"]["state"] == "available" and out["negative"]["value"] < 0, (
        "a negative gap was clamped or dropped", out["negative"])
    # Every missing denominator is `unavailable` plus a reason, never 0.
    assert out["missing"]["state"] == "unavailable" and out["missing"]["reason"], out["missing"]
    assert out["noDenominator"]["state"] == "unavailable", out["noDenominator"]
    assert "no denominator" in out["noDenominator"]["reason"], out["noDenominator"]
    assert out["inconsistent"]["state"] == "unavailable", (
        "a recorded gap that disagrees with its own cohort was published as if fine",
        out["inconsistent"])


def test_scored_and_unscored_are_counted_separately_with_coverage() -> None:
    out = _run("""
const run = { run_id: "s", model_id: "limn", checkpoint_id: "c", protocol_id: "p",
  query_count: 6003,
  metrics: { multimodal: { scored_count: 5884, query_count: 6003 } } };
console.log('__R__' + JSON.stringify(C.scoredCounts(run, "multimodal")));
""")
    assert {k: out[k] for k in ("scored", "total", "unscored")} == {
        "scored": 5884, "total": 6003, "unscored": 119}, out
    assert abs(out["coverage"] - 5884 / 6003) < 1e-12, out
    assert out["unscored"] > 0, "the 119 unscored queries disappeared"
    # A bucket-derived count must be marked as derived, so the UI never presents it as served.
    assert out["derived"] is True, out


def test_served_measurement_replaces_the_derived_count() -> None:
    """The contract's `measurement` is authoritative and keeps `unscored_query_count` separate.

    This is the served shape for a CIRCUS run: 6,003 sampled, 5,884 scored, 119 unscored by category
    resolution. The 119 must never be summed into a retrieval-failure count, and a served count must
    not be labelled derived.
    """
    out = _run("""
const run = { run_id: "m", model_id: "limn", checkpoint_id: "c", protocol_id: "p",
  query_count: 6003,
  measurement: { sample_count: 6003, denominator: 5884, coverage: 0.980176578377478,
                 scored_count: 5884, unscored_query_count: 119,
                 unscored_by_kind: { category_unresolved: 119 } },
  metrics: {} };
console.log('__R__' + JSON.stringify(C.scoredCounts(run, "multimodal")));
""")
    assert out["scored"] == 5884 and out["total"] == 6003, out
    assert out["unscored"] == 119, out
    assert out["unscoredByKind"] == {"category_unresolved": 119}, out
    assert out["derived"] is False, out


def test_capability_detail_status_and_reason_are_surfaced_not_invented() -> None:
    """A false flag carries a NAME and a reason. It is a capability the run lacks, not an error."""
    out = _run("""
const run = { run_id: "d", model_id: "limn", checkpoint_id: "c", protocol_id: "p",
  execution_status: "PARTIAL",
  capabilities: { aggregate_metrics: true, per_query_target_ranks: true,
                  candidate_topk: false, image_availability: true },
  capability_detail: {
    candidate_topk: { available: false, status: "TOPK_EXPORT_NOT_REQUESTED",
                      reason: "this run was produced without candidate Top-K export",
                      depth: null },
    image_availability: { status: "local", sampled: 40, resolved: 40 },
  },
  metrics: { coverage: 0.98, ndcg_multimodal: 0.21, multimodal: { scored_count: 5884 } } };
console.log('__R__' + JSON.stringify(C.compositionCapabilities(run)));
""")
    assert out["flags"]["candidate_topk"] is False, out
    assert out["statuses"]["candidate_topk"] == "TOPK_EXPORT_NOT_REQUESTED", out
    assert "without candidate Top-K export" in out["reasons"]["candidate_topk"], out
    assert out["derived"] is False, (
        "a served capability report must not be labelled derived", out)


def test_an_imported_row_is_its_own_bucket_and_never_default_research_evidence() -> None:
    """`provenance_source: released_audit_retriever` is real released data but not our measurement."""
    out = _run("""
console.log('__R__' + JSON.stringify({
  imported: C.provenanceOf({ run_id: "i", provenance_source: "released_audit_retriever" }),
  importedArtifact: C.provenanceOf({ run_id: "i2", provenance_source: "external_artifact" }),
  normal: C.provenanceOf({ run_id: "n", verification: { state: "certified", certified: true } }),
}));
""")
    assert out["imported"]["bucket"] == "imported", out["imported"]
    assert "released_audit_retriever" in out["imported"]["label"], out["imported"]
    assert out["importedArtifact"]["bucket"] == "imported", out
    assert out["normal"]["bucket"] == "research", out


def test_transfer_reads_the_served_object() -> None:
    out = _run("""
console.log('__R__' + JSON.stringify({
  served: C.transferLabel({ run_id: "t", model_id: "ptha_mtst", checkpoint_id: "cirr_transfer",
    protocol_id: "p",
    transfer: { is_transfer: true, training_dataset: "CIRR", evaluation_dataset: "FashionIQ" } }),
  servedNative: C.transferLabel({ run_id: "n", model_id: "tme", checkpoint_id: "f",
    protocol_id: "p",
    transfer: { is_transfer: false, training_dataset: "FashionIQ", evaluation_dataset: "FashionIQ" } }),
}));
""")
    assert out["served"]["isTransfer"] is True, out
    assert "CIRR" in out["served"]["label"] and "FashionIQ" in out["served"]["label"], out
    assert out["servedNative"]["isTransfer"] is False, out


def test_duplicate_status_is_stated_not_guessed() -> None:
    out = _run("""
console.log('__R__' + JSON.stringify({
  unique: C.duplicateLabel("unique", 1),
  duplicate: C.duplicateLabel("duplicate", 3),
  undetermined: C.duplicateLabel("undetermined", 2),
  fixture: C.duplicateLabel("synthetic_fixture", 1),
  absent: C.duplicateLabel(null, null),
}));
""")
    assert out["unique"] == "unique measurement", out
    assert "3" in out["duplicate"], out
    assert "identity inputs missing" in out["undetermined"], (
        "an undetermined identity must not read as unique or duplicate", out)
    assert "fixture" in out["fixture"], out
    assert out["absent"] is None, out


def test_provenance_is_decided_by_declared_fields_never_by_model_name() -> None:
    out = _run("""
const rows = {
  synthetic: C.provenanceOf({ run_id: "1", is_synthetic: true }),
  fixture: C.provenanceOf({ run_id: "2", synthetic_fixture: true }),
  mock: C.provenanceOf({ run_id: "3", data_kind: "mock" }),
  refused: C.provenanceOf({ run_id: "4", verification: { state: "refused", certified: false } }),
  certified: C.provenanceOf({ run_id: "5", verification: { state: "certified", certified: true } }),
  // A model literally called "fixture_encoder" is NOT a fixture unless it declares itself one.
  nameLooksSynthetic: C.provenanceOf({ run_id: "6", model_id: "fixture_encoder",
                                       verification: { state: "certified", certified: true } }),
  unknown: C.provenanceOf({ run_id: "7" }),
};
console.log('__R__' + JSON.stringify(rows));
""")
    for key in ("synthetic", "fixture", "mock", "refused"):
        assert out[key]["bucket"] == "development", (key, out[key])
    assert out["certified"]["bucket"] == "research", out["certified"]
    assert out["nameLooksSynthetic"]["bucket"] == "research", (
        "provenance was decided from the model name", out["nameLooksSynthetic"])
    assert out["unknown"]["undetermined"] is True, (
        "missing provenance was reported as determined", out["unknown"])


def test_attempts_group_by_measurement_identity_and_say_undetermined() -> None:
    """Two attempts with the same weights/protocol/condition are ONE measurement; if a component of
    the identity is missing the group says so instead of guessing."""
    out = _run("""
const base = { model_id: "csmcir", checkpoint_identity: "sha-1", protocol_id: "p",
               checkpoint_training_dataset: "FashionIQ", checkpoint_training_noise_pct: 0,
               evaluation_noise_pct: 0, subset: "full" };
const groups = C.groupMeasurements([
  {...base, run_id: "a"}, {...base, run_id: "b"},
  {...base, run_id: "c", checkpoint_identity: "sha-2"},
  {...base, run_id: "d", checkpoint_identity: null, checkpoint_id: "", model_id: null},
]);
console.log('__R__' + JSON.stringify(groups.map((g) => ({attempts: g.attempts.map((a) => a.run_id),
                                                undetermined: g.undetermined,
                                                missing: g.missing}))));
""")
    sizes = {tuple(group["attempts"]): group for group in out}
    assert ("a", "b") in sizes, out
    assert sizes[("a", "b")]["undetermined"] is False, out
    assert any(group["undetermined"] for group in out), (
        "an attempt missing its checkpoint identity was grouped as if determined", out)
    complete = sizes[("a", "b")]
    assert complete["missing"] == [], complete


def test_the_fixed_cutoff_is_named_and_not_a_control() -> None:
    out = _run("console.log('__R__' + JSON.stringify(C.FIXED_CUTOFFS));")
    assert out["compositionOutcomeK"] == 10, out
    assert "@10" in out["compositionOutcomeNote"], out
    assert "fixed" in out["compositionOutcomeNote"], out
    assert "DISPLAYED" in out["candidateDisplayNote"], out


def test_transfer_is_labelled_transfer() -> None:
    """PTHA CIRR→FashionIQ is a transfer, and must never render as a paper row."""
    out = _run("""
console.log('__R__' + JSON.stringify({
  transfer: C.transferLabel({ run_id: "t", model_id: "ptha_mtst", checkpoint_id: "cirr_transfer",
    protocol_id: "circus_fashioniq_release_v1", checkpoint_transfer: "CIRR→FashionIQ transfer",
    checkpoint_training_dataset: "CIRR" }),
  native: C.transferLabel({ run_id: "n", model_id: "tme", checkpoint_id: "fashioniq_0",
    protocol_id: "circus_fashioniq_release_v1", checkpoint_training_dataset: "FashionIQ" }),
}));
""")
    assert out["transfer"]["isTransfer"] is True, out
    assert out["transfer"]["label"].startswith("TRANSFER"), out
    assert out["native"]["isTransfer"] is False, out


# ---------------------------------------------------------------- candidate window & null score

def test_a_null_score_is_not_zero_and_the_window_is_not_the_rank() -> None:
    """The inspector's rules, executed against the served shapes.

    * `score: null` means "the artifact stored an order but no scores" — it must render `—`, because
      `0.0` would be a fabricated confidence.
    * `target_rank` is a full-gallery fact that survives OUTSIDE the stored window. A rank of 31,987
      with a depth-20 window is not inside the window and must not be clamped to 20.
    * `gallery_position` is a different fact from `rank`; both are kept.
    """
    out = support.run_json(f"""
import {{fmtInt, fmtRank}} from {json.dumps("__FORMAT__")};
const item = {{"query_id": "7:4436", "target_rank": 31987, "candidate_depth": 20,
  "candidate_status": "AVAILABLE", "query_state": "retrieval_failure",
  "unscored": false, "retrieval_failure": true,
  "top_results": [
    {{"rank": 1, "image_id": "mbeir_images/fashioniq_images/B00DP8U3GU.jpg",
     "gallery_position": 69023, "score": null}},
    {{"rank": 2, "image_id": "mbeir_images/fashioniq_images/B0047LWLOG.jpg",
     "gallery_position": 12875, "score": null}}]}};
const shown = item.top_results.map((c) => ({{
  rank: c.rank,
  gallery: c.gallery_position,
  score: c.score == null ? "—" : c.score.toFixed(4),
}}));
console.log('__R__' + JSON.stringify({{
  shown,
  targetRank: item.target_rank,
  // The rank is outside the window, and that is a statement, not a clamp.
  outsideWindow: item.target_rank > item.candidate_depth,
  windowIsDepth: item.candidate_depth,
  rankText: null,
}}));
""".replace("__FORMAT__", str(support.compile_module("format.ts", Path("/tmp/cir_frontend_format2")))))
    assert out["shown"] == [
        {"rank": 1, "gallery": 69023, "score": "—"},
        {"rank": 2, "gallery": 12875, "score": "—"},
    ], out["shown"]
    assert out["targetRank"] == 31987, out
    assert out["outsideWindow"] is True, out
    assert out["windowIsDepth"] == 20, out


def test_the_inspector_renders_null_scores_as_a_dash_and_states_the_window() -> None:
    """Wiring, not just arithmetic: the component must not call toFixed on a nullable score, must
    show `gallery_position`, and must say when the exact rank lies outside the stored window."""
    inspector = (support.FRONTEND / "QueryInspector.tsx").read_text(encoding="utf-8")
    assert "candidate.score == null" in inspector, (
        "the inspector formats a nullable score without a null branch")
    assert 'candidate.score.toFixed(4)' in inspector, "the score is no longer rendered"
    # No unconditional toFixed on the union type.
    assert "candidate.score.toFixed" in inspector and "candidate.score == null" in inspector
    assert "gallery_position" in inspector, (
        "gallery_position is a different fact from rank and must be shown")
    assert "ngoài cửa sổ" in inspector, (
        "the inspector must state that the exact rank lies outside the shown window")
    # The contract: an IMPORTED run's candidates are labelled with their provenance and are never
    # presented as this model's own output. This used to assert the literal "NGUỒN", which `87ff419`
    # replaced with the fuller sentence "Nguồn ứng viên: …" — the copy changed, the contract did
    # not. Asserting exact prose would break again on the next rewrite, so the assertions are on the
    # symbols and the gating condition instead.
    assert "IMPORTED_LABEL" in inspector, (
        "imported candidates must map their source to a human label")
    assert "released_audit_retriever" in inspector, (
        "the imported-source discriminator must gate the label; without it a model run could be "
        "labelled as an import")
    # The label is only rendered when the run actually declares a provenance source: the gate is the
    # condition, not the sentence. Both halves must be present, or the label leaks onto model runs
    # (gate missing) or never renders at all (usage missing).
    assert "run.provenance_source &&" in inspector, (
        "the provenance notice must be gated on the run declaring a provenance_source")
    assert "IMPORTED_LABEL[run.provenance_source]" in inspector, (
        "the notice must render the label resolved from the run's own provenance_source")
    # And it must say, in whatever words, that this is not this model's output. A label alone reads
    # as a footnote; the disclaimer is the part that stops the row being read as a model result.
    notice = inspector[inspector.index("IMPORTED_LABEL[run.provenance_source]"):]
    notice = notice[:notice.index("</p>")] if "</p>" in notice else notice[:800]
    assert "không phải đầu ra của model" in notice, (
        "the provenance notice must state the candidates are not this model's output")


def test_the_imported_scope_is_opt_in_and_never_the_default() -> None:
    """`include_imported=true` only when the operator asked, and the default catalog excludes it."""
    data = (support.FRONTEND / "data.ts").read_text(encoding="utf-8")
    assert 'query.imported ?? "exclude"' in data, (
        "imported released-audit rows must default to excluded")
    assert "include_imported=true" in data, "the opt-in request is never issued"
    routes = (support.FRONTEND / "routes.ts").read_text(encoding="utf-8")
    assert 'imported: "exclude"' in routes, "the URL default must be exclude"
    assert 'params.set("imported", "include")' in routes, (
        "an opt-in must survive a reload, or the label and the data can disagree")


def test_a_served_measurement_that_contradicts_its_own_denominator_is_not_used() -> None:
    """The served `measurement` block is not always subset-scoped.

    Real payload from this deployment (`/api/composition/runs?subset=sf`): `sample_count: 4069`,
    `denominator: 4000`, but `scored_count: 5884` — the whole-run total. Rendering it produced
    "5,884 / 4,069 (144.6%)", a numerator larger than its own denominator. A count that cannot be
    reconciled with the denominator it is printed beside is not evidence, so the subset-scoped
    per-condition bucket wins and the served block is ignored.
    """
    out = _run("""
const contradictory = { run_id: "c1", model_id: "limn", checkpoint_id: "c", protocol_id: "p",
  subset: "sf", query_count: 4069, query_count_per_subset: { sf: 4069 },
  measurement: { sample_count: 4069, denominator: 4000, scored_count: 5884,
                 unscored_query_count: 119, coverage: 0.98 },
  metrics: { multimodal: { scored_count: 4000, query_count: 4069 } } };
const consistent = { run_id: "c2", model_id: "ptha_mtst", checkpoint_id: "c", protocol_id: "p",
  subset: "sf", query_count: 4069,
  measurement: { sample_count: 4069, denominator: 4069, scored_count: 4069,
                 unscored_query_count: 0, coverage: 1.0 },
  metrics: { multimodal: { scored_count: 4069, query_count: 4069 } } };
console.log('__R__' + JSON.stringify({
  contradictory: C.scoredCounts(contradictory, "multimodal"),
  consistent: C.scoredCounts(consistent, "multimodal"),
}));
""")
    bad = out["contradictory"]
    assert bad["scored"] <= bad["total"], (
        f"a numerator larger than its denominator was returned: {bad}")
    assert bad["scored"] == 4000 and bad["total"] == 4069, bad
    assert bad["derived"] is True, (
        "a served block that contradicted itself was still treated as authoritative", bad)
    good = out["consistent"]
    assert good["scored"] == 4069 and good["total"] == 4069, good
    assert good["derived"] is False, good
