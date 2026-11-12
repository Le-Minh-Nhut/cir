"""R9 — a mandatory G1/G2 subcheck that FAILED can never surface as PASS.

Measured on HEAD `2e2e2d2` from the artifacts themselves (`g1_g2.encoder.*.adapter.json`):
`batch_size_1_vs_n.within_tolerance = false` in all three categories, with `max_rel`
`3.131e-03 / 7.795e-03 / 1.165e-02` against the declared `1e-3`; `non_tie_count = 1` in dress and
shirt with the artifact classifying each as `explained_by_tie_epsilon: false`; and
`population.*.max_rel` above the declared `1e-4`. Every one of those is a mandatory subcheck under
the tolerances declared in the same file.

The surface above them still read "encoder: G1 PASS, G2 PASS with declared failures", and no code
joined the two, so "PASS" was printed next to evidence of mandatory failure. These tests pin the
join, and pin the four things that are easy to conflate:

* observer matches macro (the pipeline reproducing its own number),
* input parity (G1),
* forward/scoring parity (G2),
* batch parity (1 vs N) — the subcheck that actually fails.

They also pin the rule that matters most: **the historical record is not rewritten.** The old verdict
stays readable in every artifact and in `historical`, and no tolerance is moved.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend import parity_gate as gate

REVIEW = Path(__file__).resolve().parents[1] / "docs" / "reviews" / "20261109-native15-composition-limn"
CATEGORIES = ("dress", "shirt", "toptee")


def _artifact(model: str, category: str, side: str) -> dict:
    name = (f"g1_g2.encoder.{category}.{side}.json" if model == "encoder"
            else f"g1_g2.{category}.{side}.json")
    return json.loads((REVIEW / name).read_text(encoding="utf-8"))


def _tolerances(model: str = "encoder") -> dict:
    return _artifact(model, "shirt", "native")["tolerances"]


def _write(tmp_path: Path, model: str, category: str, side: str, body: dict) -> None:
    name = (f"g1_g2.encoder.{category}.{side}.json" if model == "encoder"
            else f"g1_g2.{category}.{side}.json")
    (tmp_path / name).write_text(json.dumps(body), encoding="utf-8")


def _verdict(tmp_path: Path, **kwargs) -> gate.ParityVerdict:
    """The synthetic evidence holds one category, so the gate is asked about exactly that one."""
    return gate.evaluate_model("encoder", root=tmp_path, categories=("shirt",))


#: A hit-count block that satisfies the CURRENT rule: per-side counts at every declared cutoff, the
#: same eligible denominator, and an assertion that both sides ranked the same eligible set. A bare
#: `native_metrics_match_feature_rerank` boolean used to be enough; it is now only a cross-check.
def _metrics_detail(*, native_hits: dict | None = None, adapter_hits: dict | None = None,
                    native_count: int = 10, adapter_count: int = 10,
                    identical_set: bool = True, recorded_flag: bool = True) -> dict:
    native_hits = native_hits or {"r1": 4, "r10": 7, "r50": 9}
    adapter_hits = adapter_hits or dict(native_hits)
    return {
        "cutoffs": [1, 10, 50],
        "native": {"hits": dict(native_hits), "query_count": native_count},
        "adapter": {"hits": dict(adapter_hits), "query_count": adapter_count},
        "eligible_query_set_identical": identical_set,
        "equality": {key: native_hits[key] == adapter_hits[key] for key in native_hits},
        "recorded_flag": recorded_flag,
    }


#: The identifiers a hit-count comparison must carry on BOTH sides before it can certify anything.
_PROVENANCE = {
    "protocol": "fashioniq_val_split",
    "checkpoint_id": "testmodel_checkpoint",
    "query_order_sha256": "a" * 64,
    "gallery_order_sha256": "b" * 64,
}

def _minimal(tmp_path: Path, *, model: str = "encoder", categories: tuple[str, ...] = ("shirt",),
             batch_within: bool = True, batch_max_rel: float = 1e-4,
             non_tie: int = 0, min_gap: float = 0.0,
             population_max_rel: float = 1e-6, identity_ok: bool = True,
             checkpoint_tolerances: dict | None = None,
             relation_sha: str | None = "6a09e667f3bcc908",
             metrics_detail: dict | None = None) -> Path:
    """A synthetic evidence directory: one category, one knob per subcheck.

    `relation_sha` defaults to a digest recorded identically on BOTH sides: ENCODER draws a relation
    tensor at construction, so an artifact that records none is not a clean artifact (it cannot show
    the two sides drew the same tensor). `None` omits the key altogether.
    """
    for category in categories:
        native = {
            "category": category, "phase": "native",
            "tolerances": checkpoint_tolerances or _tolerances(model),
            "native": {
                "layer1_identity": {
                    "query_count_native": 10, "query_count_annotations": 10,
                    "gallery_count_native": 20, "gallery_count_rebuilt_from_annotations": 20,
                    "gallery_order_identical": identity_ok,
                    "reference_ids_identical": identity_ok,
                    "target_ids_identical": identity_ok,
                    "model_text_identical_to_author_composition": identity_ok,
                    "gallery_has_duplicates": False,
                },
                "native_metrics_match_feature_rerank": True,
            },
        }
        native.update(_PROVENANCE)
        native["native_metrics_parity_detail"] = (metrics_detail
                                                  if metrics_detail is not None
                                                  else _metrics_detail())
        adapter = {
            "category": category, "phase": "adapter",
            "tolerances": native["tolerances"],
            "adapter": {
                "layer2_pixels": {"adapter_decode_vs_native_val_decode": {
                    "max_abs": 0.0, "max_rel": 0.0, "bitwise_equal": True, "min_cosine": 1.0,
                    "shape": [2, 4]}},
                "layer4_forward_sample": {
                    "query_val_pixels": {"max_abs": 0.0, "max_rel": 0.0, "min_cosine": 1.0,
                                         "bitwise_equal": True, "shape": [2, 4],
                                         "within_tolerance": True},
                    "target": {"max_abs": 0.0, "max_rel": 0.0, "min_cosine": 1.0,
                               "bitwise_equal": True, "shape": [2, 4], "within_tolerance": True},
                },
                "population": {
                    "features_vs_native_val_control": {
                        "max_abs": 1e-06, "max_rel": population_max_rel, "min_cosine": 0.999999999,
                        "bitwise_equal": False, "shape": [10, 4],
                        "within_tolerance": True},
                    "gallery_vs_native": {
                        "max_abs": 1e-06, "max_rel": 1e-06, "min_cosine": 0.999999999,
                        "bitwise_equal": False, "shape": [20, 4], "within_tolerance": True},
                },
                "batch_size_1_vs_n": {
                    "max_abs": 3e-07, "max_rel": batch_max_rel, "min_cosine": 0.999999999,
                    "bitwise_equal": False, "shape": [2, 4], "within_tolerance": batch_within},
                "cache_off_restart": {"max_abs": 0.0, "max_rel": 0.0, "min_cosine": 1.0,
                                      "bitwise_equal": True, "within_tolerance": True},
                "scoring_ranking": {
                    "score_max_abs_delta_vs_dot": 1.19e-07,
                    "score_pairs_compared": 8, "score_max_abs_declared": 1e-06,
                    "score_within_tolerance": True,
                    "adapter_vs_argsort_unmasked": {"count": 0, "head": [], "non_tie_count": 0,
                                                    "non_tie_head": []},
                    "rank_differences_vs_native_val_control": {
                        "count": non_tie, "head": [7] * non_tie, "non_tie_count": non_tie,
                        "min_gap_abs": min_gap},
                },
            },
        }
        adapter.update(_PROVENANCE)
        if relation_sha is not None:
            native["relation_state"] = {"sha256": relation_sha, "matches_pinned_digest": True}
            adapter["relation_state"] = {"sha256": relation_sha, "matches_pinned_digest": True}
        _write(tmp_path, model, category, "native", native)
        _write(tmp_path, model, category, "adapter", adapter)
    return tmp_path


# ---------------------------------------------------------------- the finding, on the real artifacts

@pytest.mark.skipif(not (REVIEW / "g1_g2.encoder.shirt.adapter.json").is_file(),
                    reason="the campaign artifacts are not on this host")
def test_encoder_gate_is_not_pass_on_head() -> None:
    """The exact case that motivated this module: a declared mandatory failure and a PASS surface.

    The batch-relative subcheck is the INVARIANT failure: `batch_size_1_vs_n.max_rel` exceeds the
    `batch_max_rel` the artifact itself declares, in every category. The measured `max_rel` is a
    number this campaign legitimately re-measures, so only the relation to its own declared tolerance
    is asserted -- never the third decimal.

    `non_tie_count` is likewise measured, and it moved to 0 once both sides were seeded with the same
    relation tensor. So the invariant asserted here is the CAUSE, not the old count: with
    `relation_state.sha256` equal on both sides and zero non-tie rank changes, scoring parity has no
    residual to explain, and the verdict must still fail on the batch dimension alone.
    """
    verdict = gate.evaluate_model("encoder")
    assert verdict.status == gate.MANDATORY_FAILED
    assert verdict.passed is False
    names = {check.name for check in verdict.failures}
    assert any("batch_1_vs_n.max_rel" in name for name in names), names
    # The failing batch checks are exactly one per category, each judged against the artifact's own
    # declared tolerance and each genuinely above it.
    batch = [check for check in verdict.failures if check.name.endswith("batch_1_vs_n.max_rel")]
    assert len(batch) == 3
    for check in batch:
        assert check.declared == 1e-3
        assert check.measured > check.declared
    assert not any("non_tie_changes" in name for name in names), (
        "a zero non-tie count must not appear as a failure: " + repr(names))
    # The relation digests are the reason the rank dimension is clean, so they are the recorded
    # invariant. Identical bytes on both sides is what makes the features comparable at all.
    for category in CATEGORIES:
        native = _artifact("encoder", category, "native")
        adapter = _artifact("encoder", category, "adapter")
        digest = native["relation_state"]["sha256"]
        assert digest == adapter["relation_state"]["sha256"], category
        assert adapter["adapter"]["scoring_ranking"][
            "rank_differences_vs_native_val_control"]["non_tie_count"] == 0, category


@pytest.mark.skipif(not (REVIEW / "g1_g2.encoder.shirt.adapter.json").is_file(),
                    reason="the campaign artifacts are not on this host")
def test_the_declared_tolerance_is_what_the_artifact_declares() -> None:
    """A pass/fail must be judged against 1e-3 because the artifact says 1e-3, not because we chose it."""
    declared = _tolerances()
    assert declared["batch_max_rel"] == 1e-3
    assert declared["batch_max_abs"] == 1e-4
    assert declared["forward_max_rel"] == 1e-4
    assert declared["tie_epsilon"] == 1e-06
    verdict = gate.evaluate_model("encoder")
    numbers = {value for value in declared.values() if isinstance(value, (int, float))}
    # Every failing numeric check is judged against a tolerance this artifact declared itself.
    for check in verdict.failures:
        if isinstance(check.declared, (int, float)):
            assert check.declared in numbers, (check.name, check.declared, declared)


@pytest.mark.skipif(not (REVIEW / "g1_g2.shirt.adapter.json").is_file(),
                    reason="the campaign artifacts are not on this host")
def test_limn_gate_is_also_not_pass_on_head() -> None:
    """LIMN's own batch-relative subcheck fails the same way; it is not exempt."""
    verdict = gate.evaluate_model("limn")
    assert verdict.status == gate.MANDATORY_FAILED
    assert verdict.passed is False
    assert any("batch_1_vs_n.max_rel" in check.name for check in verdict.failures)


@pytest.mark.skipif(not (REVIEW / "g1_g2.encoder.shirt.adapter.json").is_file(),
                    reason="the campaign artifacts are not on this host")
def test_the_historical_record_survives_untouched() -> None:
    """The verdict is DERIVED. Every artifact stays readable with its own recorded booleans.

    `historical` is an ECHO, so the invariant is equality with the artifact's own recorded values --
    not the values themselves. `recorded_non_tie_rank_changes` is `0` today (both sides now draw the
    same relation tensor) and was `1` before; a literal here would pin a measurement and would have to
    be re-pinned on every legitimate re-measurement, which is exactly what a history echo must not do.
    """
    before = {name: (REVIEW / name).read_bytes()
              for name in ("g1_g2.encoder.shirt.adapter.json", "g1_g2.encoder.shirt.native.json")}
    verdict = gate.evaluate_model("encoder")
    after = {name: (REVIEW / name).read_bytes() for name in before}
    assert before == after, "the gate must never rewrite the evidence it judges"
    recorded = verdict.historical["shirt"]
    adapter = _artifact("encoder", "shirt", "adapter")["adapter"]
    scoring = adapter["scoring_ranking"]
    # The echo must equal the source, field for field, whatever the measurement currently is.
    assert recorded["recorded_non_tie_rank_changes"] == \
        scoring["rank_differences_vs_native_val_control"]["non_tie_count"]
    assert recorded["recorded_within_tolerance"]["batch_size_1_vs_n"] == \
        adapter["batch_size_1_vs_n"]["within_tolerance"]
    assert recorded["recorded_within_tolerance"]["sample_query_val_pixels"] == \
        adapter["layer4_forward_sample"]["query_val_pixels"].get("within_tolerance")
    # The declared table is echoed whole, so a reader can see what the numbers were judged against.
    assert recorded["declared_tolerances"] == \
        _artifact("encoder", "shirt", "native")["tolerances"]
    assert recorded["declared_tolerances"]["batch_max_rel"] == 1e-3
    # The old prose verdict is not the gate's output, and the gate does not repeat it as a status.
    assert verdict.status != gate.PASS


@pytest.mark.skipif(not (REVIEW / "g1_g2.encoder.shirt.adapter.json").is_file(),
                    reason="the campaign artifacts are not on this host")
def test_the_dimensions_are_not_conflated() -> None:
    """Observer/macro, input, forward, batch and scoring parity are separate answers.

    The invariant is that each dimension is DERIVED from its own mandatory checks -- a dimension is
    False exactly when one of ITS mandatory checks failed -- so the test asserts the relation between
    `dimensions` and `checks` instead of pinning which dimensions are False today. `scoring_parity`
    is the one that moved: it was False while a rank difference needed the tie epsilon to explain, and
    it is True now that both sides draw the same relation tensor. What must not move is that no
    dimension can be carried by another's checks, and that every failure lands in the dimension whose
    subcheck it is.
    """
    verdict = gate.evaluate_model("encoder")
    for dimension in gate.DIMENSIONS:
        mandatory = [check for check in verdict.checks
                     if check.mandatory and check.dimension == dimension]
        if mandatory:
            assert verdict.dimensions[dimension] is all(check.ok for check in mandatory), dimension
    for check in verdict.failures:
        assert verdict.dimensions[check.dimension] is False, check.name
    # The two failing dimensions are the ones whose subcheck is the denominator-unstable relative
    # error, and each of them fails on `max_rel` -- never on an absolute delta.
    failing = {dimension for dimension, ok in verdict.dimensions.items() if ok is False}
    assert {"batch_parity", "forward_parity"} <= failing, failing
    for check in verdict.failures:
        if check.dimension in ("batch_parity", "forward_parity"):
            assert check.name.endswith(".max_rel"), check.name
    # Input parity and the observer are separate answers, measured and never merged.
    assert verdict.dimensions["input_parity"] is True
    dimensions = {check.dimension for check in verdict.checks}
    assert {"input_parity", "batch_parity", "scoring_parity"} <= dimensions
    # The observer is separate by CONSTRUCTION, not by happening to disagree today: every observer
    # check is non-mandatory and carries its own dimension, so it can never carry forward_parity's
    # verdict. (Asserting the two dimensions merely differ would pass by coincidence whenever both are
    # False -- which is the case right now, for unrelated reasons.)
    observer = [check for check in verdict.checks if check.dimension == "observer_parity"]
    assert observer, "the encoder artifacts record an observer-vs-macro recheck"
    assert all(not check.mandatory for check in observer)
    assert all(check.name.endswith(tuple(f"_{k}" for k in ("r1", "r10", "r50")))
               for check in observer)


def test_a_model_without_evidence_is_not_run_never_pass(tmp_path: Path) -> None:
    verdict = gate.evaluate_model("hint", root=tmp_path)
    assert verdict.status == gate.NOT_RUN
    assert verdict.passed is False
    assert "no G1/G2 artifact" in verdict.detail


def test_a_partial_evidence_set_is_not_a_pass(tmp_path: Path) -> None:
    _minimal(tmp_path, categories=("shirt",))
    verdict = gate.evaluate_model("encoder", root=tmp_path)
    assert verdict.status == gate.EVIDENCE_MISSING
    assert verdict.passed is False
    assert "dress" in verdict.detail


# ---------------------------------------------------------------- the rule

def test_a_failing_batch_relative_subcheck_cannot_pass(tmp_path: Path) -> None:
    _minimal(tmp_path, batch_within=False, batch_max_rel=3.0e-03)
    verdict = _verdict(tmp_path)
    assert verdict.status == gate.MANDATORY_FAILED
    assert verdict.passed is False
    failed = [check for check in verdict.failures if check.name.endswith("batch_1_vs_n.max_rel")]
    assert len(failed) == 1
    assert failed[0].declared == 1e-3 and failed[0].measured == 3.0e-03


def test_a_non_tie_rank_change_cannot_pass(tmp_path: Path) -> None:
    _minimal(tmp_path, non_tie=2, min_gap=6.5e-04)
    verdict = _verdict(tmp_path)
    assert verdict.status == gate.MANDATORY_FAILED
    assert any("non_tie_changes" in check.name for check in verdict.failures)


def test_a_rank_change_inside_the_tie_epsilon_is_allowed(tmp_path: Path) -> None:
    """The declared rule permits a difference only when it is inside the tie epsilon."""
    _minimal(tmp_path, non_tie=0, min_gap=1e-09)
    verdict = _verdict(tmp_path)
    assert verdict.status == gate.PASS
    assert verdict.passed is True


def test_a_clean_artifact_passes(tmp_path: Path) -> None:
    _minimal(tmp_path)
    verdict = _verdict(tmp_path)
    assert verdict.status == gate.PASS
    assert verdict.passed is True and verdict.complete is True
    assert verdict.failures == []


def test_a_near_zero_denominator_relative_error_still_fails_v1(tmp_path: Path) -> None:
    """The standard is not relaxed: 7.8 on `max_rel` is a v1 mandatory failure."""
    _minimal(tmp_path, population_max_rel=7.85)
    verdict = _verdict(tmp_path)
    assert verdict.status == gate.MANDATORY_FAILED
    failed = [check for check in verdict.failures if "max_rel" in check.name]
    assert failed and failed[0].standard == gate.STANDARD_RELATIVE_V1


def test_v2_is_advisory_on_v1_data_and_cannot_certify(tmp_path: Path) -> None:
    """A post-hoc standard may be reported, never used to turn a v1 failure into a pass."""
    _minimal(tmp_path, population_max_rel=7.85)
    verdict = gate.evaluate_model("encoder", root=tmp_path, categories=("shirt",),
                                  standard=gate.STANDARD_ABS_REL_V2)
    assert verdict.status == gate.MANDATORY_FAILED
    assert verdict.passed is False
    assert verdict.advisory, "the v2 evaluation must be visible"
    assert verdict.advisory[0]["declared_before_measurement"] is False
    assert verdict.advisory[0]["applies"] is False


def test_v1_remains_the_default_standard(tmp_path: Path) -> None:
    """A v1 pair under an explicit root: the label is v1 and no breakdown invents a standard.

    The root is EXPLICIT. Reading the real evidence directory made this test depend on what happened
    to be on disk -- it broke as soon as a second standard appeared, which is a fact about the
    directory, not about the default.
    """
    _minimal(tmp_path)
    assert _verdict(tmp_path).status == gate.PASS
    result = gate.evaluate_all(root=tmp_path)
    assert result["standard"] == gate.STANDARD_RELATIVE_V1
    assert result["standards"] == {gate.STANDARD_RELATIVE_V1: ["encoder"]}
    assert result["advisory_standards"] == [gate.STANDARD_ABS_REL_V2]


def test_an_unknown_standard_is_refused(tmp_path: Path) -> None:
    _minimal(tmp_path)
    with pytest.raises(ValueError, match="unsupported parity standard"):
        gate.evaluate_model("encoder", root=tmp_path, categories=("shirt",),
                            standard="parity-vibes/v9")


def test_a_broken_identity_subcheck_cannot_pass(tmp_path: Path) -> None:
    _minimal(tmp_path, identity_ok=False)
    verdict = _verdict(tmp_path)
    assert verdict.status == gate.MANDATORY_FAILED
    assert verdict.dimensions["input_parity"] is False


def test_the_abs_rel_rule_is_the_declared_form() -> None:
    # abs(a-b) <= atol + rtol*abs(b)
    assert gate.evaluate_abs_rel(1.0 + 5e-04, 1.0, atol=1e-4, rtol=1e-3) is True
    assert gate.evaluate_abs_rel(1.0 + 2e-03, 1.0, atol=1e-4, rtol=1e-3) is False
    # A near-zero reference is bounded by atol, not by a vanishing denominator.
    assert gate.evaluate_abs_rel(1e-06, 0.0, atol=1e-4, rtol=1e-3) is True


@pytest.mark.skipif(not (REVIEW / "g1_g2.shirt.adapter.json").is_file(),
                    reason="the campaign artifacts are not on this host")
def test_a_passing_dimension_names_what_it_never_saw() -> None:
    """`True` means "every check that exists passed" -- never "fully covered".

    LIMN's artifact records no population-level forward comparison, so its forward dimension rests on
    the fixed sample alone. A reader must be able to see that instead of reading True as complete.
    """
    limn = gate.evaluate_model("limn")
    assert limn.dimensions["forward_parity"] is True
    gaps = limn.coverage["forward_parity"]["not_measured"]
    assert any("population" in gap for gap in gaps), gaps
    encoder = gate.evaluate_model("encoder")
    assert encoder.coverage["forward_parity"]["not_measured"] == []
    # Every dimension reports both sides, so an empty list is a statement, not an omission.
    for dimension, coverage in limn.coverage.items():
        assert set(coverage) == {"measured", "not_measured"}, (dimension, coverage)


@pytest.mark.skipif(not (REVIEW / "g1_g2.shirt.adapter.json").is_file(),
                    reason="the campaign artifacts are not on this host")
def test_limn_rank_checks_are_read_from_its_own_record_shape() -> None:
    """LIMN counts `unexplained_rank_differences`; reading only the ENCODER key would skip it."""
    limn = gate.evaluate_model("limn")
    names = {check.name for check in limn.checks}
    assert any("unexplained_differences" in name for name in names), names
    assert any("scores.max_abs_vs_dot" in name for name in names), names


@pytest.mark.skipif(not (REVIEW / "g1_g2.encoder.shirt.adapter.json").is_file(),
                    reason="the campaign artifacts are not on this host")
def test_the_harness_tests_max_rel_on_some_keys_and_not_others() -> None:
    """The incoherence R9 now records. It is in WHICH comparator each check calls, not in science.

    `encoder_parity.py` tests `max_abs` AND `max_rel` for the population QUERY comparisons but only
    `max_abs` for `population.gallery_vs_native` (and for `query_native_tensors`). So gallery features
    can exceed the same declared `max_rel` and still read `within_tolerance: true` while query
    features of the same shape read `false`.

    The invariant is that WIRING and its consequence: the gallery block is labelled within tolerance
    while its own `max_rel` is above the declared `forward_max_rel`, the query block is labelled
    outside, and the gate reports the disagreement. The measured ratios are re-measured by every run
    and are therefore compared to the DECLARED tolerance, never pinned as constants -- the old
    `> 1.0` / `> 0.5` thresholds recorded one run's numbers, and the gallery ratio is 0.53 today.
    """
    declared_rel = _tolerances("encoder")["forward_max_rel"]
    adapter = _artifact("encoder", "shirt", "adapter")["adapter"]
    population = adapter["population"]
    recorded = {key: value.get("within_tolerance") for key, value in population.items()
                if isinstance(value, dict)}
    # Gallery exceeds the same declared `max_rel` and is still labelled within tolerance: that
    # asymmetry IS the defect, and it holds at any measured magnitude above the tolerance.
    assert recorded["gallery_vs_native"] is True
    assert population["gallery_vs_native"]["max_rel"] > declared_rel
    # The query comparison of the same shape and dtype is labelled outside on the same tolerance.
    assert recorded["features_vs_native_val_control"] is False
    assert population["features_vs_native_val_control"]["max_rel"] > declared_rel
    # Both are denied by the backstop it was supposed to use: the absolute delta stays inside atol
    # while the relative term is what fails, i.e. a near-zero denominator drives it.
    for key in ("gallery_vs_native", "features_vs_native_val_control"):
        assert population[key]["max_abs"] <= _tolerances("encoder")["forward_max_abs"], key
    # The gate applies the declared tolerance table to both and names the disagreement.
    verdict = gate.evaluate_model("encoder")
    disagreements = {(item["comparison"], item["subcheck"])
                     for item in verdict.label_disagreements}
    assert ("shirt.population.gallery_vs_native", "max_rel") in disagreements, disagreements
    assert all(item["harness_recorded"] is True and item["gate_verdict"] is False
               for item in verdict.label_disagreements)
    # The old label is preserved verbatim on the check that disagrees.
    gallery = next(check for check in verdict.checks
                   if check.name == "shirt.population.gallery_vs_native.max_rel")
    assert gallery.harness_recorded is True and gallery.ok is False
    assert gallery.declared == declared_rel
    assert gallery.to_dict()["harness_recorded_within_tolerance"] is True


def test_evaluate_all_reports_no_passing_model_on_head() -> None:
    if not (REVIEW / "g1_g2.encoder.shirt.adapter.json").is_file():
        pytest.skip("the campaign artifacts are not on this host")
    result = gate.evaluate_all()
    assert result["passed_models"] == [], result["passed_models"]
    assert result["models"]["encoder"]["status"] == gate.MANDATORY_FAILED
    assert result["models"]["encoder"]["failure_count"] > 0
