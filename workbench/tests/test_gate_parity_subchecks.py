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


def _minimal(tmp_path: Path, *, model: str = "encoder", categories: tuple[str, ...] = ("shirt",),
             batch_within: bool = True, batch_max_rel: float = 1e-4,
             non_tie: int = 0, min_gap: float = 0.0,
             population_max_rel: float = 1e-6, identity_ok: bool = True,
             checkpoint_tolerances: dict | None = None,
             relation_sha: str | None = "6a09e667f3bcc908") -> Path:
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
    """The exact case that motivated this module: a declared mandatory failure and a PASS surface."""
    verdict = gate.evaluate_model("encoder")
    assert verdict.status == gate.MANDATORY_FAILED
    assert verdict.passed is False
    names = {check.name for check in verdict.failures}
    assert any("batch_1_vs_n.max_rel" in name for name in names), names
    assert any("non_tie_changes" in name for name in names), names
    # The failing ones are exactly the declared ones, at their declared values.
    batch = [check for check in verdict.failures if check.name.endswith("batch_1_vs_n.max_rel")]
    assert len(batch) == 3
    for check in batch:
        assert check.declared == 1e-3
        assert check.measured > check.declared
    # The three measured values are the ones the artifacts record, to full precision.
    measured = {check.measured for check in batch}
    for expected in (0.0031308995258445375, 0.0077950176161260265, 0.011646653843520266):
        assert expected in measured, (expected, measured)


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
    """The verdict is DERIVED. Every artifact stays readable with its own recorded booleans."""
    before = {name: (REVIEW / name).read_bytes()
              for name in ("g1_g2.encoder.shirt.adapter.json", "g1_g2.encoder.shirt.native.json")}
    verdict = gate.evaluate_model("encoder")
    after = {name: (REVIEW / name).read_bytes() for name in before}
    assert before == after, "the gate must never rewrite the evidence it judges"
    recorded = verdict.historical["shirt"]
    assert recorded["recorded_within_tolerance"]["batch_size_1_vs_n"] is False
    assert recorded["recorded_non_tie_rank_changes"] == 1
    assert recorded["declared_tolerances"]["batch_max_rel"] == 1e-3
    # The old prose verdict is not the gate's output, and the gate does not repeat it as a status.
    assert verdict.status != gate.PASS


@pytest.mark.skipif(not (REVIEW / "g1_g2.encoder.shirt.adapter.json").is_file(),
                    reason="the campaign artifacts are not on this host")
def test_the_dimensions_are_not_conflated() -> None:
    """Observer/macro, input, forward, batch and scoring parity are separate answers."""
    verdict = gate.evaluate_model("encoder")
    assert verdict.dimensions["input_parity"] is True
    assert verdict.dimensions["batch_parity"] is False
    assert verdict.dimensions["scoring_parity"] is False
    dimensions = {check.dimension for check in verdict.checks}
    assert {"input_parity", "batch_parity", "scoring_parity"} <= dimensions
    # Observer/macro agreement is its own dimension and never carries forward parity.
    assert "observer_parity" not in verdict.dimensions or \
        verdict.dimensions["observer_parity"] != verdict.dimensions.get("forward_parity")


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
    _minimal(tmp_path)
    assert _verdict(tmp_path).status == gate.PASS
    assert gate.evaluate_all()["standard"] == gate.STANDARD_RELATIVE_V1


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

    `encoder_parity.py:745-747` tests only `max_abs` for `layer4_forward_sample.query_native_tensors`;
    `:778-781` tests `max_abs` AND `max_rel` for the two population QUERY comparisons; `:783` tests
    only `max_abs` for `population.gallery_vs_native`. So gallery features with `max_rel` 1.36 / 7.75 /
    3.90 read `within_tolerance: true` while query features of the same shape and dtype read `false`
    on `max_rel` 0.752 / 7.850 / 3.355 -- the headline "population max_rel FAIL" depends on which
    side of the same artifact you look at.
    """
    adapter = _artifact("encoder", "shirt", "adapter")["adapter"]
    population = adapter["population"]
    recorded = {key: value.get("within_tolerance") for key, value in population.items()
                if isinstance(value, dict)}
    # Gallery fails the same declared `max_rel` and is still labelled within tolerance.
    assert recorded["gallery_vs_native"] is True
    assert population["gallery_vs_native"]["max_rel"] > 1.0
    assert recorded["features_vs_native_val_control"] is False
    assert population["features_vs_native_val_control"]["max_rel"] > 0.5
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
    assert gallery.to_dict()["harness_recorded_within_tolerance"] is True


def test_evaluate_all_reports_no_passing_model_on_head() -> None:
    if not (REVIEW / "g1_g2.encoder.shirt.adapter.json").is_file():
        pytest.skip("the campaign artifacts are not on this host")
    result = gate.evaluate_all()
    assert result["passed_models"] == [], result["passed_models"]
    assert result["models"]["encoder"]["status"] == gate.MANDATORY_FAILED
    assert result["models"]["encoder"]["failure_count"] > 0
