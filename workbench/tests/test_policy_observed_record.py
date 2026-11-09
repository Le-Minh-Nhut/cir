"""The BLAS policy an artifact observed, recorded OUTSIDE the hashed run identity.

The defect this closes: `_record_effective_blas_pin` populated a module dict that no code path
wrote anywhere durable, so an artifact recorded the pin we ASKED for (`blas_threads_pinned`) and
what the interpreter COULD do (`blas_pin_capability`) but never what the runtime DID for that run.
An artifact could not prove its own policy.

The shape of the fix matters as much as the field. The obvious move -- put the readback into
`identity` -- is wrong, and `runner.py` already documents why: `identity` is hashed before scoring
runs, so a value only known afterwards would make `run_id` depend on whether a previous run in the
same process had already pinned. The same inputs would mint different ids. So `policy_observed`
travels in the rank payload and the summary, outside the hash, and the third test below is the
regression that keeps it there.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from workbench.composition import runner


def _pin_record(**overrides):
    """A readback record shaped like the one `_blas_thread_limit` returns."""
    record = {"requested": 1, "effective": 1, "previous": 20, "setter_found": True,
              "getter_found": True, "verified": True,
              "source": "libopenblas64_p-r0-0cf96a72.3.23.dev.so", "reason": None}
    record.update(overrides)
    return record


@pytest.fixture(autouse=True)
def _clean_pin_state():
    """Every test starts and ends with an empty module dict.

    Without this, the process-global `_effective_blas_pin` would leak between tests exactly the way
    it could leak between runs -- and a test that passes only because of ordering is not evidence.
    """
    runner._reset_effective_blas_pin()
    yield
    runner._reset_effective_blas_pin()


def test_a_scored_run_records_the_readback_it_observed():
    """The record must be the RUNTIME'S answer, and it must name all three facts it can prove."""
    runner._record_effective_blas_pin(_pin_record())
    observed = runner.observed_policy_record()

    assert observed["status"] == "verified"
    assert observed["record"] is not None
    record = observed["record"]
    # Every field the pin helper can report is carried through, not a subset. A record that dropped
    # `verified` would be the original bug wearing a different name.
    for name in runner.POLICY_OBSERVED_FIELDS:
        assert name in record, f"{name} missing from the observed record"
    assert record["requested"] == 1
    assert record["effective"] == 1
    assert record["previous"] == 20
    assert record["verified"] is True
    assert record["source"].startswith("libopenblas")


def test_the_observed_value_is_not_the_requested_value_when_the_runtime_disagreed():
    """The whole point: a pin that did NOT apply must not read back as the number we asked for."""
    runner._record_effective_blas_pin(_pin_record(effective=9, verified=False,
                                                 reason="setter ran but the runtime reports 9"))
    observed = runner.observed_policy_record()

    assert observed["status"] == "unverified"
    assert observed["record"]["requested"] == 1
    assert observed["record"]["effective"] == 9, "the observed value must not be the request"
    assert observed["record"]["verified"] is False
    assert "reports 9" in (observed["reason"] or "")


def test_an_unverifiable_runtime_is_recorded_as_unverifiable_not_as_a_pass_or_a_failure():
    """`setter_found: False` (e.g. an MKL-only env) is its own verdict.

    Collapsing it into "unverified" would lose the distinction that matters operationally: this
    runtime CANNOT be pinned, so no measurement from it will ever be verifiable, as opposed to a
    runtime that can be pinned and reported a different number.
    """
    runner._record_effective_blas_pin(_pin_record(effective=None, verified=False,
                                                 setter_found=False, getter_found=False,
                                                 source=None,
                                                 reason="no OpenBLAS setter symbol"))
    observed = runner.observed_policy_record()

    assert observed["status"] == "unverifiable"
    assert observed["record"]["verified"] is False
    assert observed["record"]["setter_found"] is False


def test_a_run_that_recorded_nothing_says_so_rather_than_claiming_anything():
    """Absence of an observation is not evidence the pin failed, and must not read as `verified: false`."""
    observed = runner.observed_policy_record()

    assert observed["status"] == runner.NOT_RECORDED
    assert observed["record"] is None
    assert "absence of evidence" in observed["reason"]


def test_a_legacy_artifact_reports_absent_rather_than_a_verdict():
    """The eight artifacts written before this field exist must not be backfilled or mislabelled."""
    legacy = runner.policy_observed_from_payload({"fingerprint": "f", "identity": {}, "ranks": {}})

    assert legacy["status"] == runner.NOT_RECORDED
    assert legacy["record"] is None
    assert "predates" in legacy["reason"]

    nothing = runner.policy_observed_from_payload(None)
    assert nothing["status"] == runner.NOT_RECORDED
    assert "no rank payload" in nothing["reason"]


def test_a_modern_payload_round_trips_through_json(tmp_path: Path):
    """What is written to disk is what a reader gets back, verdict included."""
    runner._record_effective_blas_pin(_pin_record())
    payload = {"fingerprint": "f", "identity": {"model_id": "encoder"}, "ranks": {},
               "policy_observed": runner.observed_policy_record()}
    path = tmp_path / "ranks.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    read = runner.policy_observed_from_payload(json.loads(path.read_text(encoding="utf-8")))
    assert read["status"] == "verified"
    assert read["record"]["effective"] == 1
    assert read["record"]["previous"] == 20


def test_the_runner_actually_resets_the_readback_before_scoring():
    """The helper being correct is not enough -- the RUNNER must call it.

    Measured vacuity in this very file: with `_reset_effective_blas_pin()` deleted from
    `run_composition`, the behavioural test above still passed, because it invokes the helper
    directly. A helper-only test proves nothing about the production path. So the call site is
    asserted structurally, from source, together with its ORDER relative to the rank write: a reset
    that ran AFTER scoring would clear the very record it is supposed to publish.
    """
    import ast
    import inspect

    source = inspect.getsource(runner.run_composition)
    tree = ast.parse(source)

    reset_lines = [node.lineno for node in ast.walk(tree)
                   if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Name)
                   and node.func.id == "_reset_effective_blas_pin"]
    assert reset_lines, (
        "run_composition never calls _reset_effective_blas_pin(): a run that scores nothing would "
        "publish the previous model's readback as its own observation")

    observed_lines = [node.lineno for node in ast.walk(tree)
                      if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Name)
                      and node.func.id == "observed_policy_record"]
    assert observed_lines, (
        "run_composition never calls observed_policy_record(): nothing would be published")
    assert min(reset_lines) < min(observed_lines), (
        f"the reset (line {min(reset_lines)}) must precede the first observation "
        f"(line {min(observed_lines)}), or it clears the record it is meant to publish")


def test_resetting_clears_the_readback_so_a_later_run_cannot_inherit_it():
    """The scoping guarantee: `policy_observed` must describe THIS run, never the previous one."""
    runner._record_effective_blas_pin(_pin_record())
    assert runner.observed_policy_record()["status"] == "verified"

    runner._reset_effective_blas_pin()
    assert runner.observed_policy_record()["status"] == runner.NOT_RECORDED, (
        "a run that scored nothing would otherwise publish the previous model's verification")


def _identity_literal_keys() -> set[str]:
    """String keys of the `identity = {…}` literal the runner actually builds.

    Parsed from source on purpose. A test that constructs its own dict proves only that the test's
    dict is stable, which is vacuous: the bug being guarded against is a KEY ADDED TO THE REAL
    LITERAL. Reading the production assignment is the only way to catch that.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(runner.run_composition))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "identity"
                   for target in node.targets):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        return {key.value for key in node.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)}
    raise AssertionError("no `identity = {…}` literal found in run_composition")


def test_the_run_identity_does_not_depend_on_the_observed_pin():
    """THE REGRESSION. `identity` is hashed before scoring, so the readback must never enter it.

    Two assertions, because they fail for different reasons:

    1. STRUCTURAL -- the production `identity = {…}` literal must not carry `policy_observed`. This
       is the one that fails if someone adds the field to the real payload.
    2. BEHAVIOURAL -- two runs identical in every hashed input must produce the same digest no
       matter what the module dict holds, including a stale verified record from a previous run.
    """
    keys = _identity_literal_keys()
    assert "blas_threads_pinned" in keys and "blas_pin_capability" in keys, (
        "the identity literal was not found as expected; this check cannot verify anything")
    assert "policy_observed" not in keys, (
        "policy_observed was added to the hashed identity. The identity is digested BEFORE scoring "
        "runs, so the run id would then depend on whether a previous run in this process had already "
        "pinned -- the mutable-state defect runner.py:1481-1486 records. Keep it in the payload.")
    assert "blas_pin_applied" not in keys, (
        "a per-call readback field reached the hashed identity; same defect")

    def identity():
        return {
            "model_id": "encoder", "checkpoint_id": "fashioniq",
            "gallery_identity": "g", "query_order_identity": "q",
            "query_scope_resolution": "s", "blas_threads_pinned": runner.BLAS_THREADS_PINNED,
            "blas_pin_capability": runner._blas_pin_capability(),
            "cache_verify_mode": "full", "subset_memberships": ["full"],
        }

    runner._reset_effective_blas_pin()
    clean = runner._digest(identity())

    # A previous run in the same process left a readback behind. The identity must not notice.
    runner._record_effective_blas_pin(_pin_record())
    stale = runner._digest(identity())

    assert clean == stale, (
        "the run id changed because a previous run had pinned: the observed policy must stay "
        "outside the hashed identity")
    assert runner.observed_policy_record()["status"] == "verified", (
        "the readback IS available to the payload; it is only excluded from the hash")
