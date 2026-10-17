#!/usr/bin/env python3
"""Verify one recorded reproduction run end to end.

Exit 0 only when every ingredient of the run corroborates every other:

* the aggregate report exists at the reports root under its expected name and parses;
* the report's own run/model/checkpoint/protocol identity is exactly the requested one;
* the HMAC evidence ledger verifies and holds an entry for this run whose report digest
  equals the report file's sha256, with return code 0 and completion
  ``INVOCATION_COMPLETED``;
* the completion proof exists and passes the evaluator's own ``validate_run_manifest``
  (checksum and environment identity included);
* the run is real evidence: a report that declares itself synthetic, or whose ``data_kind`` is
  not a real experiment, is refused as a self-test;
* the run carries the metric evidence its own protocol can produce -- for LIMN a completed
  extraction, three complete FashionIQ categories with finite Recall@10/@50 inside [0, 100],
  and a macro that equals the arithmetic ``workbench/replay/common.py::macro_aggregate``
  defines over those categories; for every other registered model a reproduced side for at
  least one metric its verified committed reference publishes under this run's own condition,
  or an explicit refusal where no such reference exists.

Nothing is re-implemented here: the ledger reader, the hasher, the proof validator, the metric
extraction and the registry helpers are the ones the evaluator itself uses. The one arithmetic
this module re-derives -- the LIMN macro -- is re-derived *from the report's own category
numbers* precisely so the report's claimed aggregate is not taken on its word.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import workbench.scripts.evaluate_models as em
from workbench.backend import benchmark_references as benchmarks
from workbench.backend import paper_gate
from workbench.backend.operator_config import resolve_config
from workbench.backend.registry import checkpoint_by_id, model_by_id

#: Categories the LIMN macro is defined over. A report missing one of them covers a partial
#: benchmark, so its macro is not the paper's macro.
LIMN_CATEGORIES = ("dress", "shirt", "toptee")

#: The extraction state that says the metrics were actually read out of the evaluator's output.
EXTRACTION_COMPLETED = "EXTRACTED_VERIFIED"

#: A registry-declared synthetic evaluator prints its own declared aggregate
#: (`metrics_extraction.py` / `evaluate_models.py::_synthetic_aggregate`). Its numbers were
#: genuinely extracted, but from a self-test, so the run is complete evidence of an execution
#: and never evidence about the paper. Kept distinct from EXTRACTION_COMPLETED on purpose.
EXTRACTION_SYNTHETIC = "EXTRACTED_SYNTHETIC"

#: Absolute bound, in Recall percentage points, on the disagreement allowed between a macro
#: aggregate and the categories it must be the mean of. It is deliberately the same bound the
#: extraction parser itself applies when it validates a declared aggregate against the
#: categories of the same run (``workbench/backend/metrics_extraction.py:173``, ``:272``,
#: ``:359``): the report's numbers are copies of what that parser produced, so re-deriving them
#: here cannot legitimately differ at all, while 0.05 pp is four orders of magnitude below the
#: smallest deviation the 5 percent acceptance gate could ever act on. A tighter float64
#: epsilon would reject a report whose evaluator printed its macro rounded to two decimals --
#: an honest report -- without catching any inconsistency the parser's own bound misses.
MACRO_TOLERANCE_PP = 0.05

def _is_recall(value: object) -> bool:
    """A Recall is a finite number inside [0, 100]; a bool is not a number here."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and 0.0 <= float(value) <= 100.0

def _regression_pin(model_id: str) -> str | None:
    try:
        return model_by_id(model_id).get("upstream_commit_sha")
    except (KeyError, ValueError, OSError):
        return None

def verify(run_id: str, model_id: str, checkpoint_id: str, protocol_id: str) -> dict:
    config = resolve_config()
    reports = em.reports_root(config)
    report_path = reports / f"{model_id}_{checkpoint_id}_{run_id}_aggregate.json"
    reasons: list[str] = []

    report: dict | None = None
    if not report_path.is_file():
        reasons.append(f"aggregate report not found: {report_path}")
    else:
        try:
            parsed = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            reasons.append(f"aggregate report does not parse: {error}")
        else:
            if not isinstance(parsed, dict):
                reasons.append("aggregate report is not a JSON object")
            else:
                report = parsed
                # Exact identity: a substring or prefix match would let a report for another
                # run answer for this one.
                for key, expected in (("run_id", run_id), ("model_id", model_id),
                                      ("checkpoint_id", checkpoint_id),
                                      ("protocol_id", protocol_id)):
                    if str(report.get(key)) != expected:
                        reasons.append(
                            f"report {key} is {report.get(key)!r}, expected {expected!r}")

    # The key state is reported on its own line, before the chain verdict, so a reader can tell an
    # ambiguity the operator must fix apart from a corrupt chain -- and so every existing refusal
    # reason keeps the exact wording its consumers already match on.
    witness: dict | None = None
    try:
        em.ledger_hmac_key(config)
    except em.LedgerKeyError as key_error:
        reasons.append(f"the evidence ledger key is unusable: {key_error}")
    try:
        ledger = em.read_evidence_ledger(config)
    except em.LedgerKeyError as error:
        ledger = None
        key_err_msg = f"the evidence ledger key is unusable: {error}"
        if key_err_msg not in reasons:
            reasons.append(key_err_msg)
    if ledger is None:
        reasons.append("the evidence ledger is missing or its HMAC chain does not verify")
    else:
        matching = [entry for entry in ledger if entry.get("run_id") == run_id]
        if not matching:
            reasons.append(f"the evidence ledger holds no entry for run {run_id}")
        else:
            witness = matching[-1]
            if report_path.is_file():
                if str(witness.get("report_digest") or "") != em.sha256_file(report_path):
                    reasons.append(
                        "the evidence ledger's report digest does not match the report file")
            else:
                reasons.append("the evidence ledger cannot be corroborated: no report file")
            if witness.get("return_code") != 0:
                reasons.append("the evidence ledger records return code "
                               f"{witness.get('return_code')!r}, expected 0")
            if witness.get("invocation_status") != "INVOCATION_COMPLETED":
                reasons.append("the evidence ledger records invocation_status "
                               f"{witness.get('invocation_status')!r}, "
                               "expected 'INVOCATION_COMPLETED'")

    # The evaluator's own validator, on its own terms: same function the pipeline calls, which
    # resolves bundle members exactly as the run did and requires the checksum and the
    # environment identity.
    validator = em.completion_validator(model_id, checkpoint_id, protocol_id,
                                        _regression_pin(model_id), config)
    try:
        proof_ok, proof_reason, _ = validator({"run_id": run_id})
    except Exception as error:                     # a validator that cannot answer is not a pass
        proof_ok, proof_reason = False, f"completion proof validation failed: {error}"
    if not proof_ok:
        reasons.append(f"completion proof rejected: {proof_reason}")

    metric_reasons, reference_outcome = metric_evidence_reasons(
        model_id, checkpoint_id, protocol_id, report)
    reasons.extend(metric_reasons)

    # Two different questions were being answered by one field. "Did this run execute and leave
    # a corroborated artefact?" is what the queue must know before publishing `completed` — a
    # synthetic self-test satisfies it. "Is this run admissible as evidence about the paper?" is
    # what the launcher and the release report must know — a synthetic self-test can never
    # satisfy it. Collapsing them made an honestly labelled self-test look like a failed
    # execution, and made the launcher's success claim depend on a question it was not asking.
    evidence_reasons = _evidence_kind_reasons(report)
    corroborated = not reasons and not metric_reasons
    is_evidence = corroborated and not evidence_reasons
    return {
        # Kept for the existing consumers that only need "is this a corroborated run": the
        # launcher already refuses to print a success claim unless this is true, and the queue
        # publishes `completed` on it.
        "verified": corroborated,
        # The stricter, separate answer, so nothing has to infer it from the weaker one.
        "is_research_evidence": is_evidence,
        "reasons": reasons,
        "evidence_reasons": evidence_reasons,
        "report_path": str(report_path) if report_path.is_file() else None,
        # The report's own identity, not a restatement of the request: the requested identity has
        # already been checked against it above, and a reader needs the commit the run came from
        # and the runtime it ran on to judge comparability.
        "identity": em.report_identity(report),
        # The execution facts a reader must see stated, not assumed. Deliberately a separate
        # object: `identity` stays exactly the comparability shape `report_identity` defines,
        # which consumers compare by value. The aggregate report carries the invocation status;
        # the return code is recorded by the evidence ledger, the run's corroborating witness.
        "execution": {
            "execution_status": (report or {}).get("execution_status"),
            "invocation_status": ((report or {}).get("invocation_status")
                                  or (witness or {}).get("invocation_status")),
            "return_code": (witness or {}).get("return_code"),
        },
        "metrics": {"category": (report or {}).get("category_metrics") or {},
                    "observed": (report or {}).get("observed_metrics") or {}},
        "gate": (report or {}).get("benchmark_gate") or {},
        # The registry's own outcome for this model, not a restatement of the
        # reasons: a consumer distinguishing "no publishable comparison exists"
        # from "verification failed" reads this.
        "reference_outcome": reference_outcome,
    }

def _observed_recall(value: object) -> float | None:
    """A recordable recall, or None: the report's own numbers must be usable as evidence."""
    return float(value) if _is_recall(value) else None


def _category_reasons(categories: object, *, required: bool = True) -> list[str]:
    """Three complete FashionIQ categories for LIMN, or validated carried categories for others."""
    reasons: list[str] = []
    if not isinstance(categories, dict) or not categories:
        if required:
            return ["the report carries no category metrics for limn"]
        return []
    target = LIMN_CATEGORIES if required else list(categories.keys())
    for category in target:
        metrics = categories.get(category)
        if not isinstance(metrics, dict):
            if required:
                reasons.append(f"category {category} is missing from the report")
            continue
        for key in ("r10", "r50"):
            if required or key in metrics:
                value = metrics.get(key)
                if not _is_recall(value):
                    reasons.append(f"category {category} {key} is not a finite Recall "
                                   f"in [0, 100]: {value!r}")
    return reasons


def _macro_reasons(categories: dict, observed: object) -> list[str]:
    """The report's macro must equal the mean of its own categories.

    The arithmetic is the one the replay defines (``workbench/replay/common.py::macro_aggregate``):
    each macro Recall is the unweighted mean of the three categories, and the macro mean is the
    mean of the two macro Recalls. Re-deriving it from the categories -- rather than trusting the
    aggregate the report states -- is what makes an internally contradictory report refuse.
    """
    if not isinstance(observed, dict):
        return ["the report records no macro aggregate to check against its categories"]
    means = {key: sum(float(categories[category][key]) for category in LIMN_CATEGORIES)
            / len(LIMN_CATEGORIES) for key in ("r10", "r50")}
    means["mean"] = (means["r10"] + means["r50"]) / 2
    reasons = []
    for key in ("r10", "r50", "mean"):
        value = _observed_recall(observed.get(key))
        if value is None:
            reasons.append(f"the report's macro {key} is not a finite Recall in [0, 100]: "
                           f"{observed.get(key)!r}")
            continue
        if abs(value - means[key]) > MACRO_TOLERANCE_PP:
            reasons.append(f"the report's macro {key} ({value!r}) is inconsistent with its "
                           f"categories (their mean is {means[key]!r}, "
                           f"tolerance {MACRO_TOLERANCE_PP} pp)")
    return reasons


def _evidence_kind_reasons(report: dict | None) -> list[str]:
    """A synthetic or non-experiment report is a self-test, never evidence about reproduction.

    The evaluator records its synthetic declaration as ``is_synthetic``
    (``evaluate_models.py:1560``); ``data_kind`` is the vocabulary the gate itself refuses on
    (``paper_gate.py:311``). Either one being present and non-experimental is disqualifying,
    because both are written by the run about itself and a self-test can print any numbers.
    """
    report = report or {}
    if report.get("is_synthetic"):
        return ["the report declares itself synthetic (is_synthetic), so it is a pipeline "
                "self-test and not evidence about reproduction"]
    data_kind = report.get("data_kind")
    if data_kind is not None and str(data_kind) != "experiment":
        return [f"the report's data_kind is {data_kind!r}, which is not a real experiment"]
    return []


def _has_usable_metric(report: dict | None) -> bool:
    """Whether the report carries at least one finite Recall in [0, 100]."""
    if not isinstance(report, dict):
        return False
    observed = report.get("observed_metrics")
    if isinstance(observed, dict) and any(_is_recall(v) for v in observed.values()):
        return True
    categories = report.get("category_metrics")
    if isinstance(categories, dict):
        for metrics in categories.values():
            if isinstance(metrics, dict) and any(_is_recall(v) for v in metrics.values()):
                return True
    return False


def _extraction_reasons(report: dict | None) -> list[str]:
    """Metric extraction must actually have been performed.

    A registry-declared synthetic evaluator writes its metrics under its own declared status:
    the extraction really happened, the self-test nature of the run is a separate question asked
    by `_evidence_kind_reasons`. Accepting only the real-experiment status made a completed
    self-test look as though nothing had been read from its output at all.
    """
    report = report or {}
    reasons: list[str] = []
    if report.get("metric_extraction_available") is not True:
        reasons.append("the report records metric_extraction_available="
                       f"{report.get('metric_extraction_available')!r}: no metric extraction "
                       "was performed for this run")
    status = report.get("extraction_status")
    if status not in (EXTRACTION_COMPLETED, EXTRACTION_SYNTHETIC):
        reasons.append(f"the report's extraction_status is {status!r}, not "
                       f"{EXTRACTION_COMPLETED!r}: the run's metrics were not extracted")
    if not reasons and not _has_usable_metric(report):
        reasons.append("the report claims metric extraction succeeded but carries no usable metrics")
    return reasons


def _limn_reasons(report: dict | None) -> list[str]:
    """The LIMN completion contract: extraction happened, three categories, consistent macro."""
    report = report or {}
    reasons: list[str] = _extraction_reasons(report)
    categories = report.get("category_metrics")
    category_reasons = _category_reasons(categories)
    reasons.extend(category_reasons)
    # The macro can only be re-derived when every category it is defined over is usable; a
    # missing category is already named, and comparing against a partial mean would invent one.
    # When no extraction happened there is no aggregate to re-derive either, and saying so twice
    # buries the reason that matters.
    if (not category_reasons and isinstance(categories, dict)
            and report.get("metric_extraction_available") is True):
        reasons.extend(_macro_reasons(categories, report.get("observed_metrics")))
    return reasons


def _condition(model_id: str, checkpoint_id: str) -> tuple[int | None, str | None]:
    """The declared training-noise condition of the checkpoint this run evaluated."""
    try:
        checkpoint = checkpoint_by_id(model_by_id(model_id), checkpoint_id)
    except (KeyError, ValueError, OSError, TypeError):
        return None, None
    noise = checkpoint.get("checkpoint_training_noise_pct")
    return (int(noise) if isinstance(noise, int) else None), str(checkpoint.get("bundle_id") or
                                                                checkpoint_id)


def _publishes_a_metric(row: benchmarks.BenchmarkReference) -> bool:
    """Whether one committed row prints any metric a run could be compared against."""
    return any(row.value_for(key) is not None for key in paper_gate.METRIC_KEYS)


def _committed_outcome(model_id: str, checkpoint_training_noise_pct: int | None,
                       protocol_id: str) -> tuple[benchmarks.ReferenceSelection | None, str | None]:
    """The registry's own answer for this model, or `(None, reason)` if it could not be read."""
    try:
        return benchmarks.select_reference_outcome(
            model_id, checkpoint_training_noise_pct=checkpoint_training_noise_pct,
            protocol_id=protocol_id), None
    except (FileNotFoundError, ValueError, OSError) as error:
        return None, f"{model_id}: the benchmark reference registry could not be read: {error}"


def _non_limn_reasons(model_id: str, checkpoint_id: str, protocol_id: str,
                      report: dict | None) -> tuple[list[str], str]:
    """Every other registered model, judged against its own committed reference.

    Two things are required, and they are different things. First the *registry* must hold a
    comparable row: a verified reference publishing at least one metric under the condition this
    run declares. Second the *run* must carry that side of the comparison -- the published
    metrics present in ``observed_metrics`` as finite Recalls in [0, 100] -- on top of the
    contract every report owes whatever it claims: extraction was performed and succeeded, its
    categories are usable, and its macro equals the mean of its own categories. A run that prints
    a perfect number cannot cover for a report whose own arithmetic does not add up, and a run
    whose metrics were never extracted has no numbers to compare at all.

    Returns ``(reasons, outcome)``. When the registry has no comparable row for this model, the
    reasons name the registry's own outcome rather than inventing one: a model whose paper never
    published the configuration it evaluated has no publishable comparison, permanently, and is
    reported that way -- not as a run whose evidence is wrong, and not as a comparison. An
    ambiguous registry (two verified rows binding one condition) is refused with the ambiguity
    itself and the competing rows named, so the operator knows the registry is what to resolve.

    A measured *deviation* is deliberately not a refusal here. It is a research finding about a
    real execution, and the recipe for it is already on the report as the gate's own
    ``ALERT_METRIC_DEVIATION``; republishing a finding as a failed invocation would say the run
    did not happen when it did. What this refuses is a run that cannot be compared at all.
    """
    report = report or {}
    noise, checkpoint_label = _condition(model_id, checkpoint_id)
    selection, registry_error = _committed_outcome(model_id, noise, protocol_id)
    if selection is None:
        return [registry_error or f"{model_id}: the benchmark reference registry could not be read"], benchmarks.SELECTION_UNREGISTERED
    reference, outcome = selection.reference, selection.outcome
    if reference is not None and _publishes_a_metric(reference):
        published = [key for key in paper_gate.METRIC_KEYS if reference.value_for(key) is not None]
        observed = report.get("observed_metrics")
        if not isinstance(observed, dict) or not any(_is_recall(observed.get(key))
                                                    for key in published):
            return [f"{model_id}: reference {reference.reference_id} publishes "
                    f"{', '.join(published)} but the report carries none of them as a finite "
                    "Recall in observed_metrics, so nothing was compared"], outcome
        reasons = list(_extraction_reasons(report))
        reasons.extend(f"the report's {key} is not a finite Recall in [0, 100]: {observed[key]!r}"
                       for key in published
                       if key in observed and not _is_recall(observed[key]))
        categories = report.get("category_metrics")
        if not reasons and isinstance(categories, dict) and categories:
            category_reasons = _category_reasons(categories, required=False)
            reasons.extend(category_reasons)
            # The macro can only be re-derived from categories the report actually carries; a
            # partial table would otherwise invent a mean the report never claimed.
            if not category_reasons and all(category in categories for category in LIMN_CATEGORIES):
                reasons.extend(_macro_reasons(categories, observed))
        return reasons, outcome

    # A row that matches this run's own condition and prints a metric makes the "nothing
    # publishes under this condition" sentence false, so it is not written -- the selection
    # outcome above already says what the trouble is. A row that matched nothing is not evidence
    # about this run's condition, which is why this is asked of `matched`, not of `candidates`.
    matched_publishes = any(_publishes_a_metric(row) for row in selection.matched)
    if outcome == benchmarks.SELECTION_NO_PUBLISHED_COMPARISON:
        # Whether this is permanent depends on the model, not on the outcome: a variant whose
        # authors never published a FashionIQ number has nothing to compare against, forever,
        # while a model whose paper does print numbers is missing a committed row that can be
        # added. The registry says which, so the report says which.
        permanence = ("a permanent property of what its author published"
                      if selection.author_published_no_reference
                      else "a gap in the committed reference table, not a property of the run")
        tail = (f"so a run of {model_id} is reported as having no publishable comparison, "
                f"{permanence} -- not as a failure of this run and not as a comparison")
    elif matched_publishes:
        tail = ("so this run cannot carry the published-vs-reproduced comparison its protocol "
                "would need until the registry is resolved as named above")
    else:
        tail = (f"no verified committed reference publishes a metric for checkpoint "
                f"{checkpoint_label} under protocol {protocol_id}, so this run cannot carry the "
                "published-vs-reproduced comparison its protocol would need")
    return [f"{selection.reason}; {tail}"], outcome

def metric_evidence_reasons(model_id: str, checkpoint_id: str, protocol_id: str,
                            report: dict | None) -> tuple[list[str], str | None]:
    """Why this run's metrics are not usable evidence, and the registry outcome behind it.

    LIMN gets the contract its own evaluator can actually produce (a completed extraction, three
    categories, a consistent macro); every other registered model gets the contract its own
    committed reference supports, or an explicit refusal named by the registry's own selection
    outcome. The registry decides which, so no model is exempt by name. The outcome is returned
    beside the reasons so a caller can tell "this model has no publishable comparison" apart from
    "this run failed verification" without reading English.
    """
    report = report or {}
    outcome: str | None = None
    if model_id == "limn":
        return _limn_reasons(report), None
    reference_reasons, outcome = _non_limn_reasons(model_id, checkpoint_id, protocol_id, report)
    if outcome == benchmarks.SELECTION_UNREGISTERED:
        # A model with no row in the committed reference table cannot be held to a comparison
        # that will never exist. It is still held to the contracts that do apply to every run —
        # a completed metric extraction and finite, in-range metrics — so "no published
        # comparison" never becomes a licence to certify unextracted or impossible numbers.
        # LIMN's category/macro contract is deliberately not imposed: this model's own protocol
        # does not publish per-category figures, and inventing that requirement would refuse
        # honest runs of an evaluator the workbench legitimately supports.
        honest = list(_extraction_reasons(report))
        observed = report.get("observed_metrics") or {}
        honest.extend(f"the report's {key} is not a finite Recall in [0, 100]: {value!r}"
                      for key, value in observed.items()
                      if key in paper_gate.METRIC_KEYS and not _is_recall(value))
        categories = report.get("category_metrics")
        if not honest and isinstance(categories, dict) and categories:
            honest.extend(_category_reasons(categories, required=False))
            if not honest and all(category in categories for category in LIMN_CATEGORIES):
                honest.extend(_macro_reasons(categories, observed))
        return honest, outcome
    return reference_reasons, outcome

def parser_for() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="invocation run identity to verify")
    parser.add_argument("--model", required=True, help="registered model id")
    parser.add_argument("--checkpoint", required=True, help="checkpoint (or bundle) id")
    parser.add_argument("--protocol", required=True, help="evaluation protocol id")
    parser.add_argument("--json", action="store_true",
                        help="print the machine-readable verdict and nothing else")
    return parser

def main(argv: list[str] | None = None) -> int:
    args = parser_for().parse_args(argv)
    verdict = verify(args.run_id, args.model, args.checkpoint, args.protocol)
    if args.json:
        print(json.dumps(verdict))
    elif verdict["verified"]:
        print(f"[OK] run {args.run_id} is corroborated: {verdict['report_path']}")
    else:
        for reason in verdict["reasons"]:
            print(f"[FAIL] {reason}", file=sys.stderr)
    return 0 if verdict["verified"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
