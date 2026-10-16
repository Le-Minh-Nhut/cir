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
* for LIMN, every FashionIQ category carries a finite Recall@10/@50 inside [0, 100].

Nothing is re-implemented here: the ledger reader, the hasher, the proof validator and the
registry helpers are the ones the evaluator itself uses.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import workbench.scripts.evaluate_models as em
from workbench.backend.operator_config import resolve_config
from workbench.backend.registry import model_by_id

#: Categories the LIMN macro is defined over. A report missing one of them covers a partial
#: benchmark, so its macro is not the paper's macro.
LIMN_CATEGORIES = ("dress", "shirt", "toptee")

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

    ledger = em.read_evidence_ledger(config)
    witness: dict | None = None
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

    if model_id == "limn":
        categories = (report or {}).get("category_metrics")
        if not isinstance(categories, dict) or not categories:
            reasons.append("the report carries no category metrics for limn")
        else:
            for category in LIMN_CATEGORIES:
                metrics = categories.get(category)
                if not isinstance(metrics, dict):
                    reasons.append(f"category {category} is missing from the report")
                    continue
                for key in ("r10", "r50"):
                    value = metrics.get(key)
                    if not _is_recall(value):
                        reasons.append(f"category {category} {key} is not a finite Recall "
                                       f"in [0, 100]: {value!r}")

    return {
        "verified": not reasons,
        "reasons": reasons,
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
    }

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
