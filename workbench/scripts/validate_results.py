#!/usr/bin/env python3
"""Validate canonical result JSON artifacts without modifying them."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workbench.backend.index import load_runs
from workbench.backend.operator_config import resolve_config
from workbench.backend.schemas.results import admission, ResultRun
from workbench.replay.publication import verify_run_identity

PROVENANCE_FIELDS = ("upstream_commit", "checkpoint_sha256", "command_digest", "environment_digest")


def _missing_provenance(result: ResultRun) -> list[str]:
    return [field for field in PROVENANCE_FIELDS if not getattr(result.run, field)]


def _report_provenance(runs: list[ResultRun], strict_real: bool, output: Callable[[str], None]) -> bool:
    """Report the ONE admission policy's verdict (F-13).

    Previously this file applied its own provenance rule while `index.py` applied a different one,
    so the same artifact could be "valid" to the validator and "verified" to the index. Both now
    read `schemas.results.admission`, so the two entry points cannot disagree.

    A development fixture is admitted and reported as such; an unverified experiment is admitted but
    never promoted. `--strict-real` makes incomplete experiment provenance an error, which is what a
    release gate wants; without it the state is reported as a warning so the operator can still
    inspect the workspace.
    """
    from workbench.backend.schemas.results import (AWAITING_CORROBORATION, PER_QUERY_UNVERIFIED,
                                               PER_QUERY_VERIFIED, SYNTHETIC_OR_MOCK)

    valid = True
    for result in runs:
        state, reasons = admission(result)
        if state == SYNTHETIC_OR_MOCK:
            continue
        if state == PER_QUERY_UNVERIFIED:
            # "awaiting corroboration" is not a PROVENANCE defect: the artifact may be complete and
            # merely not yet checked by the verifier. --strict-real asks whether the artifact records
            # what its verification needs, so that reason alone must not fail it (review finding:
            # every valid experiment was being BLOCKED).
            provenance_reasons = [r for r in reasons if r != AWAITING_CORROBORATION]
            if not provenance_reasons:
                output(f"[OK] {result.run.run_id}: provenance complete; awaiting verifier corroboration")
                continue
            level = "BLOCKED" if strict_real else "WARN"
            output(f"[{level}] {result.run.run_id}: {'; '.join(provenance_reasons)}")
            valid = valid and not strict_real
        else:
            # A parity state that is not VERIFIED means the artifact's numbers are not established:
            # FAILED is a defect, and PENDING means nobody checked. `--strict-real` is a release
            # gate, so it must reject both -- accepting PENDING there let an unchecked artifact
            # through the gate (found by a verifier). Without the flag both are reported as warnings
            # so an operator can still inspect a workspace in progress.
            unresolved = state != PER_QUERY_VERIFIED
            level = "BLOCKED" if (unresolved and strict_real) else "WARN"
            output(f"[{level}] {result.run.run_id}: {state}: {'; '.join(reasons)}")
            valid = valid and not (unresolved and strict_real)
    return valid


def validate_root(root: Path, strict_real: bool = False, output: Callable[[str], None] = print) -> bool:
    """Validate every result file below root using index loading rules.

    Applies the SAME per-artifact identity check as `validate_file`, so the two entry points cannot
    disagree about one file (F-13: `validate_file` used to refuse a relabelled artifact while
    `validate_root` loaded it through `load_runs` and passed it).
    """
    try:
        runs = load_runs(root)
    except Exception as error:
        output(f"[BLOCKED] {root}: {error}")
        return False
    mismatched = [(result.run.run_id, mismatch) for result in runs
                  for mismatch in verify_run_identity(result)]
    for run_id, mismatch in mismatched:
        output(f"[BLOCKED] {run_id}: {mismatch}")
    valid = _report_provenance(runs, strict_real, output) and not mismatched
    output(f"[OK] validated {len(runs)} runs under {root}")
    return valid


def validate_file(path: Path, strict_real: bool = False, output: Callable[[str], None] = print) -> bool:
    """Validate one result artifact without changing it.

    Schema validity is not identity consistency: `run_id` is derived from the artifact's own
    provenance, so it is recomputed here. An artifact whose identity was edited to another run's
    would otherwise pass every schema check.
    """
    try:
        result = ResultRun.model_validate_json(path.read_text())
    except Exception as error:
        output(f"[BLOCKED] {path}: {error}")
        return False
    if mismatches := verify_run_identity(result):
        for mismatch in mismatches:
            output(f"[BLOCKED] {path}: {mismatch}")
        return False
    valid = _report_provenance([result], strict_real, output)
    output(f"[OK] validated {path}")
    return valid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selectors = parser.add_mutually_exclusive_group(required=True)
    selectors.add_argument("--file", type=Path)
    selectors.add_argument("--root", type=Path)
    selectors.add_argument("--all", action="store_true", help="Validate all canonical result artifacts.")
    parser.add_argument("--strict-real", action="store_true", help="Treat missing real-artifact provenance as errors.")
    args = parser.parse_args(argv)
    if args.file is not None:
        valid = validate_file(args.file, args.strict_real)
    else:
        valid = validate_root(resolve_config().WORKBENCH_RESULTS_ROOT if args.all else args.root, args.strict_real)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
