"""The evaluator output a sandbox run records, in the format the parser actually reads.

A fixture that writes the digits it wants into the *report* and leaves `stdout.log` as prose is no
longer enough: the verifier re-parses the run's own digest-bound output with the same parser the
evaluator used, and compares it against the numbers the report claims. That check exists because a
report can attest to its own extraction -- `metric_extraction_available` and `extraction_status` are
fields the run writes about itself -- and the output is the one piece of the run the report cannot
answer for.

So a fixture has to emit what an evaluator emits. This module is that one place, so the format lives
in a single file rather than in every suite's copy of it.
"""
from __future__ import annotations

import json
from pathlib import Path

LIMN_CATEGORIES = ("dress", "shirt", "toptee")

def limn_aggregate_line(categories: dict[str, dict[str, float]],
                        return_code: int = 0) -> str:
    """One LIMN structured-JSON line, exactly as `limn.py` prints it.

    `aggregate.complete` is the evaluator's own statement that all three categories were measured,
    and the parser refuses a line without it -- as it should, since the macro only exists for the
    complete benchmark.
    """
    r10 = sum(values["r10"] for values in categories.values()) / len(LIMN_CATEGORIES)
    r50 = sum(values["r50"] for values in categories.values()) / len(LIMN_CATEGORIES)
    payload = {
        "model_id": "limn",
        "return_code": return_code,
        "categories": [{"category": name, "r1": values.get("r1"),
                        "r10": values["r10"], "r50": values["r50"]}
                       for name, values in categories.items()],
        "aggregate": {"macro_r10": round(r10, 4), "macro_r50": round(r50, 4),
                      "macro_mean": round((r10 + r50) / 2, 4), "complete": True},
    }
    return json.dumps(payload)

def write_run_output(run_directory: Path, categories: dict[str, dict[str, float]],
                     *, prefix: str = "evaluating\n",
                     return_code: int = 0) -> tuple[str, str, str]:
    """Write `stdout.log`/`stderr.log` for a run and return the logs and their evidence digest.

    The digest is the one the completion proof must carry: it is computed over the two logs with
    `json.dumps(..., sort_keys=True)` exactly as the runner computes it, so the fixture's proof and
    report stay consistent with the bytes on disk.
    """
    import hashlib

    run_directory.mkdir(parents=True, exist_ok=True)
    stdout = prefix + limn_aggregate_line(categories, return_code=return_code) + "\n"
    (run_directory / "stdout.log").write_text(stdout, encoding="utf-8")
    (run_directory / "stderr.log").write_text("", encoding="utf-8")
    logs = {"stdout": stdout, "stderr": ""}
    digest = hashlib.sha256(json.dumps(logs, sort_keys=True).encode("utf-8")).hexdigest()
    return stdout, "", digest
