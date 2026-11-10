"""Failing-first probes for the LIMN diagnostic evidence validator (workbench/backend/limn_diagnostic.py).

Each case copies the real evidence tree into a temp repo root, applies ONE mutation, and asks the
production reader what it says. A mutation that the reader still calls COMPLETE is a guard gap.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPO = Path("/home/heheboiz/data/cir")
REVIEW = REPO / "workbench" / "docs" / "reviews" / "20261109-native15-composition-limn"
sys.path.insert(0, str(REPO))

from workbench.backend import limn_diagnostic  # noqa: E402


def config_for(root: Path):
    return SimpleNamespace(CIR_REPO_ROOT=str(root), FASHIONIQ_ROOT=str(REPO / "data/FashionIQ"))


def stage(tmp: Path) -> Path:
    target = tmp.joinpath(*limn_diagnostic.EVIDENCE_RELATIVE)
    target.mkdir(parents=True)
    for path in REVIEW.glob("limn_dual_scope*"):
        if path.is_file():
            shutil.copy2(path, target / path.name)
    return target


def read(evidence: Path) -> dict:
    limn_diagnostic._RECORD_CACHE.clear()
    record = limn_diagnostic.load_diagnostic(config=config_for(evidence.parents[3]))
    return {"status": record["status"], "complete": record["complete"],
            "defects": sorted({item["code"] for item in record["defects"]})}


def mutate_truncate_window(evidence: Path) -> None:
    """Keep depth=100 but store ONE candidate: a window that cannot be reproduced from the file."""
    path = evidence / "limn_dual_scope.shirt.multimodal.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows[0]["candidates"] = rows[0]["candidates"][:1]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def mutate_duplicate_qid(evidence: Path) -> None:
    """A second row for the same qid, with a different rank: which row does the reader believe?"""
    path = evidence / "limn_dual_scope.toptee.multimodal.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    clone = json.loads(json.dumps(rows[0]))
    clone["target_rank"] = clone["target_rank"] + 1000
    clone["target_score"] = clone["target_score"] - 0.5
    rows.append(clone)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def main() -> int:
    report = {}
    for name, mutate in (("baseline", None), ("truncated_window", mutate_truncate_window),
                         ("duplicate_qid_row", mutate_duplicate_qid)):
        with tempfile.TemporaryDirectory() as tmp:
            evidence = stage(Path(tmp))
            if mutate is not None:
                mutate(evidence)
            report[name] = read(evidence)
    print("M03VALIDATOR:" + json.dumps(report, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
