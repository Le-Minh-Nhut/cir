"""R6 audit: feed malformed evidence to the diagnostic loaders on a SCRATCH COPY.

Nothing under workbench/docs/reviews/20261109-native15-composition-limn is touched: the whole
directory is copied to /tmp and the config's CIR_REPO_ROOT is pointed at the copy. The loaders take
`config`, so this exercises the same code the HTTP endpoints call.

For each mutation: does the SUMMARY endpoint (load_diagnostic) accept it, and does the CANDIDATES
endpoint (load_candidates) accept it? "accepted" = returned a payload with no DiagnosticEvidenceError.
"""
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

from workbench.backend.limn_diagnostic import (DiagnosticEvidenceError, load_candidates,
                                               load_diagnostic)

SRC = Path("workbench/docs/reviews/20261109-native15-composition-limn")
WORK = Path(tempfile.mkdtemp(prefix="r6-scratch-"))


def fresh():
    """A clean scratch copy; returns (config, evidence_dir)."""
    root = Path(tempfile.mkdtemp(dir=WORK))
    dst = root / "workbench" / "docs" / "reviews" / "20261109-native15-composition-limn"
    dst.parent.mkdir(parents=True)
    shutil.copytree(SRC, dst)
    return SimpleNamespace(CIR_REPO_ROOT=str(root)), dst


def probe(label, cfg, summary_fn=None, cand_fn=None):
    s = c = "-"
    if summary_fn:
        try:
            load_diagnostic(config=cfg)
            s = "ACCEPTED"
        except DiagnosticEvidenceError as e:
            s = f"REFUSED ({str(e)[:90]})"
    if cand_fn:
        try:
            load_candidates(*cand_fn[0], depth=cand_fn[1], config=cfg)
            c = "ACCEPTED"
        except DiagnosticEvidenceError as e:
            c = f"REFUSED ({str(e)[:90]})"
    print(f"{label:52s} summary={s:22s} candidates={c}")


Q = "7:3038"
SCOPE, COND = "shirt", "multimodal"

# 0. control
cfg, d = fresh()
probe("0 control (unmodified copy)", cfg, True, ((Q, SCOPE, COND), 100))

# 1. empty payload: queries {}
cfg, d = fresh()
p = d / "limn_dual_scope.json"; j = json.loads(p.read_text()); j["queries"] = {}
p.write_text(json.dumps(j))
probe("1 empty payload (queries={})", cfg, True, ((Q, SCOPE, COND), 100))

# 2. mismatched checkpoint: summary says shirt uses toptee's sha
cfg, d = fresh()
p = d / "limn_dual_scope.json"; j = json.loads(p.read_text())
shirt = j["checkpoints"]["shirt"]["sha256"]
j["checkpoints"]["shirt"]["sha256"] = j["checkpoints"]["toptee"]["sha256"]
j["checkpoints"]["shirt"]["checkpoint_id"] = "base_iter0_toptee"
p.write_text(json.dumps(j))
probe("2 summary checkpoint id/sha swapped (shirt<-toptee)", cfg, True, ((Q, SCOPE, COND), 100))

# 3. export checkpoint sha disagrees with the summary's declared checkpoint
cfg, d = fresh()
e = d / f"limn_dual_scope.{SCOPE}.{COND}.jsonl"
rows = [json.loads(l) for l in e.read_text().splitlines() if l.strip()]
for r in rows:
    r["checkpoint_sha256"] = "0" * 64
e.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
probe("3 export checkpoint_sha256 = 0*64 (wrong checkpoint)", cfg, True, ((Q, SCOPE, COND), 100))

# 4. wrong qid / scope / condition (already refused by the endpoint) - recorded for completeness
cfg, d = fresh()
probe("4 wrong qid", cfg, None, (("7:9999", SCOPE, COND), 100))
probe("4 wrong scope", cfg, None, ((Q, "dress", COND), 100))
probe("4 wrong condition", cfg, None, ((Q, SCOPE, "bogus"), 100))

# 5. rank/score contradiction: export target_rank moved, target_score unchanged
cfg, d = fresh()
e = d / f"limn_dual_scope.{SCOPE}.{COND}.jsonl"
rows = [json.loads(l) for l in e.read_text().splitlines() if l.strip()]
for r in rows:
    if r["qid"] == Q:
        r["target_rank"] = 77           # rank 3 -> 77, window untouched
e.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
probe("5 rank/score contradiction (rank 3->77, window kept)", cfg, True, ((Q, SCOPE, COND), 100))

# 5b. rank that contradicts its own is_target row in the window
cfg, d = fresh()
e = d / f"limn_dual_scope.{SCOPE}.{COND}.jsonl"
rows = [json.loads(l) for l in e.read_text().splitlines() if l.strip()]
for r in rows:
    if r["qid"] == Q:
        r["target_rank"] = 1            # window rank1 is NOT the target
e.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
probe("5b rank=1 while window rank1 is_target=false", cfg, True, ((Q, SCOPE, COND), 100))

# 6. duplicate ID: same qid twice in one export, second row different rank
cfg, d = fresh()
e = d / f"limn_dual_scope.{SCOPE}.{COND}.jsonl"
rows = [json.loads(l) for l in e.read_text().splitlines() if l.strip()]
dup = json.loads(json.dumps(rows[0])); dup["target_rank"] = 999
e.write_text("\n".join(json.dumps(r) for r in rows + [dup]) + "\n")
try:
    out = load_candidates(Q, SCOPE, COND, depth=100, config=cfg)
    print(f"{'6 duplicate qid row (2nd says rank 999)':52s} summary=-                      "
          f"candidates=ACCEPTED returning rank={out['target_rank']} (first match wins, silently)")
except DiagnosticEvidenceError as err:
    print(f"{'6 duplicate qid row':52s} candidates=REFUSED ({err})")

# 7. broken depth: depth_stored=100 but only 3 candidates stored
cfg, d = fresh()
e = d / f"limn_dual_scope.{SCOPE}.{COND}.jsonl"
rows = [json.loads(l) for l in e.read_text().splitlines() if l.strip()]
for r in rows:
    r["candidates"] = r["candidates"][:3]
e.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
probe("7 broken depth (depth=100, 3 candidates stored)", cfg, True, ((Q, SCOPE, COND), 100))

# 7b. depth requested > stored, and depth=0 / negative
cfg, d = fresh()
for dep in (0, -5, 500):
    try:
        out = load_candidates(Q, SCOPE, COND, depth=dep, config=cfg)
        print(f"{'7b depth=' + str(dep):52s} candidates=ACCEPTED n={len(out['candidates'])} "
              f"depth_stored={out['depth_stored']}")
    except DiagnosticEvidenceError as err:
        print(f"{'7b depth=' + str(dep):52s} candidates=REFUSED ({err})")

# 8. one export file deleted -> summary refuses (documented behaviour, verify)
cfg, d = fresh()
(d / f"limn_dual_scope.{SCOPE}.{COND}.jsonl").unlink()
probe("8 one per-scope export deleted", cfg, True, None)

# 9. export exists but is EMPTY (0 bytes) -> summary only stats it
cfg, d = fresh()
(d / f"limn_dual_scope.{SCOPE}.{COND}.jsonl").write_text("")
probe("9 per-scope export truncated to 0 bytes", cfg, True, ((Q, SCOPE, COND), 100))

# 10. export is unparseable JSON -> summary only stats it
cfg, d = fresh()
(d / f"limn_dual_scope.{SCOPE}.{COND}.jsonl").write_text("{not json\n")
probe("10 per-scope export is unparseable JSON", cfg, True, ((Q, SCOPE, COND), 100))

print()
print("scratch root:", WORK)
