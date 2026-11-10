"""Baseline: pinned third-party provenance from each repo's own provenance.json.

Reports model ids, upstream commit, local_fixes, and a manifest verification
(sha256 of every file listed in file_manifest, compared to the recorded digest).
"""
import hashlib
import json
from pathlib import Path

TP = Path("workbench/third_party")
rows = []
for repo in sorted(p for p in TP.iterdir() if p.is_dir()):
    prov = repo / "provenance.json"
    if not prov.is_file():
        rows.append((repo.name, "NO provenance.json", "", "", "", ""))
        continue
    d = json.loads(prov.read_text(encoding="utf-8"))
    fixes = d.get("local_fixes") or []
    fix_desc = "; ".join(f"{f.get('path')} ({f.get('origin')})" for f in fixes) or "-"
    manifest = d.get("file_manifest") or {}
    bad, missing = [], []
    for rel, want in manifest.items():
        f = repo / rel
        if not f.is_file():
            missing.append(rel)
            continue
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        if h != want:
            bad.append(rel)
    verdict = "MANIFEST_OK" if not bad and not missing else f"bad={len(bad)} missing={len(missing)}"
    rows.append((repo.name, d.get("upstream_commit", "?"), d.get("upstream_commit_date", "?"),
                 ",".join(d.get("model_ids") or []), fix_desc, f"{verdict} ({len(manifest)} files)",
                 ",".join((bad + missing)[:5])))

w = max(len(r[0]) for r in rows)
for r in rows:
    print(f"{r[0]:<{w}} | commit={r[1]} | date={r[2]} | models={r[3]} | local_fixes={r[4]} | {r[5]}")
    if len(r) > 6 and r[6]:
        print(f"{'':<{w}} |   mismatched: {r[6]}")
