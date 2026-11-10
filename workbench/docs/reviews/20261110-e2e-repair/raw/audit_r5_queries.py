"""R5 audit: reconcile qid -> positive DID -> asset -> rank across all 12 cells.

Reads the published dual-scope summary and the six per-scope exports.
No ?q= jq selectors on these artifacts: plain json.load / line-wise json.loads only.
"""
import json
from pathlib import Path

ROOT = Path("workbench/docs/reviews/20261109-native15-composition-limn")
summary = json.loads((ROOT / "limn_dual_scope.json").read_text(encoding="utf-8"))

print("== summary.queries ==")
for qid, rec in summary["queries"].items():
    print(f"qid={qid} release={rec.get('release')} target_gallery_id={rec.get('target_gallery_id')} "
          f"target_position_in_gallery={rec.get('target_position_in_gallery')}")
    for scope, conds in rec["scopes"].items():
        for cond, cell in conds.items():
            print(f"   {scope:7s} {cond:22s} rank={cell.get('target_rank')!s:6s} "
                  f"score={cell.get('target_score')!r} gallery={cell.get('gallery_size_scored')} "
                  f"export={Path(str(cell.get('top100_export') or '')).name}")

print()
print("== per-scope exports: first rows ==")
for scope in ("shirt", "toptee"):
    for cond in ("multimodal", "text_only_black_image", "image_only_empty_text"):
        p = ROOT / f"limn_dual_scope.{scope}.{cond}.jsonl"
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"-- {p.name}: rows={len(rows)}")
        for r in rows:
            c = r.get("candidates") or []
            top3 = [(x.get("rank"), x.get("gallery_id"), round(float(x.get("score", 0.0)), 6)) for x in c[:3]]
            print(f"   qid={r.get('qid')!r} target_rank={r.get('target_rank')} "
                  f"target_score={r.get('target_score')!r} depth={r.get('depth')} "
                  f"gallery_size={r.get('gallery_size')} n_cand={len(c)}")
            print(f"      keys={sorted(r.keys())}")
            print(f"      top3={top3}")
