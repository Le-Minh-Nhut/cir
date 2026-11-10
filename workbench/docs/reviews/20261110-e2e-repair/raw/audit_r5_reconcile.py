"""R5 audit part 2: cross-check every cell's rank/score against its own top-100 window,
and check the summary's target_did against the release's positive_ids per qid."""
import json
from pathlib import Path

ROOT = Path("workbench/docs/reviews/20261109-native15-composition-limn")
S = json.loads((ROOT / "limn_dual_scope.json").read_text(encoding="utf-8"))

print("== A. summary target_did vs release positive_ids ==")
for qid, rec in S["queries"].items():
    rel = rec["release"]
    print(f"qid={qid} positive_ids={rel['positive_ids']} target_gallery_id={rec['target_gallery_id']}")
    for scope, conds in rec["scopes"].items():
        for cond, cell in conds.items():
            print(f"   {scope:7s} {cond:22s} target_did={cell.get('target_did')!r} "
                  f"MATCH={cell.get('target_did') in rel['positive_ids']}")

print()
print("== B. per-cell rank/score vs the stored window (is_target row) ==")
for scope in ("shirt", "toptee"):
    for cond in ("multimodal", "text_only_black_image", "image_only_empty_text"):
        p = ROOT / f"limn_dual_scope.{scope}.{cond}.jsonl"
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            qid = r["qid"]
            cands = r["candidates"]
            flags = [c for c in cands if c.get("is_target")]
            in_window = r["target_rank"] <= len(cands)
            detail = "n/a (rank beyond stored depth)"
            ok = "n/a"
            if in_window:
                c = cands[r["target_rank"] - 1]
                same_score = abs(float(c["score"]) - float(r["target_score"])) < 1e-6
                detail = (f"rank{r['target_rank']} gid={c['gallery_id']} is_target={c.get('is_target')} "
                          f"score={float(c['score']):.6f} target_score={float(r['target_score']):.6f}")
                ok = "PASS" if (c.get("is_target") and same_score) else "FAIL"
            print(f"{scope:7s} {cond:22s} qid={qid} rank={r['target_rank']:5d} "
                  f"n_is_target_flags={len(flags)} window_check={ok}")
            print(f"        {detail}")
            # score monotonicity of the stored window
            sc = [float(c["score"]) for c in cands]
            print(f"        window_sorted_desc={all(sc[i] >= sc[i+1] for i in range(len(sc)-1))} "
                  f"ranks_1..n={cands[0]['rank']}..{cands[-1]['rank']}")
