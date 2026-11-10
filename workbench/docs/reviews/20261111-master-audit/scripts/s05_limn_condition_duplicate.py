"""S-05: does the production LIMN run feed the SAME image+caption under all three labels?

The old harness bug (`d3b_limn_aggregate_scan.py:50`) calls `adapter.encode_queries(paths, texts, cats)`
with the ORIGINAL path and the ORIGINAL caption inside the `for cond in CONDS` loop -- no call to
`_stage_query_path`/`adapt_query_inputs` anywhere in that file. All three conditions therefore score
identical inputs, which is why the scan reported changed=0 for all three.

This script answers the PRODUCTION question from the production artifacts, not from the harness:
read the per-condition ranks of the published LIMN run and check whether the three conditions are
identical. Identical ranks under three labels would be the harness bug propagated; different ranks
are evidence the production path really applied the intervention (it goes through `_stage_query_path`
at runner.py:668).
"""
import json
import sys
from pathlib import Path

RUN = Path("/home/heheboiz/data/cir/workbench/artifacts/composition/runs/"
           "10173dc45dea8c90fd2a284b/ranks/limn_base_iter0_all_categories.json")
payload = json.loads(RUN.read_text(encoding="utf-8"))
ranks = payload["ranks"]
conds = sorted(ranks)
qids = sorted(ranks[conds[0]])
print("run:", RUN)
print("conditions:", conds)
print("qids per condition:", {c: len(ranks[c]) for c in conds})

pairs = [("multimodal", "text_only_black_image"), ("multimodal", "image_only_empty_text"),
         ("text_only_black_image", "image_only_empty_text")]
for a, b in pairs:
    same = sum(1 for q in qids if ranks[a].get(q) == ranks[b].get(q))
    print(f"{a} vs {b}: identical_ranks={same}/{len(qids)} "
          f"identical_fraction={same/len(qids):.4f}")

# The signature of the harness bug: ALL THREE identical. Anything else is not that bug.
all_same = all(ranks[a].get(q) == ranks[b].get(q) == ranks[conds[2]].get(q) for q in qids)
print("all_three_conditions_identical:", all_same)
print("verdict:", "HARNESS BUG PROPAGATED" if all_same
      else "NOT propagated: the three conditions carry different ranks")

# Do the production frames even mention the ablation conclusion?
for path in Path("/home/heheboiz/data/cir/workbench/docs/reviews").rglob("*.json"):
    if path.name.startswith("d3b_limn"):
        print("harness artifact present:", path)
