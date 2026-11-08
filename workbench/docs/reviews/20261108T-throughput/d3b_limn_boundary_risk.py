import json, math
d=json.load(open('workbench/artifacts/composition/runs/10173dc45dea8c90fd2a284b/ranks/limn_base_iter0_all_categories.json'))
print("LIMN published artifact - boundary proximity (a +-1 rank move only changes R@K if it straddles K)\n")
for cond, ranks in d['ranks'].items():
    scored=[r for r in ranks.values() if r is not None]
    if not scored: continue
    print(f"--- {cond}: {len(scored)} scored")
    for K in (10,50,100):
        at=[r for r in scored if abs(r-K)<=1]
        print(f"    K={K:<4} ranks within +-1 of boundary: {len(at)}   values={sorted(at)[:8]}")
    worst=max(abs(1/math.log2(r+1)-1/math.log2(r+2)) for r in scored)
    print(f"    worst single-query |dndcg| from a +-1 move: {worst:.3e}  (mean impact /6003 = {worst/len(scored):.3e})")
    print()
