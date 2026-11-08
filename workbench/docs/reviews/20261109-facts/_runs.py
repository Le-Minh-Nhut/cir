import os, glob, subprocess, json

os.chdir('/home/heheboiz/data/cir')
runs = sorted(glob.glob('workbench/artifacts/composition/runs/*'))
out = []
for r in runs:
    out.append("== %s" % r)
    ranks = os.path.join(r, 'ranks')
    out.append("  ranks dir exists: %s" % os.path.isdir(ranks))
    if os.path.isdir(ranks):
        for f in sorted(os.listdir(ranks)):
            fp = os.path.join(ranks, f)
            out.append("    ranks/%-45s %10d bytes" % (f, os.path.getsize(fp)))
    cand = os.path.join(r, 'candidates')
    out.append("  candidates dir exists: %s" % os.path.isdir(cand))
    if os.path.isdir(cand):
        for d in sorted(os.listdir(cand)):
            dp = os.path.join(cand, d)
            if os.path.isdir(dp):
                out.append("    candidates/%s/" % d)
                for f in sorted(os.listdir(dp)):
                    fp = os.path.join(dp, f)
                    if os.path.isfile(fp):
                        n = sum(1 for _ in open(fp, 'rb'))
                        out.append("      %-45s %10d bytes  lines=%d" % (f, os.path.getsize(fp), n))
            else:
                out.append("    candidates/%s  %d bytes" % (d, os.path.getsize(dp)))
    out.append("")
open('workbench/docs/reviews/20261109-facts/_runs.txt', 'w', encoding='utf-8').write("\n".join(out))
print("wrote", len(out), "lines")
