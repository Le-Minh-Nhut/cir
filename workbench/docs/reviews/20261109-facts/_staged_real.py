import os, json
from pathlib import Path
from workbench.backend.registry import load_registry, checkpoint_path

os.chdir('/home/heheboiz/data/cir')

root = None
for line in open('workbench/config/workbench.env', encoding='utf-8'):
    if line.startswith('WORKBENCH_CHECKPOINT_ROOT='):
        root = Path(line.split('=', 1)[1].strip())
print("WORKBENCH_CHECKPOINT_ROOT =", root)
print("exists:", root.is_dir())

reg = load_registry()
models = reg['models']

def rule(model_id, ck, configured):
    conventional = configured / model_id
    rel = ck.get('checkpoint_base_relative')
    base = conventional if (conventional.exists() or not rel) else configured / rel
    if base == conventional:
        return 'conventional'
    return 'checkpoint_base_relative'

def table(configured, label):
    print()
    print("=" * 110)
    print("ROOT (%s): %s" % (label, configured))
    print("exists:", configured.is_dir())
    print("%-22s %-28s %-10s %-8s %s" % ("model_id", "checkpoint_id", "rule", "exists", "resolved path"))
    n = staged = by_conv = by_rel = 0
    rows = []
    for m in models:
        for ck in (m.get('checkpoint_variants') or []):
            n += 1
            p = checkpoint_path(m['model_id'], ck, root=configured)
            ex = p.exists()
            r = rule(m['model_id'], ck, configured)
            if ex:
                staged += 1
                by_conv += (r == 'conventional')
                by_rel += (r == 'checkpoint_base_relative')
            rows.append((m['model_id'], ck.get('checkpoint_id'), os.path.relpath(p, configured), ex, r))
            print("%-22s %-28s %-10s %-8s %s" % (
                m['model_id'], ck.get('checkpoint_id'), r, ex, os.path.relpath(p, configured)))
    print()
    print("%s: variants=%d staged=%d not_staged=%d | staged-by-conventional=%d staged-by-base_relative=%d"
          % (label, n, staged, n - staged, by_conv, by_rel))
    return rows, n, staged, by_conv, by_rel

r1 = table(root, "CIRCUS cohort root")
native = root.parent.parent / 'checkpoints'   # sibling of cohort/, not child
r2 = table(native, "native cohort sibling")

print()
print("=" * 110)
print("Per-variant: which root resolves it")
print("%-22s %-28s %-14s %-14s %-10s %s" % ("model_id", "checkpoint_id", "CIRCUS", "native", "agree", "note"))
disagree = 0
for a, b in zip(r1[0], r2[0]):
    if a[3] != b[3]:
        disagree += 1
    print("%-22s %-28s %-14s %-14s %-10s %s" % (
        a[0], a[1], a[3], b[3], a[3] == b[3], ("" if a[4] == b[4] else "rule %s vs %s" % (a[4], b[4]))))
print("rows where the two roots disagree on exists():", disagree)
print("union staged (resolvable under at least one root):",
      sum(1 for a, b in zip(r1[0], r2[0]) if a[3] or b[3]))
