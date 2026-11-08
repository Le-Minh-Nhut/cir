import os
os.chdir('/home/heheboiz/data/cir')
out = []
for p in ['workbench/docs/reviews/20261108T212500+0700-sweep/full_suite_c2d8cdeb.txt',
          'workbench/docs/reviews/20261108T225338+0700-sweep/full_suite_1129c58b.txt']:
    lines = open(p, encoding='utf-8').read().split('\n')
    fails = [l for l in lines if l.startswith('FAILED ')]
    out.append("==== %s  FAILED-count=%d" % (p, len(fails)))
    for i, l in enumerate(fails, 1):
        # keep the id only (up to first ' - ')
        out.append("  %2d %s" % (i, l[7:].split(' - ')[0]))
    out.append("")
open('workbench/docs/reviews/20261109-facts/_fails.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('lines', len(out))
