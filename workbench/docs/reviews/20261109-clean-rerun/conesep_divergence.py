"""Why did conesep's rank move for 17 queries? Show the neighbourhood of the changed rows."""
import json
import sys

OLD = ('workbench/artifacts/composition/runs/30f6de423861cc1d6c78c361/'
       'candidates/conesep_fiq_n02/{cond}.jsonl')
NEW = ('workbench/artifacts/composition/runs/a4f786d8af4d46ac2a85ab9c/'
       'candidates/conesep_fiq_n02/{cond}.jsonl')
CHANGED = {
    'multimodal': ['7:3039', '7:3744', '7:5116'],
    'image_only_empty_text': ['7:1521', '7:4552', '7:4681', '7:4937', '7:5118',
                              '7:5131', '7:5419', '7:5673', '7:5680', '7:968'],
    'text_only_black_image': ['7:2495', '7:3254', '7:4513', '7:5844'],
}


def load(path, want):
    out = {}
    with open(path) as stream:
        for line in stream:
            row = json.loads(line)
            if row['qid'] in want:
                out[row['qid']] = row
            if len(out) == len(want):
                break
    return out


for condition, qids in CHANGED.items():
    want = set(qids)
    old = load(OLD.format(cond=condition), want)
    new = load(NEW.format(cond=condition), want)
    for qid in qids:
        a, b = old[qid], new[qid]
        print(f'{condition} {qid} target_rank {a["target_rank"]} -> {b["target_rank"]}')
        for index in range(len(a['topk'])):
            pa, sa = a['topk'][index]
            pb, sb = b['topk'][index]
            if pa != pb:
                print(f'   first divergence at window index {index}: '
                      f'old pos={pa} score={sa!r} | new pos={pb} score={sb!r}')
                lo, hi = max(0, index - 1), min(len(a['topk']), index + 2)
                for j in range(lo, hi):
                    qa, ta = a['topk'][j]
                    qb, tb = b['topk'][j]
                    print(f'     [{j}] old {qa}:{ta!r}   new {qb}:{tb!r}   score_equal={ta == tb}')
                break
        else:
            print('   top-100 window identical; rank differs outside the stored window')
