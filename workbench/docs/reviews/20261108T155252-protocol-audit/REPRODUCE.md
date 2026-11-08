# Protocol audit — REPRODUCE

Audit timestamp: `20261108T155252` (+0700). Evidence dir: `workbench/docs/reviews/20261108T155252-protocol-audit/`.

## What was executed vs read

| step | how |
|---|---|
| Clone 16 repos at registry pins | **EXECUTED** — `raw/clone_upstream_pins.sh`, log `raw/clone.log`. `git clone --no-checkout --filter=blob:none`, explicit `git fetch origin <sha>` when the object is absent, `git checkout --detach <sha>`. All 16 `CHECKOUT_OK`; `git status --porcelain` empty for all 16 afterwards. |
| Pin resolution | **EXECUTED** — every registry `upstream_commit_sha` is already a full 40-char SHA (`raw/pins.tsv`); no prefix expansion was needed. |
| Cited-line verification | **EXECUTED** — `ast` parse of each cited file, printing the cited line's text and its innermost enclosing `def`/`class` (`raw/cite_check.json`). This is how the four wrong citations were found. |
| Gallery counts | **EXECUTED** — our `src/evaluation/fashioniq.py::build_fashioniq_gallery` on the real `data/FashionIQ` splits: full 3817/6346/5373/15536, pair union 2628/3089/2902/8619. Cross-checked against direct JSON lengths. |
| Rank semantics | **EXECUTED** — `recall_at_k` vs `recall_at_k_reference_excluded` on `[REF, WRONG, TARGET]`: target rank 3 kept, rank 2 excluded. |
| Vendored-vs-clean divergence | **EXECUTED** — `diff -r --exclude=.git --exclude=__pycache__ --exclude='*.pyc'` per repo, full unified diffs in `raw/vendored_diffs.txt`. |
| Patch digests | **EXECUTED** — sha256 of each live vendored file vs registry `derived_sha256` and provenance `delivered_sha256`; all 18 match. |
| Upstream protocol classification | **READ** — pinned source at the pinned commit. No upstream evaluator was run: it needs each repo's own environment, checkpoints, and the GPU, and a cohort evaluation owns the GPU (`pgrep -af composition_eval` → pid 264541). |
| Reported paper numbers | **NOT VERIFIED** — out of scope and impossible without those environments. |

## Reproduce from scratch

```bash
# 1. audit root + pins (registry is the source of truth)
A=/tmp/cir-upstream-audit-$(date +%Y%m%dT%H%M%S); mkdir -p "$A"
workbench/envs/ctl312h/bin/python - <<'PY' > /tmp/pins.tsv
import yaml
d = yaml.safe_load(open("workbench/registry/models.yaml"))
seen = {}
for m in d["models"]:
    seen[m["source_dir"]] = (m["upstream_repo_url"], m["upstream_commit_sha"])
for k in sorted(seen):
    print(k, seen[k][0], seen[k][1], sep="\t")
PY

# 2. clone + detach (4 at a time; host is shared)
bash workbench/docs/reviews/20261108T155252-protocol-audit/raw/clone_upstream_pins.sh "$A" /tmp/pins.tsv

# 3. confirm every clone is pristine at its pin
for d in $(cut -f1 /tmp/pins.tsv); do (cd "$A/$d" && printf "%s\t%s\t%s\n" "$d" "$(git rev-parse HEAD)" "$(git status --porcelain | wc -l)"); done

# 4. counts + rank semantics through OUR evaluator
workbench/envs/ctl312h/bin/python - <<'PY'
import sys, json
sys.path.insert(0, "src")
import torch
from evaluation.fashioniq import (recall_at_k, recall_at_k_reference_excluded, build_fashioniq_gallery)
g = ["REF", "WRONG", "TARGET"]; s = torch.tensor([[0.9, 0.8, 0.7]])
assert recall_at_k(s, ["TARGET"], g, 1) == 0.0 and recall_at_k(s, ["TARGET"], g, 3) == 100.0
assert recall_at_k_reference_excluded(s, ["TARGET"], ["REF"], g, 1) == 0.0
assert recall_at_k_reference_excluded(s, ["TARGET"], ["REF"], g, 2) == 100.0
root, split_root = "data/FashionIQ", "data/FashionIQ/image_splits"
class Ann:
    def __init__(self, r, t): self.reference_id, self.target_id = r, t
for c in ("dress", "shirt", "toptee"):
    caps = json.load(open(f"{root}/captions/cap.{c}.val.json"))
    ann = [Ann(x["candidate"], x["target"]) for x in caps]
    print(c, len(build_fashioniq_gallery("fashioniq_original_split", split_root, c, ann, "val")),
          len(build_fashioniq_gallery("fashioniq_val_split", split_root, c, ann, "val")))
PY

# 5. vendored divergence
V=workbench/third_party
for d in CSMCIR MTST_PTHA CLIP4Cir SPRC TME AirKnow ConeSep HABIT INTENT DCNet HINT ENCODER PAIR CLVCNet TGCIR LIMN; do
  echo "######## $d"
  diff -r -q --exclude=.git --exclude=__pycache__ --exclude='*.pyc' "$A/$d" "$V/$d" | grep -v '^Only in'
done
```

## Reading order for a reviewer

1. `PROTOCOL_MATRIX.md` — the verified classification with quoted evidence, then the registry reconciliation.
2. `PROTOCOL_FINDINGS.md` — what was wrong (citations, entrypoints, the per-model protocol binding), the eight
   traps, and the vendored-patch verdict.
3. `REGISTRY_CORRECTIONS.patch` — proposed, **not applied**. Protocol section: none. Only two missing
   `approved_source_patches` declarations.
4. `CLONE_MANIFEST.tsv` — pins + clone paths, so this survives `/tmp` being cleared.
5. `raw/` — `clone.log`, `pins.tsv`, `cite_check.json`, `group{A,B,C}_quotes.txt`,
   `vendored_diffs.txt`, `provenance_local_fixes.txt`, `clone_upstream_pins.sh`.

## Scope boundaries honoured

- Nothing committed, nothing pushed; `cir` git state untouched.
- No GPU work; the running `composition_eval.py` (pid 264541) was never touched or duplicated.
- `workbench/third_party/**` read only — never edited, never `git pull`ed. Upstream sources were read from the
  `/tmp` clones, not from the vendored trees.
- `workbench/config/workbench.env` untouched.
- Read-only on `workbench/registry/**`, `workbench/composition/**`, `workbench/backend/**`,
  `workbench/replay/**`, `workbench/tests/**`.
- New files: this evidence dir + `workbench/scripts/audit/clone_upstream_pins.sh`.
