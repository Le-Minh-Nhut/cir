# Two audited false-PASS paths — reproduced on `b4d25c7`

Reproduction lane, 2026-11-13. **No production file was edited, no commit was made.** Everything below
is an observation of the tree at `b4d25c7` under the control interpreter
`workbench/envs/ctl312h/bin/python` (CPython 3.12.11, numpy 1.26.4). Scripts live beside this file and
in `/tmp/`; each one's full output is beside it as `<script>.observed.txt`.

| case | verdict |
|---|---|
| **A** V2 chunk coverage | **REPRODUCED** |
| **B** Hit-count identity | **REPRODUCED** |

Both defects are in the reader, not the writer. In A the writer is correct and the reader accepts a
record that has been stripped of the only evidence of its own failure. In B the *producers* derive the
flag from the counts correctly (`limn_hits.py:172`, `third_model_parity.py:1037`) and the reader throws
the derivation away, trusting the boolean alone.

---

## A. V2 chunk coverage

### A.1 As written: the offender is judged and the pair fails

Command (`/tmp/repro_a.py`, inlined at the end):

```
workbench/envs/ctl312h/bin/python /tmp/repro_a.py
```

Observed, verbatim:

```
WRITER scope: {"components_total": 4146, "components_emitted": 513, "coverage": "chunked", "mismatch_count": 1, "max_abs": 0.009999990463256836, "worst_normalized": 166.66650772094727, "offenders": [3600], "head_last_index": 511, "offender_emitted": true}

=== A1 as-written record (offender inside the emitted chunk) ===
status           : PARITY_MANDATORY_SUBCHECK_FAILED
passed           : False
applied_standard : parity-absolute-relative/v2
detail           : 1 mandatory subcheck(s) failed: shirt.components.sample_forward.query. A mandatory failure is never reported as a pass, whatever a prose surface above it says
component checks : [('shirt.components.sample_forward.query', False, 1, 513, 4146, 'chunked', 166.66650772094727, [3600])]
failures         : ['shirt.components.sample_forward.query']
missing          : []
```

The scope is built by the repo's own writer (`t4_parity.component_scope`, 4,146 components, head
`0..511`, single offender at flat index `3600`), so the chunk rule in `V2_STANDARD.md` is honoured:
`components_emitted 513 == 512 head + 1 offender`, `coverage: chunked`, `worst_normalized 166.67 > 1`.
The reader derives the same offender and the pair fails. **This half is correct.**

### A.2 The defect: offender rows removed from the preview → `PARITY_PASS`

Two variants, in the same run and same output file as above:

```
=== A2a offender rows deleted, mismatch_count/offenders left as recorded ===
status           : PARITY_EVIDENCE_MALFORMED
detail           : malformed parity artifact: shirt.component_parity.sample_forward.query.mismatch_count is 1 but the emitted components hold 0 violation(s) at []: the record contradicts the values it carries

=== A2b offender rows deleted + mismatch_count/offenders zeroed (digests+worst kept) ===
status           : PARITY_PASS
passed           : True
applied_standard : parity-absolute-relative/v2
detail           : every mandatory subcheck of every mandatory dimension was measured and held under its own declared tolerance
component checks : [('shirt.components.sample_forward.query', True, 0, 512, 4146, 'chunked', 0.0, [])]
failures         : []

recorded worst_normalized still: 166.66650772094727 max_abs: 0.009999990463256836 coverage: chunked components_emitted/total: 512 / 4146
```

A2a shows the *only* thing that ever caught the deletion: `parity_gate.py:487-490`, the
`mismatch_count`-vs-derived-count cross-check. Delete one number and the guard is gone. A2b — the
adversarial variant the audit names — is **`PARITY_PASS`, `passed: True`**, while the record it is
built from still declares `worst_normalized: 166.67` on a 4,146-component scope whose only two-hundred-
and-fifty-second-part was ever inspected (512 of 4,146 = 12.3%).

Responsible code path:

* `workbench/backend/parity_gate.py:487-490` — the cross-check that A2a trips and A2b walks past.
  The reader compares `mismatch_count` to the offenders it derived **from the emitted rows only**.
  Nothing compares either number to `offenders[]`, and nothing compares `worst_normalized` to the
  violation the emit rule promises to include.
* `workbench/backend/parity_gate.py:514-519` — `worst_normalized` is validated **only when
  `coverage == "full"`**; under `chunked` it is merely required to be *at least* the derived value,
  so `166.67` beside a derived `0.0` is accepted. `5.0` beside a derived `0.0` is accepted too (A3 P4).
* `workbench/backend/parity_gate.py:468-469` + `402-412` — the three `sha256` values are checked for
  *shape* only: 64 hex characters. They are never recomputed, and the reader holds no array to
  recompute them against, so a chunk is not bound to the full array it claims to summarize.
* `workbench/backend/parity_gate.py:475` — the verdict is `not offenders`, i.e. "no violation among
  the rows the artifact chose to emit". `ok` therefore never means "the scope is clean"; it means
  "the preview is clean".

### A.3 What the reader validates vs trusts

`_component_scope_check` (`parity_gate.py:414-539`). Split by what a PASS actually rests on:

| property | status | where |
|---|---|---|
| `dimension` ∈ `{forward_parity}` | **validated** (refuses others) | `:426-431`, `:366` |
| tolerances taken from the artifact's own table, per dimension | **validated** (block cannot loosen it) | `:432-437` |
| `components_total` > 0, int | **validated** | `:438-441` |
| `shape` non-negative ints, `prod(shape) == components_total` | **validated** — product only, never element order | `:442-450` |
| `indices` ints, `0 <= i < total` | **validated** | `:387-397` |
| `indices` ascending and unique | **validated** | `:398-400` |
| `components_emitted == len(indices) == len(reference) == len(actual)` | **validated** | `:456-461` |
| `components_emitted <= components_total` | **validated** | `:459-461` |
| `coverage ∈ {full, chunked}` | **validated** | `:462-464` |
| `coverage == "full"` ⟺ `emitted == total` | **validated** | `:465-467` |
| each of `sha256.{reference,actual,delta}` is 64 hex | **validated (shape only)** | `:402-412`, `:468-469` |
| per-component finiteness, non-bool, numeric | **validated** | `:370-383` |
| rule `abs(a-b) <= atol + rtol*|b|` on **emitted pairs only** | **validated** | `:474-482` |
| `mismatch_count` == derived offender count | **validated** | `:484-490` |
| `offenders[].index` set == derived offender set | **validated** | `:491-502` |
| `max_abs` vs derived | **validated for `full`; one-sided (`>=`) for `chunked`** | `:505-512` |
| `worst_normalized` vs derived | **validated for `full`; one-sided (`>=`) for `chunked`** | `:513-519` |
| **chunk ordering / contiguity** | **trusted** — `sorted(set(range(512))∪offenders)` is the writer's rule, never re-derived | absent |
| **head presence** (`0..511` ⊆ indices when chunked) | **trusted** — indices `[3000..3099]` PASS (A3b P2) | absent |
| **coverage completeness** (chunk accounts for `total`) | **trusted** — `1 of 4146` PASSes (A3b P10) | absent |
| **gap / overlap detection inside a scope** | **trusted** — 144-index hole in the head PASSes (A3b P3) | absent |
| **digest recomputation** | **trusted** — three `'a'*64` values PASS (A3 P5) | absent |
| **`components_total` against anything outside the record** | **trusted** — relabelling 4146→512 as `full` PASSes (A4) | absent |
| **`shape` element order / alignment** | **trusted** — `[6,691]` beside `[4146]` data PASSes (A3 P6) | absent |
| **`worst_normalized > 1` ⇒ an offender must be emitted** | **trusted** — `5.0` with `mismatch_count 0` PASSes (A3 P4) | absent |
| **scopes recorded vs scopes the aggregates cover** | **trusted** — a v2 pair may record one scope and fail no other | `:959-1008` |

The gaps, restated as the two sentences that matter:

1. **`ok` is derived from the preview, never from the array.** Every derived number
   (`offenders`, `worst`, `derived_max_abs`) is computed over `indices/reference/actual` as emitted.
   There is no field anywhere in the reader that says "this chunk is a faithful, complete sample of
   the scope" — so a smaller preview is a smaller claim, and a smaller claim is easier to satisfy.
2. **The digests name the binding the never-completed check would need.** `V2_STANDARD.md:76-78`
   ("A reviewer holding the run's arrays can prove the emitted chunk came from them; the chunk itself is
   not a new measurement") makes the reader's position explicit: the digest is for an *external*
   reviewer. Inside the gate, a 64-hex string is all that is inspected, so the artifact is the only
   authority on its own completeness — which is exactly the authority the write side was trusted with and the read side must not accept.

Observed gap probes (all on the writer's own clean 4,146-component chunked record, one change each):

```
P1 chunked, only 100 of 4146 emitted (no head guarantee)         -> PARITY_PASS   compared/total=100/4146 cov=chunked
P2 chunked, indices 3000..3099 only (no head, no gap check)      -> PARITY_PASS   compared/total=100/4146 cov=chunked
P3 chunked, head has a 144-index hole (no completeness check)    -> PARITY_PASS   compared/total=368/4146 cov=chunked
P4 chunked, worst_normalized=5.0 recorded, mismatch_count=0      -> PARITY_PASS   compared/total=512/4146 cov=chunked
P5 all three sha256 replaced by 'aaaa…' (digests never verified) -> PARITY_PASS   compared/total=512/4146 cov=chunked
P6 shape rewritten to [6,691] (only the product is checked)      -> PARITY_PASS   compared/total=512/4146 cov=chunked
P7 duplicate index (control: refused)                            -> PARITY_EVIDENCE_MALFORMED
P8 NaN component (control: refused)                              -> PARITY_EVIDENCE_MALFORMED
P9 components_total+shape shrunk to 512, coverage left chunked   -> PARITY_EVIDENCE_MALFORMED
```

and the tenth, from `/tmp/repro_a4.py` — a chunked record whose `components_total`/`shape` are relabelled
`512`/`[512]` with `coverage: full`, `worst_normalized` zeroed, the real 4,146-element `sha256` values
kept, and no offender rows:

```
A4 chunked->full relabel, total claim 512, sha256 still of the real 4146 arrays, no offenders
   status: PARITY_PASS | compared/total: 512 / 512 | coverage: full
```

i.e. the `components_total` that A3 P9 catches only when the coverage label contradicts it becomes
unfalsifiable once the label is relabelled with it. Nothing else in the document records 4,146.

### A.4 Shipped evidence, for calibration

The live evidence on this tree has exactly one chunked v2 producer output, and it is the shape the
defect needs (`hint`, three categories):

```
g1_g2.hint.dress.adapter.json  | sample_forward.query  | total 1280  emitted 1280 cov full    off 0 worst 0.0
g1_g2.hint.dress.adapter.json  | sample_forward.target | total 40960 emitted 512  cov chunked off 0 worst 0.0
```

`40960 → 512` is 1.25% coverage, and the reader reports `ok` for it
(`hint` gate run: `('dress.components.sample_forward.target', True, 'chunked', 512, 40960, 0.0)`). That
is not itself a false PASS — the writer emitted every offender, and there is none — but it is the exact
shape in which A2b occurs, and no field on that record would have to change except two integers and two
list entries for the pair to pass with a violation hiding at index 512.

**A verdict: REPRODUCED.** The as-written record fails correctly; the adversarial variant
(offender rows removed, `mismatch_count`/`offenders` silenced, digests and `worst_normalized` untouched)
returns `PARITY_PASS` with `worst_normalized: 166.67` recorded on a 4,146-component scope.
Responsible: `parity_gate.py:487-490` (cross-check derived from the emitted rows only),
`parity_gate.py:514-519` (`worst_normalized` unvalidated under `chunked`), `parity_gate.py:402-412`
(digests never recomputed), `parity_gate.py:474-482` (`ok` from the preview, not the scope).

---

## B. Hit-count identity

### B.1 The flag's derivation and the code path that reads it

Producer side, where the flag is *derived from counts*:

* `workbench/docs/reviews/20261112-parity-repair/scripts/limn_hits.py:172`
  `artifact["native_metrics_match_feature_rerank"] = all(result["equality"].values())`
  with `result["equality"]` built at `limn_hits.py:137` as
  `{f"r{k}": native["hits"][f"r{k}"] == adapter["hits"][f"r{k}"] for k in CUTOFFS}`.
* `workbench/docs/reviews/20261112-parity-repair/scripts/third_model_parity.py:1037`
  `document["native_metrics_match_feature_rerank"] = bool(ranking["hit_counts_equal"])`,
  and `:949-950`
  `block["hit_counts"] = {"native": _hits(control["ranks"]), "adapter": _hits(native_style)}` /
  `block["hit_counts_equal"] = block["hit_counts"]["native"] == block["hit_counts"]["adapter"]`.

Reader side, where the derivation is discarded:

```
886:def _metrics_checks(native: dict[str, Any], category: str) -> list[ParityCheck]:
887:    """Hit-count equality. A non-boolean flag is malformed, never "not applicable"."""
888:    if "native_metrics_match_feature_rerank" not in native:
889:        return []
890:    value = native.get("native_metrics_match_feature_rerank")
891:    if not isinstance(value, bool):
...
895:        name=f"{category}.metrics.hit_count_equality", dimension="metrics_parity", gate="G2",
896:        declared="exact hit-count equality", measured=value,
897:        ok=value is True, mandatory=True)]
```

`grep native_metrics_parity_detail workbench/backend/` → **no match**. The reader never opens the
`native_metrics_parity_detail` block the producer writes beside the flag; the flag *is* the check.
`parity_gate.py:1316-1318` then makes `metrics_parity` `all(check.ok ...)` over that one check, so a
`true` boolean is the entire evidence for the dimension.

### B.2 Proven: `true` beside differing per-side counts

`/tmp/repro_b1.py` on the **real shipped artifacts**:

```
REAL dress pair, unmodified:
  status: PARITY_MANDATORY_SUBCHECK_FAILED
  dress metrics: [('dress.metrics.hit_count_equality', True, True, True)]
  dress detail counts: {'r1': 375, 'r10': 1032, 'r50': 1498}

REAL dress pair, adapter r1 raised 375 -> 376 (a rerun under a different mask), flag untouched:
  flag kept: True | detail native r1: 375 | detail adapter r1: 376 | equality: {'r1': False, 'r10': True, 'r50': True}
  status: PARITY_MANDATORY_SUBCHECK_FAILED
  dress metrics: [('dress.metrics.hit_count_equality', True, True, True)]
  dress dimension metrics_parity: True
  failures: ['dress.batch_1_vs_n.max_rel', 'shirt.batch_1_vs_n.max_rel', 'toptee.batch_1_vs_n.max_rel']

Same but with the flag flipped to False (proves the flag alone decides):
  status: PARITY_MANDATORY_SUBCHECK_FAILED
  dress metrics: [('dress.metrics.hit_count_equality', False, False, True)]
```

The `dress` pair carries `native r1 = 375` and `adapter r1 = 376` in its own recorded detail, its own
`equality.r1` is `False`, and the gate still reports
`dress.metrics.hit_count_equality = ok:True, measured:True` and `metrics_parity: True`. The
`PARITY_MANDATORY_SUBCHECK_FAILED` above is **not** from the metrics dimension — it is the three
pre-existing `batch_1_vs_n.max_rel` failures (`dress/…` 0.0114 vs declared 1e-3). On a tree where those
batch failures were fixed, this pair would be `PARITY_PASS` with one side at 375 and the other at 376.

`/tmp/repro_b2.py` isolates it on a minimal LIMN-shaped pair (full real fixture otherwise unchanged):

```
B2a flag=true WITH a detail block saying native 375 / adapter 376, equality r1=False
   status          : PARITY_PASS
   passed          : True
   metrics_parity  : True
   metrics checks  : [('dress.metrics.hit_count_equality', True, True)]
   detail block read by the gate?  False

B2b flag=true, NO detail block at all (a bare assertion)
   status          : PARITY_PASS
   passed          : True

B2c flag=false (control: the reader does react to the boolean)
   status          : PARITY_MANDATORY_SUBCHECK_FAILED
   metrics checks  : [('dress.metrics.hit_count_equality', False, False)]

B2d flag absent (LIMN before limn_hits: absence is INCOMPLETE, not a pass)
   status          : PARITY_INCOMPLETE
   metrics checks  : []
```

`B2b` is the sharper statement: the flag does not have to contradict anything — it can arrive with no
counts *anywhere* in the document and still be the whole `metrics_parity` evidence.

Where the specific `375` vs `376` can and cannot come from, measured by `/tmp/repro_b3.py` on the stored
dress adapter features and native masks:

```
dress queries: 2017 gallery: 2628
masked,   strict hard-hit R@1      : 375
masked,   tie-counted R@1         : 382
unmasked, strict hard-hit R@1     : 348
unmasked, tie-counted R@1         : 354
artifact recorded native hits     : {'r1': 375, 'r10': 1032, 'r50': 1498}
artifact recorded adapter hits    : {'r1': 375, 'r10': 1032, 'r50': 1498}
```

So `375` is the masked/strict count and `376` (`[INFERENCE]`) is a one-count difference no convention on
this file reproduces — it is the kind of off-by-one a different tie tolerance, a mask applied to one side
only, or a rerun with a different `tie_epsilon` produces. That is precisely the point: the flag cannot
tell those apart from a clean run, and the artifact records nothing that would.

### B.3 The identity fields the LIMN pair lacks

Actual keys, `g1_g2.{dress,shirt,toptee}.native.json` and `.adapter.json`
(`g1_g2.limn.<category>.*.json` — the name this brief uses — **does not exist**; LIMN ships under the
legacy bare names, resolved by `parity_gate.py:278-291`):

LIMN **native** top level: `category, phase, tolerances, layer1_identity,
native_metrics_match_feature_rerank, native_metrics_parity_detail`.
LIMN **adapter** top level: `category, phase, tolerances, layer2_4_forward`.

| field a strict certification needs | LIMN pair | note |
|---|---|---|
| per-side hit counts at each declared cutoff | **PRESENT but single-sided in the artifact that carries it** — `native_metrics_parity_detail.native.hits{r1,r10,r50}` and `.adapter.hits{…}` | reader never reads this block |
| per-side eligible query counts / denominators | **PRESENT** — `.native.query_count` 2017, `.adapter.query_count` 2017 | informational; not checked against either side's own record |
| eligible query **membership** | **text only** — `eligible_query_set_identical: true` + a prose `eligible_query_set_evidence` string | a boolean about itself, no digest, no id list |
| per-cutoff equality derivation | **PRESENT** — `equality {r1,r10,r50}` | discarded by the reader in favour of the boolean |
| protocol id | **ABSENT** | `IDENTITY_FIELDS` at `parity_gate.py:649` names `protocol_id`/`protocol`; LIMN carries neither, and `_identity_cross_checks:666-668` **skips a field both sides omit** |
| model / checkpoint identity | **ABSENT** | same: `model_id`, `checkpoint_id`, `checkpoint_identity` all missing, silently skipped |
| query order digest | **ABSENT** | `query_order_sha256` missing |
| gallery order digest | **ABSENT** | `gallery_order_sha256` missing |
| `layer1_identity` on the adapter side | **ABSENT** | LIMN adapter has no identity block at all; `_identity_checks` reads native only |
| gallery/query count agreement | **PRESENT** — `query_count_native 2017 / query_count_annotations 2017`, `gallery_count_native 2628 / gallery_count_rebuilt_from_annotations 2628` | the one identity pair that is real |
| `sha?`/`digest`-ish keys anywhere in either document | **ABSENT** — `[]` | verified by scan |
| `tolerances.metrics` | **PRESENT** — `"exact hit-count equality"` on both sides | the declared rule the flag is supposed to certify |

For contrast, HINT (the newest producer) carries `model_id`, `checkpoint_id`, `checkpoint`,
`query_order_sha256`, `gallery_order_sha256`, `protocol`, `standard`, `capture_state` on **both** sides,
and ENCODER carries `model_id` + `protocol` on both. LIMN is the only evidenced model where all seven
`IDENTITY_FIELDS` are absent from the adapter and four are absent from the native — so
`_identity_cross_checks` emits **no check at all** for LIMN's identity, and `input_parity` rests solely
on the native `layer1_identity` booleans (which LIMN writes about itself).

The consequence for B: the pair cannot state, in machine-checkable form, that both sides ranked the same
queries over the same gallery under the same protocol with the same checkpoint. `hit_count_equality`
is therefore not redundant with anything — it *is* the whole metric dimension — and it arrives as a
boolean.

### B.4 What the corrected rule must be

1. **Equality is DERIVED, never read.** The gate recomputes `hits{native, adapter}` at each cutoff from
   the recorded per-side counts (and, where per-query ranks are available, from the ranks), then
   derives `equality[k] = native[k] == adapter[k]` itself. A recorded boolean contributes nothing to
   the verdict.
2. **The flag is an ASSERTION to cross-check.** `native_metrics_match_feature_rerank` stays a mandatory
   check whose `measured` is the flag, but its `ok` is `flag == derived_equality` — the flag may only
   confirm the derivation. Any disagreement, in either direction, is a **contradictory artifact**:
   `_Malformed` / `PARITY_EVIDENCE_MALFORMED`, never `ok` and never a mere failure of one check, because
   the document cannot be both right and wrong.
3. **No counts, no check.** A flag with no per-side counts behind it is not evidence. Today it is a
   PASS (B2b); under the corrected rule it must be `INCOMPLETE` (`metrics_parity` unmeasured), which is
   what the same artifact reads before `limn_hits.py` runs (B2d).
4. **Denominators and membership are part of the identity.** Hit-count equality is meaningless across
   different query sets. The pair must record eligible query count **per side** plus a digest of the
   eligible id sequence per side (`query_order_sha256` already exists in `IDENTITY_FIELDS`), and the
   equality check must be gated on those two digests being present and equal. LIMN today records
   neither, so under the corrected rule LIMN's metric dimension cannot be `metrics_parity: True` — it
   must read `INCOMPLETE` until the producer emits them.

**B verdict: REPRODUCED.** `native_metrics_match_feature_rerank: true` certifies
`metrics_parity: True` while the same document records `native r1 = 375` and `adapter r1 = 376` and
`equality.r1: false`, and the pair reaches `PARITY_PASS` on a fixture with no other failing dimension.
Responsible: `parity_gate.py:886-897` (the flag *is* the check), `parity_gate.py:1316-1318` (one check
becomes the dimension), and the silent skips at `parity_gate.py:666-668`.

---

## Exact regression cases the fix must add

Phrased as assertions. Each is a `pytest` test; A's belong in
`workbench/tests/test_parity_absolute_relative_v2.py`, B's in a new
`workbench/tests/test_gate_metrics_identity.py`.

**A — chunk coverage**

1. `assert verdict.status == gate.MANDATORY_FAILED` for the writer's own 4,146-component record with one
   offender at index `3600` (the A1 fixture) — the as-written case must keep failing.
2. `assert verdict.status == gate.MALFORMED` and `"worst_normalized" in verdict.detail` when the
   offender rows are deleted while `worst_normalized == 166.67` is left recorded on a `chunked` scope
   (the A2b fixture). **This is the defect: it currently returns `PARITY_PASS`.**
3. `assert verdict.passed is False` when `coverage == "chunked"` and
   `recorded["worst_normalized"] > 1.0` and `recorded["mismatch_count"] == 0` — a recorded violation
   above the rule's own unit threshold with no offending coordinate emitted is self-contradictory.
4. `assert verdict.status == gate.MALFORMED` when `coverage == "chunked"` and
   `set(indices) ⊉ set(range(COMPONENT_HEAD))` — the chunk rule promises the first 512 components, so a
   chunk without them is not the artifact the standard describes (A3b P2).
5. `assert verdict.status == gate.MALFORMED` when `coverage == "chunked"` and
   `components_emitted == len(indices)` but `indices` has a gap inside `[0, COMPONENT_HEAD)` — i.e.
   `indices[:COMPONENT_HEAD] == list(range(COMPONENT_HEAD))` must hold (A3b P3, P11).
6. `assert verdict.status == gate.MALFORMED` when `components_emitted < COMPONENT_HEAD` and
   `components_total > COMPONENT_FULL_LIMIT` — a chunk that cannot be the documented chunk (A3b P10).
7. `assert verdict.status == gate.MALFORMED` when `coverage == "chunked"` and any recorded `sha256`
   value is not the digest the scope's own arrays produce, **once the reader is given the arrays**;
   otherwise `assert verdict.coverage_bound is False` — i.e. the check must be reachable or the check
   must be absent from the claim, not present-but-vacuous (A3 P5).
8. `assert verdict.status == gate.MALFORMED` when a scope relabelled `coverage: "full"` with
   `components_total: 512` carries `sha256` values that no 512-element array could produce while the
   document also records `components_total: 4146` elsewhere — the total must be singular per scope
   (A4).
9. `assert check.measured["offending_coordinates"]` is non-empty and `check.ok is False` whenever
   `recorded["offenders"]` is non-empty, at any coverage — a recorded offender may never coexist with
   `ok: True`.
10. `assert verdict.status == gate.MALFORMED` when `indices` is not a prefix-union of
    `range(COMPONENT_HEAD)` and the recorded offender indices, i.e. the emit rule is re-derived rather
    than trusted (A3 P1, P2, P6).
11. `assert evaluate_abs_rel` was applied to `len(record["indices"])` components and
    `measured["components_total"] == len(full_array)` when the reader is handed the full array — i.e.
    the verdict is derived over the scope, not the preview.

**B — hit-count identity**

12. `assert verdict.status == gate.MALFORMED` when `native_metrics_match_feature_rerank is True` while
    `native_metrics_parity_detail.native.hits != .adapter.hits` at any declared cutoff — the flag must
    equal the derived equality (the B2a fixture with `375` vs `376`).
13. `assert verdict.status == gate.MALFORMED` when
    `native_metrics_match_feature_rerank != (native_hits == adapter_hits)` for the recorded detail block,
    in **either** direction (a `false` flag beside equal counts is equally contradictory).
14. `assert verdict.missing == ["metrics_parity"]` (i.e. `INCOMPLETE`, not PASS) when the flag is present
    and `native_metrics_parity_detail` is absent — a bare boolean with no counts behind it is not
    evidence.
15. `assert check.measured["derived_equality"] == {"r1": …, "r10": …, "r50": …}` and
    `check.ok == (recorded_flag == all(derived_equality.values()))` for the shipped LIMN dress pair, so
    the flag's role as an assertion is visible in the record.
16. `assert verdict.status != gate.PASS` when the two sides' `eligible_query_set_identical` is `true` but
    no per-side `query_order_sha256` is recorded — membership must be digest-bound, not asserted.
17. `assert verdict.missing == ["metrics_parity"]` for the shipped LIMN artifacts as they stand, because
    LIMN records neither `protocol_id`, `checkpoint_id`, nor either order digest — the dimension cannot
    be `True` on identity that was never stated.
18. `assert verdict.status != gate.PASS` when `native.query_count != adapter.query_count` in the parity
    detail, even with equal hit counts — equal hits over different denominators is not equality.
19. `assert _metrics_checks` reads `native_metrics_parity_detail` — assert directly on
    `inspect.getsource(gate._metrics_checks)`, in the style of
    `test_evaluate_abs_rel_has_a_production_caller` at `test_parity_absolute_relative_v2.py`, so the
    block cannot silently become unread again.

---

## Scripts (inline)

### `/tmp/repro_a.py`

```python
"""Repro A: a chunked v2 record whose only offender is outside the 512 head, then the same
record with that offender's rows removed from the emitted preview. Writer = the repo's own."""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path("/home/heheboiz/data/cir")
sys.path.insert(0, str(REPO))
from workbench.backend import parity_gate as gate

SCRIPTS = REPO / "workbench/docs/reviews/20261109-native15-composition-limn/scripts"
spec = importlib.util.spec_from_file_location("producer_t4", SCRIPTS / "t4_parity.py")
t4 = importlib.util.module_from_spec(spec)
sys.modules["producer_t4"] = t4
spec.loader.exec_module(t4)

TOL = {"identity": "exact, zero tolerance", "forward_max_abs": 1e-05, "forward_max_rel": 1e-04,
       "forward_min_cosine": 0.9999999, "batch_max_abs": 1e-04, "batch_max_rel": 1e-03,
       "score_max_abs": 1e-06, "tie_epsilon": 1e-06, "metrics": "exact hit-count equality"}
TOTAL = 4146
SCOPE = "sample_forward.query"

reference = np.full(TOTAL, 0.5, dtype=np.float32)
actual = reference + np.float32(1e-9)
OFFENDER = 3600                              # outside the first 512, ascending
actual[OFFENDER] = np.float32(0.5 + 1e-2)    # 1e-2 > 1e-5 + 1e-4*0.5 = 6e-5 -> violation

block = t4.component_block({SCOPE: t4.component_scope(
    SCOPE, "forward_parity", actual, reference,
    atol=TOL["forward_max_abs"], rtol=TOL["forward_max_rel"])})
rec = block["scopes"][SCOPE]
print("WRITER scope:", json.dumps({
    "components_total": rec["components_total"], "components_emitted": rec["components_emitted"],
    "coverage": rec["coverage"], "mismatch_count": rec["mismatch_count"],
    "max_abs": rec["max_abs"], "worst_normalized": rec["worst_normalized"],
    "offenders": [o["index"] for o in rec["offenders"]],
    "head_last_index": rec["indices"][511], "offender_emitted": OFFENDER in rec["indices"]}))


def native():
    return {"category": "shirt", "phase": "native", "model_id": "testmodel",
            "tolerances": dict(TOL), "standard": gate.STANDARD_ABS_REL_V2,
            "native": {
                "layer1_identity": {
                    "query_count_native": 10, "query_count_annotations": 10,
                    "gallery_count_native": 20, "gallery_count_rebuilt_from_annotations": 20,
                    "gallery_order_identical": True, "reference_ids_identical": True,
                    "target_ids_identical": True,
                    "model_text_identical_to_author_composition": True,
                    "gallery_has_duplicates": False},
                "native_metrics_match_feature_rerank": True}}


def fwd(value=0.0):
    return {"max_abs": value, "max_rel": value, "min_cosine": 1.0, "bitwise_equal": False,
            "shape": [2, 4], "within_tolerance": value <= TOL["forward_max_abs"]}


def adapter(components):
    body = {"layer2_pixels": {"adapter_decode_vs_native_val_decode": {
                "max_abs": 0.0, "max_rel": 0.0, "bitwise_equal": True, "min_cosine": 1.0,
                "shape": [2, 4]}},
            # No `sample_forward`/`population` aggregate for the component scope's comparison:
            # the component record is this pair's ONLY forward evidence (LIMN's shape, per
            # parity_gate.py:1275 "the LIMN artifact records no population forward delta at all").
            "batch_size_1_vs_n": {"max_abs": 3e-07, "max_rel": 1e-06, "min_cosine": 0.9999999,
                                  "bitwise_equal": False, "shape": [2, 4],
                                  "within_tolerance": True},
            "scoring_ranking": {
                "score_max_abs_delta_vs_dot": 1.19e-07, "score_pairs_compared": 8,
                "score_max_abs_declared": 1e-06, "score_within_tolerance": True,
                "adapter_vs_argsort_unmasked": {"non_tie_count": 0},
                "rank_differences_vs_native_val_control": {"count": 0, "non_tie_count": 0,
                                                           "min_gap_abs": 0.0}}}
    return {"category": "shirt", "phase": "adapter", "model_id": "testmodel",
            "tolerances": dict(TOL), "standard": gate.STANDARD_ABS_REL_V2,
            "adapter": body, "component_parity": components}


def run(label, components):
    root = Path(tempfile.mkdtemp(prefix="repro-a-"))
    for name, doc in (("g1_g2.testmodel.shirt.native.json", native()),
                      ("g1_g2.testmodel.shirt.adapter.json", adapter(components))):
        (root / name).write_text(json.dumps(doc), encoding="utf-8")
    v = gate.evaluate_model("testmodel", root=root, categories=("shirt",))
    derived = [c for c in v.checks if c.name.startswith("shirt.components.")]
    print(f"\n=== {label} ===")
    print("status           :", v.status)
    print("passed           :", v.passed)
    print("applied_standard :", v.applied_standard)
    print("detail           :", v.detail)
    print("component checks :", [(c.name, c.ok, c.measured["mismatch_count"],
                                  c.measured["components_compared"],
                                  c.measured["components_total"],
                                  c.measured["coverage"],
                                  c.measured["worst_normalized"],
                                  c.measured["offending_coordinates"]) for c in derived])
    print("failures         :", [c.name for c in v.failures])
    print("missing          :", v.missing)
    return v


run("A1 as-written record (offender inside the emitted chunk)", block)

# A2a: delete ONLY the offender's rows from the preview, leave mismatch_count/offenders as recorded.
mutated = json.loads(json.dumps(block))
m = mutated["scopes"][SCOPE]
drop = m["indices"].index(OFFENDER)
for key in ("indices", "reference", "actual"):
    del m[key][drop]
m["components_emitted"] = len(m["indices"])
run("A2a offender rows deleted, mismatch_count/offenders left as recorded", mutated)

# A2b: also silence the two summary fields that contradict (offenders list, mismatch_count).
#       coverage stays "chunked", totals/digests/worst_normalized/max_abs untouched.
mutated2 = json.loads(json.dumps(mutated))
m2 = mutated2["scopes"][SCOPE]
m2["mismatch_count"] = 0
m2["offenders"] = []
run("A2b offender rows deleted + mismatch_count/offenders zeroed (digests+worst kept)", mutated2)
print("\nrecorded worst_normalized still:", mutated2["scopes"][SCOPE]["worst_normalized"],
      "max_abs:", mutated2["scopes"][SCOPE]["max_abs"],
      "coverage:", mutated2["scopes"][SCOPE]["coverage"],
      "components_emitted/total:", mutated2["scopes"][SCOPE]["components_emitted"],
      "/", mutated2["scopes"][SCOPE]["components_total"])
```

The remaining scripts (`repro_a3.py`, `repro_a3b.py`, `repro_a4.py`, `repro_b1.py`, `repro_b2.py`,
`repro_b3.py`) are stored verbatim beside this file and in `/tmp/`, with their observed output in
`*.observed.txt`. They share the fixture builders shown above; only the mutation under test differs.

## Evidence directory

```
workbench/docs/reviews/20261113-final-acceptance/raw/parity-repro/
  REPRO.md
  repro_a.py               repro_a.observed.txt     (A1, A2a, A2b)
  repro_a3.py              repro_a3.observed.txt    (P1..P9)
  repro_a3b.py             repro_a3b.observed.txt   (P2, P3, P10, P11 coverage probes)
  repro_a4.py              repro_a4.observed.txt    (chunked->full relabel)
  repro_b1.py              repro_b1.observed.txt    (real artifacts, 375 vs 376)
  repro_b2.py              repro_b2.observed.txt    (B2a..B2d minimal pair)
  repro_b3.py              repro_b3.observed.txt    (R@1 under four conventions)
```

Read-only context used, not modified: `workbench/backend/parity_gate.py`,
`workbench/tests/test_parity_absolute_relative_v2.py`,
`workbench/docs/reviews/20261112-parity-repair/V2_STANDARD.md`,
`workbench/docs/reviews/20261111-master-audit/raw/verify/`,
`workbench/docs/reviews/20261112-parity-repair/scripts/limn_hits.py`,
`workbench/docs/reviews/20261112-parity-repair/scripts/third_model_parity.py`,
`workbench/docs/reviews/20261109-native15-composition-limn/g1_g2.*.json` artifacts.
`workbench/docs/reviews/20261113-final-acceptance/LEDGER.md` did not exist at the time of this run.
`workbench/config/workbench.env` and `CLEANUP_REPORT.md.save` were not touched.
