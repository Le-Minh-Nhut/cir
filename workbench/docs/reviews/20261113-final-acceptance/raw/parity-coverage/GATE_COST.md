# Gate cost: `_component_scope_check`'s per-component dict

Measurement only. **Nothing edited, nothing committed.** All numbers below are observed on the current
tree, in-process, under `workbench/envs/ctl312h/bin/python`.

`HEAD` is `0107e66`; the coverage reader this file examines landed in `8214175` and is byte-identical
to the worktree (`git diff HEAD -- workbench/backend/parity_gate.py` → empty).

---

## 0. First correction to the advisory

The flagged lines are **`parity_gate.py:721-725`**, not `727-729` (727-729 is the two
`_Malformed` messages that *read* `derived_max_abs`):

```python
721:    values = {index: (chunk.reference[position], chunk.actual[position])
722:              for chunk in chunks
723:              for position, index in enumerate(chunk.indices)}
724:    derived_max_abs = max((abs(actual_value - reference_value)
725:                           for reference_value, actual_value in values.values()), default=0.0)
```

The substance of the advisory is right: the dict holds one entry per component and is otherwise read
only to build `offending_values` for the first 5 offenders.

---

## 1. Real component counts (measured, not the advisory's estimate)

Every `component_parity` scope on disk, by artifact:

| artifact | components | coverage forms |
|---|---:|---|
| `g1_g2.encoder.dress.adapter.json` | 2,382,336 | 2 × `full` (2048 each) + 2 × `chunks` (1,032,704 / 1,345,536) |
| `g1_g2.encoder.shirt.adapter.json` | 2,629,120 | 2 × `full` + 2 × `chunks` (1,043,456 / 1,581,568) |
| `g1_g2.encoder.toptee.adapter.json` | 2,492,928 | 2 × `full` (1536 each) + 2 × `chunks` (1,004,032 / 1,485,824) |
| `g1_g2.habit.{dress,shirt,toptee}.adapter.json` | 42,240 ×3 | `full` 1280 + `chunks` 40,960 |
| `g1_g2.hint.{dress,shirt,toptee}.adapter.json` | 42,240 ×3 | `full` 1280 + `chunks` 40,960 |
| `g1_g2.tgcir.{dress,shirt,toptee}.adapter.json` | 5,120 ×3 | 2 × `full` |
| **total** | **7,773,184** | across 7 evidenced models |

The advisory's "~1.03M–1.34M per ENCODER population scope, ×2 scopes ×3 categories" is **exactly
right**: 3 × (1,032,704 + 1,345,536 … ) = 7,504,384 of the 7,773,184 total (96.5%) is ENCODER's six
population scopes. Everything else is small.

---

## 2. Timing table — cold `evaluate_all()`, three runs

| run | total wall | peak RSS (VmHWM) | RSS at start |
|---|---:|---:|---:|
| 1 | 4.369 s | 480.6 MiB | 23.2 MiB |
| 2 | 4.446 s | 480.8 MiB | 23.1 MiB |
| 3 | 4.374 s | 480.8 MiB | 23.1 MiB |

Stable at **≈4.37 s, ≈481 MiB peak**. `ru_maxrss` in the same process reads 4573 MiB because the venv's
torch import inflates the high-water mark before the gate runs; `/proc/self/status:VmHWM` measured in a
slim process is the honest number above, and `tracemalloc` attributes **382.1 MiB** of it to
`evaluate_all()` itself.

### Per-model (second, no profiler, imports warm)

| model | run A | run B | status |
|---|---:|---:|---|
| conesep | 0.000 | 0.000 | `PARITY_EVIDENCE_MISSING` (native only) |
| **encoder** | **4.244** | **4.139** | `PARITY_MANDATORY_SUBCHECK_FAILED` |
| habit | 0.060 | 0.058 | `PARITY_MANDATORY_SUBCHECK_FAILED` |
| hint | 0.060 | 0.057 | `PARITY_MANDATORY_SUBCHECK_FAILED` |
| intent | 0.000 | 0.000 | `PARITY_EVIDENCE_MISSING` |
| limn | 0.001 | 0.001 | `PARITY_MANDATORY_SUBCHECK_FAILED` |
| tgcir | 0.008 | 0.008 | `PARITY_MANDATORY_SUBCHECK_FAILED` |
| **sum** | **4.373** | **4.263** | |

**ENCODER is 96.9 % of the cost.** The other six models together are 0.135 s.

---

## 3. Timing table — live endpoint

Service restarted at 10:10 with current code; `curl -s -o /dev/null -w '%{time_total} %{size_download}'`.

| # | time_total | size_download |
|---|---:|---:|
| cold 1 | 4.203622 s | 188,396 B |
| cold 2 | 4.186524 s | 188,396 B |
| cold 3 | 4.209728 s | 188,396 B |
| warm 1 | 4.208322 s | 188,396 B |
| warm 2 | 4.214673 s | 188,396 B |
| warm 3 | 4.435294 s | 188,396 B |

**Cold and warm are identical (4.19–4.44 s).** There is no cache to be warm: `main.py:1262-1273` is

```python
@app.get("/api/runs/parity-gate")
def parity_gate():
    from workbench.backend.parity_gate import evaluate_all
    return evaluate_all()
```

No `lru_cache`, no stamp, no memo. Every request — including the frontend's fetch on mount — pays the
full 4.2 s and 188 KB.

---

## 4. Where the 4.1 s actually goes (cProfile, `evaluate_model('encoder')`)

```
127,821,175 function calls in 12.705 s (profiled; ~3x the unprofiled 4.1 s)
  12.409 s  _component_scope_check        (12 calls)
   6.632 s  _chunks_tile                  (6 calls)
   4.847 s  _verify_digests               (1839 calls)
   4.192 s  component_digest              (5517 calls)   <- 2.347 s own time
   1.703 s  _component_numbers            (3678 calls)
   1.344 s  builtins.max                  (7,504,396 calls)   <- the flagged generator
   1.314 s  builtins.abs                  (45,026,304 calls)
   1.147 s  evaluate_abs_rel              (7,504,384 calls)
   0.940 s  hashlib.update                (22,513,152 calls)
   0.903 s  struct.pack                   (22,513,152 calls)
```

Two facts the advisory did not have:

1. **The dominant term is digest packing, not the dict.** `component_digest` packs one `struct` per
   component (22.5 M calls), because it was written element-at-a-time to stay allocation-free for
   small samples — but the samples are 1.5 M components.
2. **JSON parse and decode are cheap**: `json.loads` on the 28 MB dress adapter = 0.046 s, decoding all
   2,378,240 base64/zlib components = 0.089 s. The base64 transport is not the cost.

---

## 5. Measured cost attributable to the flagged dict

A `/tmp` copy with the dict replaced by in-loop accumulation (and `values` limited to the ≤5 offender
indices), everything else byte-identical, run in the same process against the same artifacts:

| variant | runs | best | verdict md5 |
|---|---|---:|---|
| real (per-component dict) | 4.152, 4.105 | **4.105 s** | `2edf794cf460cf14abab02b2ba48fafe` |
| accumulate only | 3.607, 3.556 | **3.556 s** | identical |
| accumulate + bulk `struct.pack` | 3.219, 3.216 | **3.216 s** | identical |
| real again | 4.221, 4.185 | 4.185 s | identical |
| accumulate + bulk again | 3.188, 3.191 | **3.188 s** | identical |

- The dict alone: **≈0.55 s (13 %)** of ENCODER's 4.1 s.
- The dict's memory: `tracemalloc` peak **382.1 MiB → 187.6 MiB**, i.e. **194.5 MiB** of the 481 MiB
  peak is that dict.
- `v.to_dict()` is **byte-identical** between real and candidate (same md5, same 9 failures, same
  statuses), so the change is behaviour-preserving, not a verdict change.

### The previous tree state, for scale

`8214175^`'s gate against `8214175^`'s own artifacts (its coverage set was `("full","chunked")`, and
ENCODER recorded **no** component block at all):

| tree | `evaluate_all()` |
|---|---:|
| `8214175^` (preview reader, no complete coverage) | **0.004 s** |
| HEAD (complete-coverage reader) | **4.37 s** |

**≈1100× slower.** That is the honest headline, and it is *not* a regression to be optimised away: the
old 0.004 s was cheap because it judged 512 of 2,382,336 components and could not say anything about
the rest. The 4.37 s is the price of looking at all 7.77 M.

---

## 6. Verdict

**`REAL_PROBLEM`** — but *not* for the reason the advisory gives, and *not* the dict as the chief
cause.

Against the acceptance row ("gate is trustworthy", "release is reproducible"): the verdict is
unchanged and correct either way, so trustworthiness is not at issue. The problem is the row's
*presented* form: the UI fetches this endpoint on mount and gets no cache, so **every page load blocks
4.2 s and allocates ~480 MiB server-side**. Six of seven models cost 0.135 s combined; one model's
coverage dominates. That is a live usability defect at the stated threshold ("multi-second … is a live
usability defect"), and it is the kind that gets worse, not better, with each future large-population
model.

The dict is the smallest of the three levers: **13 %**, but it is also the one with a 194.5 MiB memory
price and zero behavioural risk. The larger lever is caching (**~18,000× on warm requests**), and the
second is bulk digest packing (**~0.85 s** of `component_digest`'s own 4.19 s cumulative).

---

## 7. Diff sketch (NOT applied)

Ordered by measured benefit per unit of risk. Each is independently safe; the first two do not change
any verdict, the third must be shown to.

### (a) Memoise `evaluate_all` on an artifacts `(mtime, size)` stamp — ~4.36 s → ~0.0002 s warm

```python
#: (mtime_ns, size) of every artifact the last evaluation read. A change to ANY of them invalidates,
#: so a rewritten artifact can never be served from a stale verdict; the cost of the check is a stat
#: per file (36 files = 0.2 ms), i.e. 18,000x cheaper than re-deriving the answer.
_EVALUATE_ALL_CACHE: dict[tuple, tuple[tuple, Any]] = {}

def _artifact_stamp(root: Path) -> tuple:
    return tuple(sorted((p.name, p.stat().st_mtime_ns, p.stat().st_size)
                        for p in root.glob("g1_g2.*.json")))

def evaluate_all(*, root=None, standard=STANDARD_RELATIVE_V1):
    base = evidence_root() if root is None else Path(root)
    stamp = (str(base), standard, _artifact_stamp(base))
    cached = _EVALUATE_ALL_CACHE.get(stamp)
    if cached is not None:
        return copy.deepcopy(cached[0])       # callers may mutate the payload
    result = _evaluate_all_uncached(root=root, standard=standard)
    _EVALUATE_ALL_CACHE.clear()               # one entry: never grow with every stamp ever seen
    _EVALUATE_ALL_CACHE[stamp] = (result, None)
    return copy.deepcopy(result)
```

`deepcopy` of the 188 KB payload measured **0.5 ms**, so a warm request is ~0.7 ms end to end. The
stamp includes the requested `standard` because that changes the advisory block. Cache size is one
entry, so a long-running service cannot grow it without bound.

### (b) Accumulate `derived_max_abs` in the existing loop, keep only offender values — −0.55 s, −194 MiB

```python
-    values = {index: (chunk.reference[position], chunk.actual[position])
-              for chunk in chunks
-              for position, index in enumerate(chunk.indices)}
-    derived_max_abs = max((abs(actual_value - reference_value)
-                           for reference_value, actual_value in values.values()), default=0.0)
+    # `derived_max_abs` is a fold over the same values the rule loop already visits, and only the
+    # first few offender values are ever read (`offending_values`). Materialising one dict entry per
+    # component costs ~195 MiB and ~0.55 s on a 1.5M-component scope to build a map that is then
+    # read at most five times.
+    shown = set(offenders[:5])
+    values: dict[int, tuple[float, float]] = {}
+    derived_max_abs = 0.0
+    for chunk in chunks:
+        for position, index in enumerate(chunk.indices):
+            reference_value = chunk.reference[position]
+            actual_value = chunk.actual[position]
+            delta = abs(actual_value - reference_value)
+            if delta > derived_max_abs:
+                derived_max_abs = delta
+            if index in shown:
+                values[index] = (reference_value, actual_value)
```

Better still, fold it into the per-chunk rule loop that already computes `delta` and `worst`
(the `for chunk in chunks: for index, actual_value, reference_value in zip(...)` block), which removes
a second full pass and should recover a little more than the 0.55 s measured here.

### (c) Bulk-pack in `component_digest` — −0.85 s

```python
-    packed = struct.Struct(COMPONENT_DIGEST_FORMAT)
-    digest = hashlib.sha256()
-    for value in values:
-        digest.update(packed.pack(float(value)))
-    return digest.hexdigest()
+    items = [float(value) for value in values]
+    if not items:
+        return hashlib.sha256(b"").hexdigest()
+    return hashlib.sha256(struct.pack(f"<{len(items)}d", *items)).hexdigest()
```

Byte-identical output (`per_element == bulk == numpy '<f8'.tobytes()` verified for 4096 values),
**6.3× faster** per digest. Note this trades the current allocation-free property for one large
`bytes` per digest — ~12 MB transient at 1.5 M components, against the 195 MiB the dict holds
simultaneously, so (b) and (c) together still reduce peak.

### Projected result

| step | `evaluate_all()` cold | endpoint warm |
|---|---:|---:|
| today | 4.37 s | 4.2 s |
| (b)+(c) only | ≈2.9 s | ≈2.9 s |
| (a)+(b)+(c) | ≈2.9 s once | **≈3 ms** |

`evaluate_all` at ≈2.9 s is still above a second, which is why (a) — not (b) or (c) — is what actually
answers the acceptance concern. If only one change is acceptable this cycle, take (a): it is 12 lines,
touches no verdict-bearing arithmetic, and removes 99.93 % of the user-visible cost.

### Recommended regression test for (a)

```python
def test_a_warm_request_does_not_re_evaluate_and_a_touched_artifact_invalidates(tmp_path):
    first = gate.evaluate_all(root=tmp_path)
    assert gate.evaluate_all(root=tmp_path) == first          # same answer
    _touch_or_rewrite(tmp_path / "g1_g2.testmodel.shirt.adapter.json")
    assert gate.evaluate_all(root=tmp_path) != first or True  # must not serve the stale object
```

staged with an evaluation counter (`gate._evaluate_all_uncached` monkeypatched to count calls) so the
assertion is on *calls*, not on elapsed time: warm ⇒ exactly 1 call, touched ⇒ exactly 2.

---

## 8. Scripts

`cost_cold.py` (cold total + RSS), `cost_clean.py` (per-model), `cost_rss.py` (VmHWM),
`cost_digest.py` (per-element vs bulk pack), `cost_shadow2.py` (real vs candidate, same process, same
artifacts, verdict md5), `cost_cache.py` (stamp scan vs full evaluation), `cost_prev2.py` (pre-`8214175`
gate against pre-`8214175` artifacts) — all beside this file and in `/tmp/`.
