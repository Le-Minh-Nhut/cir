# Gate cost: implemented (a) + (b), measured

Supersedes the sketch in `GATE_COST.md`. **Nothing committed.** Two files changed.

`(c)` (bulk `struct.pack`) is **not** implemented, as instructed — the measured cost is unchanged on
the path that matters because (a) makes the warm path 1.5 ms.

---

## (b) The per-component dict is gone

Before (`parity_gate.py` pre-change), two full passes over every component:

```python
values = {index: (chunk.reference[position], chunk.actual[position])
          for chunk in chunks
          for position, index in enumerate(chunk.indices)}          # one dict entry per component
derived_max_abs = max((abs(a - r) for r, a in values.values()), default=0.0)   # second pass
```

After: one pass, inside the loop that already computed `delta`:

```python
shown_values: dict[int, tuple[float, float]] = {}
for chunk in chunks:
    compared += len(chunk.indices)
    for index, actual_value, reference_value in zip(chunk.indices, chunk.actual, chunk.reference):
        if not evaluate_abs_rel(actual_value, reference_value, atol=atol, rtol=rtol):
            offenders.append(index)
            if len(shown_values) < 5:
                shown_values[index] = (reference_value, actual_value)
        denominator = atol + rtol * abs(reference_value)
        delta = abs(actual_value - reference_value)
        if delta > derived_max_abs:
            derived_max_abs = delta
        if denominator > 0.0:
            if delta / denominator > worst:
                worst = delta / denominator
        elif delta > 0.0:
            unbounded = True
```

`offending_values` reads `shown_values`, which holds at most five entries. The two `_Malformed`
messages that read `derived_max_abs` and `worst` are textually unchanged.

| | before | after (b only) |
|---|---:|---:|
| `evaluate_all()` cold | 4.37 s | **2.90 s** |
| peak RSS (`VmHWM`) | 480.6 MiB | **270.4 MiB** |
| `evaluate_all()` payload md5 | `da701905419b8b9833966f637ac521a8` | `da701905419b8b9833966f637ac521a8` |
| `evaluate_model('encoder')` md5 | `2edf794cf460cf14abab02b2ba48fafe` | `2edf794cf460cf14abab02b2ba48fafe` |

`byte_identical: true`, both payloads 200,370 bytes.

---

## (a) `evaluate_all` is memoised, inside `parity_gate` — no `main.py` change

The memo lives in `parity_gate.evaluate_all`, so **`main.py:1271-1273` needs no edit**:

```python
1271:    from workbench.backend.parity_gate import evaluate_all
1272:
1273:    return evaluate_all()
```

That call now hits the cache. `evaluate_all_uncached` is the plain derivation and is what the tests
compare against, so an in-process caller that wants a forced re-read has a named way to ask.

```python
def _artifact_stamp(base: Path) -> tuple[tuple[str, int, int], ...]:
    if not base.is_dir():
        raise FileNotFoundError(f"{base} is not a readable directory")
    return tuple(sorted((path.name, path.stat().st_mtime_ns, path.stat().st_size)
                        for path in base.glob("g1_g2.*.json")))

_EVALUATE_ALL_CACHE: list[tuple[tuple, dict[str, Any]]] = []      # ONE entry

def evaluate_all(*, root=None, standard=STANDARD_RELATIVE_V1):
    base = root if root is not None else evidence_root()
    try:
        stamp = (str(base), standard, _artifact_stamp(base))
    except OSError:
        return evaluate_all_uncached(root=root, standard=standard)
    if _EVALUATE_ALL_CACHE and _EVALUATE_ALL_CACHE[0][0] == stamp:
        return copy.deepcopy(_EVALUATE_ALL_CACHE[0][1])
    result = evaluate_all_uncached(root=root, standard=standard)
    _EVALUATE_ALL_CACHE.clear()
    _EVALUATE_ALL_CACHE.append((stamp, result))
    return copy.deepcopy(result)
```

| | value |
|---|---:|
| cold (first request) | 2.90–3.00 s |
| **warm (every later request)** | **1.4–1.5 ms** |
| warm speedup | ≈2,000× |
| stamp scan (36 files, stat only) | 0.2 ms |
| deepcopy of the 200,370 B payload | 0.5 ms |
| cache entries after mixed traffic | 1 |

### Stamp exactness — why it cannot go stale

The gate reads artifacts in exactly three places, all name-resolved under `base`:
`_discover_models` (`base.iterdir()`), `_artifact_paths` (`base / name`), and `_read` on the two paths
`_artifact_paths` returned. Every name matches `g1_g2.*.json`, so the glob covers the entire read set:
an artifact appearing, disappearing, or being rewritten changes the stamp.

Deliberate choices, each because the alternative is a stale verdict:

* `mtime_ns`, not `st_mtime` — two writes in one second must not share a stamp.
* **No TTL.** A clock cannot tell whether a file changed; a time-based expiry would add staleness
  the stamp rules out exactly.
* `standard` is in the key — it changes the advisory block, so a v2 request must not serve the v1 entry.
* `str(base)` is in the key — two roots must not alias.
* A missing/unlistable root **raises out of the stamp** and the call derives without caching.
  (`glob` returns `[]` for an absent directory, which would have cached "no artifacts" forever — the
  test `test_an_unlistable_evidence_root_derives_without_caching` caught that before it shipped.)
* One entry only, `clear()` on miss: a long-running service cannot accumulate verdicts.

**Residual gap, stated rather than hidden:** a write that preserves both `mtime_ns` and size (a
timestamp-restoring tool, or a same-length edit plus `touch -r`) would not be seen. Nothing in this
repo writes artifacts that way (`cp -p`, `git checkout`, and the producers all bump mtime). Hashing
every artifact instead would see even that, at 94 MB of reads per request — the cost this memo exists
to remove. If you want that guarantee instead, the stamp is one function and the change is local.

---

## Tests — asserted on calls and bytes, never on elapsed time

All in `workbench/tests/test_parity_absolute_relative_v2.py`:

| test | asserts |
|---|---|
| `test_a_warm_request_derives_once_and_returns_the_same_payload` | 2 requests → **1** entry into `evaluate_all_uncached`; payload equal |
| `test_touching_an_artifact_re_derives` | `os.utime` on one artifact → **2** derivations; verdict unchanged (cache misses, answer doesn't move) |
| `test_rewriting_an_artifact_is_never_served_from_the_cache` | publishing an offender into the record → 2 derivations and the status stops being PASS |
| `test_a_caller_mutating_the_payload_cannot_poison_the_next_request` | mutate returned dict → still **1** derivation, next payload clean |
| `test_the_memoised_payload_is_byte_identical_to_the_derivation` | `json.dumps(memoised, sort_keys=True) == json.dumps(uncached, …)` |
| `test_the_requested_standard_is_part_of_the_key` | v1 then v2 → 2 derivations; `standard` (applied) identical |
| `test_the_cache_holds_one_entry_only` | mixed traffic → `len(_EVALUATE_ALL_CACHE) == 1` |
| `test_an_unlistable_evidence_root_derives_without_caching` | missing root → cache stays empty |

Plus the whole-root byte-identity proof, run out of band against the pre-change module:

```
pre_change_md5   da701905419b8b9833966f637ac521a8
post_change_md5  da701905419b8b9833966f637ac521a8   (b only)
                 da701905419b8b9833966f637ac521a8   (a + b, cold and warm)
byte_identical   True   (200,370 B both sides)
```

Manual verification of the four memo properties (`/tmp/memo_check.py`):

```
calls_cold 1 | calls_warm 1 | calls_after_touch 2 | calls_after_mutation 2
equal_cold_warm true | equal_after_touch true | mutation_isolated true | entries 1
```

---

## Test counts

| selection | result |
|---|---|
| `test_parity_absolute_relative_v2.py` | **71 passed** |
| `test_parity_absolute_relative_v2.py` + `test_gate_parity_subchecks.py` + `test_gate_contract_negatives.py` | **139 passed** |
| `pytest workbench/tests -k "parity or gate"` | **385 passed, 1 skipped, 2952 deselected** |

No timing assertion anywhere.

---

## Changed paths

```
workbench/backend/parity_gate.py                    +97 −7
workbench/tests/test_parity_absolute_relative_v2.py +128
```

`workbench/backend/main.py` **not changed** (the memo is inside `evaluate_all`, so `main.py:1273`
inherits it). No artifact touched, no tolerance moved, nothing committed.
