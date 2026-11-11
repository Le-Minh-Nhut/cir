# 90a — the third condition was a whole-run abort for `tgcir`, and a lie in the plan for `clvc_net`

Branch `research/cir-failure-workbench`, 2026-11-12. Owner of this file plus a narrow fix in
`workbench/composition/model_clvc_net.py` and `workbench/composition/model_tgcir.py` (their two
parity test files updated with the reason cited). Not touched: the worker modules' scoring logic,
`runner.py`, `registry/models.yaml`, `workbench/config/workbench.env`,
`test_composition_adapters_missing.py`, `workbench/frontend/`. No GPU job launched for this section;
no full run launched.

Every claim below is read out of the code or executed. Check interpreter `workbench/envs/ctl312h/bin/python`.

---

## 1. Does the runner ask the adapter, or run every condition regardless?

**It asks.** Two separate places read `mode_supported`:

```
1666:            supported = {}
1667:            for condition in conditions:
1668:                try:
1669:                    supported[condition] = adapter.mode_supported(condition)
1670:                except Exception as error:
1671:                    supported[condition] = (False, f"{type(error).__name__}: {error}")
```

and the condition loop consumes that answer instead of iterating `conditions`:

```
1920:                active_conditions = [condition for condition in conditions if supported[condition][0]
1921:                                     and (not retry_conditions or condition in retry_conditions)]
1922:                for condition in conditions:
1923:                    if not supported[condition][0] and condition not in mode_status:
1924:                        mode_status[condition] = {"status": "ABLATION_UNSUPPORTED",
1925:                                                  "reason": supported[condition][1]}
```

The gallery is encoded only `if active_conditions:` (1928) and the encode loop iterates
`active_conditions` (1940), so an unsupported condition costs no encode and is recorded as
`ABLATION_UNSUPPORTED` with the adapter's own reason. The Top-K writer set is filtered the same way
(`export_conditions`, 1835-1837), so an unsupported condition writes no candidate file and is not
counted as a missing export.

The answer also enters the run identity (`"mode_support": {condition: supported[condition] ...}`,
1736), so a run whose adapter changed its answer mints a new run id instead of reusing ranks.

**Why that was not enough for `image_only_empty_text` before this change:** both adapters answered
`True` for every declared condition, because `mode_supported` only consulted a structural refusal.
So the runner did launch the condition and the only thing that could stop it was the per-query path.

## 2. `clvc_net` — was already degrading per query; the defect was the declaration

```
92:  class Refused(RuntimeError):
306:    empty = [index for index, text in enumerate(texts) if not str(text).strip()]
308:        raise Refused(f"adapter_empty_edit_unrepresentable: model=clvc_net rows={empty[:10]} ...")
```

Measured, not inferred:

```
Refused mro: ['Refused', 'RuntimeError', 'Exception', 'BaseException', 'object']
is AdapterWorkerError: False
is RuntimeError: True
```

`Refused` is raised adapter-side, inside `_run`, before any worker job is written. `Refused` is NOT
an `AdapterWorkerError`, so the `except Exception` at 692 takes the per-query branch and records a
named error per query — it never reaches the unconditional re-raise at 693-694:

```
692:        except Exception as batch_error:
693:            if isinstance(batch_error, AdapterWorkerError):
694:                raise
```

So `clvc_net` degrades to 6003 named `adapter_empty_edit_unrepresentable` failures. What was wrong
is that the plan said the condition was supported and then every query in it failed.

### The fix

One shared predicate, used by BOTH `mode_supported` and `status().per_mode`, so the two cannot
disagree:

```python
def _mode_support(condition: str, reason: str | None) -> tuple[bool, str | None]:
    if condition not in CONDITIONS:
        return False, f"unknown composition condition: {condition!r}"
    if condition == "image_only_empty_text":
        return False, "; ".join(item for item in (reason, EMPTY_TEXT_REASON) if item)
    return (reason is None), reason
```

(the comment in the file names `runner.py:1920-1925` and `text_model.py:84-86`).

Measured on a CONSTRUCTED adapter with the operator's config:

```
multimodal              -> (True, None)
text_only_black_image   -> (True, None)
image_only_empty_text   -> (False, "CLVC-Net cannot encode an empty edit: TextLSTMModel.forward
                                 tokenises to zero tokens and indexes x[0][0] (text_model.py:84-86),
                                 which raises IndexError for every row, so image_only_empty_text is
                                 unrepresentable in this model's text tower rather than merely
                                 unsupported by the adapter")
status per_mode         -> {'multimodal': (True, None), 'text_only_black_image': (True, None),
                            'image_only_empty_text': (False, "<same reason>")}
```

`status.ready` is unchanged (it reads the structural blockers, not the per-mode map). The registry
still declares python 3.7.6/torch 1.6.0 for the historical contract; `adapted_runtime` makes the
gate pass, exactly as 89 measured.

## 3. `tgcir` — the abort was real, and it was a bug in the correction, not in the model

`Corrections.correct` raised `ValueError` whenever `_candidates("")` returned nothing. The worker's
`run_one` catches any exception and replies failure (`model_tgcir.py` worker loop), `_run` turns a
failed reply into `AdapterWorkerError`, and the runner re-raises that unconditionally (693-694) — so
`image_only_empty_text` would have killed the model's entire run AFTER the 74 381-image gallery
encode. That is the exact failure mode the lead raised, and it was correct.

`""` is not ambiguous, and it is representable. Measured in the model's own interpreter:

```
workbench/envs/tgcir39-la/bin/python -c "import clip; print(clip.tokenize(['']))"
tokenize([""]) -> (1, 77) dtype torch.int32
first 5 ids: [49406, 49407, 0, 0, 0]   nonzero count: 2   SOT 49406, EOT at index 1
```

i.e. a valid `(SOT, EOT)` sequence, which is what `model.py:120` `clip.tokenize(txt)` accepts — the
text tower is a caption STRING encoder, not a caption-presence check.

### The fix

`correct()` now returns `""` for an input that is blank after `strip()`, before the category lookup:

```
182:            if not str(text).strip():
...                corrected.append("")
191:                continue
```

The reasoning is in the code comment: every candidate dictionary maps a blank string to `""`, so
there is nothing for the lookup to choose between; and refusing it here aborts the run through
`AdapterWorkerError` (693-694). The NON-blank ambiguity refusal is untouched.

Measured:

```
1) correct(["", "is solid white and is a lighter color"], [None, None])
   -> ["", "is solid white and is a lighter color"]
   correct(["", <same>], ["dress", "shirt"]) -> ["", "is solid white and is a lighter color"]
   multimodal              -> ["is solid white and is a lighter color"]
   text_only_black_image   -> ["is solid white and is a lighter color"]
   image_only_empty_text   -> [""]

2) ambiguity guard, synthetic two-category disagreement on a real stem (fixture deleted afterwards)
   natural ambiguous rows in _by_text: 0
   raise: tgcir caption text 0 ('a probe caption') is ambiguous: categories ['dress', 'shirt']
          correct it to ['probe dress form', 'probe shirt form']
   fixture removed; same caption now raises a different stated reason:
   raise: tgcir caption text 0 ('a probe caption') resolves to no category, so no correction
          dictionary can be chosen
```

There is no natural ambiguous row in the val dictionaries (0 of `_by_text`), which is what 88 §6
already reported; the guard is exercised on a synthetic fixture so it cannot rot.

**Consequence for a TG-CIR full run:** condition 3 now runs, with `corrected_first == ""` and a
different `corrected_digest` than the other two conditions — the empty text reaches the model as
empty, once. What it measures is a real TG-CIR forward on `""`, not a substitution.

## 4. Tests

```
workbench/envs/ctl312h/bin/python -m pytest \
  workbench/tests/test_clvc_net_composition_parity.py \
  workbench/tests/test_tgcir_composition_parity.py \
  workbench/tests/test_circus_env_gate_omp.py -q
64 passed in 1.46s
```

Adjacent shared files that read the same adapters, checked because the change is in their surface:

```
workbench/envs/ctl312h/bin/python -m pytest \
  workbench/tests/test_circus_adapters_rest.py \
  workbench/tests/test_composition_adapters_missing.py -q
55 passed in 7.34s
```

New assertions (all in the two owned parity files):

* `test_the_empty_text_condition_is_declared_unsupported_by_name` — `mode_supported` answers
  `(False, EMPTY_TEXT_REASON)` for the empty-text condition and `(True, None)` for the other two.
* `test_status_per_mode_agrees_with_mode_supported_for_the_empty_condition` — the shared predicate,
  including the structural-reason prefix form.
* `test_a_blank_caption_is_transported_as_blank_rather_than_refused` — `["", real]` returns
  `["", real]`, with and without an explicit category, and `"   "` → `""`.
* `test_a_non_blank_ambiguous_caption_is_still_refused` — the ambiguity branch still raises, and the
  same caption refuses for a different reason once the fixture is removed.

No existing assertion was weakened or deleted. `test_empty_edit_is_refused_by_name` (CLVC
`_refuse_empty_edits`) still passes unchanged: the adapter-side per-query refusal is kept, because
a plan that skips the condition is not a guarantee that no caller ever hands it an empty text.

## 5. The held run

`clvc_net` smoke, launched before this fix and **RUNNING** (not paused, not cancelled) at the time of
writing. The earlier "PAUSED (not cancelled) on instruction" line in this document was wrong and is
corrected here: `/proc/1248457/status` reported `State: R (running)` with `PPid: 1248099` while
`work/clvc_net/features/` held `gallery_dress.npy` and `gallery_shirt.npy` with the dress sidecar
already written. A reader must not think a GPU job was parked — it was consuming the card.

| field | value |
|---|---|
| run dir | `workbench/artifacts/composition/runs/889bc330d60e8cbab956944f` |
| PID (control) | 1248099 |
| PID (GPU child) | 1248457, 1320 MiB |
| start (UTC) | 2026-11-11T17:26:57Z |
| stage at time of writing | gallery encode, second of three scopes (`gallery_dress.npy` 609 329 280 B + `gallery_shirt.npy`, dress sidecar written); observed 2026-11-11T18:50:55Z |
| conditions | all three launched under the OLD `mode_supported`; the empty-text condition is the one this fix now declares unsupported |

That smoke was started under the pre-fix code, so its `image_only_empty_text` will show 5 named
`adapter_empty_edit_unrepresentable` failures rather than `ABLATION_UNSUPPORTED`. It is evidence
about the old behaviour, not about the fix.

## Còn chặn

1. **No full run in this section.** The two `--subsets full` runs are still held. The fix makes them
   launchable without the abort; it produces no metric.
2. **`runner.py` is unchanged**, so a future adapter that raises `AdapterWorkerError` on an input it
   cannot encode would still abort the run. That is by design (a worker failure is not a per-query
   fact), but it means `mode_supported` must be truthful for every adapter — the reason this file
   exists.
3. **`clvc_net` still fails per query if the condition is reached anyway** (`_refuse_empty_edits`
   kept). The declaration is a plan-level signal; it is not a guard against a direct caller.
4. **`tgcir`'s empty-text numbers are unmeasured.** 88 §4 measured rank movement on 5 queries with an
   empty text through the worker; the aggregate is still unmeasured.
