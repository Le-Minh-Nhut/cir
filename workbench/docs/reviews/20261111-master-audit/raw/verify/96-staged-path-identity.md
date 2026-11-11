# 96 — The staged query image must keep the REFERENCE STEM, not the query id

Branch `research/cir-failure-workbench`, 2026-11-12. Owner of this file plus a narrow change in
`workbench/composition/runner.py` (`_stage_query_path` only) and its tests. Not touched:
`model_tgcir.py`, `model_clvc_net.py`, `models.yaml`, `workbench.env`, `workbench/frontend/`.
Every GPU command under `workbench/scripts/gpu_job.sh` with `MAX_WAIT=3600`; ports 8000/5173 never bound.

---

## 1. The defect, as measured

TG-CIR full/smoke run `9f115afeb58eb1adf75d88ba` reported, verbatim:

| condition | status | unscored | reason |
|---|---|---|---|
| `multimodal` | `COMPLETED` | 0 | — |
| `text_only_black_image` | `MODEL_UNAVAILABLE` | **6003** | `AdapterWorkerError: adapter_worker_job_failed: model=tgcir phase=queries error=ValueError: tgcir caption text 1 ('Is grey with a design on the back.') resolves to no category, so no correction dictionary can be chosen` |
| `image_only_empty_text` | `MODEL_UNAVAILABLE` | **6003** | same |

Mechanism. `_stage_query_path` wrote the blacked-out reference to
`work_root/queries/<condition>/<qid>.png`, so the adapter received a path whose
`Path(path).stem` is the **qid** (`7-1`), not an image id. `Corrections.correct` resolves a caption's
category in the order: explicit caller category → `(reference stem, raw text)` → raw text → reference
stem. The runner passes `categories=[None…]` for TG-CIR (`category_scopes = (None,)`), and the release
sentence is not an exact `_by_text` key, so the STEM was the only remaining resolver — and staging had
destroyed it. `_candidates` returned `{}` → `ValueError` in the worker → `_run` raised
`AdapterWorkerError` → `runner._encode_queries` re-raised it unconditionally, zeroing the condition.

Measured on the release, exactly **3 of 6003** queries resolve only by stem:

```
('7:3938', 'Is grey with a design on the back.', '7:20350')
('7:4723', 'Is an off the shoulder top.',          '7:63668')
('7:5123', 'Fades from red to orange.',            '7:70795')
```

These are the same three rows 88 §6 named. Under `multimodal` the real image path arrives, the stem
resolves, and the condition completes — which is why only the staged condition died.

## 2. The fix

```python
staged = work_root / "queries" / condition / f"{Path(path).stem}.png"
```

`staged.parent.mkdir(parents=True, exist_ok=True)` is kept. Nothing about the pixels, the text
transport, or `model_tgcir.py`'s refusal changed: the adapter is right to demand a category and its
stem-based resolution is pinned behaviour.

### Why the shared filename is safe — verified, not assumed

`inputs.black_image_like` is `Image.new(image.mode, image.size, 0)` (`inputs.py:14-16`), so the staged
bytes depend ONLY on the source's mode and size. Two queries with the same stem refer to the same
source file (`image_resolver` maps an identity to one path), therefore the same mode and size, therefore
identical bytes: the overwrite cannot change what an earlier query encoded. The qid is not needed in the
name because `_encode_queries` keeps its own index→path mapping.

## 3. Proof

### 3a. Unit tests (added to `workbench/tests/test_composition_topk_runner.py`)

* `test_the_staged_text_only_image_is_named_by_the_source_stem_not_the_qid` — for qid `7:1` and source
  `B005X4PL1G.jpg`, the staged path's `.stem == "B005X4PL1G"`, suffix `.png`, text unchanged, pixels
  all-zero at the source's 8×6 size; and `image_only_empty_text` returns the ORIGINAL path untouched.
* `test_two_queries_sharing_a_reference_collide_safely` — two qids sharing one reference stage to the
  same file, and the second write does not change the first query's bytes.

Staged filename examples (shape): `<work_root>/queries/text_only_black_image/B005X4PL1G.png`,
`…/B00AKLK08G.png` — the reference image stem, one PNG per distinct reference.

### 3b. TG-CIR smoke, post-fix — run `1eee4c0adbc93f7708205a5e`

5 queries, `--topk-depth 100`, `--subsets full`, 409 s under the GPU lock.
`execution_status: COMPLETED`.

| condition | status | unscored |
|---|---|---|
| `multimodal` | `COMPLETED` | 0 |
| `text_only_black_image` | **`COMPLETED`** | **0** |
| `image_only_empty_text` | **`COMPLETED`** | **0** |

Per-condition: `multimodal` R@10 20.0 / R@50 40.0; `text_only` R@10 0.0 / R@50 0.0; `image_only`
R@10 0.0 / R@50 0.0 (all 5/5 scored — a 5-query sample is a plumbing instrument, not a metric).
`topk_export`: `{"depth": 100, "partial": false, "missing_query_count": 0, "status":
"TOPK_EXPORT_COMPLETE"}` — the first TG-CIR run in this campaign with all three windows written.

### 3c. `corrected_digest` per condition — measured through the real worker

Worker replies are deleted after each run, so this was measured on the same query through the same
adapter (qid `7:1`, one real reference image, under the GPU lock):

```
qid 7:1 | query_txt 'Is shiny and silver with shorter sleeves and fit and flare.'
multimodal            corrected_first='is shiny and silver with shorter sleeves and fit and flare'
                      digest=3174ba7f60c00d56554a629f80c34599f39a0a768ead99cfa27ee06376f4dbd1
text_only_black_image corrected_first='is shiny and silver with shorter sleeves and fit and flare'
                      digest=3174ba7f60c00d56554a629f80c34599f39a0a768ead99cfa27ee06376f4dbd1
image_only_empty_text corrected_first=''
                      digest=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

The empty-text digest differs from the other two, and `corrected_first == ''` — the empty caption
reaches the model as empty, exactly once, which is what commit `5cb0103` made possible. The two
non-empty conditions share a digest, correctly: they carry the SAME text and differ only in pixels.

### 3d. CLVC smoke, post-fix — run `d541e40cde1a949026f98854` (regression check)

CLVC also stages for this condition, so it had to be re-checked. `execution_status: PARTIAL`,
exit 1 by declared ablation.

| condition | status | unscored |
|---|---|---|
| `multimodal` | `COMPLETED` | 0 |
| `text_only_black_image` | **`COMPLETED`** | **0** |
| `image_only_empty_text` | `ABLATION_UNSUPPORTED` (with `EMPTY_TEXT_REASON`) | — |

`topk_export` covers only `["multimodal", "text_only_black_image"]` with `missing_query_count: 0`,
`partial: false`, `status: TOPK_EXPORT_COMPLETE` — the unsupported condition is not listed and is not
counted as a missing window. No regression: `text_only_black_image` still completes, and the
empty-text condition is still declared unsupported rather than attempted.

### 3e. Tests

```
workbench/envs/ctl312h/bin/python -m pytest \
  workbench/tests/test_composition_topk_runner.py \
  workbench/tests/test_composition_api.py \
  workbench/tests/test_clvc_net_composition_parity.py \
  workbench/tests/test_tgcir_composition_parity.py -q
62 passed, 5 warnings
```

No existing test asserted the old call granularity; nothing was weakened or deleted. One assertion in
my own new test used `Image.getdata()`, replaced with `getextrema()` to avoid the Pillow 14
deprecation rather than ship a warning.

## Còn chặn

1. **The two `--subsets full` runs have not been started** under this fix at the time of writing. The
   numbers they must record are: wall time, images encoded (74 381 × 3 scopes for CLVC; TG-CIR encodes
   the same gallery once, one category-agnostic scope), s/img, peak VRAM, per-condition metrics and
   `condition_status`.
2. **A second, distinct TG-CIR failure was observed** in the retry attempt of `9f115afe`
   (`adapter_worker_exited … exit_status=1`, worker stderr already deleted). The staged-stem fix is
   necessary but was not proven sufficient for that path; the post-fix smoke (§3b) completed all three
   conditions, so it is not reproducible on the 5-query sample. Unresolved.
3. **`multimodal` and `text_only_black_image` share a `corrected_digest`** and that is correct, not a
   bug — same text, different pixels. Recorded so a reader does not read the equality as a defect.
