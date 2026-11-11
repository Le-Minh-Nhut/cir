# R05 — a gutted annotation file is not evidence of absence

Defect: `caption_scope_index` used JSON validity as its only test. `cap.toptee.val.json` emptied to
`[]` parses, and so does one row with a wrong-typed `candidate`. Both produced a unique surviving
holder, so `resolve_query_category` returned `"shirt"` for a genuinely ambiguous pair instead of
`None`. The loader had the same hole through a different door: `assign_garment_categories` accepted
an emptied split because `len({...}) == len([])`, silently deleting a category's identities so its
queries routed elsewhere.

Files touched: `workbench/composition/data.py`, `workbench/composition/model_limn.py`,
`workbench/tests/test_limn_annotation_provenance_r05.py` (new),
`workbench/tests/test_limn_scope_routing.py` (one fixture line), `workbench/tests/test_circus_cli.py`
and `workbench/tests/test_circus_data.py` (fixtures that wrote `[]` now write honest non-empty
rows). `runner.py` untouched.

## 1. Schema validation and the pin

`data.ANNOTATION_PIN` holds, per file, the committed `sha256` and row count. Provenance:
`workbench/docs/reviews/20261111-master-audit/raw/m02/qid_scope_map.json`
(`sha256.caption_files`, `sha256.loader_identity_support_files`), re-measured on disk. Covers the
three `cap.<category>.val.json`, three `split.<category>.val.json`, three `cap.<category>.test.json`.

`data.inspect_annotation(path, *, shape, pin)` returns an `AnnotationReport` whose `problem` is
`None` only when every row is well formed (string non-empty `candidate`; non-empty list of strings
`captions`; split shape: non-empty strings), the file is non-empty, and count+digest match the pin.
Otherwise `unreadable` / `malformed` (with the offending row index and reason) / `empty` /
`count_mismatch` / `digest_mismatch`. `inspect_annotation_set` keeps order;
`describe_annotation_reports` is the single published diagnostic block.

`CaptionScopeIndex` now carries `unreadable`, `malformed`, `empty`, `mismatch`, `counts`,
`digests`, `complete`, the derived `incomplete`, and `report()`. Keys are categories: the pre-existing
tests at `test_limn_scope_routing.py:214,224` already asserted
`index.unreadable == frozenset({"toptee"})`, so the category form is the established one.
`resolve_query_category` refuses whenever `incomplete` is true, i.e. for all three states the task
names, because in each the index cannot distinguish "no row here" from "file never read / gutted".

`pin` defaults: `caption_scope_index(..., pin=None)` means "this root has no pinned bytes" (the
hand-built fixtures); the ONE released-data caller, `Adapter.annotation_pin = ANNOTATION_PIN`,
passes the pin, so live routing refuses an off-pin file. `assign_garment_categories` takes
`annotation_pin=None` and resolves it through `resolved_annotation_pin(data, pin)`: explicit argument
wins, else `{}` when `provenance["is_synthetic"]`, else `ANNOTATION_PIN`.

Proof: `test_empty_annotation_file_refuses_routing`, `test_malformed_row_refuses_routing`,
`test_counts_and_digests_are_enforced_against_the_pin`, `test_loader_refuses_a_real_set_that_is_off_pin`,
`test_pin_is_data_owned_and_matches_the_committed_scope_map` (reads the committed map and compares).

## 2. Loader refuses a partial identity set

`assign_garment_categories` inspects all six identity files (3 splits + 3 test captions) before
touching any query. Any unusable file, or a split whose stems repeat, returns the dataset with every
`category=None` and `category_resolution.status = "unavailable"` carrying
`unavailable_reasons: {file: problem}` and the full `annotation_provenance`. No category is ever
built from a subset. Proof: `test_loader_emptied_split_makes_assignment_unavailable` (empties
`split.toptee.val.json`, asserts `None` + `unavailable` + reason `empty`, then restores it and shows
the original ambiguity returning), `test_loader_names_every_unusable_file_and_never_partially_assigns`,
`test_loader_reports_a_dishonest_duplicate_split`.

## 3. Loader routes are marked unverified, never re-decided

`resolve_query_scope(index, query, names)` returns the decision record:
`{qid, scope, reason, provenance_incomplete, holding_categories, annotation_complete}`. `reason` is
`loader` / `annotation:<category>` / `unresolved:<why>` where `<why>` is
`no_reference_identity`, `annotation_index_incomplete`, `no_annotation_row`,
`annotation_names_two_categories`. The loader route is still taken first and unchanged; only
`provenance_incomplete` flips to `true` when the annotation set cannot confirm it. A plain `dict`
index is treated as incomplete by `_index_complete` (it has no provenance to prove anything).
Proof: `test_loader_assigned_category_is_reported_as_unverified_when_the_index_is_incomplete`,
`test_a_plain_dict_index_cannot_prove_completeness`.

## 4. Exported mapping

`Adapter.query_scope` records each decision; `Adapter.query_scope_resolution()` returns
`{model_id, resolver, annotation (full per-file report), counts_by_reason, unresolved, decisions,
scope_by_qid, provenance_incomplete_count, query_count}`. `Adapter.close()` writes it to
`<artifacts>/composition/runs/<run_id>/scope/query_scope_limn.json` — beside `ranks/`, so it lands
in the run directory the inventory already enumerates, is written before `runner.py`'s
`shutil.rmtree(work_root)`, and needs no runner change. The write is best-effort (an `OSError`
returns `None`): routing provenance is a record, never a scoring input.

No `runner.py` change required: the artifact is self-contained, and `runner.py` already copies
`data.provenance["category_resolution"]` into the summary verbatim, so `annotation_provenance` and
`unavailable_reasons` appear there with no edit.

Proof: `test_adapter_exports_a_per_query_rationale_map`.

## 5. `7:3038` and `7:3790`

Both stay unscored on the real committed data, each with
`reason = unresolved:annotation_names_two_categories`, `holding_categories = ("shirt", "toptee")`,
`provenance_incomplete = false`, and a `candidates` block naming each category file, its row count,
whether it was usable, and whether it holds the pair. No category is picked. Proof:
`test_real_unresolved_pair_has_its_own_diagnostics`.

## Real annotation files validated

| file | rows | sha256 |
|---|---|---|
| `data/FashionIQ/captions/cap.dress.val.json` | 2017 | `5e5117d45695df9c3ca91fac3e0bc49e6422ae9c8b3793f0ac9b3ab83adb7de9` |
| `data/FashionIQ/captions/cap.shirt.val.json` | 2038 | `7b7ca3797b85dddfd83e74cdb4454cb63e3e7c09acef2590155efbacea6d7feb` |
| `data/FashionIQ/captions/cap.toptee.val.json` | 1961 | `b4e09f428e6c255cc4b4ec46e77908305ab83075d82840c960be81af15cdc5eb` |
| `data/FashionIQ/captions/cap.dress.test.json` | 2024 | `190240b3b4d115cd4f80de65ab78ca18a3f8581c041cbeea6f5a1f57ee4f5a48` |
| `data/FashionIQ/captions/cap.shirt.test.json` | 2038 | `27c7551b005d4befa53135a7ddee5e31c97821eafd3ceb3fec514e7ac0ad83a7` |
| `data/FashionIQ/captions/cap.toptee.test.json` | 2056 | `db822aa52340ce1f09c3000064f79b60b6c476b50b98bf34200dd2996c06e146` |
| `data/FashionIQ/image_splits/split.dress.val.json` | 3817 | `21ff91d53c23859da91bfd49f3acc139b7f3a3dc944fc4e4a80194468cf14bab` |
| `data/FashionIQ/image_splits/split.shirt.val.json` | 6346 | `b82effbf7352a6f828b38c45eb32e53d726bd4d6fd81d6290c81eeb2c68e3234` |
| `data/FashionIQ/image_splits/split.toptee.val.json` | 5373 | `ee42b2505275dd7a19b11ddb88c9264aab2d77f4e59b075fb1c7c3095d02f412` |

Good path unchanged on real data: index `complete`, 6003 keys, loader `resolved 5884 / ambiguous 119
/ unresolved 0`; of the 119, exactly 2 stay unscored (`7:3038`, `7:3790`), 117 route (shirt 46,
toptee 71). Proof: `test_a_complete_pin_matching_set_resolves_exactly_as_before` plus the pre-existing
`test_all_119_unscored_queries_are_classified`, `test_rescoring_arithmetic_closes_at_6003`.

## Probe, before / after

`/tmp/protocol_pure_probe_r05.py` (probe copied to `/tmp/`, imports repointed at
`/home/heheboiz/data/cir`). Before-run uses a HEAD snapshot of `data.py`/`model_limn.py` at
`/tmp/r05_before/`.

```text
case                  HEAD        after
both_valid            None        None
toptee_empty          'shirt'     None     <- defect
toptee_invalid_row    'shirt'     None     <- defect
complete set, 1 holder  (n/a)     'shirt'  <- control, still resolves
loader toptee emptied 'shirt'     None     <- defect
```

`toptee_empty` reports `empty: ["dress","toptee"]` (`dress` is `[]` in the probe's own fixture, which
is now correctly unusable too), `toptee_invalid_row` reports
`malformed: {"toptee": "row 0: candidate is int, expected non-empty string"}`.

## Tests

`workbench/envs/ctl312h/bin/python -m pytest workbench/tests -k "limn or categor or scope" -q --basetemp=/tmp/r05_bt`
→ **failure set identical to baseline; 376 passed, 2 failed, 2 skipped** in 74.9 s.

Two failures, both pre-existing on this host and unrelated to R05 (identical on the pre-change
baseline):
`test_circus_tme_lifecycle.py::test_encoder_and_limn_failed_child_latch_and_preserve_log[encoder]`
and `[limn]` — `RuntimeError: execution environment not verified [UNCONFIGURED/UNCONFIGURED]:
WORKBENCH_ENCODER_PYTHON=None` / `WORKBENCH_LIMN_PYTHON=None`. They need model interpreters this host
does not configure; no model or GPU job was launched.

Existing tests whose FIXTURES encoded the old behaviour (explicitly, not silently):
`test_circus_cli.py:69-72` wrote `[]` for every non-dress split and every `cap.*.test.json`;
`test_circus_data.py:239-244` wrote `[]` for `split.*.val.json` and `cap.toptee.test.json`;
`test_limn_scope_routing.py:143` had no `annotation_pin` on an `object.__new__(Adapter)` seam. Each
now writes honest non-empty filler / passes `{}` where the bytes are synthetic. No assertion about
R05 behaviour was weakened: `test_circus_data.py:247` still asserts `status == "ambiguous"` and
`ambiguous_count == 6003`, it just no longer relies on `[]` meaning "no garment here".
