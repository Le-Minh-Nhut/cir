# R04 — reusable replay cache roots now prove their provenance

Defect: `workbench/replay/common.py::reusable_cache_root` yielded whatever cache root the parent
named. The native constructors take their warm branch on the mere presence of `fashion_iq_data.json`
and unpickle `test_{queries,targets}_<cat>.pkl` without re-running the transform, so a root that
symlinked `OLD_ANNOTATIONS` and held tensors from a foreign CLIP transform satisfied a request for a
different dataset root, loaded the foreign pixels and ran the declared transform zero times. The
docstring claimed `assert_alignment` refused such a cache; nothing did.

## Manifest contract

`CACHE_MANIFEST_NAME = "workbench_cache_manifest.json"`, `CACHE_MANIFEST_SCHEMA_VERSION = 1`,
written by `write_cache_manifest()` with a `NamedTemporaryFile` in the same directory plus
`os.replace` (and `fsync`), so a crashed warm pass leaves the previous complete manifest or none.

Fields: `schema_version`; `model_id`; `checkpoint.{checkpoint_id, expected_sha256}`; `transform`
(descriptor built by `describe_transform()` from the caller's actual transform object — class path
plus `size`/`ratio`/`scale`/`mean`/`std`/`interpolation`); `annotations.<name>.{cache_entry,
is_symlink, requested_path, resolved_path, digest}`; `gallery.<cat>.{count, digest, first, last}`;
`state.{seed, eval_mode, noise_ratio}`; `tensors.{file: sha256}`; `cache_files`; `written_by`;
`written_at`.

`resolved_path` is `Path.resolve()`, so a symlink is recorded as its target and a link whose target
has moved no longer compares equal — the probe's first failure step.

## Refusal rule

`reusable_cache_root` raises `CacheManifestError(cache_root, mismatches)`, where `mismatches` names
every differing field. Refused on: missing manifest, schema-version change, any required field
difference (model, checkpoint, transform, annotation requested/resolved path or digest, gallery
count/order/digest, seed/eval state), a tensor digest that no longer matches the manifest or a tensor
on disk the manifest does not describe, and any input symlink whose `resolve()` is not
`(dataset_root / name).resolve()` (checked first, always).

Refuse rather than silently rebuild: a silently reused foreign cache costs a published wrong
ranking; a rebuild costs minutes of CPU the operator can see. Opt-out:
`WORKBENCH_CACHE_MANIFEST_OPT_OUT=1` restores trust-whatever-is-present, and the run records
`cache_manifest_status: not_verified_opt_out` plus `cache_manifest_opted_out: true` — never silent.

Statuses: `verified`, `rebuilt` (cold, or fresh isolated root), `not_verified_opt_out`. On the way out
of a cold construction the manifest for the tensors it just wrote is written, so the next child of the
same run verifies instead of being refused.

## Consumers

`airknow.py`, `tgcir.py`, `clvc_net.py` pass `requirement=cache_requirement(...)` built from their own
model id, the transform object they actually apply, their checkpoint id/digest and
`state={seed, eval_mode, noise_ratio}`, and record `provenance.cache_manifest` with
`cache_manifest_status`, `cache_manifest_reason`, `cache_manifest_path`, `cache_manifest_digest`,
`cache_manifest_scope`, `cache_manifest_checked_fields`, `cache_manifest_opted_out`.

## warm_replay_cache.py and `cache_files: []`

`TENSOR_WRITING_MODELS = ("airknow", "tgcir", "pair")` — those constructors save the pickles.
CLVC-Net's does not (`CLVCNet/datasets.py:152` keeps tensors in memory), so `--model clvc_net` reports
`cache_reusable: false`, `cache_files: []` and a `reason` instead of implying a cache. The campaign
claim "CLVC reuse explained by tensor reuse" is therefore unsupported: there were never any files.
For warmable models the script verifies any pre-existing manifest, discards tensors it cannot verify,
runs the constructor, and writes the manifest from the files that were actually produced.

## Probe

`/tmp/r04probe/cache_and_absence_probe.py` (copy, imports repointed at the repo; the only call change
is passing `requirement=cache_requirement(requested_dataset, model_id="airknow")` and catching
`CacheManifestError`). Before: `effective_annotation_identity="OLD_ANNOTATIONS"`,
`loaded_query_pixels="FOREIGN_CLIP_TRANSFORM_TENSOR"`, `new_targetpad_transform_calls=0`,
`error=null`. After: `CacheManifestError` naming
`annotations.captions.resolved_path`, `annotations.image_splits.resolved_path`,
`annotations.resized_image.resolved_path`; nothing was loaded.

## Tests

`workbench/tests/test_r04_cache_manifest.py` (16 tests): foreign symlink refused, manifest over
foreign links refused, rewritten tensor refused by digest, different transform refused, matching
manifest accepted+recorded, moved symlink refused, no manifest refused, schema change refused, gallery
count change refused, opt-out explicit+visible, cold root reports `rebuilt`, isolated root has no
manifest, atomic write, both annotation paths recorded, gallery order/count, compare names every
field. `test_seeded_dataset_construction.py::test_a_run_shares_one_cache_between_its_children` updated
to write the manifest and assert `verified` (it previously asserted the defect: sharing without proof).
