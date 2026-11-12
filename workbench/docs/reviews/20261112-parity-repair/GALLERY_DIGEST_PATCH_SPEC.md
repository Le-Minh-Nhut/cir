# Patch spec — one key, two `gallery_identity_sha256` formulas

Status: spec only. `workbench/scripts/composition_eval.py` and `workbench/composition/runner.py`
are owned by another pass; this document is the change request, not the change.

## Defect (confirmed)

`gallery_identity_sha256` names two different digests of the same gallery, so a digest published by
one branch of one command can never equal the digest of a completed run of that same command.

| where | function | payload | value on this release |
|---|---|---|---|
| plan / no-runnable-spec branch | `workbench/scripts/composition_eval.py:298-300` (`_plan`) | `{"protocol_id", "image_ids"}` | `0bfd43b9cc0162885f06c60cfc751602ee9f3bed2fb4d206ffcebe3f7b443462` |
| plan / no-runnable-spec branch | `workbench/scripts/composition_eval.py:426-429` (inline copy in `main`, the `if specs else {...}` branch) | same as above | `0bfd43b96…` |
| every completed run | `workbench/composition/runner.py:512-515` (`_gallery_fingerprint`) | `{"protocol_id", "ordered_gallery", "candidate_names"}` | `1db68c59d96f014cdbecf936e108653de98abef0bd9a98a8abc2fe08b5dec4b4` |

Evidence both values describe one gallery: `gallery_count` is `74381` in both, and the same run
`7b1fa85007bffdab931f7f19` publishes `1db68c59…` in its rank artifact while
`workbench/docs/reviews/20261111-master-audit/raw/compose_probe/composition_plan.log:342` publishes
`0bfd43b9…` for the identical release. Recorded already in
`raw/verify/104-clvc-full-image-only.md:65-69`.

The consequence is not cosmetic: a plan digest and a run digest are the only fields a caller can
compare to decide "this plan produced this run". Publishing two formulas under one key makes that
comparison permanently false, and a consumer that starts relying on the plan value silently pins
itself to the plan branch.

## Resolution — reuse `_gallery_fingerprint`

`gallery_identity_sha256` should mean exactly one thing: **the runner's gallery fingerprint**, the
value every completed run records. Both plan-branch sites must call the runner's helper instead of
re-deriving a second formula.

### Change 1 — `workbench/scripts/composition_eval.py`, `_plan` (lines 298-300)

Replace

```python
    gallery_identity = hashlib.sha256(json.dumps(
        {"protocol_id": data.gallery.protocol_id, "image_ids": data.gallery.image_ids},
        sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
```

with

```python
    from workbench.composition.runner import _gallery_fingerprint

    gallery_identity = _gallery_fingerprint(data)
```

and leave line 305 (`"gallery_identity_sha256": gallery_identity`) untouched.

### Change 2 — the same file, the blocked-report branch (lines 426-429)

The `if specs else {...}` fallback repeats the formula inline. Replace the inline
`hashlib.sha256(json.dumps({...}, ...))` expression with a call to the same helper, computed once
before the `run_composition(...) if specs else {...}` expression, e.g.

```python
    gallery_identity = _gallery_fingerprint(data)
    report = run_composition(...) if specs else {
        "run_count": 0, "models": [], "gallery_count": len(data.gallery.image_ids),
        "gallery_identity_sha256": gallery_identity,
        ...}
```

`_gallery_fingerprint` is `workbench/composition/runner.py:512` and is already imported by name in
this module's imports of `runner` (both `run_composition` and `_checkpoint_identity` come from
there), so no new dependency edge is introduced. `hashlib` remains used elsewhere in the file; if
it does not, drop the import — do not leave it dead.

### Why not the alternative (rename the blocked-branch key)

Renaming the blocked-branch/plan key to `gallery_image_id_sha256` is a valid second option and it
also breaks no published digest — but it keeps two formulas alive and forces every future consumer
to know which of the two it holds. Reusing the runner's helper removes the formula instead of
documenting it, and it makes the plan-checkable equality "does this plan's digest equal the run's"
true by construction.

## Which choice breaks no published digest

**Both.** Neither the plan digest nor the run digest is embedded in any stored artifact that is
re-verified on read:

* `gallery_identity_sha256` is written into summaries and candidate manifests
  (`runner.py:2508`, `runner.py:2710`, `topk.py:209`) by the runner and is only ever compared by a
  reader against another value read from the same file
  (`workbench/backend/composition.py:1431` compares it to the manifest it came with).
* The plan document is printed to stdout; it is not an input to run-id derivation
  (`runner.py:1820` computes its own `gallery_hash`).
* No on-disk artifact stores `0bfd43b9…` except the audit transcripts under `workbench/docs/`,
  which are historical records of what that binary printed and must not be rewritten.

So changing the plan branch to emit `1db68c59…` invalidates nothing already on disk. Historical
transcripts keep the old value and should say so if anyone re-reads them; the fix is forward-only.

## Test to add

In `workbench/tests/` (a new `test_gallery_identity_one_formula.py`, or appended to
`test_composition_topk_runner.py`, which already owns the manifest digest at line 136):

```python
def test_plan_and_run_publish_the_same_gallery_digest():
    """The plan branch and a completed run must publish ONE digest for one gallery."""
    from workbench.backend.composition import load_cached_release
    from workbench.composition.runner import _gallery_fingerprint
    from workbench.scripts import composition_eval

    data = load_cached_release()
    assert data is not None
    plan = composition_eval._plan([], [], data, test_config(), ["full"], ("multimodal",))
    assert plan["gallery_identity_sha256"] == _gallery_fingerprint(data)
    # today: 0bfd43b9…  != 1db68c59…  -- the fix is what makes this pass
```

`_plan` takes `(pairs, unavailable, data, config, subsets, conditions)`; `pairs=[]` is enough
because the digest is computed after the model loop and the loop body is skipped for no pairs.
`test_config()` is whatever `WorkbenchConfig` fixture the neighbouring composition tests use. If
constructing a config is awkward, the same assertion can be made against the blocked branch by
monkeypatching every model's readiness false.

```python
def test_the_blocked_report_branch_publishes_the_run_digest(...):
    # force `specs == []` (every model blocked) and assert the printed report's
    # gallery_identity_sha256 equals _gallery_fingerprint(data) -- the value a completed run records.
```

Both must fail on today's source (`0bfd43b9… != 1db68c59…`) and pass after the change. A third
assertion worth keeping: the value is stable across two calls in one process, i.e. it is a
function of the loaded release and not of dict ordering (`candidate_names` is sorted in
`_gallery_fingerprint`, so this holds).
