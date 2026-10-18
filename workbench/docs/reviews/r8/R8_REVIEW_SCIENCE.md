# R8 Review — SCIENCE (LIMN ranking, parity, checkpoint identity, protocol invariants)

Perspective: scientific semantics. Target: `workbench/replay/limn.py`, `limn_export.py`,
`limn_publish.py`, `workbench/scripts/evaluate_models.py` (LIMN observation path + `manifest_digest`),
`workbench/registry/models.yaml`. Method: full `git diff` read, end-to-end trace of the LIMN
observation path, pinned-source inspection, and digest-level probes with
`.venv/bin/python` (commands and outputs quoted below). No file under the repo was modified; no
full test run was executed.

Base facts.

- `git status --short`: modified = `pc.sh`, 4 docs, `workbench/scripts/evaluate_models.py`,
  `workbench/tests/test_handoff_commands.py`, `workbench/tests/test_pc_launcher.py`.
  **`workbench/replay/`, `workbench/backend/`, and `workbench/registry/` are untouched**
  (`git status --short workbench/replay/` is empty).
- `git diff --stat workbench/scripts/evaluate_models.py` → `20 ++` / `0 --`: a single pure
  insertion, hunk `@@ -1747,6 +1747,26 @@`, inside `_run_observation`
  (`workbench/scripts/evaluate_models.py:1632`).
- Pinned LIMN source: `git -C workbench/third_party/LIMN rev-parse HEAD` →
  `7d7bc9b116f594a65ac22457491edf28a88d3c3e`, exactly `upstream_commit_sha`
  (`workbench/registry/models.yaml:916`). Dirty state is `__pycache__` only.

---

## CLAIMS ATTACKED

### C1. "The `evaluate_models.py` change only ADDS a binding check and alters no scientific quantity."

**Verdict: HOLDS (attack failed).**

The diff is +20/−0. The insertion occupies
`workbench/scripts/evaluate_models.py:1748-1769`, inside `_run_observation`, between the
existing run-id/model/protocol artifact binding and the artifact digest computation:

```python
planned_digest = plan.bundle_manifest_digest or manifest_digest(_plan_members(plan))
if planned_digest:
    recorded_digest = payload["run"].get("bundle_manifest_digest")
    if recorded_digest != planned_digest:
        return None, (
            f"observer artifact records bundle_manifest_digest={recorded_digest!r}, but this "
            f"run planned {planned_digest!r}: ..."), None, None
```

Every scientific quantity is computed elsewhere, and none of those files is in the diff:

- **Ranking** — native `test()` call: `limn.py:201-203` (`replay_category`), and
  `limn_export.py:observe_and_verify` (same loaded objects). Observer re-derivation:
  `limn_observer.py:rank_queries` (L2-normalize, dot, `NATIVE_REFERENCE_MASK = -10e10` at
  `limn_observer.py:38`, `np.argsort(-sims)` at `limn_observer.py:162`). Files untouched.
- **Recall / parity** — `limn_observer.py:74-77` (`recall(k) = 100 * sum(target_rank <= k)/n`)
  and `cross_check_with_native` (`limn_observer.py:352-371`, exact-count comparison at k ∈ {1,10,50}).
  Untouched.
- **Gallery membership** — `limn_observer.py:276-280`: gallery is `dataset.test_targets`
  (the native ordered unique union of references+targets), reused verbatim. Untouched.
- **Reference eligibility** — mask, not removal: `limn_observer.py:149-150` and
  `limn_observer.py:346-347` (`eligible_count = len(gallery_ids) - 1`); native twin at
  `LIMN/test.py` (`sims[i, test_targets_id.index(t['source_img_id'])] = -10e10`). Untouched.
- **Checkpoint loading** — `common.py:43-55` (`load_trusted_whole_model` re-hashes before
  unpickling) and `limn.py:171`. Untouched.

The new code runs strictly **after** the child observer and publisher have produced the
artifact, and its only effect on failure is to return `(None, note, None, None)`. Downstream,
that flips at most `per_query_state` to `PER_QUERY_EXPORT_FAILED`
(`evaluate_models.py:1586-1595`); the report's `observed_metrics`
(`evaluate_models.py:1956`), `benchmark_gate` (`_benchmark_gate_verdict(plan, extraction, code)`,
`evaluate_models.py:1926` — depends only on plan/extraction/return code, all computed before the
observation), and the report's own `bundle_manifest_digest`
(`evaluate_models.py:1836-1837`, derived from the plan, not the artifact) are all independent of
it. The new code cannot alter a metric, a ranking, a gallery, an eligibility, or a load.

No behaviour other than the new refusal changed: there is no other hunk.

### C2. "The published artifact still records the correct bundle identity."

**Verdict: HOLDS (attack failed), with the binding gap the patch closes being real.**

Chain of custody, all verified:

1. Plan side (outside the payload): `guarded_plan` builds the manifest over the three members
   on disk and hashes them before the observer runs — `evaluate_models.py:393-401`:
   `names = [f"0_{cat}_best_model.pt" for cat in ("dress", "shirt", "toptee")]`,
   `checkpoint_bundle = build_artifact_manifest(bundle_root, names)`,
   `bundle_manifest_digest = manifest_digest(checkpoint_bundle)`.
   `build_artifact_manifest` hashes actual bytes
   (`evaluate_models.py:1125-1150`: `sha256_file(member_path) if member_path.is_file() else None`),
   and a missing member is a plan blocker, so a plan can never exist over absent members
   (`runtime_plan_blockers` "checkpoint bundle incomplete", `workbench/backend/runtime.py:2467+`
   region, cited at `evaluate_models.py:347,372`).
2. Observer side: the payload's `checkpoint_members` is built from `LIMN_CHECKPOINTS`
   (`limn_export.py:93-98`), which is derived from the registry (`limn.py:23-28`,
   `LIMN_CHECKPOINTS = {category: {...next(item for item in MODEL["checkpoint_variants"] ...)}}`) —
   no hardcoded digests in code. Those exact bytes were hash-verified before any observation
   (`load_trusted_whole_model`, `common.py:49-51`).
3. Publisher side: `bundle_digest = manifest_digest(members)` over the payload members
   (`limn_publish.py:101`), recorded as `run.bundle_manifest_digest` via
   `publish_observed_categories(... bundle_manifest_digest=bundle_digest ...)`
   (`limn_publish.py:113`; field at `workbench/backend/schemas/results.py:122`, inside the
   `run` block the parent reads — `ResultRun.run`, `results.py:147-151`).
4. Report side: `evaluate_models.py:1836-1837` records
   `plan.bundle_manifest_digest or manifest_digest(_plan_members(plan))`.

Both sides route through the single `manifest_digest`
(`evaluate_models.py:1154-1172`), which keys identity on each member's own filename + sha256,
not the caller's map key:

```python
identity = sorted(
    ({"filename": member.get("filename") or name, "sha256": member.get("sha256")}
     for name, member in manifest.items()),
    key=lambda entry: str(entry["filename"]),
)
return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
```

Keying invariance and on-disk parity, measured:

```
$ .venv/bin/python - <<'EOF'  (manifest_digest over filename-keyed manifest vs
  category-keyed map vs payload member shape; registry alignment; None-sha sensitivity)
keying-invariant: True 230984dab6ae7675
payload-shape-parity: True
missing-file-differs: True
registry-aligned: True
source pin: True
plan-members-present: True
```

```
$ .venv/bin/python (sha256 of the three on-disk checkpoints under
  workbench/artifacts/checkpoints/limn vs registry expected_sha256)
0_dress_best_model.pt ON-DISK MATCH
0_shirt_best_model.pt ON-DISK MATCH
0_toptee_best_model.pt ON-DISK MATCH
```

**The two digests cannot now silently diverge.** The payload-vs-payload check
(`limn_publish.py:79-88`) is not a binding by itself, because the observer is the writer of the
staged `checkpoint_members` AND merely echoes the parent-supplied
`planned_bundle_manifest_digest` (`limn_export.py:145`); a rogue observer could stage the wrong
members and declare a matching planned digest. The new parent check closes exactly that hole:
its planned value is derived outside the payload, from the pre-observer disk hash
(`evaluate_models.py:1761-1769`, comment at 1755-1760 states this precisely).

**The refusal is reachable**: trigger = any published artifact whose
`run.bundle_manifest_digest` ≠ the plan's digest (corrupted/tampered/stale artifact, or the
rogue-observer scenario above). `_run_observation` is the only observation entry point, invoked
at `evaluate_models.py:1910` under `code == 0 and capability`, and on the official path
capability is non-None (LIMN is the only model declaring `per_query_observer`,
`models.yaml:904`, read via `per_query_capability`, `evaluate_models.py:1598-1604`) and
`checkpoint_id == "base_iter0_all_categories"` (the single-category escape,
`evaluate_models.py:1901-1906`, disables the observer instead). `planned_digest` is non-None on
that path (always set at `evaluate_models.py:401`), so the `if planned_digest:` guard is active.

**It does not fire on a legitimate run**: on the official path the recorded digest is
`manifest_digest(payload members)`, whose members' sha256s are the registry digests, which equal
the on-disk hashes (measured MATCH), which equal the bytes `build_artifact_manifest` hashed at
plan time (same three files under the same bundle root; the run never writes to the checkpoint
directory — it writes only under the results root `[INFERENCE]` on run-time immutability of
`workbench/artifacts/checkpoints/limn`). Measured digest equality across all three keyings
(`keying-invariant: True`, `payload-shape-parity: True`) and `missing-file-differs: True`
(a member whose file vanished would make the digests disagree → refusal, not silent acceptance).

### C3. "Protocol C for LIMN is unchanged."

**Verdict: HOLDS (attack failed).**

- The registry is not in the diff. `models.yaml:916` pins
  `upstream_commit_sha: 7d7bc9b116f594a65ac22457491edf28a88d3c3e`; protocol fields
  `native_protocol: fashioniq_val_split`, `supported_protocols: [fashioniq_val_split]`,
  `literature_split_label: val`, `evaluation_noise_pct: 0` are all present and unmodified
  (`models.yaml:913-922,963`).
- The workbench-side files that implement or mirror the protocol — `limn.py`,
  `limn_export.py`, `limn_publish.py`, `limn_observer.py`, `publication.py`,
  `workbench/backend/*` — are all untouched (`git status` empty for those paths).
- Protocol C (union gallery in order of first appearance, reference-then-target; reference
  masked at −10e10, never removed) as documented in
  `workbench/docs/reviews/final_model_audit/SCIENTIFIC_PROTOCOL_AUDIT.md:31-36` matches the
  pinned native source observed in place:
  - gallery construction: `LIMN/datasets.py` FashionIQ `val` —
    `test_targets_id` appends `source_img_id` then `target_img_id` "if not already present",
    in query order (the union block just before `class Shoes`);
  - mask + full-order sort + hit counting: `LIMN/test.py` —
    `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10`,
    `nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]`,
    `if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]: r += 1`,
    `r = 100 * r / len(nn_result)` for k ∈ {1, 10, 50}.
- The new parent-side check cannot change protocol semantics: it operates on the published
  artifact after the fact and has no write path to any metric, ranking, or protocol field.
- `pc.sh` scientific parameters are unmodified: `MODEL="limn"` (`pc.sh:21`),
  `"top_k": 200` (`pc.sh:1512`), `FINAL_PROTOCOL:-fashioniq_val_split` (`pc.sh:2085,2405`);
  the `pc.sh` diff hunks touching the embedded-Python readers change only HOW the JSON bodies
  are passed (file path vs argv element) and subshell-wrapping of verifier calls — no
  top-k, protocol, or checkpoint-selection value changed in any hunk.

### C4. "The three checkpoint members and their digests are still what the registry declares."

**Verdict: HOLDS (attack failed).**

Registry declaration (`workbench/registry/models.yaml`, lines as observed):

| checkpoint_id | filename | expected_sha256 |
|---|---|---|
| `base_iter0_dress` (960) | `0_dress_best_model.pt` (961) | `96a335d4eb8b1e77d35575aab9177637208e2eb6c59e12f833f67e5cb0e7fc1c` (966) |
| `base_iter0_shirt` (969) | `0_shirt_best_model.pt` (970) | `9260c7f77d3017d8cc10e6652cc8cfe275dbc9704366fa2bf581a24808930d74` (975) |
| `base_iter0_toptee` (978) | `0_toptee_best_model.pt` (979) | `79f663a14d89d455fdc3612b0b36d011c06d7957d1475a654755e1053a6cece2` (984) |

Bundle: `base_iter0_all_categories`, `required_checkpoint_ids: [base_iter0_dress,
base_iter0_shirt, base_iter0_toptee]`, `structure: category_specific_whole_models`,
`mapping_status: VERIFIED_METADATA` (`models.yaml:987-992` region).

Code references, all verified consistent:

- `limn.py:23-28` — `LIMN_CHECKPOINTS` is derived from `MODEL["checkpoint_variants"]`
  (single source, registry-driven); `limn.py:38-50` (`validate_checkpoint_bundle`) blocks on
  missing file, empty file, SHA mismatch, or non-`VERIFIED_METADATA` mapping; `limn.py:171`
  re-verifies the hash at load time via `load_trusted_whole_model`.
- `evaluate_models.py:395-396` — the only hardcoded member names in the planner:
  `[f"0_{cat}_best_model.pt" for cat in ("dress", "shirt", "toptee")]` — identical to the
  registry filenames (measured: `registry-aligned: True`).
- `limn_export.py:93-98` — payload members pulled from the same `LIMN_CHECKPOINTS`.

On-disk reality (measured): all three files under
`workbench/artifacts/checkpoints/limn/` hash to exactly the registry `expected_sha256`
(`ON-DISK MATCH` × 3, command output quoted in C2).

### C5. "Top-K contract: Pass@K = target_rank ≤ K, Fail@K = target_rank > K, over the COMPLETE eligible ranking, never from a truncated Top-K."

**Verdict: HOLDS (attack failed).**

- Rank is 1-based over the FULL native ranking: `limn_observer.py:162-166` —
  `order = np.argsort(-sims[index, :])`, then "The full ranking is walked, so a target beyond
  the saved depth still gets a true rank":
  `rank = int(np.flatnonzero(order == gallery_index)[0]) + 1`. The truncated
  `order[:top_k]` slice feeds only the stored `top_results` display list, never `rank`.
- Pass predicate: `limn_observer.py:74-77` —
  `return 100 * sum(1 for q in self.queries if q.target_rank <= k) / len(self.queries)`.
- Fail predicate: `workbench/backend/storage.py:85-92` — `failed_at` → SQL `target_rank > ?`,
  `passed_at` → `target_rank <= ?` (complementary, enforced mutually exclusive at
  `storage.py:125-129`), over the authoritative stored `target_rank`.
- Saved depth is bounded by the eligible count, not the raw gallery:
  `limn_observer.py:310-318` (`eligible_depth = min(top_k, len(gallery_ids) - 1)`) so a masked
  reference can never appear as a "retrieved" candidate (F-15); `eligible_count =
  len(gallery_ids) - 1` at `limn_observer.py:347`.
- Contract pinned by test with boundary ranks (K−1, K, K+1) and a deliberately truncated saved
  depth (50) against rank-500 queries: `workbench/tests/test_failure_thresholds.py:1-13,
  43-49, 125-141`.
- None of these files is in the diff (`git status` empty for `workbench/backend/`,
  `workbench/tests/test_failure_thresholds.py`). The patch touched none of them.

---

## CONFIRMED DEFECTS

None. No P0/P1/P2 scientific defect found in the changed code.

Non-defect observations (noted, no action required):

- **P3/NOTE** — Standalone CLI export without an orchestrator: a payload staged without
  `--planned-bundle-manifest-digest` and without `--run-id` passes both digest checks (the
  `if planned_digest:` guards, `limn_publish.py:80`, `evaluate_models.py:1762`) and publishes
  with a self-declared bundle digest. It is neutralized by identity, not by digest: the
  `dev-`-prefixed run id forces `data_kind="mock"` (`limn_publish.py`, the
  `data_kind="experiment" if not str(payload["run_id"]).startswith("dev-") else "mock"` line),
  so a development export can never ride on an experiment. BY_DESIGN, consistent with the
  exporter's docstring (`limn_export.py`, "Without it the payload is marked as a development
  run").
- **P3/NOTE** — The `_plan_members(plan)` fallback in the new check
  (`evaluate_models.py:1761`) is unreachable on the official LIMN path, where
  `plan.bundle_manifest_digest` is always set at `evaluate_models.py:401`; it is consistent
  with the plan value by construction (`_plan_members`, `evaluate_models.py:1109-1122`,
  re-derives the same filename-keyed manifest from the same on-disk files).

## DISPROVED ALLEGATIONS

- **C1 attacked, disproven-as-allegation**: no ranking, Recall, gallery membership, reference
  eligibility, or checkpoint loading changed. The diff is a single 20-line insertion in
  `_run_observation`'s post-hoc artifact-admission stage; every scientific computation lives in
  files outside the diff (`git status --short workbench/replay/` → empty).
- **C2 attacked, disproven-as-allegation**: the artifact's recorded bundle identity is correct
  and cannot silently diverge from the aggregate report's. Both values are
  `manifest_digest` (single implementation, filename-keyed, `evaluate_models.py:1154-1172`)
  over members whose sha256s are registry-pinned, hash-verified at load
  (`common.py:49-51`), and measured equal on disk. The payload-vs-payload weakness is real and
  is now closed by the parent's outside-payload binding (`evaluate_models.py:1761-1769`).
- **C3 attacked, disproven-as-allegation**: Protocol C (union gallery, −10e10 mask, full-order
  argsort, hit counting at k ∈ {1,10,50}, top_k 200, `fashioniq_val_split`) is byte-identical in
  the pinned source (commit `7d7bc9b`, verified in place) and no protocol-bearing file was
  modified.
- **C4 attacked, disproven-as-allegation**: member filenames and digests in code match the
  registry (measured `registry-aligned: True`), and the on-disk checkpoints hash to exactly the
  registry digests (measured `ON-DISK MATCH` × 3).
- **C5 attacked, disproven-as-allegation**: the Top-K contract is computed from the complete
  eligible ranking (`limn_observer.py:162-166`), Pass@K/Fail@K predicates are exactly
  `target_rank <= K` / `target_rank > K` (`limn_observer.py:76`, `storage.py:89,92`), and no
  such code was touched by the patch.

## UNVERIFIED

- **Live end-to-end LIMN run** (full GPU path: plan → evaluator → observer → publisher →
  parent binding): NOT executed. Requires CUDA
  (`limn.py:121-123`: `blockers.append("LIMN native test requires CUDA; ...")` when
  `torch.cuda.is_available()` is false) and the full FashionIQ dataset; not available in this
  environment. Attempted instead: static trace of the whole path plus the digest-level probes
  quoted in C2, which verify the binding's arithmetic on the real on-disk checkpoints. The
  statement "the refusal does not fire on a legitimate run" is therefore verified by
  construction + measured digest parity, not by a live run.
- **`pc.sh` embedded-reader semantics** (file-path vs argv): outside this perspective's scope
  (runtime perspective); only the scientific parameters (`top_k`, protocol, checkpoint
  selection) were checked, and they are unmodified (C3).

---

## VERDICT

REJECTED_FINDING — all five attacked claims hold; no scientific defect confirmed. The one
residual risk is UNVERIFIED, not a defect: the live end-to-end path could not be executed
(no GPU), so the new binding was verified by full static trace plus measured digest parity on
the real checkpoints (`230984dab6ae7675…` across all keyings; all three on-disk members match
their registry digests).

Regression-test proposal (one line): assert, over the real checkpoint directory, that
`manifest_digest(build_artifact_manifest(bundle_root, ["0_dress_best_model.pt",
"0_shirt_best_model.pt", "0_toptee_best_model.pt"]))` equals the digest computed from a
category-keyed member map in the payload's `checkpoint_members` shape, and that replacing any
member's sha256 with `None` changes the digest.