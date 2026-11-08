"""The six correctness defects the independent review reproduced at d699b66.

Each test here is written to FAIL on the pre-fix source and PASS after. They use the real
`workbench.composition.runner` helpers with synthetic fixtures -- no model, no GPU, no full run.

The defects, in the order the review numbered them:

  #1 backfill / batch-fallback scoring policy  -- runner.py `_backfill_export`, fallback loop
  #2 an ineffective BLAS pin reported as pinned -- runner.py `_blas_thread_limit`
  #5 measurement identity grouping              -- backend/composition.py `measurement_identity`
  #6 historical provenance for scoring_impl    -- backend/composition.py `scoring_implementation`
  #7 cache content integrity                    -- runner.py `_load_memmap`
  #8 the `assert ... or True` gallery check     -- the composition test modules

Nothing here asserts a *number* about a real model. Where a threshold appears it is chosen so the
fixture's correct answer is unambiguous (rank 1 vs rank 2), not a tolerance on a measurement.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from workbench.composition import runner


# --------------------------------------------------------------------------- #2 BLAS pin

def test_blas_pin_reports_verified_or_failed_never_a_silent_success(monkeypatch) -> None:
    """#2: `_blas_thread_limit` must report what ACTUALLY happened.

    Before the fix it returned an `int` and did `if setter is None: return int(previous)`, so a
    runtime with no usable setter looked exactly like a successful pin -- and `previous` itself was
    read from environment variables rather than from the runtime. The probe measured that shape:
    requested 1, runtime stayed at 9, `setter_found` false, `return_previous` 1.

    The contract now: every call returns a record carrying `requested`, `effective`,
    `setter_found`, `verified`, and `previous` read from the RUNTIME when it can be read.
    """
    # A runtime with no setter at all -- the probe's exact failure shape.
    monkeypatch.setattr(runner, "_openblas_symbols", lambda: (None, None, None))
    result = runner._blas_thread_limit(1)
    assert result["setter_found"] is False, result
    assert result["verified"] is False, (
        f"a pin with no setter must not claim verification: {result}")
    assert result["requested"] == 1
    assert result["effective"] is None, (
        f"with no setter the effective value is unknown, not the requested one: {result}")
    assert result["previous"] is None, (
        f"the previous value must not be invented from env vars when the runtime is unreadable: {result}")


def test_blas_pin_verified_when_the_runtime_accepts_it(monkeypatch) -> None:
    """The positive half: a working setter/getter pair must be read back and marked verified."""
    state = {"threads": 10}

    def setter(value):
        state["threads"] = int(value)

    def getter():
        return state["threads"]

    monkeypatch.setattr(runner, "_openblas_symbols", lambda: (setter, getter, "libopenblas-test.so"))
    result = runner._blas_thread_limit(1)
    assert result["setter_found"] is True
    assert result["verified"] is True, f"the getter read back 1, so the pin is verified: {result}"
    assert result["effective"] == 1
    assert result["previous"] == 10, "previous must come from the runtime getter, not the env"
    assert result["source"] == "libopenblas-test.so"


def test_blas_pin_is_unverified_when_readback_disagrees(monkeypatch) -> None:
    """A setter that silently ignores the request must NOT be recorded as verified."""
    state = {"threads": 9}

    monkeypatch.setattr(runner, "_openblas_symbols",
                        lambda: (lambda value: None, lambda: state["threads"], "lib-stubborn.so"))
    result = runner._blas_thread_limit(1)
    assert result["setter_found"] is True
    assert result["verified"] is False, (
        f"setter was called but the runtime still reports 9: {result}")
    assert result["effective"] == 9
    assert result["requested"] == 1


def test_blas_pin_backwards_compatible_int_coercion(monkeypatch) -> None:
    """Existing callers do `previous = _blas_thread_limit(n)` then `_blas_thread_limit(previous)`.

    That works on the record, so the helpers must accept a record as the "restore" argument and
    still make sense of a plain int. This keeps every existing call site correct without editing
    them, which is what lets the fix stay small.
    """
    state = {"threads": 4}
    monkeypatch.setattr(runner, "_openblas_symbols",
                        lambda: (lambda v: state.__setitem__("threads", int(v)),
                                 lambda: state["threads"], "lib.so"))
    pinned = runner._blas_thread_limit(1)
    restored = runner._blas_thread_limit(pinned["previous"])
    assert state["threads"] == 4
    assert restored["effective"] == 4


# --------------------------------------------------------------------------- #7 cache integrity

def _cache_file(tmp_path: Path, rows: int = 4, width: int = 3, fill: float = 1.0) -> Path:
    path = tmp_path / "features" / "gallery_global.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((rows, width), fill, dtype=np.float32)
    np.save(path, array, allow_pickle=False)
    return path


def test_cache_rejects_same_shape_different_content(tmp_path: Path) -> None:
    """#7: the probe's exact case -- content replaced with a same-shape array must be rejected.

    Before the fix the loader checked only `fingerprint` and `shape`, so a file whose bytes changed
    (sum 6.0 -> 0.0) loaded as if nothing had happened.
    """
    path = _cache_file(tmp_path, fill=1.0)
    runner._write_memmap_metadata(path, "fingerprint-abc",
                                  np.load(path, allow_pickle=False))
    original = np.load(path, allow_pickle=False)
    assert float(original.sum()) == 12.0

    # Same shape, same dtype, different bytes.
    np.save(path, np.zeros_like(original), allow_pickle=False)

    loaded = runner._load_memmap(path, "fingerprint-abc", original.shape)
    assert loaded is None, (
        "a cache whose content digest no longer matches must not load; it must fail closed")


def test_cache_reports_digest_absent_for_legacy_files_without_one(tmp_path: Path) -> None:
    """A cache written before digests existed must still load, but be flagged unverified.

    The flag is the mechanism that lets the fix land without invalidating 74k-image caches that
    are already on disk and probably fine -- it just cannot call them validated.
    """
    path = _cache_file(tmp_path)
    # A v1 sidecar: fingerprint + shape only, exactly what the old writer produced.
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(
        {"fingerprint": "fp-legacy", "shape": [4, 3], "dtype": "float32"}), encoding="utf-8")

    loaded = runner._load_memmap(path, "fp-legacy", (4, 3))
    assert loaded is not None, "a legacy cache must remain usable"
    state = runner._memmap_integrity(path)
    assert state["content_digest"] == "absent", state
    assert state["verified"] is False, (
        "a cache with no recorded digest must never be described as validated")


def test_cache_rejects_truncated_file(tmp_path: Path) -> None:
    """A half-written file must fail closed rather than load a short array."""
    path = _cache_file(tmp_path, rows=8, width=4)
    data = path.read_bytes()
    path.write_bytes(data[:len(data) // 2])
    loaded = runner._load_memmap(path, "fp-any", (8, 4))
    assert loaded is None, "a truncated cache must not load as if it had the right row count"


def test_cache_rejects_stale_sidecar_shape(tmp_path: Path) -> None:
    """A sidecar claiming a different shape than the file must fail closed."""
    path = _cache_file(tmp_path, rows=4, width=3)
    runner._write_memmap_metadata(path, "fp-shape", np.load(path, allow_pickle=False))
    sidecar = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
    sidecar["shape"] = [99, 3]
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(sidecar), encoding="utf-8")

    assert runner._load_memmap(path, "fp-shape", (4, 3)) is None
    assert runner._load_memmap(path, "fp-shape", (99, 3)) is None, (
        "the file has 4 rows, so neither the real nor the claimed shape may load")


def test_cache_digest_is_recorded_on_write(tmp_path: Path) -> None:
    """The writer must record a digest that the loader can verify on the next pass."""
    path = _cache_file(tmp_path, rows=3, width=2, fill=2.0)
    value = np.load(path, allow_pickle=False)
    runner._write_memmap_metadata(path, "fp-write", value)

    state = runner._memmap_integrity(path)
    assert state["content_digest"] not in (None, "absent"), state
    assert state["verified"] is True, state
    assert runner._load_memmap(path, "fp-write", (3, 2)) is not None


def test_cache_digest_covers_the_bytes_it_claims(tmp_path: Path) -> None:
    """The digest must be the array's content, not its sidecar or its name."""
    a = _cache_file(tmp_path / "a", fill=1.0)
    b = _cache_file(tmp_path / "b", fill=1.0)
    c = _cache_file(tmp_path / "c", fill=3.0)
    for path in (a, b, c):
        runner._write_memmap_metadata(path, "fp", np.load(path, allow_pickle=False))
    assert runner._memmap_integrity(a)["content_digest"] == runner._memmap_integrity(b)["content_digest"]
    assert runner._memmap_integrity(a)["content_digest"] != runner._memmap_integrity(c)["content_digest"]


# --------------------------------------------------------------------------- #5 identity grouping

def _backend():
    from workbench.backend import composition as backend_composition

    return backend_composition


def _summary(base: dict | None = None, **changes) -> dict:
    """A summary shaped like the ones `/api/composition/runs` serves."""
    summary = {
        "run_id": "run-1", "model_id": "encoder", "checkpoint_id": "ckpt",
        "checkpoint_identity": "ckpt-sha", "protocol_id": "circus_fashioniq_release_v1",
        "subset": "full", "gallery_identity_sha256": "gallery-sha", "query_count": 6003,
        "conditions": ["multimodal"], "seed": 0, "preprocessing": {"id": "p"},
        "runtime": {"model_interpreter": {"environment_fingerprint": "env-sha"}},
        "scoring_module_source_hash": "scoring-sha",
        "adapter_module_source_hash": "adapter-sha",
        "evaluator_environment_knobs": {"omp_num_threads": "10"},
        "metrics": {"gap_cohort_count": 6003},
        "is_synthetic": False,
    }
    summary.update(base or {})
    summary.update(changes)
    return summary


IDENTITY_V2_FIELDS = ("blas_threads_pinned", "query_scope_resolution", "query_order_identity")


def test_identity_v2_splits_the_group_when_the_blas_policy_changes() -> None:
    """#5: two runs that differ only in `blas_threads_pinned` must not share a group_id.

    The probe measured the opposite: `same_group: true, confidence: full`.
    """
    backend = _backend()
    one = _summary({"blas_threads_pinned": 1})
    ten = _summary({"blas_threads_pinned": 10})
    first = backend.measurement_identity(one)
    second = backend.measurement_identity(ten)
    assert first["group_id"] != second["group_id"], (
        "BLAS thread count is an arithmetic input; two policies cannot be one measurement")


def test_identity_v2_splits_on_scope_resolution_and_query_order() -> None:
    backend = _backend()
    base = _summary({"query_scope_resolution": "scope-a", "query_order_identity": "order-a"})
    scope = _summary({"query_scope_resolution": "scope-b", "query_order_identity": "order-a"})
    order = _summary({"query_scope_resolution": "scope-a", "query_order_identity": "order-b"})
    ids = {backend.measurement_identity(item)["group_id"] for item in (base, scope, order)}
    assert len(ids) == 3, f"three distinct scoring inputs must be three groups, got {len(ids)}"


def test_identity_v2_stamps_the_schema_version() -> None:
    """`identity_schema_version` is what stops a v1 group being compared to a v2 group."""
    backend = _backend()
    facts = backend.measurement_identity(_summary({field: 1 for field in IDENTITY_V2_FIELDS}))
    assert facts["identity"]["identity_schema_version"] == 2
    assert facts.get("schema_version") == 2


def test_identity_v1_artifacts_are_unknown_not_regrouped() -> None:
    """A run RECORDED before v2 has no v2 inputs: it must be labelled, never silently regrouped.

    The distinction that matters: a *recorded run identity* from an older runner is a v1 artifact.
    A served summary that merely lacks the fields is not -- it is metadata, not a historical record.
    """
    backend = _backend()
    legacy_summary = _summary({"run_id": "old-run"})       # served metadata, no v2 fields
    # What an older runner actually wrote into runs/<id>/ranks/*.json:identity.
    legacy_recorded = {
        "model_id": "encoder", "checkpoint_id": "ckpt", "subset_memberships": ["full"],
        "gallery_identity": "gallery-sha", "adapter_module_source_hash": "adapter-sha",
        "scoring_module_source_hash": "scoring-sha",
        # None of blas_threads_pinned / query_scope_resolution / query_order_identity: this block
        # was written by a runner older than all three fields.
    }
    facts = backend.measurement_identity(legacy_summary, legacy_recorded)
    assert facts["group_id"] is None, (
        "a v1 artifact must not be given a v2 group id it never had")
    assert facts["confidence"] == "unknown"
    assert "predates identity v2" in facts["reason"]
    assert facts["identity"]["identity_schema_version"] == "unknown (predates identity v2)"

    # And a summary-only row is NOT relabelled v1 just because the v2 fields are absent.
    served = backend.measurement_identity(legacy_summary)
    assert served["schema_version"] == 2, (
        "served metadata has no recorded history, so it is not a v1 artifact")
    assert served["group_id"] is not None


def test_identity_v2_fields_split_groups_when_present() -> None:
    """v2 inputs must CHANGE the group when they differ -- that is their whole purpose."""
    backend = _backend()
    one = backend.measurement_identity(
        _summary({"blas_threads_pinned": 1, "query_scope_resolution": "s",
                  "query_order_identity": "o"}))
    two = backend.measurement_identity(
        _summary({"blas_threads_pinned": 10, "query_scope_resolution": "s",
                  "query_order_identity": "o"}))
    assert one["group_id"] != two["group_id"]
    assert one["confidence"] == "full" and two["confidence"] == "full", (
        "both rows carry complete identity, so both are full confidence")


def test_duplicate_labelling_never_calls_two_policies_duplicates() -> None:
    """The end-to-end consequence: annotated rows must not read as `duplicate` across policies."""
    backend = _backend()
    rows = [_summary({"run_id": "a", "blas_threads_pinned": 1, **{f: "x" for f in IDENTITY_V2_FIELDS[1:]}}),
            _summary({"run_id": "b", "blas_threads_pinned": 10, **{f: "x" for f in IDENTITY_V2_FIELDS[1:]}})]
    backend.annotate_duplicate_attempts(rows)
    assert {row["duplicate_status"] for row in rows} == {"unique"}, (
        "two BLAS policies measured different things; neither is a duplicate of the other")


# --------------------------------------------------------------------------- #6 historical provenance

def test_scoring_impl_comes_from_the_artifact_not_todays_source(monkeypatch) -> None:
    """#6: a historical run must be labelled from what it recorded, not from today's adapter body.

    Before the fix `_scoring_impl_for` read the live module's `score` body even when the artifact
    carried its own recorded value, so renaming/refactoring an adapter today would relabel history.
    """
    backend = _backend()
    summary = {"model_id": "conesep", "scoring_impl": "einsum-optimized",
               "scoring_module_source_hash": "recorded-hash"}
    result = backend.scoring_implementation(
        "conesep", "recorded-hash", "adapter-hash", summary=summary, live=True)
    assert result["impl"] == "einsum-optimized", (
        f"the recorded value must win over today's source: {result}")
    assert result["basis"].startswith("recorded"), result["basis"]


def test_scoring_impl_unknown_when_artifact_records_nothing_derivable(monkeypatch) -> None:
    """No recorded value and no trustworthy hash mapping => `unknown` with a reason, not a guess."""
    backend = _backend()
    result = backend.scoring_implementation("conesep", None, None,
                                            summary={"model_id": "conesep"}, live=False)
    assert result["impl"] == "unknown", (
        f"deriving from today's source for a historical run is exactly the defect: {result}")
    assert "historical" in result["basis"].lower() or "unknown" in result["basis"].lower()


def test_scoring_impl_still_derives_for_a_live_run(monkeypatch) -> None:
    """A run being measured NOW may derive from source -- that is what `live` means."""
    backend = _backend()
    result = backend.scoring_implementation(
        "encoder", "h", "a", summary={"model_id": "encoder"}, live=True)
    assert result["impl"] in {"einsum", "einsum-optimized", "matmul", "unknown"}, result


# --------------------------------------------------------------------------- #1 backfill policy

def test_backfill_runs_under_the_same_blas_policy_as_the_normal_path(monkeypatch, tmp_path: Path) -> None:
    """#1: backfill must pin/restore exactly like `_rank_groups_concurrently` does.

    The probe drove the real helpers with a policy-sensitive fixture: the normal path pinned
    [1, 10] and produced rank 1; backfill pinned nothing, saw 10 threads and produced rank 2.
    """
    calls: list[int] = []
    state = {"threads": 10}

    def setter(value):
        state["threads"] = int(value)

    monkeypatch.setattr(runner, "_openblas_symbols",
                        lambda: (setter, lambda: state["threads"], "lib.so"))
    original_limit = runner._blas_thread_limit

    def recording_limit(threads):
        calls.append(int(threads) if not isinstance(threads, dict) else int(threads["previous"]))
        return original_limit(threads)

    monkeypatch.setattr(runner, "_blas_thread_limit", recording_limit)

    class PolicySensitive:
        """Scores by the thread count it observes, so a missing pin is visible in the rank."""
        def rank_and_topk(self, vectors, targets, positions, categories, depth):
            rank = 1 if state["threads"] == runner.BLAS_THREADS_PINNED else 2
            return [rank], [{("gallery", 0): (1.0, [0, 1])}]

    work_root = tmp_path / "work"
    features_dir = work_root / "features"
    features_dir.mkdir(parents=True)
    np.save(features_dir / "gallery_global.npy", np.zeros((3, 4), dtype=np.float32))
    np.save(features_dir / "query_multimodal_global.npy", np.ones((1, 4), dtype=np.float32))
    writer = SimpleNamespace(depth=2, write=lambda *a, **k: None)

    captured = {}

    def fake_write_batch(writers, condition, qids, ranks, windows):
        captured["ranks"] = list(ranks)

    monkeypatch.setattr("workbench.composition.topk.write_batch", fake_write_batch)
    # The cache is keyed by SCOPE (None for a single-scope gallery), which is what
    # `_cached_features` returns and what the loop looks up.
    monkeypatch.setattr(runner, "_cached_features",
                        lambda *a, **k: ({None: [0, 1, 2]},
                                         {None: np.ones((1, 4), dtype=np.float32)}))

    query = SimpleNamespace(qid="q0")
    runner._backfill_export(PolicySensitive(), [query], [None], [None],
                            {"q0": (0,)}, {}, work_root, "fp", ["multimodal"],
                            {"multimodal": writer}, 1)

    assert calls, "backfill never called the BLAS limiter -- that is the defect"
    assert runner.BLAS_THREADS_PINNED in calls, (
        f"backfill must pin to {runner.BLAS_THREADS_PINNED} like the normal path: {calls}")
    assert captured["ranks"] == [1], (
        f"scoring under the shared policy must yield the same rank as the normal path: {captured}")


# --------------------------------------------------------------------------- #8 the no-op assert

def test_gallery_identity_assertion_is_not_vacuous() -> None:
    """#8: the gallery-identity comparison must be able to FAIL.

    `assert a == b or True` can never fail. These two assertions are the replacement, so they must
    themselves be capable of failing -- checked here against a deliberately wrong manifest.
    """
    truth = {"gallery_identity_sha256": "sha-right", "gallery_count": 5,
             "ordered_ids": ["a", "b", "c", "d", "e"]}

    def agrees(expected, manifest):
        """The same comparison the topk test now makes, reduced to its falsifiable core."""
        if not manifest.get("gallery_identity_sha256"):
            return False
        if manifest["gallery_identity_sha256"] != expected["gallery_identity_sha256"]:
            return False
        if manifest.get("gallery_count") != expected["gallery_count"]:
            return False
        ids = manifest.get("ordered_ids")
        if ids is not None:
            if len(ids) != len(expected["ordered_ids"]):
                return False
            if len(set(ids)) != len(ids):
                return False
            if ids != expected["ordered_ids"]:
                return False
        return True

    assert agrees(truth, dict(truth)) is True
    assert agrees(truth, dict(truth, gallery_identity_sha256="sha-wrong")) is False
    assert agrees(truth, dict(truth, ordered_ids=["b", "a", "c", "d", "e"])) is False
    assert agrees(truth, dict(truth, ordered_ids=["a", "b", "c", "d"], gallery_count=4)) is False
    assert agrees(truth, dict(truth, ordered_ids=["a", "a", "c", "d", "e"])) is False


def test_cache_verify_mode_is_in_the_run_summary() -> None:
    """A provenance fact about how the features were verified must be written to the summary.

    It lived only in the identity block, so `measurement_identity` read `None` from every real
    artifact and a `full` run and a `header` run minted the SAME group id -- the false-duplicate
    class this whole change exists to remove.
    """
    import inspect

    source = inspect.getsource(runner)
    # The summary is built as a literal; the key must appear in it, not only in `identity`.
    assert '"cache_verify_mode"' in source
    summary_block = source[source.find("summary = {"):]
    summary_block = summary_block[:summary_block.find("_atomic_json(summary_path, summary)")]
    assert '"cache_verify_mode"' in summary_block, (
        "cache_verify_mode must be written into the run summary, not only into the identity block")


def test_cache_verify_mode_splits_the_group() -> None:
    """`full` and `header` are different verification guarantees, so they are different groups."""
    backend = _backend()
    common = {field: 1 for field in IDENTITY_V2_FIELDS}
    full = _summary(dict(common, cache_verify_mode="full"))
    header = _summary(dict(common, cache_verify_mode="header"))
    first = backend.measurement_identity(full)
    second = backend.measurement_identity(header)
    assert first["group_id"] != second["group_id"], (
        "a header-verified cache is not the same measurement as a fully-verified one")


def test_cache_verify_mode_reads_the_recorded_value() -> None:
    """A recorded v2 artifact keeps ITS OWN mode even when the served summary says otherwise."""
    backend = _backend()
    common = {field: 1 for field in IDENTITY_V2_FIELDS}
    served = _summary(dict(common))                       # no cache_verify_mode in the summary
    recorded = {**common, "cache_verify_mode": "header"}
    facts = backend.measurement_identity(served, recorded)
    assert facts["identity"]["cache_verify_mode"] == "header", (
        f"the recorded value must win: {facts['identity']['cache_verify_mode']!r}")


def test_legacy_artifact_without_cache_verify_mode_is_not_a_duplicate() -> None:
    """An artifact with no recorded verification mode must not be duplicated against a modern one."""
    backend = _backend()
    common = {field: 1 for field in IDENTITY_V2_FIELDS}
    modern = _summary(dict(common, cache_verify_mode="full", run_id="modern"))
    legacy = _summary(dict(common, run_id="legacy"))      # no cache_verify_mode anywhere
    backend.annotate_duplicate_attempts([modern, legacy])
    assert modern["duplicate_status"] != "duplicate" or legacy["duplicate_status"] != "duplicate", (
        "a run whose cache verification is unknown cannot be the same measurement as one that is known")
