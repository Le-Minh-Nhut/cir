"""The runner must use the adapter's scope resolver, and must not let a resume hide it.

`Adapter.query_scope()` claimed the runner called it; it did not, so 119 unscored FashionIQ
queries stayed unscored. Two things have to hold together, and the second is the one that is easy
to miss: calling the resolver alone leaves the run identity unchanged (`_query_fingerprint` keys on
the loader's `query.category`, still `None`), so `--resume` matches the old ranks artifact and
silently reuses the 119 stale nulls. These checks pin the resolver contract and the identity move.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from workbench.backend.composition import load_cached_release
from workbench.composition import runner

SCOPES = ("dress", "shirt", "toptee")
REPO = Path(__file__).resolve().parents[2]

#: The finished run whose block reads "6003 unscoped" on a run that scored all 6003. IMMUTABLE
#: history: the fix is forward-only, so this file asserts the old block is still there.
FROZEN = (REPO / "workbench/artifacts/composition/summary/"
          "tgcir_tgcir_archive_unresolved_7b1fa85007bffdab931f7f19/full.json")
FROZEN_RUN_ID = "7b1fa85007bffdab931f7f19"

def _query(qid: str, category: str | None, text: str = "is blue and is long") -> SimpleNamespace:
    return SimpleNamespace(qid=qid, index=int(qid.split(":")[-1]) if ":" in qid else 0,
                           query_txt=text, reference_id=qid, category=category,
                           positive_ids=(), subsets=("full",))

class _Resolving:
    """Adapter shape: declares scopes and answers `query_scope`, recording what it was handed."""

    category_scopes = SCOPES
    answers: dict[str, str | None] = {}
    seen: list[tuple[str, object]] = []

    def query_scope(self, query, candidate_names=None):
        type(self).seen.append((query.qid, candidate_names))
        return type(self).answers.get(query.qid)

def test_resolver_is_consulted_when_the_loader_left_the_category_unset() -> None:
    _Resolving.answers = {"7:1": "shirt"}
    _Resolving.seen = []
    adapter = _Resolving()
    query = _query("7:1", None)
    scope, reason = runner._scope_for_query(adapter, query, {"7:1": "img/B000S.jpg"})
    assert (scope, reason) == ("shirt", None)
    # The release candidate table reached the resolver; it cannot key on (stem, text) without it.
    assert _Resolving.seen == [("7:1", {"7:1": "img/B000S.jpg"})]

def test_resolver_value_outside_declared_scopes_is_unresolved_not_trusted() -> None:
    _Resolving.answers = {"7:2": "shoes"}
    adapter = _Resolving()
    scope, reason = runner._scope_for_query(adapter, _query("7:2", None))
    assert scope is None
    assert reason == ("query 7:2: category unresolved has no adapter embedding scope "
                      f"{SCOPES!r}; batch_indices=(2,), qids=['7:2']")
    assert runner._is_stable_failure(reason)

def test_resolver_returning_none_leaves_the_query_unscored() -> None:
    _Resolving.answers = {"7:3": None}
    adapter = _Resolving()
    scope, reason = runner._scope_for_query(adapter, _query("7:3", None))
    assert scope is None
    assert "category unresolved has no adapter embedding scope" in reason

def test_loader_assigned_category_wins_and_the_resolver_is_not_asked() -> None:
    """Re-deriving what the release loader decided is out of scope, and a contradiction must lose."""
    _Resolving.answers = {"7:4": "toptee"}
    _Resolving.seen = []
    adapter = _Resolving()
    assert runner._scope_for_query(adapter, _query("7:4", "shirt")) == ("shirt", None)
    assert _Resolving.seen == []

def test_adapter_without_the_hook_keeps_the_previous_behaviour() -> None:
    adapter = SimpleNamespace(category_scopes=SCOPES)
    scope, reason = runner._scope_for_query(adapter, _query("7:5", None))
    assert scope is None
    assert "category unresolved has no adapter embedding scope" in reason

def test_a_category_agnostic_adapter_resolves_to_the_global_scope_with_no_reason() -> None:
    """`(None,)` means ONE embedding matrix, not "could not scope": the scope is `global`.

    This is the distinction that made `7b1fa850…` report 6003 unscoped queries on a run that
    scored all 6003. `(None, None)` is a RESOLVED scope; `(None, "<reason>")` is a failure, and
    only the second may be counted as unscoped.
    """
    agnostic = SimpleNamespace(category_scopes=(None,))
    assert runner._scope_for_query(agnostic, _query("7:6", None)) == (None, None)
    # Same answer for a query the loader DID categorise: a single-matrix adapter ignores it.
    assert runner._scope_for_query(agnostic, _query("7:7", "shirt")) == (None, None)
    assert runner._scope_name(None) == "global"
    # ...while the declared-scope branch above still returns a reason. Two branches, two answers.
    scoped = SimpleNamespace(category_scopes=SCOPES)
    assert runner._scope_for_query(scoped, _query("7:8", None))[1] is not None

def test_the_recorded_block_counts_a_global_scope_as_scored_not_unscoped() -> None:
    """End to end through the runner's own record builder: the block and the resolver agree."""
    agnostic = SimpleNamespace(category_scopes=(None,))
    queries = [_query("7:9", None), _query("7:10", "shirt")]
    scopes = agnostic.category_scopes
    assignments = [runner._scope_for_query(agnostic, query, {}) for query in queries]
    block = runner._scope_resolution_record(agnostic, queries, scopes, assignments)
    assert block["category_agnostic"] is True
    assert block["still_unscoped"] == 0
    assert block["unscoped_qids"] == []
    assert block["global_scoped"] == 2
    # The invariant that must hold for TG-CIR, asserted on the block itself rather than on a run.
    assert block["still_unscoped"] == len(block["unscoped_qids"])

def test_resolved_scope_map_changes_the_run_identity() -> None:
    queries = [_query("7:1", None), _query("7:2", None), _query("7:3", "dress")]
    with_resolution = runner._scope_resolution_identity(queries, ["shirt", "toptee", "dress"])
    all_null = runner._scope_resolution_identity(queries, [None, None, "dress"])
    assert with_resolution != all_null, "the resolver's answer must move the identity"
    # Same map, different query order: same experiment, same identity.
    assert runner._scope_resolution_identity(list(reversed(queries)),
                                             ["dress", "toptee", "shirt"]) == with_resolution
    # One query changing scope is enough to mint a new identity.
    assert runner._scope_resolution_identity(queries, ["shirt", "toptee", "shirt"]) != with_resolution

def test_resolution_digest_reaches_the_identity_dict_through_the_helper() -> None:
    """The helper is what `identity` carries, so a resumed old run can never match a resolved one."""
    queries = [_query("7:1", None), _query("7:2", None)]
    resolved = runner._scope_resolution_identity(queries, ["shirt", "toptee"])
    unresolved = runner._scope_resolution_identity(queries, [None, None])
    base = {"model_id": "limn", "checkpoint_id": "base_iter0_all_categories"}
    assert (runner._digest(dict(base, query_scope_resolution=resolved))
            != runner._digest(dict(base, query_scope_resolution=unresolved)))


class _ScopeRecordingAdapter:
    """Minimal encoder that records the scopes the runner hands to the adapter."""

    category_scopes = SCOPES

    def __init__(self) -> None:
        self.scopes_seen: list[str | None] = []

    def encode_queries(self, image_paths, texts, categories):
        self.scopes_seen.extend(categories)
        return np.zeros((len(image_paths), 4), dtype=np.float32)


def test_encode_queries_hands_the_adapter_the_resolved_scope(tmp_path) -> None:
    """Without this the resolver would be cosmetic: `query.category` is still None at encode time."""
    from PIL import Image

    source = tmp_path / "ref.png"
    Image.new("RGB", (3, 2), (1, 2, 3)).save(source)
    queries = [_query("7:1", None), _query("7:2", None)]
    adapter = _ScopeRecordingAdapter()
    output = tmp_path / "query.npy"
    runner._encode_queries(adapter, queries, [source, source], [0, 1], "multimodal", tmp_path,
                           output, "fingerprint", 4, 2, ["shirt", "toptee"])
    assert adapter.scopes_seen == ["shirt", "toptee"]


def test_encode_queries_without_a_scope_map_still_uses_the_loader_category(tmp_path) -> None:
    from PIL import Image

    source = tmp_path / "ref.png"
    Image.new("RGB", (3, 2), (1, 2, 3)).save(source)
    queries = [_query("7:1", "dress")]
    adapter = _ScopeRecordingAdapter()
    runner._encode_queries(adapter, queries, [source], [0], "multimodal", tmp_path,
                           tmp_path / "query.npy", "fingerprint", 4, 1)
    assert adapter.scopes_seen == ["dress"]

def _record(adapter, queries) -> dict:
    """The runner's own record builder over the real resolver, no encoding involved."""
    scopes = tuple(getattr(adapter, "category_scopes", (None,)))
    assignments = [runner._scope_for_query(adapter, query, {}) for query in queries]
    return runner._scope_resolution_record(adapter, queries, scopes, assignments)

class _CategoryAgnostic:
    """TG-CIR's shape: ONE embedding matrix, no per-category routing, no resolver hook."""

    category_scopes = (None,)

class _PerCategory:
    """LIMN's shape: a real set of garment scopes; queries outside it stay genuinely unscoped."""

    category_scopes = ("dress", "shirt", "toptee")

def test_category_agnostic_queries_are_scored_not_unscoped() -> None:
    queries = [_query(f"7:{index}", None) for index in range(1, 6)]
    block = _record(_CategoryAgnostic(), queries)
    assert block["category_agnostic"] is True
    assert block["adapter_routed"] == 0, "no NAMED matrix was chosen; that is the narrow meaning"
    assert block["global_scoped"] == 5
    assert block["still_unscoped"] == 0
    assert block["unscoped_qids"] == []
    assert block["adapter_routed"] + block["global_scoped"] + block["still_unscoped"] == len(queries)

def test_a_query_with_no_scope_in_a_per_category_adapter_is_still_unscoped() -> None:
    """The other direction: a real reason must keep counting, or the fix would hide failures."""
    queries = [_query("7:1", "dress"), _query("7:2", "shoes")]
    block = _record(_PerCategory(), queries)
    assert block["category_agnostic"] is False
    # `loader_scoped` counts the loader's own assignments, INCLUDING a category the adapter does
    # not declare: the loader decided it, so it is not an adapter routing decision.
    assert block["loader_scoped"] == 2
    assert block["adapter_routed"] == 1
    assert block["still_unscoped"] == 1
    assert block["unscoped_qids"] == ["7:2"]
    assert len(block["unscoped_qids"]) == block["still_unscoped"]

def test_the_mixed_case_separates_all_three_states() -> None:
    """Loader-scoped + resolver-routed + genuinely unscopable in ONE run (LIMN's real shape)."""
    class _Resolving(_PerCategory):
        answers = {"7:3": "toptee"}

        def query_scope(self, query, candidate_names=None):
            return type(self).answers.get(query.qid)

    queries = [_query("7:1", "dress"), _query("7:2", None), _query("7:3", None),
               _query("7:4", "shoes")]
    block = _record(_Resolving(), queries)
    assert block["category_agnostic"] is False
    assert block["resolver"].endswith(".query_scope")
    assert block["loader_scoped"] == 2, "7:1 dress + 7:4 shoes, both assigned by the loader"
    assert block["adapter_routed"] == 2, "7:1 by the loader, 7:3 by the resolver"
    # `global_scoped` is 0: a per-category adapter has no global matrix, so an unscoped query is a
    # failure, not a placement.
    assert block["global_scoped"] == 0
    assert block["still_unscoped"] == 2
    assert block["unscoped_qids"] == ["7:2", "7:4"]
    assert block["still_unscoped"] == len(block["unscoped_qids"])
    # `adapter_routed + global_scoped + still_unscoped` covers every query exactly once; the
    # `loader_scoped` count is descriptive and overlaps `adapter_routed` (7:1 is both).
    assert block["adapter_routed"] + block["global_scoped"] + block["still_unscoped"] == len(queries)

def test_the_old_artifact_keeps_its_block_and_still_contradicts_itself() -> None:
    """Immutable history: `7b1fa850…` is NOT rewritten, and this test says so out loud."""
    if not FROZEN.is_file():
        pytest.skip(f"{FROZEN} is not present on this host")
    summary = json.loads(FROZEN.read_text(encoding="utf-8"))
    block = summary["adapter_query_scope_resolution"]
    assert summary["run_id"] == FROZEN_RUN_ID
    # The recorded (misleading) values, preserved verbatim: rewriting them would falsify history.
    assert block["still_unscoped"] == 6003
    assert block["adapter_routed"] == 0
    assert block["category_agnostic"] if "category_agnostic" in block else True
    assert "category_agnostic" not in block, (
        "a pre-fix artifact must not be retrofitted with the new key: its block is what the old "
        "code printed, and that is the evidence")
    assert summary["category_scopes"] == [None]
    assert sum(row["unscored_count"] for row in summary["condition_status"].values()) == 0
    assert block["still_unscoped"] != sum(row["unscored_count"]
                                          for row in summary["condition_status"].values()), (
        "the contradiction this fix removes is preserved as history, not repaired in place")

def test_the_real_global_run_replayed_through_the_new_path_has_no_unscoped_queries() -> None:
    """The forward-only proof: same release, same adapter SHAPE, corrected block, no GPU.

    The adapter's shape is read off the real class (`model_tgcir.Adapter.category_scopes ==
    (None,)`) rather than guessed; the queries are the release's real 6003, with the garment
    assignment the CLI applies before planning (`assign_garment_categories`). Instantiating the
    model itself needs its checkpoint and a GPU, which is the only part left to a live re-run — the
    routing arithmetic needs neither.
    """
    data = load_cached_release()
    if data is None:
        pytest.skip("the CIRCUS release is not loaded on this host")
    from workbench.backend.operator_config import resolve_config
    from workbench.composition import data as composition_data, model_tgcir

    config = resolve_config()
    fashioniq = getattr(config, "FASHIONIQ_ROOT", None)
    if not fashioniq or not Path(fashioniq).is_dir():
        pytest.skip("the FashionIQ annotation root is not present on this host")
    data = composition_data.assign_garment_categories(data, Path(fashioniq))

    assert model_tgcir.Adapter.category_scopes == (None,), (
        "this replay is only faithful while TG-CIR really declares one global matrix")
    queries = tuple(data.queries)
    assert len(queries) == 6003, f"expected the full released query set, got {len(queries)}"

    block = _record(SimpleNamespace(category_scopes=model_tgcir.Adapter.category_scopes), queries)
    assert block["category_agnostic"] is True
    assert block["still_unscoped"] == 0, (
        "a category-agnostic adapter cannot leave a query unscoped: every query has the global "
        "matrix to score against")
    assert block["unscoped_qids"] == []
    assert block["adapter_routed"] + block["global_scoped"] + block["still_unscoped"] == len(queries)
    # The number the served block must now carry instead of "6003 unscoped" — the value spec
    # option (a) would have called `adapter_routed`.
    assert block["global_scoped"] == len(queries) == 6003
    # `loader_scoped` is DESCRIPTIVE and overlaps `global_scoped` here, deliberately: the loader did
    # assign 5 884 garment categories, and this adapter ignores them when choosing a matrix. It is
    # not a third partition member, and the real artifact records the same 5 884 for the same run.
    assert block["loader_scoped"] == 5884
    assert block["adapter_routed"] == 0

def test_the_real_release_still_has_genuinely_ambiguous_queries() -> None:
    """The fix must not claim routing is perfect: 119 real queries stay ambiguous at the loader."""
    data = load_cached_release()
    if data is None:
        pytest.skip("the CIRCUS release is not loaded on this host")
    from workbench.backend.operator_config import resolve_config
    from workbench.composition import data as composition_data

    fashioniq = getattr(resolve_config(), "FASHIONIQ_ROOT", None)
    if not fashioniq or not Path(fashioniq).is_dir():
        pytest.skip("the FashionIQ annotation root is not present on this host")
    # `category_resolution` is a fact about the ASSIGNED release (what the CLI loads just before
    # planning), not about the raw loader output.
    data = composition_data.assign_garment_categories(data, Path(fashioniq))
    resolution = data.provenance.get("category_resolution") or {}
    assert resolution.get("ambiguous_count"), "this release must record its ambiguous count"
    # Real data, not a tolerance: the loader's own ambiguous set is non-empty and smaller than the
    # query count, which is why "global scope" and "unresolved category" are different states.
    ambiguous = resolution.get("ambiguous_qids") or ()
    assert resolution["ambiguous_count"] == len(ambiguous)
    assert 0 < len(ambiguous) < len(data.queries)

def test_the_served_block_equals_the_recorded_block_on_the_frozen_artifact(monkeypatch) -> None:
    """`GET /api/composition/runs/<id>` must SERVE the recorded block, never reconcile it.

    The number was wrong where it was measured, so the backend must not paper over it: a served
    value that disagreed with the artifact on disk would make the defect unfalsifiable from the API
    and would diverge from `derive_routing_cohort_gap.py`, which replays the artifact directly.
    The frozen run keeps its old, misleading block — and the API says exactly that.
    """
    if not FROZEN.is_file():
        pytest.skip(f"{FROZEN} is not present on this host")
    from workbench.backend import main as M
    from workbench.backend.composition import _SUMMARY_FILE_CACHE
    import json as _json

    recorded = _json.loads(FROZEN.read_text(encoding="utf-8"))["adapter_query_scope_resolution"]
    for cache in (_SUMMARY_FILE_CACHE,):
        cache.clear()
    served = M.composition_run(run_id=FROZEN_RUN_ID, subset="full")
    block = served["adapter_query_scope_resolution"]
    assert block["still_unscoped"] == recorded["still_unscoped"] == 6003
    assert block["unscoped_qids"] == recorded["unscoped_qids"]
    assert block["adapter_routed"] == recorded["adapter_routed"] == 0
    assert "category_agnostic" not in block, (
        "the served payload must not invent the key for an artifact that predates it")
    # ...and the contradiction is still visible through the API, which is the honest answer for a
    # run measured before the fix. No backend code rewrites it.
    assert sum(row["unscored_count"] for row in served["condition_status"].values()) == 0

def test_the_served_block_exposes_the_new_keys_for_a_new_shape_summary(tmp_path, monkeypatch) -> None:
    """A summary written by the FIXED runner is served verbatim, new keys included."""
    import json as _json

    from workbench.backend import main as M
    from workbench.backend.composition import _SUMMARY_FILE_CACHE

    block = {"resolver": None, "loader_scoped": 3, "category_agnostic": True,
             "adapter_routed": 0, "global_scoped": 4, "still_unscoped": 0, "unscoped_qids": []}
    summary = {"schema_version": 1, "model_id": "tgcir", "checkpoint_id": "ck", "run_id": "newshape",
               "subset": "full", "protocol_id": "p", "conditions": ["multimodal"],
               "query_count": 4, "gallery_count": 4, "execution_status": "COMPLETED",
               "gallery_identity_sha256": "g" * 64, "metrics": {"coverage": 1.0},
               "condition_status": {"multimodal": {"status": "COMPLETED", "unscored_count": 0}},
               "category_scopes": [None], "adapter_query_scope_resolution": block}
    path = tmp_path / "composition" / "summary" / "tgcir_ck_newshape" / "full.json"
    path.parent.mkdir(parents=True)
    path.write_text(_json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr(M, "resolve_config",
                        lambda: SimpleNamespace(WORKBENCH_ARTIFACTS_ROOT=tmp_path,
                                                FASHIONIQ_ROOT=str(tmp_path / "fiq")))
    _SUMMARY_FILE_CACHE.clear()
    served = M.composition_run(run_id="newshape", subset="full")
    _SUMMARY_FILE_CACHE.clear()
    assert served["adapter_query_scope_resolution"] == block
