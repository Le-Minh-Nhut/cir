"""The runner must use the adapter's scope resolver, and must not let a resume hide it.

`Adapter.query_scope()` claimed the runner called it; it did not, so 119 unscored FashionIQ
queries stayed unscored. Two things have to hold together, and the second is the one that is easy
to miss: calling the resolver alone leaves the run identity unchanged (`_query_fingerprint` keys on
the loader's `query.category`, still `None`), so `--resume` matches the old ranks artifact and
silently reuses the 119 stale nulls. These checks pin the resolver contract and the identity move.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from workbench.composition import runner

SCOPES = ("dress", "shirt", "toptee")

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
