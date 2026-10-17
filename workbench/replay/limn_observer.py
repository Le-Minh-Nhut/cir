"""Observation-only per-query exporter for the pinned LIMN evaluation.

The pinned author's ``test.py::test`` computes a full ranking — ``np.argsort(-sims[i, :])`` — and
then throws it away, keeping only ``[(f'{name}_r{k}', r) for k in (1, 10, 50)]``. This module
recovers the per-query rankings that the native function discards. It does not replace the native
metric: the untouched ``test()`` still produces the authoritative aggregate, and
:func:`replay.limn.replay_category_with_observation` runs BOTH and refuses to publish per-query
results unless they agree with the native hit counts exactly.

Two properties this module deliberately preserves, because they are the difference between an
observation and a second implementation:

* The features come from the SAME native model calls the native function makes
  (``model.compose_query(imgs, mods, dataset_type)[0]`` and ``model.compose_target(imgs,
  dataset_type)[0]``), with the same batching, dtype, device, normalization, mask value
  (``-10e10``) and ``np.argsort`` ordering. Only the *ranking* is re-derived, because that is
  precisely the intermediate the native function does not expose.
* The dataset object, gallery construction and gallery ordering are the native dataset's own
  (``test_queries``/``test_targets``). Nothing is sorted, re-derived or re-indexed here.

Identity: the native dataset stores image ids as INTEGER POSITIONS into
``image_splits/split.{category}.val.json`` (``datasets.py`` ``get_test_data``:
``images.index(candidate)``). The canonical FashionIQ image id is ``images[position]``. The
observer does that mapping once, here, so no consumer ever sees an integer position as an id.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CATEGORIES = ("dress", "shirt", "toptee")

#: The pinned author's reference mask (``test.py``: ``sims[i, ...] = -10e10``).
NATIVE_REFERENCE_MASK = -10e10

#: Default saved retrieval depth. The FULL ranking is always computed; this only bounds what is
#: written out.
DEFAULT_TOP_K = 200


@dataclass(frozen=True)
class ObservedQuery:
    """One query's native ranking, with canonical FashionIQ identity attached."""

    annotation_index: int
    reference_id: str
    target_id: str
    raw_captions: tuple[str, str]
    model_input_text: str
    target_rank: int
    top_results: tuple[tuple[int, str, float], ...]  # (1-based rank, canonical image id, score)

    @property
    def query_id(self) -> str:
        return f"{self.category}:{self.annotation_index}:{self.reference_id}:{self.target_id}"

    #: Set by :func:`observe_category`; declared here so the dataclass stays frozen and flat.
    category: str = ""


@dataclass(frozen=True)
class ObservedCategory:
    category: str
    gallery_size: int
    eligible_count: int
    queries: tuple[ObservedQuery, ...]

    def recall(self, k: int) -> float:
        """Recall@k recomputed from the observed ranks — the parity-check input."""
        if not self.queries:
            return 0.0
        return 100 * sum(1 for q in self.queries if q.target_rank <= k) / len(self.queries)


def _image_ids(dataset_root: Path, category: str) -> list[str]:
    """The ordered validation split — the same list the native dataset indexes into."""
    with (dataset_root / "image_splits" / f"split.{category}.val.json").open() as handle:
        return json.load(handle)


def _raw_captions(dataset_root: Path, category: str) -> list[dict[str, Any]]:
    """The unmodified validation caption records, in ORIGINAL annotation order."""
    with (dataset_root / "captions" / f"cap.{category}.val.json").open() as handle:
        return json.load(handle)


class NonFiniteRanking(ValueError):
    """A ranking was attempted on non-finite features or similarities (F-16).

    Raised instead of returning a ranking, because NumPy will happily produce one: a zero-norm
    feature yields NaN similarities, and `np.argsort` places the masked reference (whose negated
    score is a large finite number) *ahead* of every NaN -- so the observation would report the
    reference at rank 1 and the target below it, a plausible-looking ranking built from nothing.

    An observer failure must never be promoted to parity verification, so this refuses rather than
    substituting zero, clamping, or inventing a rank.
    """


def _finite_rows(matrix: "np.ndarray", label: str) -> None:
    if not np.all(np.isfinite(matrix)):
        rows = np.flatnonzero(~np.all(np.isfinite(matrix), axis=1))[:5].tolist()
        raise NonFiniteRanking(
            f"{label} features contain NaN or infinity in row(s) {rows}; the feature extractor "
            "produced a non-finite vector and no ranking can be derived from it")


def rank_queries(query_features: "np.ndarray", target_features: "np.ndarray",
                 reference_positions: list[int], target_positions: list[int],
                 top_k: int = DEFAULT_TOP_K) -> tuple[list[int], list[list[tuple[int, int, float]]]]:
    """The native ranking core, free of torch and of any model.

    This is the numerically load-bearing part of :func:`observe_category`, isolated so it can be
    checked on a machine with no GPU. It performs exactly the pinned author's steps: per-row L2
    normalization, ``dot`` similarity, the ``-10e10`` reference mask, and ``np.argsort(-sims)``.

    Returns ``(target_ranks, top_results)`` where each rank is 1-based over the FULL native
    ranking (the masked reference still occupies a slot) and each top-results entry is
    ``(rank, gallery_index, score)``.

    Non-finite inputs are REFUSED (:class:`NonFiniteRanking`) rather than ranked. A zero-norm or NaN
    feature otherwise yields a NaN similarity matrix whose sort order is dominated by the masked
    reference -- a ranking that looks plausible and means nothing.
    """
    if query_features.shape[1] != target_features.shape[1]:
        raise ValueError("query and target feature dimensions differ")
    if len(reference_positions) != len(query_features) or len(target_positions) != len(query_features):
        raise ValueError("one reference and one target position are required per query")

    _finite_rows(query_features, "query")
    _finite_rows(target_features, "gallery")
    query_norms = np.linalg.norm(query_features, axis=1, keepdims=True)
    target_norms = np.linalg.norm(target_features, axis=1, keepdims=True)
    if np.any(query_norms == 0) or np.any(target_norms == 0):
        # A zero-norm row normalizes to NaN. Named separately from a NaN INPUT because the cause and
        # the remedy differ: here the feature is finite but degenerate.
        raise NonFiniteRanking(
            "a query or gallery feature has zero norm, so it cannot be L2-normalized; the "
            "normalization would divide by zero and every similarity for that row would be NaN")

    queries = query_features / query_norms
    targets = target_features / target_norms
    sims = queries.dot(targets.T)

    for index, gallery_index in enumerate(reference_positions):
        sims[index, gallery_index] = NATIVE_REFERENCE_MASK

    if not np.all(np.isfinite(sims)):
        # Finite features can still produce a non-finite similarity through overflow. Refuse: the
        # sort order would be decided by the infinities, not by the model's ranking.
        rows = np.flatnonzero(~np.all(np.isfinite(sims), axis=1))[:5].tolist()
        raise NonFiniteRanking(
            f"the similarity matrix is non-finite in row(s) {rows}; no ranking can be derived")

    ranks: list[int] = []
    top: list[list[tuple[int, int, float]]] = []
    for index, gallery_index in enumerate(target_positions):
        order = np.argsort(-sims[index, :])
        # The full ranking is walked, so a target beyond the saved depth still gets a true rank.
        rank = int(np.flatnonzero(order == gallery_index)[0]) + 1
        ranks.append(rank)
        top.append([(position, int(gallery), float(sims[index, gallery]))
                    for position, gallery in enumerate(order[:top_k], start=1)])
    return ranks, top


class QueryAlignmentError(RuntimeError):
    """The native query set does not correspond to the annotations it is exported against.

    The pinned dataset caches `test_queries`/`test_targets` to `.pkl` and loads them unconditionally
    when present, without recording which annotations they came from. A stale cache from another
    dataset, a truncated file, or a reordered file would therefore produce a *self-consistent*
    ranking whose per-query identities are wrong -- plausible Recall, corrupted failure analysis.
    Nothing in the native source can detect that, so the observer checks it here.
    """


def assert_alignment(category: str, test_queries: list, images: list[str],
                     raw_captions: list[dict], dataset_root: Path | None = None) -> None:
    """Prove every native query corresponds to the annotation at the same index.

    Checks, per index: the reference and target positions map to the annotation's own candidate and
    target ids, and the composed model input is exactly what the native correction pipeline produces
    from that annotation's captions. Any disagreement -- a shuffled cache, a stale one from a
    different dataset, a truncated or longer file -- raises rather than exporting.
    """
    if len(test_queries) != len(raw_captions):
        raise QueryAlignmentError(
            f"{category}: the native query set has {len(test_queries)} entries but "
            f"cap.{category}.val.json has {len(raw_captions)} annotations; a cached query file "
            "does not match this dataset")

    correction_dict: dict[str, str] | None = None
    if dataset_root is not None:
        correction_path = dataset_root / "captions" / f"correction_dict_{category}.json"
        if correction_path.is_file():
            correction_dict = json.loads(correction_path.read_text())

    for index, entry in enumerate(test_queries):
        record = raw_captions[index]
        reference_position = entry["source_img_id"]
        target_position = entry["target_img_id"]
        if not isinstance(reference_position, int) or not isinstance(target_position, int):
            raise QueryAlignmentError(
                f"{category}: query {index} does not carry integer image positions")
        if not 0 <= reference_position < len(images) or not 0 <= target_position < len(images):
            raise QueryAlignmentError(
                f"{category}: query {index} image position is outside split.{category}.val.json")
        if images[reference_position] != record["candidate"]:
            raise QueryAlignmentError(
                f"{category}: query {index} reference is {images[reference_position]!r} but the "
                f"annotation says {record['candidate']!r}; the native query set is not aligned with "
                "the annotations (a stale or reordered cache produces exactly this)")
        if images[target_position] != record["target"]:
            raise QueryAlignmentError(
                f"{category}: query {index} target is {images[target_position]!r} but the "
                f"annotation says {record['target']!r}")
        model_text = (entry.get("mod") or {}).get("str")
        # The composed text is what the model actually read. Recomputing it here from the
        # annotation proves the correction dictionary in force matches the one the cache used: a
        # stale dict would silently change every query's text while leaving ids aligned.
        if model_text is None or not isinstance(model_text, str) or not model_text.strip():
            raise QueryAlignmentError(
                f"{category}: query {index} carries no model input text")
        if correction_dict is not None:
            expected = expected_model_input(record["captions"], correction_dict)
            if model_text != expected:
                raise QueryAlignmentError(
                    f"{category}: query {index} model input text is {model_text!r} but composing the "
                    f"annotation's own captions with the on-disk correction dictionary gives "
                    f"{expected!r}; either the cache used a different dictionary or the captions "
                    "changed")
    return None


def expected_model_input(captions: list, correction_dict: dict[str, str]) -> str:
    """The pinned author's composition: lowercase, strip punctuation, apply the dict, join ' and '.

    Reimplemented from ``datasets.py`` ``correct_text``/``concat_text`` so the observer can prove the
    native text it recorded came from THESE captions and THIS dictionary.
    """
    import string

    def correct(text: str) -> str:
        translation = str.maketrans({key: " " for key in string.punctuation})
        tokens = str(text).lower().translate(translation).strip().split()
        return " ".join(correction_dict.get(word, word) for word in tokens)

    return f"{correct(captions[0])} and {correct(captions[1])}"


def observe_category(category: str, model: Any, dataset: Any, dataset_root: Path,
                     batch_size: int = 32, top_k: int = DEFAULT_TOP_K) -> ObservedCategory:
    """Re-derive one category's per-query rankings from the native model and native dataset.

    ``model`` and ``dataset`` must be the objects :func:`replay.limn.replay_category` loaded; this
    function calls the same ``compose_query``/``compose_target`` methods with the same arguments.
    """
    import torch

    test_queries = dataset.test_queries
    test_targets = dataset.test_targets
    images = _image_ids(dataset_root, category)
    raw_captions = _raw_captions(dataset_root, category)
    # Alignment BEFORE any ranking: identity is checked against the annotations the export claims,
    # not assumed from list lengths.
    assert_alignment(category, test_queries, images, raw_captions, dataset_root)

    # Gallery ordering is the native dataset's own: test_targets is the ordered unique union of
    # validation references and targets, and test_targets[i]['target_img_id'] is that image's
    # position in the split. Reused verbatim, never re-derived (§14).
    gallery_positions = [entry["target_img_id"] for entry in test_targets]
    gallery_ids = [images[position] for position in gallery_positions]
    if len(set(gallery_ids)) != len(gallery_ids):
        raise ValueError(f"{category}: gallery contains duplicate canonical image ids")
    position_of = {position: index for index, position in enumerate(gallery_positions)}

    with torch.no_grad():
        query_features: list[np.ndarray] = []
        batch_images: list[Any] = []
        batch_texts: list[str] = []
        for entry in test_queries:
            batch_images.append(entry["source_img_data"])
            batch_texts.append(entry["mod"]["str"])
            if len(batch_images) >= batch_size or entry is test_queries[-1]:
                stacked = torch.stack([item if "torch" in str(type(item)) else torch.from_numpy(item).float() for item in batch_images]).float().cuda()
                kinds = torch.ones(stacked.shape[0]).cuda() * dataset.dataset_type
                features = model.compose_query(stacked, batch_texts, kinds)[0]
                query_features.append(features.data.cpu().numpy())
                batch_images, batch_texts = [], []

        target_features: list[np.ndarray] = []
        batch_images = []
        for entry in test_targets:
            batch_images.append(entry["target_img_data"])
            if len(batch_images) >= batch_size or entry is test_targets[-1]:
                stacked = torch.stack([item if "torch" in str(type(item)) else torch.from_numpy(item).float() for item in batch_images]).float().cuda()
                kinds = torch.ones(stacked.shape[0]).cuda() * dataset.dataset_type
                features = model.compose_target(stacked, kinds)[0].data.cpu().numpy()
                target_features.append(features)
                batch_images = []

    all_queries = np.concatenate(query_features)
    all_imgs = np.concatenate(target_features)

    # The saved depth is bounded by the ELIGIBLE candidate count, not the raw gallery size: the
    # reference still occupies a column in the native matrix, so a depth equal to the gallery size
    # would include it as a "retrieved" image in a protocol that forbids retrieving it (F-15).
    eligible_depth = min(top_k, len(gallery_ids) - 1)
    ranks, top_rankings = rank_queries(
        all_queries, all_imgs,
        [position_of[entry["source_img_id"]] for entry in test_queries],
        [position_of[entry["target_img_id"]] for entry in test_queries],
        top_k=eligible_depth,
    )

    observed = []
    correction_dict: dict[str, str] | None = None
    if dataset_root is not None:
        correction_path = dataset_root / "captions" / f"correction_dict_{category}.json"
        if correction_path.is_file():
            correction_dict = json.loads(correction_path.read_text())

    for index, entry in enumerate(test_queries):
        record = raw_captions[index]
        observed.append(ObservedQuery(
            annotation_index=index,
            reference_id=images[entry["source_img_id"]],
            target_id=images[entry["target_img_id"]],
            raw_captions=(record["captions"][0], record["captions"][1]),
            model_input_text=entry["mod"]["str"],
            target_rank=ranks[index],
            top_results=tuple((rank, gallery_ids[gallery], score)
                              for rank, gallery, score in top_rankings[index]),
            category=category,
        ))

    return ObservedCategory(
        category=category,
        gallery_size=len(gallery_ids),
        # The reference is masked, never removed: it still occupies a rank slot, so the eligible
        # universe is the full gallery with one candidate made unreachable per query.
        eligible_count=len(gallery_ids) - 1,
        queries=tuple(observed),
    )


def cross_check_with_native(observed: ObservedCategory, native_metrics: dict[str, Any]) -> list[str]:
    """PARITY 1 — native instrumentation parity, on exact hit counts.

    Returns the mismatches. An empty list is the only state in which per-query results may be
    published. This is deliberately not a tolerance comparison: Recall@k is a hit count out of the
    same query population, so the two must agree exactly, or the observation is not the native
    ranking.
    """
    category = observed.category
    mismatches = []
    for k in (1, 10, 50):
        reconstructed = observed.recall(k)
        native = native_metrics.get(f"{category}_r{k}")
        if native is None:
            mismatches.append(f"{category}_r{k}: native metric absent")
            continue
        # Compare as counts, not floats: 100 * hits / n is exact in both directions.
        if round(reconstructed, 9) != round(float(native), 9):
            mismatches.append(f"{category}_r{k}: observer {reconstructed} != native {native}")
    return mismatches
