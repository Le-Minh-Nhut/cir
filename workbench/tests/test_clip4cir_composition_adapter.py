from __future__ import annotations

import numpy as np

from workbench.composition.model_clip4cir_rn50x4_fullft import Adapter as FullFtAdapter
from workbench.composition.model_combiner_rn50x4_noft import Adapter as NoFtAdapter


def test_both_variants_score_normalized_clip4cir_cosine() -> None:
    queries = np.zeros((2, 640), dtype=np.float32)
    queries[0, 0] = 7
    queries[1, 1] = 3
    targets = np.zeros((3, 640), dtype=np.float32)
    targets[0, 0] = 2
    targets[1, 1] = 4
    targets[2, 0] = 1
    expected = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
    for adapter_type in (NoFtAdapter, FullFtAdapter):
        adapter = adapter_type.__new__(adapter_type)
        np.testing.assert_allclose(adapter.score(queries, targets), expected)


def test_empty_gallery_keeps_query_rows() -> None:
    adapter = NoFtAdapter.__new__(NoFtAdapter)
    scores = adapter.score(np.zeros((2, 640)), np.zeros((0, 640)))
    assert scores.shape == (2, 0)

def test_small_vectors_use_torch_normalize_epsilon() -> None:
    adapter = NoFtAdapter.__new__(NoFtAdapter)
    query = np.zeros((1, 640), dtype=np.float32)
    query[0, 0] = 1e-20
    target = np.zeros((1, 640), dtype=np.float32)
    target[0, 0] = 1
    scores = adapter.score(query, target)
    np.testing.assert_allclose(scores, [[1e-8]], rtol=1e-6, atol=0)
