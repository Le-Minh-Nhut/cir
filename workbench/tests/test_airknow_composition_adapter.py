from __future__ import annotations

import numpy as np
import pytest

from workbench.composition import model_airknow


def test_native_max_token_dot_preserves_singleton_axes() -> None:
    adapter = model_airknow.Adapter.__new__(model_airknow.Adapter)
    queries = np.zeros((1, 256), dtype=np.float32)
    queries[0, 3] = 2
    targets = np.zeros((1, 32, 256), dtype=np.float32)
    targets[0, 31, 3] = 3
    np.testing.assert_array_equal(adapter.score(queries, targets), [[6]])
    np.testing.assert_array_equal(adapter.score(queries, targets.reshape(1, 8192)), [[6]])


def test_reference_eligibility_probe_uses_one_token_without_squeeze() -> None:
    adapter = model_airknow.Adapter.__new__(model_airknow.Adapter)
    measured = adapter.measured_reference_eligibility()
    assert measured == {
        "measured": True,
        "reference_eligible": True,
        "reason": None,
        "reference_rank": 1,
        "method": "score+rank on a synthetic gallery whose reference is the exact match",
    }


def test_invalid_native_feature_shapes_are_rejected() -> None:
    adapter = model_airknow.Adapter.__new__(model_airknow.Adapter)
    with pytest.raises(ValueError, match="Air-Know expects"):
        adapter.score(np.zeros((1, 256)), np.zeros((1, 512)))
    with pytest.raises(ValueError, match="Air-Know expects"):
        adapter.score(np.zeros((1, 256)), np.zeros((1, 0, 256)))
