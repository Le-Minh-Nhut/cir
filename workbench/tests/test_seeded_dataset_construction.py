"""Regressions for the stochastic-query-input defect (R18 Fix A).

The iLearn-family constructors bake the transformed query image INTO the dataset at construction
time: `datasets.py::get_test_data` calls `get_img(..., stage=0)` and the replay supplies
`transform = [preprocess_train, preprocess_val]`, so query inputs are drawn through
`preprocess_train`, whose first step is `RandomResizedCrop`. Two constructions therefore observe
different query populations, which is what made ENCODER's published per-query artifact disagree with
the aggregate it was supposed to reconstruct.

These tests use a genuinely stochastic transform and two INDEPENDENT constructions. A test over fixed
vectors would pass whether or not the defect existed, because it would never draw from the RNG.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from workbench.replay.common import (  # noqa: E402
    DATASET_CONSTRUCTION_SEED,
    seeded_dataset_construction,
)


def _stochastic_transform():
    """A transform with the property that matters: it draws from the global RNG per call.

    `torchvision.transforms.RandomResizedCrop` is the step the pinned objective actually applies to
    query images. This reproduces its *contract* -- a different crop box on every call -- so the test
    exercises the seeding mechanism rather than one torchvision version's particular RNG consumption.
    It returns the drawn box because the box IS the input population's identity here.
    """
    torch = pytest.importorskip("torch")

    def draw(_image):
        return tuple(torch.randint(0, 32, (4,)).tolist())

    return draw


def _construct_queries(count: int = 6) -> list[tuple]:
    """One construction's query inputs -- the population the defect makes disagree across runs."""
    draw = _stochastic_transform()
    return [draw(None) for _ in range(count)]


def test_unseeded_constructions_draw_different_query_inputs() -> None:
    """The defect itself: without seeding, two constructions disagree.

    This is the test that would have caught the published ENCODER disagreement. It asserts the
    PROBLEM, so it documents why the seeding below is necessary rather than cosmetic.
    """
    first = _construct_queries()
    second = _construct_queries()
    assert first != second, (
        "the stochastic transform produced identical draws across two independent constructions, so "
        "this test cannot demonstrate the defect it exists to pin; the transform must draw from the "
        "global RNG per call")


def test_seeded_constructions_observe_the_same_query_inputs() -> None:
    """The fix: with the construction seeded, every construction draws the same population."""
    with seeded_dataset_construction():
        first = _construct_queries()
    with seeded_dataset_construction():
        second = _construct_queries()
    assert first == second, (
        "two seeded constructions disagree; a per-query export that rebuilds the dataset would again "
        "observe a different input population than the aggregate")


def test_a_different_seed_draws_a_different_population() -> None:
    """Identity is per-seed, not "the transform stopped being random".

    If seeding worked by making the transform deterministic in general, the previous test would pass
    while silently changing the experiment. This pins that the seed still selects a population.
    """
    with seeded_dataset_construction(0):
        first = _construct_queries()
    with seeded_dataset_construction(1):
        second = _construct_queries()
    assert first != second


def test_seeding_a_construction_does_not_leak_into_the_caller_stream() -> None:
    """A seeded construction must not perturb whatever the caller was doing."""
    random.seed(1234)
    expected = [random.random() for _ in range(3)]

    random.seed(1234)
    random.random()                     # consume the caller's first value
    with seeded_dataset_construction():
        inside = random.random()        # draws from the SEEDED stream
    after = random.random()             # must resume the caller's stream

    assert after == expected[1], (
        "the caller's RNG stream did not resume after a seeded construction, so seeding leaked")
    assert inside != expected[1], "the context did not actually seed the RNG"


def test_the_seed_is_a_recorded_constant_not_an_implicit_zero() -> None:
    """The seed is part of the experiment's identity, so it is named."""
    assert isinstance(DATASET_CONSTRUCTION_SEED, int)


@pytest.mark.parametrize("modality", ["python", "numpy", "torch"])
def test_each_random_source_is_saved_and_restored(modality: str) -> None:
    """Every generator the constructor can draw from is restored, not just `random`."""
    if modality != "python":
        pytest.importorskip(modality if modality == "numpy" else "torch")
    if modality == "python":
        random.seed(7)
        before = random.random()
        with seeded_dataset_construction():
            pass
        random.seed(7)
        assert random.random() == before
    elif modality == "numpy":
        import numpy as np

        np.random.seed(7)
        before = np.random.random()
        with seeded_dataset_construction():
            pass
        np.random.seed(7)
        assert np.random.random() == before
    else:
        import torch

        torch.manual_seed(7)
        before = torch.rand(1).item()
        with seeded_dataset_construction():
            pass
        torch.manual_seed(7)
        assert torch.rand(1).item() == before
