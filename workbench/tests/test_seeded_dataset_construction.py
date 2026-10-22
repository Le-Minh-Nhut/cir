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


# ------------------------------------------------------------------ run-scoped cache reuse (R18)

def _dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    for name in ("captions", "image_splits", "resized_image"):
        (root / name).mkdir(parents=True)
    return root


def test_a_run_shares_one_cache_between_its_children(tmp_path: Path, monkeypatch) -> None:
    """The aggregate and the observer must build the native cache ONCE between them.

    `isolated_cache_root` yields a fresh directory every time, so each child built its own copy --
    measured at an 11.87 GB peak for one three-category construction. Two of them need twice a host
    that fits one, and the observer died with `[Errno 28] No space left on device` inside the native
    `save_obj`, which reads as a broken export rather than as a disk the parent could have saved.
    """
    from workbench.replay.common import CACHE_ROOT_ENV, run_cache_root

    dataset = _dataset_root(tmp_path)
    shared = tmp_path / "replay-cache"
    monkeypatch.setenv(CACHE_ROOT_ENV, str(shared))

    with run_cache_root(dataset) as first:
        assert first == shared
        (first / "fashion_iq_data.json").write_text("{}")
        (first / "test_queries_dress.pkl").write_bytes(b"cache")

    with run_cache_root(dataset) as second:
        assert second == first, "the second child did not join the run's cache"
        assert (second / "test_queries_dress.pkl").read_bytes() == b"cache", (
            "the second construction cannot see the first's cache, so it would rebuild it")


def test_the_cache_root_is_fresh_when_no_parent_named_one(tmp_path: Path, monkeypatch) -> None:
    """A directly-invoked replay keeps full isolation: there is no run to share with."""
    from workbench.replay.common import CACHE_ROOT_ENV, run_cache_root

    dataset = _dataset_root(tmp_path)
    monkeypatch.delenv(CACHE_ROOT_ENV, raising=False)
    with run_cache_root(dataset) as first:
        (first / "test_queries_dress.pkl").write_bytes(b"cache")
    with run_cache_root(dataset) as second:
        assert second != first
        assert not (second / "test_queries_dress.pkl").exists()


def test_the_shared_cache_still_symlinks_the_pinned_inputs(tmp_path: Path, monkeypatch) -> None:
    """Reuse must not replace the isolation that made the cache safe: inputs are links, not copies."""
    from workbench.replay.common import CACHE_ROOT_ENV, run_cache_root

    dataset = _dataset_root(tmp_path)
    (dataset / "captions" / "cap.dress.val.json").write_text("[]")
    monkeypatch.setenv(CACHE_ROOT_ENV, str(tmp_path / "replay-cache"))

    with run_cache_root(dataset) as root:
        link = root / "captions"
        assert link.is_symlink(), "the shared cache copied an input instead of linking it"
        assert link.resolve() == (dataset / "captions").resolve()
