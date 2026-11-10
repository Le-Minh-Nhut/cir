"""G-02: the R3 case script cannot answer the question it reports on.

`workbench/docs/reviews/20261110-e2e-repair/scripts/pc_r3_real_case.py` is the artifact the campaign
cites for the ENCODER rank comparison. Read against HEAD it does four things that make its verdict
unfounded, and each is reproduced below as a fixture rather than argued:

1. **The mask is not passed.** `rank_and_order(row, target, mask_position=None)` is the default, and
   the script calls it without a mask -- yet the entry it writes says `"rank_used_one_mask": True`
   on every row. LIMN/ENCODER both mask the reference out of their own ranking; a rank computed
   without that mask is not the same rank.
2. **The gap probe reads a DIFFERENT row than the ranking did.** `probe_gap` takes the unmasked row
   while the masked rank used `row[mask] = -10e10`, so the "gap to the crossing candidate" is
   measured on a matrix the rank was not computed from.
3. **The crossing candidates are read from the stale order slice.** `order[lo - 1:hi]` is an index
   range in ONE ordering; when the two ranks differ by more than the entries that actually swapped,
   that slice names candidates that never crossed.
4. **Targets are joined by exact vector equality against the gallery.** One of eleven targets is
   identifiable in the dress/shirt/toptee sample, so the script reports 10/11 as "target not
   identifiable" instead of joining by ID. A vector join also silently succeeds when two gallery
   rows are bit-identical.

Nothing here touches the published artifacts: the tests build their own matrices and assert the
corrected rule, which is what a re-run must use.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REVIEW = Path(__file__).resolve().parents[1] / "docs" / "reviews" / "20261109-native15-composition-limn"
RAW = REVIEW / "raw" / "encoder_parity"
TIE_EPSILON = 1e-06


def _unit(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), np.finfo(np.float32).tiny)


def _rank(row: np.ndarray, target: int, mask_position: int | None) -> tuple[int, np.ndarray]:
    """The rank of `target` in `row`, with `mask_position` excluded exactly as the model does."""
    ranked = row.copy()
    if mask_position is not None:
        ranked[mask_position] = -10e10
    order = np.argsort(-ranked, kind="stable")
    return int(np.flatnonzero(order == target)[0]) + 1, order


def test_a_masked_rank_differs_from_an_unmasked_one_so_the_two_cannot_be_conflated() -> None:
    """The fixture the R3 script gets wrong: the reference sits above the target and is masked out.

    Unmasked, the target is rank 2; masked, it is rank 1. The script reports the masked-looking
    label while computing the unmasked number.
    """
    row = np.array([1.0, 0.5, 0.1], dtype=np.float32)  # reference, target, other
    target, reference = 1, 0
    unmasked_rank, _ = _rank(row, target, mask_position=None)
    masked_rank, _ = _rank(row, target, mask_position=reference)
    assert unmasked_rank == 2 and masked_rank == 1
    assert unmasked_rank != masked_rank, (
        "if masking could not move this rank the fixture would prove nothing")


def _probe_gap(row: np.ndarray, target: int, lo: int, hi: int) -> float:
    """The harness's own probe: max |row[window] - row[target]| over the order slice.

    Copied in shape from `pc_r3_real_case.probe_gap` / `pc_r3_tie_harness.probe_gap`: the window is
    an `argsort` slice and the gap is the max absolute difference inside it.
    """
    window = np.argsort(-row, kind="stable")[lo - 1:hi]
    return float(np.abs(row[window] - row[target]).max()) if len(window) else 0.0


def test_the_gap_must_be_probed_on_the_same_row_the_rank_was_computed_from() -> None:
    """Probing the UNMASKED row puts the masked reference in the window.

    The rank is computed with the reference excluded (`row[reference] = -10e10`, which `argsort`
    orders last), so the reference cannot be a competitor. Probing the unmasked row for the "gap to
    the crossing candidate" scans a window that contains its real score, and reports a gap of ~0.9
    against a declared tie epsilon of 1e-6 -- for a comparison whose only participant is an image
    the ranking never considered.
    """
    row = np.array([1.0, 0.1, 0.1], dtype=np.float32)  # reference, target, tied other
    target, reference = 1, 0
    masked = row.copy()
    masked[reference] = -10e10

    unmasked_rank, _ = _rank(row, target, mask_position=None)
    masked_rank, _ = _rank(masked, target, mask_position=None)
    assert unmasked_rank == 2 and masked_rank == 1, (unmasked_rank, masked_rank)

    lo, hi = sorted((masked_rank, unmasked_rank))
    gap_unmasked_probe = _probe_gap(row, target, lo, hi)
    gap_masked_probe = _probe_gap(masked, target, lo, hi)
    # Unmasked, the window's widest member is the REFERENCE at 1.0 -- 0.9 away from the target, so
    # the script declares a non-tie. Masked, the only other member is an exact tie, gap 0.0.
    assert gap_unmasked_probe > 0.5, gap_unmasked_probe
    assert gap_masked_probe == 0.0, gap_masked_probe
    assert gap_unmasked_probe != gap_masked_probe, (
        "probing a different row than the rank used flips the tie verdict: "
        f"{gap_unmasked_probe!r} vs {gap_masked_probe!r}")


def test_the_crossing_candidates_must_come_from_the_pairwise_definition() -> None:
    """The harness reads crossings off an `argsort` slice of the UNMASKED matrix.

    `pc_r3_real_case.py` ranks with no mask and then reads `order[lo-1:hi]` from the same unmasked
    row, so the reference -- which the real ranking excludes -- occupies rank 1 of that slice and is
    reported as a crossing candidate. Once the window also spans more than one position, the slice
    names whoever happens to sit there, not who actually passed the target.
    """
    target, reference = 1, 0
    # The reference has the highest score; the target drops from above to below two candidates.
    row_a = np.array([9.0, 0.35, 0.30, 0.29], dtype=np.float32)
    row_b = np.array([9.0, 0.25, 0.30, 0.29], dtype=np.float32)
    rank_a, _ = _rank(row_a, target, mask_position=reference)
    rank_b, _ = _rank(row_b, target, mask_position=reference)
    assert (rank_a, rank_b) == (1, 3), (rank_a, rank_b)

    lo, hi = sorted((rank_a, rank_b))
    slice_names = {int(p) for p in np.argsort(-row_a, kind="stable")[lo - 1:hi] if p != target}
    pairwise = {p for p in range(len(row_a)) if p not in (target, reference)
                and (row_a[p] > row_a[target]) != (row_b[p] > row_b[target])}

    assert pairwise == {2, 3}, pairwise
    assert reference in slice_names, (
        f"the unmasked slice leaks the masked reference as a 'crossing candidate': {sorted(slice_names)}")
    assert slice_names != pairwise, (
        "the stale-slice read and the pairwise definition disagree: "
        f"slice={sorted(slice_names)} actual={sorted(pairwise)}")


def test_targets_are_joined_by_id_not_by_exact_vector_equality() -> None:
    """A vector join collapses when two gallery rows are identical, and fails when they are not."""
    gallery = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)  # rows 0 and 2 equal
    target = np.array([1.0, 0.0], dtype=np.float32)
    matches = np.flatnonzero(np.all(gallery == target, axis=1))
    assert matches.size == 2, "the fixture needs a duplicated row for this to mean anything"

    by_vector = int(matches[0]) if matches.size == 1 else None
    assert by_vector is None, (
        "a vector join cannot resolve a duplicated gallery row; only an ID (position) join can")
    by_id = 2  # the annotation names gallery position 2
    assert by_id in matches.tolist()


@pytest.mark.skipif(not (RAW / "native.dress.npz").is_file(),
                    reason="the R3 npz evidence is not on this host")
def test_the_recorded_sample_cannot_identify_the_targets_by_vector() -> None:
    """The campaign's own numbers: 1 of 11 sampled targets is identifiable by exact vector equality.

    This is why the script reports ten "target not identifiable" rows, and why those rows carry no
    rank comparison at all. It is a property of the JOIN, not of the models.
    """
    identifiable = total = 0
    for category in ("dress", "shirt", "toptee"):
        path = RAW / f"native.{category}.npz"
        if not path.is_file():
            pytest.skip(f"{path.name} missing")
        native = np.load(path)
        gallery = _unit(native["gallery"])
        targets = _unit(native["sample_target"])
        total += len(targets)
        for row in targets:
            identifiable += int(np.all(gallery == row, axis=1).sum() == 1)
    assert total == 11, total
    assert identifiable == 1, (
        f"{identifiable}/{total} targets are identifiable by exact vector equality; the R3 script "
        "reports the rest as unidentifiable rather than joining by ID")
