"""S-02: the BLAS pin must not claim a pin a CONCURRENT job quietly undid.

The audit measured the pin not taking effect at all (request 1, runtime stayed at 9). On this host
the pin DOES take effect (`raw/gate_s02_pool_probe.txt`: 20 -> 1 -> 20, read back through
`openblas_get_num_threads64_` on the library `/proc/self/maps` shows as mapped). What is still
unprotected is the other half of the same claim: the pool is PROCESS-GLOBAL, and nothing stops a
second scorer -- a sibling run in the same process, a library that sets its own thread count --
from changing it while a pinned scoring region is between its set and its restore. The arithmetic
then runs at a thread count the artifact did not declare, and the artifact still says `verified`.

These tests fail on HEAD because `verified` is decided from the readback taken immediately after
the setter, never again before the restore.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.composition import runner

requires_openblas = pytest.mark.skipif(
    not callable(runner._openblas_symbols()[1]),
    reason="this interpreter's OpenBLAS exposes no readback getter")


def test_a_concurrent_writer_during_the_pinned_region_is_not_reported_verified() -> None:
    """A sibling that sets the pool to 10 while we are inside must clear `verified`, not hide."""
    if not callable(runner._openblas_symbols()[1]):
        pytest.skip("no OpenBLAS readback getter on this interpreter")

    observed: dict[str, object] = {}
    stolen = threading.Event()

    def sibling() -> None:
        # A concurrent job that does NOT go through the scope: the case the guard cannot exclude
        # (another library, another process-wide setter) and therefore must DETECT.
        if stolen.wait(timeout=5):
            runner._blas_thread_limit(10)

    thread = threading.Thread(target=sibling, daemon=True)
    thread.start()
    try:
        with runner.blas_pin_scope() as pin:
            pooled_inside = runner._current_openblas_threads()
            stolen.set()
            # Give the sibling time to take the pool; a race we lose is a race we must not claim.
            for _ in range(200):
                if runner._current_openblas_threads() != pooled_inside:
                    break
                threading.Event().wait(0.005)
            observed["during"] = runner._current_openblas_threads()
    finally:
        thread.join(timeout=5)
        runner._blas_thread_limit(20)

    observed["changed_by_sibling"] = observed.get("during") != 1
    if not observed["changed_by_sibling"]:
        pytest.skip("the sibling could not take the pool on this runtime; nothing to observe")
    assert pin["verified"] is False, (
        "the pool was changed by another thread while the pin was in effect, so the pin did not "
        f"govern the arithmetic: {pin}")
    assert pin.get("drift_detected") is True, pin
    assert "concurrent" in (pin.get("reason") or "").lower(), pin


def test_the_pin_is_restored_after_the_scope_even_when_the_body_raises() -> None:
    if not callable(runner._openblas_symbols()[1]):
        pytest.skip("no OpenBLAS readback getter on this interpreter")
    before = runner._current_openblas_threads()
    with pytest.raises(RuntimeError):
        with runner.blas_pin_scope() as pin:
            assert runner._current_openblas_threads() == pin["effective"]
            raise RuntimeError("scoring blew up")
    assert runner._current_openblas_threads() == before, "a raising body must still restore the pool"


def test_an_undisturbed_scope_reports_verified() -> None:
    if not callable(runner._openblas_symbols()[1]):
        pytest.skip("no OpenBLAS readback getter on this interpreter")
    with runner.blas_pin_scope() as pin:
        assert runner._current_openblas_threads() == 1
    assert pin["verified"] is True, pin
    assert pin["drift_detected"] is False, pin


def test_two_pinned_scopes_in_one_process_serialise_instead_of_interleaving() -> None:
    """Two concurrent scopes must not see each other's pool. Serialised, or one is unverified."""
    if not callable(runner._openblas_symbols()[1]):
        pytest.skip("no OpenBLAS readback getter on this interpreter")
    seen: list[int | None] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            with runner.blas_pin_scope() as pin:
                seen.append(runner._current_openblas_threads())
                assert pin["effective"] == runner._current_openblas_threads()
        except BaseException as error:  # noqa: BLE001 - surfaced by the assert below
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not errors, errors
    assert seen == [1, 1], f"each scope must compute at the pinned count, saw {seen}"
    runner._blas_thread_limit(20)
