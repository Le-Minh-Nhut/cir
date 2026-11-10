"""S-02: read the effective OpenBLAS pool before, inside and after the pin context.

Mechanism, and its limitation: `openblas_get_num_threads64_` ctypes'd out of the openblas library
`/proc/self/maps` shows as MAPPED by this process (not merely named by a glob). `threadpoolctl` is
not installed, so this is the only readback that speaks for the arithmetic actually running. The
value is process-global; it says nothing about another process.
"""
import ctypes, glob, os, sys
sys.path.insert(0, "/home/heheboiz/data/cir")
import numpy as np
from workbench.composition import runner

root = os.path.dirname(os.path.dirname(np.__file__))
mapped = sorted({line.split()[-1] for line in open("/proc/self/maps") if "openblas" in line.lower()})
globbed = sorted(glob.glob(os.path.join(root, "numpy.libs", "*openblas*")))
setter, getter, source = runner._openblas_symbols()
print("openblas_mapped_by_process:", mapped)
print("openblas_globbed_by_helper:", globbed)
print("helper_source:", source)
print("source_is_the_mapped_library:", os.path.basename(mapped[0]) == source if mapped else None)
print("pool_before:", runner._current_openblas_threads())
pin = runner._blas_thread_limit(runner.BLAS_THREADS_PINNED)
print("record:", {k: pin[k] for k in ("requested", "previous", "effective", "verified", "source")})
print("pool_inside:", runner._current_openblas_threads())
restored = runner._blas_thread_limit(pin)
print("pool_after:", runner._current_openblas_threads())
print("restored_verified:", restored["verified"], "restored_to:", restored["effective"])
print("BLAS_THREADS_PINNED:", runner.BLAS_THREADS_PINNED)
print("torch_threads_untouched_by_blas_pin:", __import__("torch").get_num_threads())
