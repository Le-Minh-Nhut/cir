"""Step 0 gate: prove the runtime OpenBLAS pin works in this interpreter."""
import ctypes, glob, json, os, sys
import numpy as np

root = os.path.dirname(os.path.dirname(np.__file__))
out = {"interpreter": sys.executable, "python": sys.version.split()[0],
       "numpy": np.__version__,
       "env_openblas": os.environ.get("OPENBLAS_NUM_THREADS"),
       "env_omp": os.environ.get("OMP_NUM_THREADS"),
       "setter_found": False, "getter_found": False, "source": None,
       "previous": None, "effective_after_1": None, "verified": False,
       "restored": None, "get_matches_set": None}
for path in sorted(glob.glob(os.path.join(root, "numpy.libs", "*openblas*"))):
    try:
        lib = ctypes.CDLL(path)
    except OSError:
        continue
    setter = (getattr(lib, "openblas_set_num_threads64_", None)
              or getattr(lib, "openblas_set_num_threads", None))
    getter = (getattr(lib, "openblas_get_num_threads64_", None)
              or getattr(lib, "openblas_get_num_threads", None))
    if setter is None:
        continue
    out["setter_found"] = True
    out["getter_found"] = callable(getter)
    out["source"] = os.path.basename(path)
    if callable(getter):
        out["previous"] = int(getter())
    setter(1)
    if callable(getter):
        out["effective_after_1"] = int(getter())
        out["verified"] = out["effective_after_1"] == 1
        if out["previous"] is not None:
            setter(out["previous"])
            out["restored"] = int(getter())
            out["get_matches_set"] = out["restored"] == out["previous"]
    break
print(json.dumps(out, indent=1))
