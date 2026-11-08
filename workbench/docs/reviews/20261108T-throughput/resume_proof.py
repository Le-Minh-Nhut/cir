import sys, json
sys.path.insert(0,"/home/heheboiz/data/cir")
from workbench.composition.runner import _digest, BLAS_THREADS_PINNED

# Reproduce the identity subset that carries the arithmetic policy, as the runner builds it.
base = {"model_id":"encoder","checkpoint_id":"fashioniq","scoring_module_source_hash":"abc"}
pinned = dict(base, blas_threads_pinned=BLAS_THREADS_PINNED)
unpinned = dict(base)                    # field ABSENT = pre-pin artifact
other = dict(base, blas_threads_pinned=10)

fp_pin, fp_unpin, fp_10 = _digest(pinned), _digest(unpinned), _digest(other)
print("pinned=1   fingerprint:", fp_pin[:24])
print("absent     fingerprint:", fp_unpin[:24])
print("pinned=10  fingerprint:", fp_10[:24])
print("pin != absent :", fp_pin != fp_unpin)
print("pin != 10     :", fp_pin != fp_10)

# An EXISTING published artifact carries no such field, so it cannot match the new fingerprint.
art = "workbench/artifacts/composition/runs/6a528e877c328772e2c7bb28/ranks/encoder_fashioniq.json"
ident = json.load(open(art)).get("identity", {})
print(f"\npublished encoder identity has blas_threads_pinned: {'blas_threads_pinned' in ident}")
print(f"published encoder fingerprint (as recorded): {json.load(open(art)).get('fingerprint','')[:24]}")
print(f"new-code fingerprint would be:               {fp_pin[:24]}")
print("=> resume cannot reuse it (fingerprints differ)")
json.dump({"pinned":fp_pin,"absent":fp_unpin,"pinned10":fp_10}, open("/tmp/resume_proof.json","w"), indent=2)
