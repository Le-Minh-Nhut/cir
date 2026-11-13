import hashlib, struct, time, json, sys
import numpy as np
vals = list(np.random.rand(4096))
pack1 = struct.Struct("<d")
def per_element(values):
    d = hashlib.sha256()
    for v in values: d.update(pack1.pack(float(v)))
    return d.hexdigest()
def bulk(values):
    return hashlib.sha256(struct.pack(f"{len(values)}d", *values)).hexdigest()
def bulk_bytes(values):
    return hashlib.sha256(struct.pack(f"<{len(values)}d", *values)).hexdigest()
assert per_element(vals)==bulk(vals)==bulk_bytes(vals)
N=200
t=time.perf_counter()
for _ in range(N): per_element(vals)
a=time.perf_counter()-t
t=time.perf_counter()
for _ in range(N): bulk(vals)
b=time.perf_counter()-t
numpy_f8 = hashlib.sha256(np.ascontiguousarray(np.array(vals),dtype="<f8").tobytes()).hexdigest()
print(json.dumps({"identical": per_element(vals)==numpy_f8==bulk(vals),
 "per_element_s_per_4096": round(a/N,6), "bulk_s_per_4096": round(b/N,6),
 "speedup": round(a/b,1), "projected_saving_s_for_5517_digests":
   round((a-b)/N*5517/ (5517/200),2) if False else round((a-b)/N*(22513152/4096),2)}))
