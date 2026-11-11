"""Checkpoint resolution + digests, ONLY through workbench.backend.registry.checkpoint_path()."""
import sys, hashlib
sys.path.insert(0,'/home/heheboiz/data/cir')
from workbench.backend.registry import (load_registry, checkpoint_by_id, checkpoint_path,
                                       checkpoint_is_present)
from workbench.backend.operator_config import resolve_config
cfg = resolve_config()
print("checkpoint root:", cfg.WORKBENCH_CHECKPOINT_ROOT)
reg = load_registry()
for mid in ("clvc_net", "tgcir"):
    model = next(m for m in reg["models"] if m["model_id"] == mid)
    cp = checkpoint_by_id(model, model["checkpoint_variants"][0]["checkpoint_id"])
    p = checkpoint_path(mid, cp, cfg.WORKBENCH_CHECKPOINT_ROOT)
    print("###", mid, cp["checkpoint_id"])
    print("  path:", p)
    print("  present:", checkpoint_is_present(p, cp, model_id=mid, config=cfg))
    if p.is_file():
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        print("  sha256:", digest, "declared:", cp.get("expected_sha256"), "match:", digest == cp.get("expected_sha256"))
        print("  size:", p.stat().st_size, "declared:", cp.get("expected_size_bytes"))
    elif p.is_dir():
        total = 0
        for member in cp.get("bundle_members") or []:
            f = p / member["filename"]
            digest = hashlib.sha256(f.read_bytes()).hexdigest()
            total += f.stat().st_size
            print("   ", member["filename"], f.stat().st_size,
                  "MATCH" if digest == member["expected_sha256"] else "MISMATCH")
        print("  total bytes:", total)
