"""Phase A: freeze the registry facts each protocol row cites, as text for the matrix."""
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/heheboiz/data/cir")
from workbench.backend.registry import load_registry, model_by_id, checkpoint_path
from workbench.backend.operator_config import resolve_config

r = load_registry()
print("# protocols")
print(json.dumps(r["protocols"], indent=1, sort_keys=True))
print("# models")
for m in r["models"]:
    mid = m["model_id"]
    print("##", mid)
    for k in ("native_protocol","literature_split_label","replay_type","replay_status",
              "checkpoint_serialization","category_model_policy","command_status",
              "reproduction_status","per_query_observer","per_query_checkpoint",
              "approved_source_patches","fashioniq_layout","preparation_id",
              "fashioniq_image_extensions","canonical_dataset_link","category_checkpoint_structure"):
        v = m.get(k)
        if v is not None:
            print("  ", k, "=", json.dumps(v)[:300])
    env = m.get("environment") or {}
    print("   env:", json.dumps({k: env.get(k) for k in ("python","pytorch","torchvision","cuda","confidence","python_env_var")}))
    if env.get("adapted_runtime"):
        print("   adapted:", json.dumps(env["adapted_runtime"]))
    cfg = resolve_config()
    for c in (m.get("checkpoint_variants") or []):
        slim = {k: c.get(k) for k in ("checkpoint_id","filename","expected_sha256","status",
               "checkpoint_mapping_status","checkpoint_training_noise_pct","evaluation_noise_pct",
               "artifact_type","required_files","bundle_members")}
        print("   variant", json.dumps(slim)[:600])
        try:
            print("     resolved_path:", checkpoint_path(mid, c))
        except Exception as e:
            print("     resolved_path: ERROR", type(e).__name__, e)
