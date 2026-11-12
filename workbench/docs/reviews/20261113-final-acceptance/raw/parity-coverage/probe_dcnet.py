
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/heheboiz/data/cir")
from workbench.backend.operator_config import resolve_config
from workbench.backend.registry import load_registry, checkpoint_availability
from workbench.backend.preflight import preflight_blockers
from workbench.backend.runtime import resolve_model_interpreter
model_id = 'dcnet'
config = resolve_config()
registry = load_registry()
model = next(row for row in registry["models"] if row["model_id"] == model_id)
variable = (model.get("environment") or {}).get("python_env_var")
resolved = resolve_model_interpreter(model, config)
print(json.dumps({
    "model_id": model_id,
    "python_env_var": variable,
    "resolver_returned": resolved,
    "interpreter_exists": bool(resolved) and Path(resolved).is_file(),
    "preflight_blockers": preflight_blockers(model, config),
    "adapter_module": model.get("composition_adapter"),
    "adapter_module_exists": bool(model.get("composition_adapter")) and (
        Path("/home/heheboiz/data/cir") / (str(model["composition_adapter"]).replace(".", "/") + ".py")
    ).is_file(),
    "checkpoints": [
        {"checkpoint_id": variant.get("checkpoint_id"),
         "path": checkpoint_availability(model, variant, None)["checkpoint_path"],
         "downloaded": checkpoint_availability(model, variant, None)["checkpoint_downloaded"],
         "mapping_verified": checkpoint_availability(model, variant, None)["mapping_verified"]}
        for variant in (model.get("checkpoint_variants") or [])
    ],
}, indent=1))
