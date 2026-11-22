import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))
from workbench.backend.operator_config import resolve_config
from workbench.backend.registry import model_by_id
from workbench.backend.runtime import require_verified_model_interpreter, check_environment_isolation
import json

config = resolve_config()
model = model_by_id("dcnet")
interpreter = require_verified_model_interpreter(model, require_cuda=False, config=config)
ok, reason = check_environment_isolation(Path(interpreter), {})
print(json.dumps({"controller_executable": sys.executable,
                  "controller_prefix": sys.prefix,
                  "worker_interpreter": interpreter,
                  "worker_isolation": ok,
                  "worker_isolation_reason": reason}, sort_keys=True))
if not ok:
    raise SystemExit(1)
