"""Integration: the pinned processor path, then a checkpoint whose class was never cached.

The R10 evidence for this boundary used a miniature package and loaded a class the test had already
imported, which cannot distinguish "the package context survives" from "the class happened to be in
`sys.modules`". This runs the real thing:

* the producer is a SEPARATE process that builds the checkpoint against the pinned checkout and then
  exits, so nothing it imported survives;
* the consumer is a fresh process that calls the production wrapper's own `_load_replay_modules`
  (processor setup, registry, class resolution) and only then deserialises the checkpoint, which
  must resolve a pinned class path that was never cached in that interpreter.

A generated fixture checkpoint is used because the author weights are not on this host. The digest
is computed from the produced file and checked by the production loader, so the trust path is real
even though the weights are not.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from workbench.backend.registry import model_by_id  # noqa: E402

LAVIS_FAMILY = {
    "hint": ("HINT", "lavis/models/blip2_models/HINT.py"),
    "habit": ("HABIT", "lavis/models/blip2_models/HABIT.py"),
    "intent": ("INTENT", "lavis/models/blip2_models/blip2_cir.py"),
    "conesep": ("ConeSep", "lavis/models/blip2_models/ConeSep.py"),
    "airknow": ("AirKnow", "lavis/models/blip2_models/blip2_cir.py"),
}

PRODUCER = r'''
"""Build a fixture checkpoint against the pinned checkout, then exit.

Runs as its own process: whatever it imports must not be relied on by the consumer.
"""
import sys
from pathlib import Path

repo, source_dir, class_file, out = sys.argv[1:5]
sys.path.insert(0, repo)

from workbench.replay import lavis_processors

source = Path(repo) / "workbench" / "third_party" / source_dir
lavis_processors.prepare(source)

# The class the pinned config selects, imported here so the pickle records its real dotted path.
module = __import__(str(Path(class_file).with_suffix("")).replace("/", ".").replace(".py", ""),
                    fromlist=["*"])
import torch

class_name = next(name for name in dir(module)
                  if name[0].isupper() and not name.startswith("_")
                  and getattr(getattr(module, name), "__module__", "") == module.__name__)
instance = getattr(module, class_name).__new__(getattr(module, class_name))
instance.fixture_marker = "r11-lifecycle"
torch.save(instance, out)
print(json.dumps({"class_name": class_name, "module": module.__name__}))
'''

CONSUMER = r'''
"""Fresh process: processor path first, then deserialise a class that was never imported here."""
import hashlib
import json
import sys
from pathlib import Path

repo, model_id, checkpoint, build_report = sys.argv[1:5]
sys.path.insert(0, repo)

import torch
from workbench.backend.registry import model_by_id
from workbench.replay import lavis_processors
from workbench.replay.common import load_trusted_whole_model

model = model_by_id(model_id)
source = Path(repo) / "workbench" / "third_party" / model["source_dir"]

# The production wrapper's own module loader: processors, registry, and the boundary it reports.
import importlib

wrapper = importlib.import_module(f"workbench.replay.{model_id}")
checkpoint_path = Path(checkpoint)
checkpoint_globals_before = set(sys.modules)

loaded = wrapper._load_replay_modules(source)
# The wrappers return their own shape: HINT groups `(data_utils, test_module)` into one element,
# the others return them flat. Reading the LAST element as the processor pair covers both, which is
# also how each wrapper's own `evaluate_categories` consumes it.
torch_module = loaded[0]
txt_processors, boundary = loaded[-1]

# The class the checkpoint records must NOT have been imported before the wrapper ran its own
# processor path; that is the property under test, and it is measured rather than assumed.
# The producer reports the module it pickled, which is the reliable answer: scanning the pickle
# bytes for a dotted name does not survive the protocol's framing.
# The producer reports the module it pickled: scanning the pickle bytes for a dotted name does not
# survive the protocol's framing, and a guess here would make the assertion below unfalsifiable.
_recorded_module = json.loads(Path(build_report).read_text(encoding="utf-8"))["module"]
cached_before = _recorded_module in checkpoint_globals_before
if cached_before:
    # Nothing to prove in this process: the interpreter already has the class, so a successful load
    # would say nothing about the package context surviving the wrapper's setup.
    print(json.dumps({"error": f"{_recorded_module} was already imported before the loader ran"}))
    raise SystemExit(3)

digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
instance = load_trusted_whole_model(checkpoint_path, digest, torch_module)

print(json.dumps({
    "boundary": boundary,
    "resolved_class": f"{type(instance).__module__}.{type(instance).__name__}",
    "marker": getattr(instance, "fixture_marker", None),
    "processor_type": type(txt_processors.get("image") or txt_processors.get("text")
                           or next(iter(txt_processors.values()))).__name__,
    "class_cached_before_loader": cached_before,
    "recorded_module": _recorded_module,
}))
'''


def _model_interpreter(model_id: str) -> str:
    """The interpreter the production adapter would launch, or a skip naming what is missing.

    Running this test under the workbench interpreter would test the wrong environment: the model
    contract declares its own Python and its own `transformers` (4.25.0 for all five of these), while
    the host workbench venv carries whatever the backend needs. That difference is not cosmetic --
    the workbench venv's `transformers` 4.57.6 no longer re-exports `apply_chunking_to_forward` from
    `modeling_utils`, which the pinned `med.py` imports, so the pinned class fails to import there
    and succeeds in the declared environment. The declared interpreter is therefore part of the
    boundary under test.
    """
    model = model_by_id(model_id)
    variable = (model.get("environment") or {}).get("python_env_var") or ""
    configured = os.environ.get(variable, "")
    if not configured or not Path(configured).is_file():
        pytest.skip(
            f"{model_id}: {variable or 'no interpreter variable'} is not set to an existing "
            f"interpreter, so the declared model environment cannot be exercised here; "
            f"set {variable}=<python> to run this boundary")
    return configured


@pytest.mark.parametrize("model_id", sorted(LAVIS_FAMILY))
def test_the_pinned_processor_path_precedes_an_uncached_checkpoint_class(
        model_id: str, tmp_path: Path) -> None:
    source_dir, class_file = LAVIS_FAMILY[model_id]
    checkout = REPO / "workbench" / "third_party" / source_dir
    if not checkout.is_dir():
        pytest.skip(f"pinned checkout absent: {checkout}")
    interpreter = _model_interpreter(model_id)

    producer = tmp_path / "producer.py"
    producer.write_text("import json\n" + PRODUCER, encoding="utf-8")
    consumer = tmp_path / "consumer.py"
    consumer.write_text("import json\n" + CONSUMER, encoding="utf-8")
    checkpoint = tmp_path / "fixture-ckpt.pt"

    environment = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1"}
    produced = subprocess.run(
        [interpreter, str(producer), str(REPO), source_dir, class_file, str(checkpoint)],
        capture_output=True, text=True, env=environment, check=False, timeout=300)
    assert produced.returncode == 0, (
        f"producer failed for {model_id}: {produced.stdout[-500:]} {produced.stderr[-1500:]}")
    assert checkpoint.is_file(), f"{model_id}: producer wrote no checkpoint"
    built = json.loads(produced.stdout.strip().splitlines()[-1])
    build_report = tmp_path / "built.json"
    build_report.write_text(json.dumps(built), encoding="utf-8")

    consumed = subprocess.run(
        [interpreter, str(consumer), str(REPO), model_id, str(checkpoint), str(build_report)],
        capture_output=True, text=True, env=environment, check=False, timeout=300)
    assert consumed.returncode == 0, (
        f"consumer failed for {model_id}: {consumed.stdout[-500:]} {consumed.stderr[-1500:]}")
    report = json.loads(consumed.stdout.strip().splitlines()[-1])

    # The checkpoint resolved a class that this interpreter had not imported.
    assert report["resolved_class"] == f"{built['module']}.{built['class_name']}", report
    assert report["marker"] == "r11-lifecycle"
    # The processor setup is the production boundary, and the fixture property is present.
    assert report["boundary"], report
    assert checkpoint.stat().st_size > 0
    # The property: the class the checkpoint names was NOT yet imported when this interpreter
    # started the wrapper's loader, and the load found it anyway. Without the pinned package context
    # surviving that loader, a path that was never cached is exactly what fails to resolve.
    assert report["recorded_module"] == built["module"], (
        f"the checkpoint records {report['recorded_module']}, expected {built['module']}")
    assert report["class_cached_before_loader"] is False, (
        f"{report['recorded_module']} was already imported before the wrapper ran, so this run "
        "cannot show the class resolving for the first time")


def test_the_digest_gate_still_precedes_deserialisation(tmp_path: Path) -> None:
    """The fixture path must not have weakened the trust check it runs through."""
    interpreter = _model_interpreter("hint")
    producer = tmp_path / "producer.py"
    producer.write_text("import json\n" + PRODUCER, encoding="utf-8")
    checkpoint = tmp_path / "ckpt.pt"
    environment = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1"}
    produced = subprocess.run(
        [interpreter, str(producer), str(REPO), "HINT", "lavis/models/blip2_models/HINT.py",
         str(checkpoint)],
        capture_output=True, text=True, env=environment, check=False, timeout=300)
    if produced.returncode != 0:
        pytest.skip(f"producer unavailable here: {produced.stderr[-200:]}")

    probe = tmp_path / "probe.py"
    probe.write_text(
        "import hashlib, sys\n"
        "from pathlib import Path\n"
        "repo, checkpoint = sys.argv[1:3]\n"
        "sys.path.insert(0, repo)\n"
        "import torch\n"
        "from workbench.replay.common import load_trusted_whole_model\n"
        "try:\n"
        "    load_trusted_whole_model(Path(checkpoint), '0' * 64, torch)\n"
        "except ValueError as error:\n"
        "    print('REFUSED', error)\n", encoding="utf-8")
    refused = subprocess.run([interpreter, str(probe), str(REPO), str(checkpoint)],
                             capture_output=True, text=True, env=environment, check=False,
                             timeout=300)
    assert refused.returncode == 0, refused.stderr[-800:]
    assert "REFUSED" in refused.stdout and "SHA-256 mismatch" in refused.stdout, refused.stdout
