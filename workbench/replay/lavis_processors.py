"""The pinned LAVIS runtime context: text processors, and a namespace the checkpoint can resolve in.

Two problems live here, and they are the same problem seen twice.

**Why the processor is not obtained the stock way.** The LAVIS-family wrappers (`hint`, `habit`,
`intent`, `conesep`, `airknow`) need exactly one thing from `load_model_and_preprocess`: the processor
its config selects. The stock call builds a SECOND pretrained model to obtain it --
`lavis/models/__init__.py:210` runs `model_cls.from_pretrained(...)` before `:216-220` loads the
preprocess config -- and that constructor reaches for external weights:
`Blip2Base.init_tokenizer`/`init_Qformer` read `/root/bert-base-uncased`
(`lavis/models/blip2_models/blip2.py:32,48,55`), so it needs machine-local BERT assets and multi-GB
base weights purely to hand back a text processor. `text_processors` builds it from the pinned config
instead, through the pinned classes, as described in its own docstring.

**Why the namespace has to survive the call.** The author checkpoints are whole pickled modules
(`train.py:233,279`), so unpickling calls `find_class` with paths like
`lavis.models.blip2_models.HINT.HINT`. `find_class` imports the containing package, which runs
`lavis/__init__.py` -- and that file does `from lavis.datasets.builders import *`
(`:15`), imports every model, and then **registers `library_root` again** (`:24`). The pinned
`registry.register_path` raises `KeyError: Name 'library_root' already registered` on the second
call (`lavis/common/registry.py:200-202`), so a process that had already registered the path during
processor construction dies inside `torch.load`.

`prepare` therefore installs the package root **once** and leaves it installed for the life of the
process, which is the correct scope: the wrappers run one model in one subprocess
(`evaluate_models.py:594` launches it, `workbench/replay/<model>.py` is the whole program). The
registry path registration is made idempotent for the same root, so the native `__init__` can run its
own registration without the assertion firing; a *different* checkout claiming the same name is still
refused, because that is a real conflict rather than a repeat.

Nothing about the pinned code changes: the modules imported are the checkout's own, the classes are
the authors', and `prepare` reports which of them it had to reach around and why.
"""
from __future__ import annotations

import ast
import importlib
import sys
import types
from pathlib import Path
from typing import Any

#: Package objects installed by `prepare`, so a second call can tell "already done" from "torn down".
_INSTALLED: dict[str, str] = {}
#: The checkout each installed package root belongs to, so a second checkout is refused rather than
#: silently re-pointed at the first one's modules.
_OWNER: dict[str, Path] = {}
#: The `sys.path` entries `prepare` added, so `reset` removes exactly those and nothing else.
_ADDED_PATHS: list[str] = []


def _processors_module(lavis_root: Path) -> types.ModuleType:
    """`lavis.processors` with its torchaudio-dependent sibling left out.

    `base_processor` and `blip_processors` are loaded for real -- `registry.register_processor`'s
    decorator asserts every registered class subclasses *that* `BaseProcessor` object, so a second
    copy of the file would fail the assertion. The skeleton is inserted into `sys.modules` BEFORE any
    submodule is imported, or the import would execute the pinned `__init__` that imports
    `BeatsAudioProcessor` -> `torchaudio`, which is in none of these models' declared environments and
    has nothing to do with FashionIQ.
    """
    module = types.ModuleType("lavis.processors")
    module.__path__ = [str(lavis_root / "processors")]
    sys.modules["lavis.processors"] = module
    module.BaseProcessor = importlib.import_module("lavis.processors.base_processor").BaseProcessor
    module.BlipCaptionProcessor = importlib.import_module(
        "lavis.processors.blip_processors").BlipCaptionProcessor
    return module


def _registry_for(lavis_root: Path) -> Any:
    """The pinned `registry`, with `library_root` registered idempotently for *lavis_root*.

    `lavis/__init__.py:24` registers this path on every import, and the pinned implementation raises
    when the name already exists. That assertion is correct for a package imported once and wrong for
    one imported twice in a process that already configured it, which is exactly what happens when a
    pickled checkpoint re-imports the package. Repeating the registration with the SAME root is
    therefore accepted; a different root is not.
    """
    package = sys.modules.get("lavis")
    if package is None:
        package = types.ModuleType("lavis")
        package.__path__ = [str(lavis_root)]
        sys.modules["lavis"] = package
    registry = importlib.import_module("lavis.common.registry").registry
    existing = registry.get_path("library_root")
    if existing is None:
        registry.register_path("library_root", str(lavis_root))
    elif Path(str(existing)) != Path(lavis_root):
        raise ImportError(
            f"the lavis registry already points at {existing}, but this call is for {lavis_root}: "
            "two checkouts cannot share one process, because the pinned class names would resolve "
            "into whichever imported first")
    return registry


def _patch_registry_idempotence() -> None:
    """Let the pinned `register_path` accept a repeat of the same value.

    Applied to the pinned `Registry` class itself rather than to a copy, so the native
    `lavis/__init__.py` import path benefits too. Only the same-name-same-value case changes:
    registering a name twice with different values still raises, and the surrounding validation is
    untouched.
    """
    registry_module = importlib.import_module("lavis.common.registry")
    registry_class = type(registry_module.registry)
    original = registry_class.register_path
    if getattr(original, "_workbench_idempotent", False):
        return

    def register_path(cls, name, path):
        existing = cls.mapping["paths"].get(name)
        if existing is not None and str(existing) == str(path):
            return
        # Everything else, including a genuine conflict, is the pinned behaviour.
        return original(name, path)

    register_path._workbench_idempotent = True
    register_path._workbench_original = original
    registry_class.register_path = classmethod(register_path)


def _load_preprocess_function() -> tuple[Any, str]:
    """The pinned `load_preprocess` plus a label naming which copy was reachable."""
    try:
        return importlib.import_module("lavis.models").load_preprocess, "lavis.models"
    except ImportError:
        pass
    source_file = Path(importlib.import_module("lavis.common.registry").__file__).parents[1] / "models" / "__init__.py"
    tree = ast.parse(source_file.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "load_preprocess":
            namespace: dict[str, Any] = {
                "registry": importlib.import_module("lavis.common.registry").registry,
                "BaseProcessor": sys.modules["lavis.processors"].BaseProcessor,
            }
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_file), "exec"), namespace)
            return namespace["load_preprocess"], "lavis/models/__init__.py:load_preprocess"
    raise ImportError(f"the pinned load_preprocess is not defined in {source_file}")


def reset() -> None:
    """Tear the package context down so another checkout can be prepared in this process.

    A production wrapper runs one model in one subprocess and never calls this
    (`evaluate_models.py:594`). It exists for tools that deliberately compare several checkouts --
    `workbench/scripts/lavis_processor_parity.py` is the one caller -- because the pinned package
    names are global to a process and two checkouts cannot share them.

    A reset has to undo everything `prepare` did, or the next checkout inherits state from the
    previous one:

    * the modules imported from that checkout, and the registry they registered into;
    * the `sys.path` entries `prepare` added, because leaving them means the previous checkout's
      `lavis` package is still the first thing an import finds -- the second checkout's modules
      would be shadowed by the first's, which is precisely the contamination this call exists to
      prevent. Only entries `prepare` itself added are removed, so a caller's own path setup is
      left alone.
    """
    for name in [key for key in sys.modules if key == "lavis" or key.startswith("lavis.")]:
        del sys.modules[name]
    for entry in _ADDED_PATHS:
        while entry in sys.path:
            sys.path.remove(entry)
    _ADDED_PATHS.clear()
    _INSTALLED.clear()
    _OWNER.clear()


def prepare(source_root: Path) -> dict[str, Any]:
    """Install the pinned package context for this process, once.

    Idempotent, and scoped to the checkout: calling it twice for the same source is a no-op, and
    calling it for a second checkout is refused rather than silently re-pointed.
    """
    lavis_root = (source_root / "lavis").resolve()
    owner = _OWNER.get("lavis")
    if owner is not None and owner != lavis_root:
        raise ImportError(
            f"this process already prepared the lavis context for {owner}; preparing it for "
            f"{lavis_root} in the same process would mix two checkouts' classes")
    if _INSTALLED.get("lavis") == str(lavis_root):
        return {"library_root": str(lavis_root), "installed": sorted(_INSTALLED), "reused": True}

    for entry in (str(source_root), str(lavis_root)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
            _ADDED_PATHS.append(entry)

    _registry_for(lavis_root)
    _patch_registry_idempotence()
    _processors_module(lavis_root)
    _INSTALLED["lavis"] = str(lavis_root)
    _INSTALLED["lavis.processors"] = str(lavis_root)
    _OWNER["lavis"] = lavis_root
    return {"library_root": str(lavis_root), "installed": sorted(_INSTALLED), "reused": False}


def text_processors(source_root: Path, config_relative_path: str) -> tuple[dict[str, Any], str]:
    """`(txt_processors, boundary)` for the pinned config at *config_relative_path*.

    `config_relative_path` is the value the model class publishes in its
    `PRETRAINED_MODEL_CONFIG_DICT`, e.g. `configs/models/blip2/blip2_pretrain.yaml`. The package
    context is installed by `prepare` and left installed, so the checkpoint that is loaded next can
    resolve its pinned class paths.
    """
    from omegaconf import OmegaConf

    prepare(source_root)
    lavis_root = Path(_INSTALLED["lavis"])
    config_path = lavis_root / config_relative_path
    if not config_path.is_file():
        raise FileNotFoundError(f"pinned LAVIS config missing: {config_path}")

    load_preprocess, boundary = _load_preprocess_function()
    cfg = OmegaConf.load(config_path)
    _, txt_processors = load_preprocess(cfg.preprocess)
    return txt_processors, boundary
