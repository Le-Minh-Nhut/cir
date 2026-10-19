"""The pinned LAVIS text processor, without constructing a second BLIP-2 model.

The LAVIS-family wrappers (`hint`, `habit`, `intent`, `conesep`, `airknow`) need exactly one thing
from `load_model_and_preprocess`: the processor its config selects. The stock call builds a SECOND
pretrained model to obtain it -- `lavis/models/__init__.py:210` runs `model_cls.from_pretrained(...)`
before `:216-220` loads the preprocess config -- and that constructor reaches for external weights:
`Blip2Base.init_tokenizer`/`init_Qformer` read `/root/bert-base-uncased`
(`lavis/models/blip2_models/blip2.py:32,48,55`), so it needs machine-local BERT assets and multi-GB
base weights purely to hand back a text processor.

What is used instead is the pinned machinery itself:

* the preprocess block comes from the config the model class publishes for that `model_type`
  (`PRETRAINED_MODEL_CONFIG_DICT`), resolved against the checkout exactly as
  `BaseModel.default_config_path` resolves it (`get_abs_path`, `lavis/common/utils.py:52`);
* the processor class is the pinned `BlipCaptionProcessor`, whose own `from_config` builds it, so
  prompt, max_words, punctuation stripping, whitespace collapsing and truncation are native
  behaviour rather than a reimplementation;
* the config is installed in the pinned `registry` (`register_path("library_root", ...)`), so
  `get_abs_path` finds it the way it would during a normal import;
* `load_preprocess` is executed from its own definition in the pinned
  `lavis/models/__init__.py`. On a host whose environment matches the model's declared one,
  `lavis.models` imports normally and the definition is taken from the live module; where the
  package's model registry cannot be imported (the review host has an unrelated `transformers`
  version), the same function object is compiled from its own source text with the pinned `registry`
  in scope. Either way the code executed is the authors', and which copy was used is recorded in the
  provenance the caller writes.

The one substitution is the `lavis.processors` package object: its `__init__` imports
`BeatsAudioProcessor`, whose module imports `torchaudio` at module scope, and torchaudio is in none of
these models' declared environments. `base_processor` and `blip_processors` are the real modules.

Verification boundary: everything executed here is pinned source.
`workbench/scripts/lavis_processor_parity.py` is the runnable check that this path yields the same
callable the native construction would; it prints the processor class, the boundary it was loaded
through, and the output on a fixed caption set, for every model in the family.
"""
from __future__ import annotations

import ast
import importlib
import sys
import types
from pathlib import Path
from typing import Any


def _processors_stub(lavis_root: Path) -> types.ModuleType:
    """`lavis.processors` with its torchaudio-dependent sibling left out.

    `base_processor` is loaded for real: `registry.register_processor`'s decorator asserts every
    registered class subclasses *that* class object, so a second copy of the file would fail the
    assertion. The skeleton is inserted into `sys.modules` BEFORE any submodule is imported, or the
    import would execute the pinned `__init__` that this function exists to skip.
    """
    module = types.ModuleType("lavis.processors")
    module.__path__ = [str(lavis_root / "processors")]
    sys.modules["lavis.processors"] = module
    module.BaseProcessor = importlib.import_module("lavis.processors.base_processor").BaseProcessor
    module.BlipCaptionProcessor = importlib.import_module(
        "lavis.processors.blip_processors").BlipCaptionProcessor
    return module


def _pinned_registry(lavis_root: Path) -> Any:
    """The pinned `registry`, reachable without executing `lavis/__init__.py`.

    `lavis/__init__.py:15` does `from lavis.datasets.builders import *`, which imports
    `audio_classification_datasets` and therefore `torchaudio` -- a package none of these models
    declares and none of them uses for FashionIQ. Installing the package root as a namespace stub
    before importing `lavis.common.registry` is what keeps that unrelated import out of the path; the
    module actually imported, and the path it registers, are the pinned ones.
    """
    if "lavis" not in sys.modules:
        package = types.ModuleType("lavis")
        package.__path__ = [str(lavis_root)]
        sys.modules["lavis"] = package
    registry = importlib.import_module("lavis.common.registry").registry
    if registry.get_path("library_root") is None:
        registry.register_path("library_root", str(lavis_root))
    return registry


def _load_preprocess_function() -> tuple[Any, str]:
    """The pinned `load_preprocess` plus a label naming which copy was reachable."""
    try:
        return importlib.import_module("lavis.models").load_preprocess, "lavis.models"
    except ImportError:
        pass
    source_path = Path(importlib.import_module("lavis.common.registry").__file__).parents[1] / "models"
    source_file = source_path / "__init__.py"
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


def text_processors(source_root: Path, config_relative_path: str) -> tuple[dict[str, Any], str]:
    """`(txt_processors, boundary)` for the pinned config at *config_relative_path*.

    `config_relative_path` is the value the model class publishes in its
    `PRETRAINED_MODEL_CONFIG_DICT`, e.g. `configs/models/blip2/blip2_pretrain.yaml`.
    """
    from omegaconf import OmegaConf

    lavis_root = source_root / "lavis"
    config_path = lavis_root / config_relative_path
    if not config_path.is_file():
        raise FileNotFoundError(f"pinned LAVIS config missing: {config_path}")

    previous = sys.modules.get("lavis.processors")
    saved_root = sys.modules.get("lavis")
    # The pinned package root is installed first: `lavis/__init__.py` would otherwise run
    # `from lavis.datasets.builders import *`, which reaches `torchaudio` through an unrelated
    # audio dataset, before any of this module's own imports happen.
    if saved_root is None:
        package = types.ModuleType("lavis")
        package.__path__ = [str(lavis_root)]
        sys.modules["lavis"] = package
    _pinned_registry(lavis_root)
    sys.modules["lavis.processors"] = _processors_stub(lavis_root)
    try:
        load_preprocess, boundary = _load_preprocess_function()

        cfg = OmegaConf.load(config_path)
        _, txt_processors = load_preprocess(cfg.preprocess)
    finally:
        if previous is None:
            sys.modules.pop("lavis.processors", None)
        else:
            sys.modules["lavis.processors"] = previous
        if saved_root is None:
            sys.modules.pop("lavis", None)
    return txt_processors, boundary
