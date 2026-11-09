"""CIRCUS composition adapter for TG-CIR: registered, and refused before any launch.

WHY THIS ADAPTER CANNOT RUN (measured on the pinned tree, commit 65fa78e)

* **No entrypoint.** ``third_party/TGCIR/test.py`` defines ``test(params, model, testset,
  category)`` (:6) and nothing that could be launched: the file has no ``__main__`` block, no
  ``argparse`` and no checkpoint loader. ``workbench/replay/tgcir.py`` is deliberately
  preflight-only (it never imports upstream code), so nothing in this repository turns a FashionIQ
  row into a TG-CIR score. The registry records the same fact as
  ``command_status: COMMAND_BLOCKED_SOURCE_ENTRYPOINT``.
* **CUDA-hard, unconditionally.** ``.cuda()`` is called with no device guard at ``model.py:32``,
  ``:107``, ``:120``, ``:136``, ``:244``, ``:251`` and ``test.py:31``, ``:47``, ``:101``, ``:115``,
  and the CLIP backbone is loaded with ``device='cuda'`` (``model.py:41``). There is no CPU path to
  fall back to, so a CPU-only host cannot produce a feature at all.
* **No interpreter.** ``WORKBENCH_TGCIR_PYTHON`` is unresolved, and the declared stack (torch 1.7.0
  / cu110 plus a pinned OpenAI CLIP revision) is not provisioned on this host.

Native scoring, reproduced rather than invented (``third_party/TGCIR/test.py``):

* ``:53-57`` — every query row and every target row is divided by its L2 norm;
* ``:61`` — ``sims = all_queries.dot(all_imgs.T)``: one dot product, **no max over tokens**
  (``model.py:201-210`` already mean-pools ``(B, tokens, 512)`` to ``(B, 512)`` and normalizes);
* ``:63-67`` — upstream then masks the reference row (``sims[i, ...] = -10e10``). That mask is
  **deliberately not inherited**, for the same reason ``model_pair.py`` and ``model_encoder.py`` do
  not inherit it: CIRCUS ranks over a reference-eligible full gallery. ``reference_eligible`` is
  therefore left to the base class's measured score+rank probe rather than asserted here.

Feature width is 512: ``model.py:48-49`` builds the ViT-B/16 projections 768->512 and 512->512,
and ``model.py:201-204`` reduces the token axis with ``torch.mean``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from workbench.composition.adapters import (
    AdapterStatus,
    AdapterWorkRootError,
    CompositionAdapter,
)
from workbench.composition.inputs import CONDITIONS

#: ``model.py:201-210`` mean-pools ``(B, tokens, 512)`` to ``(B, 512)`` and L2-normalizes.
DIMENSION = 512

#: Verified against the pinned sources: every one of these runs before any metric exists.
CUDA_HARD = ("model.py:32,107,120,136,244,251 and test.py:31,47,101,115 call .cuda() with no "
             "device guard, and model.py:41 loads CLIP with device='cuda'")

class Refused(RuntimeError):
    """This adapter refuses to launch: the pinned upstream contract cannot be executed here."""

class Adapter(CompositionAdapter):
    """TG-CIR's cosine scorer and encoder, refused before any launch while the blockers hold."""

    model_id = "tgcir"
    reference_eligible = True
    score_dimension = DIMENSION
    category_scopes: tuple[str | None, ...] = (None,)

    def __init__(self, config: Any = None, checkpoint_id: str | None = None) -> None:
        from workbench.backend.operator_config import resolve_config

        self.config = config if config is not None else resolve_config()
        self._model, self._checkpoint, self._interpreter = _resolve(self.config, checkpoint_id)
        super().__init__(self.config, self._checkpoint["checkpoint_id"])
        self.checkpoint = self._checkpoint
        self.checkpoint_path = Path(self._checkpoint["resolved_path"])
        self.source_root = Path(self.config.WORKBENCH_THIRD_PARTY_ROOT) / self._model["source_dir"]
        self.work_root: Path | None = None
        self._interpreter_verified = False

    @classmethod
    def status(cls, config: Any = None) -> AdapterStatus:
        from workbench.backend.operator_config import resolve_config

        return _status(cls, config if config is not None else resolve_config())

    def mode_supported(self, condition: str) -> tuple[bool, str | None]:
        if condition not in CONDITIONS:
            return False, f"unknown composition condition: {condition!r}"
        reason = _refusal(self)
        return (reason is None), reason

    def encode_queries(self, image_paths: Sequence[str], texts: Sequence[str],
                       categories: Sequence[str | None]) -> np.ndarray:
        if not (len(image_paths) == len(texts) == len(categories)):
            raise ValueError("image_paths, texts, and categories must have equal lengths")
        return _run(self, "queries", image_paths, texts, categories)

    def encode_targets(self, image_paths: Sequence[str],
                       categories: Sequence[str | None]) -> np.ndarray:
        if len(image_paths) != len(categories):
            raise ValueError("image_paths and categories must have equal lengths")
        if any(category is not None for category in categories):
            raise ValueError("TG-CIR gallery is category-agnostic; target categories must all be None")
        return _run(self, "targets", image_paths, (), categories)

    def score(self, queries: Any, targets: Any,
              categories: Sequence[str | None] | None = None) -> np.ndarray:
        """Cosine matrix: ``test.py:53-57`` normalize both sides, ``:61`` one dot product.

        No max over tokens (the encoder already mean-pooled, ``model.py:201-210``); the upstream
        reference mask (``test.py:63-67``) is deliberately absent — see the module docstring. The
        shared implementation is exactly this expression, so it is reused, not restated; the
        ``categories`` argument is forwarded because the base class needs it for scoped galleries.
        """
        return super().score(queries, targets, categories)

def _resolve(config: Any, checkpoint_id: str | None) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    from workbench.backend.model_settings import setting
    from workbench.backend.registry import checkpoint_by_id, checkpoint_path, model_by_id
    from workbench.backend.runtime import resolve_model_interpreter

    model = model_by_id("tgcir")
    checkpoint = checkpoint_by_id(
        model, checkpoint_id or model["checkpoint_variants"][0]["checkpoint_id"])
    checkpoint = dict(checkpoint, resolved_path=str(
        checkpoint_path("tgcir", checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)))
    variable = model["environment"]["python_env_var"]
    return model, checkpoint, resolve_model_interpreter(model, config) or setting(variable)

def _entrypoint_refusal(source_root: Path) -> str:
    """The pinned evaluator's missing CLI, read off the pinned file rather than asserted in prose."""
    test_py = Path(source_root) / "test.py"
    try:
        source = test_py.read_text(encoding="utf-8")
    except OSError as error:
        return f"pinned TG-CIR test.py unreadable at {test_py}: {error}"
    if "__main__" in source or "argparse" in source:
        return (f"{test_py} now exposes a command line, but this adapter has no launcher for it, "
                "so nothing here can drive TG-CIR")
    return (f"pinned {test_py} has no CLI (no __main__ block, no argparse), so its only scoring "
            "surface is test(params, model, testset, category) and no checkpoint loader exists")

def _blockers(model: dict[str, Any], checkpoint: dict[str, Any], path: Path,
              interpreter: str | None, config: Any) -> list[str]:
    """Every reason this model may not be scored, in one place for status() and _run()."""
    from workbench.backend.registry import checkpoint_is_present

    reasons: list[str] = []
    mapping = checkpoint.get("checkpoint_mapping_status")
    if mapping != "VERIFIED_METADATA":
        reasons.append(f"checkpoint mapping is {mapping} for {checkpoint['checkpoint_id']}")
    if not checkpoint_is_present(path, checkpoint, model_id=model["model_id"], config=config):
        reasons.append(f"checkpoint not staged at {path}")
    if not interpreter:
        reasons.append("no interpreter environment resolved for "
                       f"{model['environment']['python_env_var']}")
    elif not Path(interpreter).is_file():
        reasons.append(f"model interpreter missing at {interpreter}")
    reasons.append(_entrypoint_refusal(Path(config.WORKBENCH_THIRD_PARTY_ROOT) / model["source_dir"]))
    reasons.append(f"TG-CIR evaluation is CUDA-hard: {CUDA_HARD}")
    return reasons

def _refusal(adapter: Adapter) -> str:
    return "; ".join(_blockers(adapter._model, adapter._checkpoint, adapter.checkpoint_path,
                               adapter._interpreter, adapter.config))

def _run(adapter: Adapter, operation: str, image_paths: Sequence[str], texts: Sequence[str],
         categories: Sequence[str | None]) -> np.ndarray:
    """Refuse before any launch; no subprocess is ever started for this model."""
    del operation, texts, categories
    if adapter.work_root is None:
        raise AdapterWorkRootError("tgcir work_root was not set by runner")
    if not image_paths:
        return np.empty((0, DIMENSION), dtype=np.float32)
    if not adapter._interpreter:
        raise Refused("no interpreter environment resolved for WORKBENCH_TGCIR_PYTHON")
    if not adapter._interpreter_verified:
        # The shared gate is called before this adapter may launch anything: it raises on an
        # unverified environment, which is a stricter answer than this adapter's own refusal.
        adapter.require_verified_interpreter()
        adapter._interpreter_verified = True
    raise Refused(_refusal(adapter))

def _status(cls: type[Adapter], config: Any) -> AdapterStatus:
    from workbench.backend.model_settings import setting
    from workbench.backend.registry import checkpoint_by_id, checkpoint_path, load_registry, model_by_id
    from workbench.backend.runtime import resolve_model_interpreter

    model = model_by_id(cls.model_id, getattr(config, "registry", None) or load_registry())
    checkpoint = checkpoint_by_id(
        model, getattr(config, "checkpoint_id", None) or model["checkpoint_variants"][0]["checkpoint_id"])
    path = checkpoint_path(cls.model_id, checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)
    variable = model["environment"]["python_env_var"]
    interpreter = resolve_model_interpreter(model, config) or setting(variable)
    reasons = _blockers(model, checkpoint, path, interpreter, config)
    reason = "; ".join(reasons)
    return AdapterStatus(cls.model_id, False, reason,
                         tuple(str(item["checkpoint_id"]) for item in model["checkpoint_variants"]),
                         {condition: (False, reason) for condition in CONDITIONS},
                         "Cosine over mean-pooled 512-d vectors; upstream reference mask NOT "
                         "inherited (CIRCUS needs a reference-eligible gallery). No subprocess is "
                         "launched: the pinned evaluator has no CLI and is CUDA-hard.")

ADAPTER = Adapter
