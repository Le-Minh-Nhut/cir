"""CIRCUS composition adapter for CLVC-Net: registered, and refused before any launch.

WHY THIS ADAPTER CANNOT RUN (measured on the pinned tree, commit bd9b688)

* **CUDA-hard, unconditionally.** The pinned evaluator calls ``.cuda()`` with no device guard and
  no CPU path. COUNTED FROM SOURCE, not copied: ``cuda_census()`` below reports 14 source lines
  holding 15 ``.cuda()`` call expressions across ``model.py`` (2), ``test.py`` (4),
  ``text_model.py`` (2 lines, 3 expressions — ``:108`` has two) and ``train.py`` (6). Of those,
  only ``test.py``+``text_model.py`` are on the scoring path: **6 lines / 7 expressions**.
  ``model.py``'s two are in ``compute_batch_based_classification_loss_`` (training loss only) and
  ``train.py``'s six are the training loop.
* **No interpreter.** ``WORKBENCH_CLVC_PYTHON`` is unresolved on this host, and the declared stack
  (Python 3.7.6 / torch 1.6.0 / cu102) is not provisioned.
* Latent Python-2 ism, recorded but not a blocker: ``text_model.py:80`` reads ``unicode``, which
  would be a ``NameError`` on Python 3 — the ``type(x[0]) is str`` test short-circuits first for
  every string input upstream actually passes, so the forward pass survives it.

Native scoring, reproduced rather than invented (``third_party/CLVCNet/test.py``):

* a query row is ``cat(normalize(local_feature_1024), normalize(global_feature_1024))``
  (``:30`` and ``:100``) and a target row is the same concatenation from the two image encoders
  (``:48`` and ``:116``), so a row is 2048 wide with two unit-norm halves;
* ``:122-125`` (and ``:55-58``) divide each whole row by its L2 norm, ``:129`` is
  ``sims = all_queries.dot(all_imgs.T)``. **No max over tokens** and no other normalization in the
  score. Because both halves are already unit-norm, dividing the 2048-d row by its norm and taking
  the dot product is exactly ``(dot_local + dot_global) / 2`` — i.e. the ordinary cosine over the
  concatenated row, which is what ``score()`` computes;
* ``:134-135`` masks the reference row (``sims[i, ...] = -10e10``). That mask is **deliberately not
  inherited**: CIRCUS ranks over a reference-eligible full gallery, the same choice
  ``model_pair.py`` and ``model_encoder.py`` make for the identical upstream pattern.

Category is a real axis here: ``train.py:156-157`` builds one local/global pair per FashionIQ
category and six files are staged, so ``category_scopes`` is ``("dress", "shirt", "toptee")`` and a
query is scored only against its own category's gallery matrix (registry:
``category_model_policy: independent``).
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

#: ``model.py:127,199`` give each member a 1024-d feature; the evaluator concatenates the two.
DIMENSION = 2048

#: The four files the specification counts, and the two on the scoring path.
CUDA_FILES = ("model.py", "test.py", "text_model.py", "train.py")
SCORING_FILES = ("test.py", "text_model.py")

class Refused(RuntimeError):
    """This adapter refuses to launch: the pinned upstream contract cannot be executed here."""

def cuda_census(source_root: Path) -> dict[str, dict[str, int]]:
    """Per file: lines holding ``.cuda()``, and ``.cuda()`` call expressions on those lines.

    The two numbers differ where one line calls it twice (``text_model.py:108``), and that is why a
    single "N unguarded calls" figure can be read two ways. Measured here rather than asserted, so
    the number in ``status()`` cannot drift from the pinned bytes.
    """
    census: dict[str, dict[str, int]] = {}
    for name in CUDA_FILES:
        try:
            source = (Path(source_root) / name).read_text(encoding="utf-8")
        except OSError:
            continue
        lines = [line for line in source.splitlines() if ".cuda()" in line]
        census[name] = {"lines": len(lines), "calls": sum(line.count(".cuda()") for line in lines)}
    return census

def _cuda_summary(census: dict[str, dict[str, int]]) -> str:
    lines = sum(item["lines"] for item in census.values())
    calls = sum(item["calls"] for item in census.values())
    scoring = {name: census[name] for name in SCORING_FILES if name in census}
    scoring_lines = sum(item["lines"] for item in scoring.values())
    scoring_calls = sum(item["calls"] for item in scoring.values())
    detail = ", ".join(f"{name}({item['calls']})" for name, item in census.items())
    return (f"unguarded CUDA: {lines} source lines / {calls} .cuda() call expressions [{detail}]; "
            f"{scoring_lines} lines / {scoring_calls} expressions reachable from scoring")

class Adapter(CompositionAdapter):
    """CLVC-Net's cosine scorer and encoder over per-category local/global pairs."""

    model_id = "clvc_net"
    reference_eligible = True
    score_dimension = DIMENSION
    category_scopes: tuple[str | None, ...] = ("dress", "shirt", "toptee")

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
        scope = next((category for category in categories if category is not None), None)
        if any(category != scope for category in categories):
            raise ValueError("CLVC-Net encodes one category per gallery; scopes must not be mixed")
        if scope not in self.category_scopes:
            raise ValueError(f"CLVC-Net gallery scope {scope!r} is not one of "
                             f"{self.category_scopes}")
        return _run(self, "targets", image_paths, (), categories)

    def score(self, queries: Any, targets: Any,
              categories: Sequence[str | None] | None = None) -> np.ndarray:
        """Cosine over 2048-d rows: ``test.py:122-129`` — no token max, no other normalization.

        Valid only for rows the encoders produced, i.e. two unit-norm 1024-d halves each: there the
        whole-row L2 of ``:125`` is exactly ``sqrt(2)``, so the upstream expression equals the plain
        cosine this inherits. The upstream reference mask (``:134-135``) is deliberately absent.
        """
        del categories
        return super().score(queries, targets)

def _resolve(config: Any, checkpoint_id: str | None) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    from workbench.backend.model_settings import setting
    from workbench.backend.registry import checkpoint_by_id, checkpoint_path, model_by_id
    from workbench.backend.runtime import resolve_model_interpreter

    model = model_by_id("clvc_net")
    checkpoint = checkpoint_by_id(
        model, checkpoint_id or model["checkpoint_variants"][0]["checkpoint_id"])
    checkpoint = dict(checkpoint, resolved_path=str(
        checkpoint_path("clvc_net", checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)))
    variable = model["environment"]["python_env_var"]
    return model, checkpoint, resolve_model_interpreter(model, config) or setting(variable)

def _blockers(model: dict[str, Any], checkpoint: dict[str, Any], path: Path,
              interpreter: str | None, config: Any) -> list[str]:
    """Every reason this model may not be scored, in one place for status() and _run()."""
    from workbench.backend.registry import checkpoint_is_present

    reasons: list[str] = []
    mapping = checkpoint.get("checkpoint_mapping_status")
    if mapping != "VERIFIED_METADATA":
        reasons.append(f"checkpoint mapping is {mapping} for {checkpoint['checkpoint_id']}")
    # The artifact is the six-file bundle directory, so presence means all six members.
    if not checkpoint_is_present(path, checkpoint, model_id=model["model_id"], config=config):
        reasons.append(f"checkpoint bundle incomplete at {path}")
    if not interpreter:
        reasons.append("no interpreter environment resolved for "
                       f"{model['environment']['python_env_var']}")
    elif not Path(interpreter).is_file():
        reasons.append(f"model interpreter missing at {interpreter}")
    census = cuda_census(Path(config.WORKBENCH_THIRD_PARTY_ROOT) / model["source_dir"])
    if census:
        reasons.append(_cuda_summary(census))
    else:
        reasons.append("pinned CLVC-Net source not found; its CUDA usage cannot be counted")
    return reasons

def _refusal(adapter: Adapter) -> str:
    return "; ".join(_blockers(adapter._model, adapter._checkpoint, adapter.checkpoint_path,
                               adapter._interpreter, adapter.config))

def _run(adapter: Adapter, operation: str, image_paths: Sequence[str], texts: Sequence[str],
         categories: Sequence[str | None]) -> np.ndarray:
    """Refuse before any launch; no subprocess is ever started for this model."""
    del operation, texts, categories
    if adapter.work_root is None:
        raise AdapterWorkRootError("clvc_net work_root was not set by runner")
    if not image_paths:
        return np.empty((0, DIMENSION), dtype=np.float32)
    if not adapter._interpreter:
        raise Refused("no interpreter environment resolved for WORKBENCH_CLVC_PYTHON")
    if not adapter._interpreter_verified:
        # The shared gate runs before this adapter could launch anything: it raises on an
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
    reason = "; ".join(_blockers(model, checkpoint, path, interpreter, config))
    return AdapterStatus(cls.model_id, False, reason,
                         tuple(str(item["checkpoint_id"]) for item in model["checkpoint_variants"]),
                         {condition: (False, reason) for condition in CONDITIONS},
                         "Cosine over concatenated, per-half-normalized 2048-d rows; upstream "
                         "reference mask NOT inherited. Per-category local/global pairs. No "
                         "subprocess is launched: the pinned evaluator is CUDA-hard and no "
                         "interpreter is provisioned.")

ADAPTER = Adapter
