"""CIRCUS composition adapter for DCNet: registered, and refused before any launch.

WHY THIS ADAPTER CANNOT RUN (measured on the pinned tree, commit 68c79d3)

* **No registered interpreter.** ``WORKBENCH_DCNET_PYTHON`` is unset and no
  ``environment_assignment.json`` entry resolves for ``dcnet``, so there is no interpreter this
  model is declared to run in. The survey below is what was actually probed on this host, so the
  common shorthand "dcnet's only interpreter has no torch" is **not** repeated here: it is wrong.
* **The checkpoint classes DO import — in three of the four candidate interpreters.** Probed
  directly (``sys.path`` = the pinned checkout, then ``import model.model``):

  - ``workbench/envs/dcnet-prepare/bin/python`` (3.7.12): **no torch** — cannot import.
  - ``workbench/artifacts/environments/cir-005ef4e1b02999a5/bin/python`` (3.10.22, torch
    2.7.1+cu128): imports ``CENet`` and loads the staged checkpoint **strictly** (518/518 tensors,
    0 missing, 0 unexpected, ``weights_only=False``).
  - ``workbench/artifacts/environments/dcnet-377/bin/python`` (3.7.7, torch 1.4.0+cpu): imports
    ``CENet``, but the registry's approved source patch cannot run there: ``torch.load`` on this
    torch rejects the keyword (``TypeError: 'weights_only' is an invalid keyword argument for
    load()``; ``inspect.signature(torch.load)`` has no such parameter). On the pinned 2.7.1 torch
    that same call is fine, so the "``weights_only`` TypeError" is real but is a torch-version
    fact, not a defect of the checkpoint.
  - ``workbench/envs/ctl312h`` (the controller) imports it as well.

  None of those is the model's registered interpreter, and which one a run would use is the
  operator's declaration, not this adapter's guess — so this adapter names the missing declaration
  instead of picking one.
* **Scoring is not a fixed-gallery dot product, so it cannot be reproduced through the adapter
  encode/score contract at all.** See ``ScoringNotReproducible`` below.
* **Protocol conflict.** DCNet's native evaluation masks the reference
  (``trainer/trainer_joint.py:217-218``, registry ``native_protocol:
  fashioniq_full_gallery_ref_excluded``) while CIRCUS requires a reference-eligible gallery. The
  two protocols cannot both be satisfied, so this adapter declares ``reference_eligible = False``.

Scoring semantics, read off the pinned source rather than invented
(``trainer/trainer_joint.py``, used by ``test.py``):

* ``:188-195`` composition branch: ``sharded_cross_view_inner_product`` with ``l2renorm=False``,
  ``dist=False`` — i.e. a plain un-normalized inner product summed over the seven expert
  subspaces (``model/metric.py:24-67``). No max over tokens anywhere: the experts pool to
  ``N x D`` (``model/model.py:115-123``).
* ``:207-214`` correction branch, and this is the blocker: its target side is
  ``mod_text = get_correction(src_experts_new, val_experts)`` — a **query-conditioned**
  transformation of every gallery row (``model/model.py:130-157``, ``FusDiff``). It is not a
  function of the gallery alone, so it cannot be precomputed as a target matrix.
* ``:217-218`` the reference row of BOTH matrices is set to ``-10e10``;
* ``:239-240`` the metric actually reported is the fused ranking
  ``log_softmax(score_comp) + log_softmax(score_corr)`` over the gallery axis. The reference is
  therefore excluded twice: once by the mask and once by the log-softmax mass.

Because the reported ranking depends on a per-(query, gallery) correction, no ``score()`` over
two fixed matrices can be the DCNet number. ``score()`` raises ``ScoringNotReproducible`` naming
that, rather than approximating it with a cosine and calling the result DCNet.
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

#: ``CENet``'s shared embedding width — ``experts.ce_shared_dim`` in the staged config and the
#: ``same_dim`` every expert fuses into. Declared so the base class's reference probe can ATTEMPT
#: this scorer; the attempt refuses, and is reported as unmeasured rather than as a verdict.
DIMENSION = 1024

#: The registry's own declared runtime for this model, and what the approved patch needs.
DECLARED_TORCH = "1.4.0"

class Refused(RuntimeError):
    """This adapter refuses to launch: the pinned upstream contract cannot be executed here."""

class ScoringNotReproducible(RuntimeError):
    """DCNet's reported ranking cannot be expressed as scores over two fixed feature matrices."""

#: Probe result, recorded so ``status()`` and the docstring cannot disagree with it.
INTERPRETER_SURVEY = (
    "envs/dcnet-prepare (3.7.12): no torch; "
    "artifacts/environments/cir-005ef4e1b02999a5 (3.10.22, torch 2.7.1+cu128): imports CENet and "
    "loads the staged checkpoint strictly (518/518 tensors, weights_only=False); "
    "artifacts/environments/dcnet-377 (3.7.7, torch 1.4.0+cpu): imports CENet but raises "
    "TypeError on the approved weights_only opt-out"
)

class Adapter(CompositionAdapter):
    """DCNet adapter: every launch path refuses, and ``score()`` names why it cannot exist."""

    model_id = "dcnet"
    #: ``trainer_joint.py:217-218`` masks the reference; CIRCUS requires it eligible.
    reference_eligible = False
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
            raise ValueError("DCNet gallery is category-agnostic; target categories must all be None")
        return _run(self, "targets", image_paths, (), categories)

    def score(self, queries: Any, targets: Any,
              categories: Sequence[str | None] | None = None) -> np.ndarray:
        """Refuse: the reported DCNet ranking is not a score over two fixed matrices.

        Upstream ranks by ``log_softmax(score_comp) + log_softmax(score_corr)``
        (``trainer_joint.py:239-240``) where ``score_corr`` re-derives its target side from each
        query (``:207-214``). Any matrix this signature could return would be a different method,
        so it raises instead of guessing.
        """
        del queries, targets, categories
        raise ScoringNotReproducible(
            "DCNet's reported ranking fuses a composition inner product with a correction score "
            "that is re-derived per query against every gallery row "
            "(third_party/DCNet/trainer/trainer_joint.py:207-218,239-240), so it cannot be "
            "expressed as scores over the target matrix a composition adapter encodes once")

def _resolve(config: Any, checkpoint_id: str | None) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    from workbench.backend.model_settings import setting
    from workbench.backend.registry import checkpoint_by_id, checkpoint_path, model_by_id
    from workbench.backend.runtime import resolve_model_interpreter

    model = model_by_id("dcnet")
    checkpoint = checkpoint_by_id(
        model, checkpoint_id or model["checkpoint_variants"][0]["checkpoint_id"])
    checkpoint = dict(checkpoint, resolved_path=str(
        checkpoint_path("dcnet", checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)))
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
    # The artifact is the run directory, so presence means config.json plus trained_model.pth.
    if not checkpoint_is_present(path, checkpoint, model_id=model["model_id"], config=config):
        reasons.append(f"checkpoint run directory incomplete at {path}")
    if not interpreter:
        reasons.append("no interpreter environment resolved for "
                       f"{model['environment']['python_env_var']} (unset, and no provisioning "
                       "assignment records one for dcnet)")
    elif not Path(interpreter).is_file():
        reasons.append(f"model interpreter missing at {interpreter}")
    reasons.append(f"registry-declared runtime torch {DECLARED_TORCH} cannot run the approved "
                   "base_trainer.py weights_only opt-out (TypeError on that torch); surveyed: "
                   f"{INTERPRETER_SURVEY}")
    reasons.append("reported ranking is log_softmax(comp)+log_softmax(corr) over a per-query "
                   "correction branch (trainer_joint.py:207-218,239-240), so score() cannot be "
                   "reproduced from two fixed matrices and refuses")
    reasons.append("upstream masks the reference (trainer_joint.py:217-218) while CIRCUS requires a "
                   "reference-eligible gallery; the protocols conflict")
    return reasons

def _refusal(adapter: Adapter) -> str:
    return "; ".join(_blockers(adapter._model, adapter._checkpoint, adapter.checkpoint_path,
                               adapter._interpreter, adapter.config))

def _run(adapter: Adapter, operation: str, image_paths: Sequence[str], texts: Sequence[str],
         categories: Sequence[str | None]) -> np.ndarray:
    """Refuse before any launch; no subprocess is ever started for this model."""
    del operation, texts, categories
    if adapter.work_root is None:
        raise AdapterWorkRootError("dcnet work_root was not set by runner")
    if not image_paths:
        return np.empty((0, DIMENSION), dtype=np.float32)
    if not adapter._interpreter:
        raise Refused("no interpreter environment resolved for WORKBENCH_DCNET_PYTHON")
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
                         "Reference-INeligible by upstream protocol (reference masked before the "
                         "fused log-softmax rank). No subprocess is launched: scoring is not "
                         "expressible over fixed feature matrices and no interpreter is registered.")

ADAPTER = Adapter
