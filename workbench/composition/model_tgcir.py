"""CIRCUS composition adapter for TG-CIR, plus the model-interpreter worker that runs it.

WHAT IS TRUE (measured on the pinned tree, commit 65fa78e; 2026-11-11)

* **The author ships no evaluable command line.** ``third_party/TGCIR/test.py`` defines
  ``test(params, model, testset, category)`` (:6) and nothing that could be launched: no ``__main__``
  block, no ``argparse``, no checkpoint loader. ``train.py:74`` declares ``--restore_file`` and never
  reads it (one occurrence in the whole file), so the registry's
  ``command_status: COMMAND_BLOCKED_SOURCE_ENTRYPOINT`` is exact. The evaluation path THIS adapter
  drives is **workbench-added**: the pinned ``test.test`` scoring expression, reproduced directly
  rather than shelled out to.
* **The pinned artifact is a whole-module pickle.** ``Fashioniq.pt`` (607,928,637 bytes, sha256
  ``c37768f3…`` from the registry) is a ``torch.save(model)`` of ``model.VCG_CIR``, not a
  ``{'state_dict': …}`` dict; it is unpickled with ``weights_only=False`` after its digest is
  verified, never before. Upstream's ``utils.load_checkpoint`` (strict=True, key ``state_dict``) is
  never called by the pinned code and is not used here.
* **Load-time adaptations, both recorded in every reply.** The 1.7-era pickle needs
  ``torch.nn.modules.linear._LinearWithBias -> torch.nn.Linear`` (a class modern torch removed; the
  alias restores the same parameter layout) and ``batch_first=False`` on the 24
  ``nn.MultiheadAttention`` modules the 1.7 state carried no such attribute for. Both are applied by
  ``workbench/replay/tgcir.py::_load_tgcir`` and the repair loop beside it, which this adapter reuses
  instead of reimplementing, and are named in the worker's reply so a run can state them.
* **CUDA-hard, and that remains a real property of the pinned source.** ``.cuda()`` is called with no
  device guard at ``model.py:32``, ``:107``, ``:120``, ``:136``, ``:244``, ``:251`` and
  ``test.py:31``, ``:47``, ``:101``, ``:115``, and the CLIP backbone is loaded with
  ``device='cuda'`` (``model.py:41``). A device-neutral path is achievable (the replay neutralises
  ``torch.Tensor.cuda``), but the pinned release itself has no CPU mode. See ``CUDA_HARD``.

SCORING (``third_party/TGCIR/test.py``), reproduced by the pinned encoder plus ``super().score``

* query row = ``model.extract_retrieval_compose(imgs, mods)`` -> ``F.normalize(mean(tokens))``, 512-d
  (``model.py:197-199``); target row = ``model.extract_retrieval_target(imgs)`` -> the same shape
  (``:201-204``).
* ``test.py:53-61`` divides every row by its L2 norm and takes ONE dot product (no max over tokens:
  the encoder already mean-pooled). ``super().score`` is exactly that expression, so it is reused.
* ``test.py:63-67`` then masks the reference row (``-10e10``). That mask is **deliberately not
  inherited**, for the same reason ``model_pair.py`` and ``model_encoder.py`` do not inherit it:
  CIRCUS ranks over a reference-eligible gallery. ``reference_eligible`` is left to the base class's
  measured score+rank probe rather than asserted here.

CAPTION CORRECTION: ONE OWNER, AND IT IS THE WORKER

``datasets.FashionIQ`` corrects each caption token by token through the category's
``correction_dict_<category>.json`` (363/360/391 entries) and joins the pair with ``" and "``
(``datasets.py:71-80``). The correction is MATERIAL: it changes the text of 6000 of the 6003 release
queries (measured), so doing it twice, or not at all, changes every embedding.

The contract is: **``texts`` handed to ``encode_queries`` are RAW captions in the protocol's own
joined form** (``cap0 + " and " + cap1``, exactly as the CIRCUS release stores ``query_txt``). The
adapter transports them unchanged (``encode_queries`` never rewrites them: a second rewrite downstream
of the condition layer would undo ``image_only_empty_text``), and the WORKER applies the pinned
``correct_text`` once, immediately before the forward pass. The dictionary is per category while the
model is category-agnostic: with ``category_scopes = (None,)`` the caller passes ``None`` and the
worker resolves the category from the text itself (index over ``cap.<category>.val.json``; 6000/6003
release texts resolve, 25 resolve to two categories whose dictionaries produce the SAME corrected text
— measured, and an ambiguity that disagreed would be refused, not guessed). A non-None category in
``categories`` wins over the lookup.

Deviations worth naming: the pinned loader corrects each caption of the pair separately and then
joins; correcting the joined string once differs on 3 of 6016 val rows, all with an empty first
caption (``" and is grey …"`` vs ``"and is grey …"``). Both forms were measured; the joined form is used
because that is the text the CIRCUS release actually stores.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import string
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from workbench.composition.adapters import (
    AdapterStatus,
    AdapterWorkRootError,
    AdapterWorkerError,
    CompositionAdapter,
    limit_torch_threads,
    prefetch_rows,
    resolved_decode_workers,
    worker_environment,
)
from workbench.composition.inputs import CONDITIONS
from workbench.replay.common import sha256_file

#: ``model.py:197-204`` mean-pools ``(B, tokens, 512)`` to ``(B, 512)`` and L2-normalizes.
DIMENSION = 512

#: Verified against the pinned sources: every one of these runs before any metric exists. Kept
#: because it is a property of the pinned release, which this adapter does not change.
CUDA_HARD = ("model.py:32,107,120,136,244,251 and test.py:31,47,101,115 call .cuda() with no "
             "device guard, and model.py:41 loads CLIP with device='cuda'")

#: The transform ``train.py:87,90`` builds: ``clip.load('ViT-B/16').preprocess``, used for BOTH the
#: query reference and the gallery (the pinned loader has no train/val asymmetry).
TRANSFORM = "clip.load('ViT-B/16', jit=False).preprocess"

class Refused(RuntimeError):
    """This adapter refuses to launch: a declared prerequisite is missing or unreadable."""

class CheckpointDigestMismatch(ValueError):
    """The staged pickle's bytes are not the bytes the registry authorised."""

def _correct_text(text: str, correction_dict: Mapping[str, str]) -> str:
    """The pinned ``datasets.FashionIQ.correct_text`` (``datasets.py:71-75``), verbatim.

    Lowercases, maps every ``string.punctuation`` character to a space, splits, and substitutes each
    token through the category dictionary. Copied rather than imported because importing the pinned
    module needs torch, CLIP and a source checkout; ``tests/test_tgcir_composition_parity.py``
    proves this copy equal to the pinned bytes by executing them out of ``datasets.py``.
    """
    trans = str.maketrans({key: " " for key in string.punctuation})
    tokens = str(text).lower().translate(trans).strip().split()
    return " ".join([correction_dict.get(word) if word in correction_dict else word for word in tokens])

class Corrections:
    """The pinned caption correction, as a lookup, so the corrected string cannot drift.

    ``datasets.FashionIQ.correct_text``/``concat_text`` (``datasets.py:71-80``) correct each caption
    of the pair SEPARATELY and then join with ``" and "``. Correcting the joined string instead is
    NOT the same function: on 3 of 6016 val rows the first caption is empty and the two forms differ
    (``" and is grey …"`` vs ``"and is grey …"``). This class therefore precomputes the PINNED form
    per annotation row, keyed by the row's raw joined text, and returns that value -- not a
    recomputation -- when the caller hands back the row's own text. Anything else (a release
    ``query_txt`` that does not match an annotation row: 3 of 6003, measured) falls back to
    correcting the given text with the resolved category's dictionary, which is what the pinned
    function does to a single string.

    Category resolution, in order: an explicit non-None caller category; the ``(reference stem, raw
    text)`` pair; the raw text alone; the reference stem alone. A choice between categories whose
    dictionaries DISAGREE is refused rather than guessed (measured: zero such rows in the val
    annotations and in the CIRCUS release, so the refusal is a guard, not a live path).
    """

    def __init__(self, dataset_root: Path) -> None:
        self.dictionaries: dict[str, dict[str, str]] = {}
        self._by_pair: dict[tuple[str, str], dict[str, str]] = {}
        self._by_text: dict[str, dict[str, str]] = {}
        self._by_stem: dict[str, set[str]] = {}
        for category in ("dress", "shirt", "toptee"):
            self.dictionaries[category] = json.loads(
                (Path(dataset_root) / "captions" / f"correction_dict_{category}.json")
                .read_text(encoding="utf-8"))
            rows = json.loads(
                (Path(dataset_root) / "captions" / f"cap.{category}.val.json")
                .read_text(encoding="utf-8"))
            for row in rows:
                joined = " and ".join(row["captions"])
                pinned = " and ".join(_correct_text(part, self.dictionaries[category])
                                      for part in row["captions"])
                key = _correct_text(joined, {})
                if key:
                    self._by_text.setdefault(key, {})[category] = pinned
                    self._by_pair.setdefault((str(row["candidate"]), key), {})[category] = pinned
                self._by_stem.setdefault(str(row["candidate"]), set()).add(category)

    def _candidates(self, text: str, stem: str | None, category: str | None) -> dict[str, str]:
        key = _correct_text(text, {})
        if category is not None:
            return {category: self._by_pair.get((stem or "", key), {}).get(
                category, self._by_text.get(key, {}).get(
                    category, _correct_text(text, self.dictionaries[category])))}
        for table in (self._by_pair.get((stem or "", key)), self._by_text.get(key)):
            if table:
                return table
        if stem and stem in self._by_stem:
            return {name: _correct_text(text, self.dictionaries[name])
                    for name in self._by_stem[stem]}
        return {}

    def correct(self, texts: Sequence[str], categories: Sequence[str | None],
                image_paths: Sequence[str | None] | None = None) -> list[str]:
        """Correct each raw caption ONCE. The caller's ``texts`` are never rewritten upstream."""
        paths = list(image_paths or [None] * len(texts))
        corrected: list[str] = []
        for position, (text, category) in enumerate(zip(texts, categories)):
            if not str(text).strip():
                # A BLANK caption is not ambiguous: every candidate dictionary maps it to "", so
                # there is nothing for the lookup to choose between. TG-CIR is a text tower over a
                # caption STRING, and "" tokenises to a valid (SOT, EOT) sequence that the pinned
                # `clip.tokenize` handles, so the empty edit — `image_only_empty_text` — is a real
                # input this model can encode. Refusing it here would abort the whole run: `_run`
                # raises `AdapterWorkerError` and the runner re-raises that unconditionally
                # (runner.py:693-694), so the condition would kill the model's run instead of
                # degrading. Recorded verbatim: the empty string reaches the model unchanged.
                corrected.append("")
                continue
            if category is not None and category not in self.dictionaries:
                raise ValueError(f"tgcir has no correction dictionary for category {category!r}")
            stem = Path(paths[position]).stem if paths[position] else None
            forms = self._candidates(text, stem, category)
            if not forms:
                raise ValueError(
                    f"tgcir caption text {position} ({text!r}) resolves to no category, so no "
                    "correction dictionary can be chosen")
            values = set(forms.values())
            if len(values) != 1:
                raise ValueError(
                    f"tgcir caption text {position} ({text!r}) is ambiguous: categories "
                    f"{sorted(forms)} correct it to {sorted(values)}")
            corrected.append(values.pop())
        return corrected

class Adapter(CompositionAdapter):
    """Run the pinned TG-CIR model in its own interpreter; no reference-image score mask."""

    model_id = "tgcir"
    reference_eligible = True
    score_dimension = DIMENSION
    category_scopes: tuple[str | None, ...] = (None,)

    def __init__(self, config: Any = None, checkpoint_id: str | None = None) -> None:
        from workbench.backend.operator_config import resolve_config

        self.config = config if config is not None else resolve_config()
        self._model, self._checkpoint, interpreter = _resolve(self.config, checkpoint_id)
        super().__init__(self.config, self._checkpoint["checkpoint_id"])
        self.checkpoint = self._checkpoint
        self.checkpoint_path = Path(self._checkpoint["resolved_path"])
        self.source_root = Path(self.config.WORKBENCH_THIRD_PARTY_ROOT) / self._model["source_dir"]
        self.dataset_root = Path(self.config.FASHIONIQ_ROOT)
        self.interpreter = interpreter
        self.work_root: Path | None = None
        #: What the worker ACTUALLY did: adaptations, device, checkpoint digest, PID.
        self.load_state: dict[str, Any] | None = None
        self.last_reply: dict[str, Any] = {}
        self._worker_pid: int | None = None
        self._counter = 0
        self._job_dir: Path | None = None
        self._worker: subprocess.Popen[str] | None = None
        self._worker_log: Any | None = None
        self._worker_failed: str | None = None
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

    def _job_root(self) -> Path:
        root = self.require_work_root() / self.model_id
        root.mkdir(parents=True, exist_ok=True)
        if self._job_dir is None or not self._job_dir.is_dir():
            self._job_dir = root / f"worker-{uuid.uuid4().hex}"
            self._job_dir.mkdir()
        return self._job_dir

    def _exchange(self, phase: str, image_paths: Sequence[str],
                  texts: Sequence[str], categories: Sequence[str | None]
                  ) -> tuple[dict[str, Any], Path]:
        """Write one job, hand it to the worker (starting it if needed), and return its reply.

        ONE worker per adapter instance: the model is loaded once and reused for every encode call.
        The PID in each reply must equal the first one -- a silent reload per batch would make the
        run slower and, if a load adaptation changed, quietly inconsistent.
        """
        job_dir = self._job_root()
        if not self.interpreter or not Path(self.interpreter).is_file():
            variable = self._model["environment"]["python_env_var"]
            raise RuntimeError(f"model interpreter unavailable: {variable} is unset or missing")
        if not self._interpreter_verified:
            # The shared gate, unchanged, before the first launch: never relaxed, never bypassed.
            self.require_verified_interpreter()
            self._interpreter_verified = True
        if self._worker_failed is not None:
            raise AdapterWorkerError(self._worker_failed)
        self._counter += 1
        job_path = job_dir / f"job-{self._counter}.json"
        output_path = job_dir / f"output-{self._counter}.npy"
        job_path.write_text(json.dumps({
            "phase": phase, "image_paths": list(image_paths), "texts": list(texts),
            "categories": list(categories), "output_path": str(output_path),
            "checkpoint_path": str(self.checkpoint_path),
            "expected_sha256": self._checkpoint.get("expected_sha256"),
            "expected_size_bytes": self._checkpoint.get("expected_size_bytes"),
            "source_root": str(self.source_root), "dataset_root": str(self.dataset_root),
            "batch_size": self.batch_size, "transform": TRANSFORM,
        }), encoding="utf-8")
        if self._worker is not None and self._worker.poll() is not None:
            status = self._worker.returncode
            self._stop_worker()
            self._worker_failed = (
                f"adapter_worker_exited: model=tgcir phase={phase} exit_status={status} "
                f"log={job_dir / 'worker.stderr.log'}")
            raise AdapterWorkerError(self._worker_failed)
        if self._worker is None:
            self._worker_log = (job_dir / "worker.stderr.log").open("w", encoding="utf-8")
            self._worker = subprocess.Popen(
                [self.interpreter, "-m", "workbench.composition.model_tgcir",
                 "--worker-job", str(job_path)], cwd=self.config.CIR_REPO_ROOT,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._worker_log,
                text=True, bufsize=1, env=worker_environment())
        else:
            assert self._worker.stdin is not None
            self._worker.stdin.write(json.dumps({"job_path": str(job_path)}) + "\n")
            self._worker.stdin.flush()
        assert self._worker.stdout is not None
        line = self._worker.stdout.readline()
        while line and not line.startswith("CIRCUS_WORKER:"):
            line = self._worker.stdout.readline()
        if not line:
            worker = self._worker
            self._stop_worker()
            status = worker.returncode if worker is not None else None
            self._worker_failed = (
                f"adapter_worker_exited: model=tgcir phase={phase} exit_status={status} "
                f"log={job_dir / 'worker.stderr.log'}")
            raise AdapterWorkerError(self._worker_failed)
        response = json.loads(line[len("CIRCUS_WORKER:"):])
        if not response.get("ok"):
            job_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
            self._worker_failed = (
                f"adapter_worker_job_failed: model=tgcir phase={phase} "
                f"error={response.get('error_type')}: {response.get('error')}")
            self._stop_worker()
            raise AdapterWorkerError(self._worker_failed)
        pid = self._check_worker_pid(response.get("pid"))
        if response.get("load_state"):
            self.load_state = dict(response["load_state"], pid=pid,
                                   loads=int(response.get("loads") or 0))
        # Per-job facts, kept beside the load state so a run can name the EXACT text it scored and
        # the VRAM the forward actually held, instead of re-deriving either.
        self.last_reply = {key: response[key] for key in
                           ("corrected_digest", "corrected_first", "corrected", "batches", "vram")
                           if key in response}
        job_path.unlink(missing_ok=True)
        return response, output_path

    def _check_worker_pid(self, pid: Any) -> int | None:
        """Refuse a reply from a DIFFERENT worker: the model must load once per adapter instance."""
        if self._worker_pid is None:
            self._worker_pid = pid
        elif pid != self._worker_pid:
            previous, self._worker_pid = self._worker_pid, pid
            self._stop_worker()
            raise AdapterWorkerError(
                f"adapter_worker_reloaded: model=tgcir the worker PID changed from "
                f"{previous} to {pid}; the model must load once per adapter instance")
        return pid

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
        """Cosine: ``test.py:53-57`` normalize both sides, ``:61`` one dot product.

        The shared implementation is exactly this expression, so it is reused, not restated. The
        upstream reference mask (``test.py:63-67``) is deliberately absent — see the docstring.
        """
        return super().score(queries, targets, categories)

    def _stop_worker(self) -> None:
        worker = self._worker
        if worker is not None:
            if worker.stdin is not None:
                try:
                    worker.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            if worker.poll() is None:
                worker.terminate()
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
            self._worker = None
        if self._worker_log is not None:
            self._worker_log.close()
            self._worker_log = None

    def close(self) -> None:
        if self._worker is not None:
            if self._worker.stdin is not None:
                try:
                    self._worker.stdin.write(json.dumps({"stop": True}) + "\n")
                    self._worker.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
            try:
                self._worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._worker.terminate()
                try:
                    self._worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._worker.kill()
                    self._worker.wait()
            self._worker = None
        if self._worker_log is not None:
            self._worker_log.close()
            self._worker_log = None
        if self._job_dir is not None:
            # A latched failure keeps the directory: the runner preserves worker.stderr.log from it.
            if self._worker_failed is None:
                shutil.rmtree(self._job_dir, ignore_errors=True)
            self._job_dir = None

    def restart(self) -> None:
        """Kill the worker and reload on the next encode call."""
        self.close()

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

def _blockers(model: dict[str, Any], checkpoint: dict[str, Any], path: Path,
              interpreter: str | None, config: Any) -> list[str]:
    """Every reason this model may not be scored, in one place for status() and mode_supported()."""
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
    clip_cache = Path.home() / ".cache" / "clip" / "ViT-B-16.pt"
    if not clip_cache.is_file():
        reasons.append(f"OpenAI CLIP ViT-B/16 asset missing: {clip_cache}")
    # The execution gate's own answer, so a plan cannot print READY for a launch the gate refuses.
    from workbench.composition.adapters import _environment_gate_reason

    gate = _environment_gate_reason(model, config)
    if gate:
        reasons.append(gate)
    return reasons

def _refusal(adapter: Adapter) -> str | None:
    reasons = _blockers(adapter._model, adapter._checkpoint, adapter.checkpoint_path,
                        adapter.interpreter, adapter.config)
    return "; ".join(reasons) if reasons else None

def _load_verified_model(path: Path, expected_sha256: str | None,
                         expected_size_bytes: int | None = None) -> Any:
    """Hash the staged pickle BEFORE unpickling it, then hand the verified bytes to the loader.

    A whole-module pickle executes code at load time, so the digest is the only thing standing
    between an unauthorised file and this process. ``load_trusted_whole_model`` re-hashes; the extra
    pass happens once per worker and is the price of failing before ``torch.load`` is reachable.
    """
    if not expected_sha256:
        raise CheckpointDigestMismatch(
            f"no authorised SHA-256 for whole-model checkpoint {path}; refusing to unpickle")
    if not Path(path).is_file() or Path(path).stat().st_size == 0:
        raise FileNotFoundError(f"whole-model checkpoint missing or empty: {path}")
    if expected_size_bytes is not None and Path(path).stat().st_size != int(expected_size_bytes):
        raise CheckpointDigestMismatch(
            f"whole-model checkpoint size mismatch: expected {expected_size_bytes}, "
            f"found {Path(path).stat().st_size} ({path})")
    actual = sha256_file(Path(path))
    if actual != expected_sha256:
        raise CheckpointDigestMismatch(
            f"whole-model checkpoint SHA-256 mismatch: expected {expected_sha256[:16]}..., "
            f"found {actual[:16]}... ({path})")
    from workbench.replay.common import load_trusted_whole_model as loader

    return loader(Path(path), expected_sha256, _torch())

def _torch() -> Any:
    import torch

    return torch

def _vram() -> dict[str, float] | None:
    """Device memory this worker's torch allocator holds: allocated, reserved, and the peak."""
    try:
        torch = _torch()
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    mib = 2 ** 20
    return {"allocated_mib": round(torch.cuda.memory_allocated() / mib, 1),
            "reserved_mib": round(torch.cuda.memory_reserved() / mib, 1),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / mib, 1),
            "device_total_mib": round(
                torch.cuda.get_device_properties(0).total_memory / mib, 1)}

def _run(adapter: Adapter, operation: str, image_paths: Sequence[str], texts: Sequence[str],
         categories: Sequence[str | None]) -> np.ndarray:
    """Launch (or reuse) the worker and return its feature matrix, one row per input, in order."""
    if adapter.work_root is None:
        raise AdapterWorkRootError("tgcir work_root was not set by runner")
    if not image_paths:
        return np.empty((0, DIMENSION), dtype=np.float32)
    if not adapter.interpreter:
        raise Refused("no interpreter environment resolved for WORKBENCH_TGCIR_PYTHON")
    response, output_path = adapter._exchange(operation, image_paths, texts, categories)
    result = np.load(output_path, allow_pickle=False).astype(np.float32, copy=False)
    if result.ndim != 2 or result.shape != (len(image_paths), DIMENSION):
        raise AdapterWorkerError(
            f"adapter_worker_invalid_matrix: model=tgcir phase={operation} "
            f"shape={result.shape} expected=({len(image_paths)}, {DIMENSION})")
    output_path.unlink(missing_ok=True)
    del response
    return result

def _status(cls: type[Adapter], config: Any) -> AdapterStatus:
    from workbench.backend.registry import checkpoint_by_id, checkpoint_path, load_registry, model_by_id
    from workbench.backend.runtime import resolve_model_interpreter
    from workbench.backend.model_settings import setting

    model = model_by_id(cls.model_id, getattr(config, "registry", None) or load_registry())
    checkpoint = checkpoint_by_id(
        model, getattr(config, "checkpoint_id", None) or model["checkpoint_variants"][0]["checkpoint_id"])
    path = checkpoint_path(cls.model_id, checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)
    variable = model["environment"]["python_env_var"]
    interpreter = resolve_model_interpreter(model, config) or setting(variable)
    reasons = _blockers(model, checkpoint, path, interpreter, config)
    reason = "; ".join(reasons) if reasons else None
    return AdapterStatus(
        cls.model_id, reason is None, reason,
        tuple(str(item["checkpoint_id"]) for item in model["checkpoint_variants"]),
        {condition: (reason is None, reason) for condition in CONDITIONS},
        "Cosine over mean-pooled 512-d vectors; upstream reference mask NOT inherited (CIRCUS needs "
        "a reference-eligible gallery). Evaluated by a WORKBENCH-ADDED path driving the pinned "
        "encoder, not by an author CLI (test.py has none; train.py --restore_file is dead). "
        f"Pinned source stays CUDA-hard: {CUDA_HARD}")

def _worker_main(job_path: Path) -> None:
    """Load the verified TG-CIR model ONCE, then process jobs, keeping stdout for replies."""
    import os as _os

    def reply(ok: bool, **values: Any) -> None:
        print("CIRCUS_WORKER:" + json.dumps({"ok": ok, "pid": _os.getpid(), **values}), flush=True)

    try:
        job = json.loads(Path(job_path).read_text(encoding="utf-8"))
        from workbench.replay.tgcir import _load_tgcir

        source_root = Path(job["source_root"])
        torch, (clip, _datasets, _tg_test), adaptations = _load_tgcir(source_root, "")
        model = _load_verified_model(Path(job["checkpoint_path"]), job.get("expected_sha256"),
                                     job.get("expected_size_bytes"))
        unset = [module for module in model.modules()
                 if isinstance(module, torch.nn.MultiheadAttention)
                 and not hasattr(module, "batch_first")]
        for module in unset:
            module.batch_first = False
        if unset:
            adaptations.append(
                f"batch_first=False restored on {len(unset)} nn.MultiheadAttention modules "
                "(torch 1.7 state)")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            # ADAPTED RUNTIME, identical to `workbench/replay/tgcir.py`: `.cuda()` is called with no
            # device guard (`model.py:32,107,120,136`), so a CPU run has to neutralise it. Weights and
            # arithmetic are untouched. Recorded, so the run can never claim a native CUDA path.
            torch.Tensor.cuda = lambda self, *a, **k: self  # type: ignore[assignment]
            adaptations.append("torch.Tensor.cuda neutralised for a CPU run; weights/arithmetic "
                               "unchanged")
        model = model.to(device).eval()
        # The transform is device-independent, so the redundant CLIP tower (the pinned model already
        # carries its own visual weights from the pickle) is built on the CPU: no VRAM, same pixels.
        _clip_model, preprocess = clip.load("ViT-B/16", device="cpu", jit=False)
        corrections = Corrections(Path(job["dataset_root"]))
        limit_torch_threads()
        workers = resolved_decode_workers()
        digest = job.get("expected_sha256")
    except Exception as error:
        reply(False, error_type=type(error).__name__, error=str(error))
        return

    load_state = {"device": device, "adaptations": adaptations, "checkpoint_sha256": digest,
                  "checkpoint_bytes": Path(job["checkpoint_path"]).stat().st_size,
                  "device_name": (torch.cuda.get_device_name(0) if device == "cuda" else None),
                  "torch": torch.__version__, "transform": TRANSFORM}

    def run_one(current: dict[str, Any]) -> None:
        from PIL import Image

        phase = current["phase"]
        paths = current["image_paths"]
        corrected: list[str] = []
        if phase == "queries":
            corrected = corrections.correct(current["texts"], current["categories"], paths)
        batch_size = max(1, int(current.get("batch_size") or 32))
        batches = [paths[start:start + batch_size] for start in range(0, len(paths), batch_size)]

        def decode(chunk: Sequence[str]) -> Any:
            images = []
            for path in chunk:
                with Image.open(path) as image:
                    images.append(preprocess(image.copy().convert("RGB")))
            return torch.stack(images)

        features = []
        try:
            for position, batch in enumerate(prefetch_rows(decode, batches, workers)):
                start = position * batch_size
                stop = start + len(batch)
                with torch.no_grad():
                    batch = batch.to(device)
                    if phase == "queries":
                        encoded = model.extract_retrieval_compose(batch, corrected[start:stop])
                    else:
                        encoded = model.extract_retrieval_target(batch)
                features.append(encoded.detach().float().cpu().numpy())
            result = (np.concatenate(features, axis=0) if features
                      else np.empty((0, DIMENSION), dtype=np.float32))
            np.save(current["output_path"], result.astype(np.float32, copy=False), allow_pickle=False)
            reply(True, load_state=load_state, loads=1,
                  corrected=len(corrected), batches=len(batches),
                  # The EXACT strings handed to the model, digested: a run can prove which text it
                  # scored without shipping 15k captions, and a double correction changes this value.
                  corrected_digest=(hashlib.sha256("\n".join(corrected).encode("utf-8")).hexdigest()
                                    if corrected else None),
                  corrected_first=(corrected[0] if corrected else None),
                  vram=_vram())
        except Exception as error:
            reply(False, error_type=type(error).__name__, error=str(error))

    run_one(job)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("stop"):
            break
        try:
            run_one(json.loads(Path(request["job_path"]).read_text(encoding="utf-8")))
        except Exception as error:
            reply(False, error_type=type(error).__name__, error=str(error))

if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker-job":
        raise SystemExit(
            "usage: python -m workbench.composition.model_tgcir --worker-job JOB.json")
    _worker_main(Path(sys.argv[2]))

ADAPTER = Adapter
