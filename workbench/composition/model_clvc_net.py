"""CIRCUS composition adapter for CLVC-Net: one resident worker per adapter instance.

The staged artifacts are **whole pickled modules**, not state dicts (``train.py:274-275``), so a
forward pass needs the pinned classes importable: the worker inserts the pinned source root on
``sys.path`` and unpickles with ``weights_only=False`` *after* the author's sha256 matched.

Shape of the evaluation (``third_party/CLVCNet/test.py``, FashionIQ branch):

* a query row is ``cat(normalize(local.compose_img_text(imgs, mods)[1]),
  normalize(global.compose_img_text(imgs, mods)[0]))`` (``:100``) and a target row is
  ``cat(normalize(local.extract_img_feature(imgs)[1]),
  normalize(global.extract_img_feature(imgs)[1]))`` (``:116``) — note the local member takes
  index ``[1]`` and the global member ``[0]``, because their return tuples differ;
* so a row is 2048 wide with two unit-norm 1024-d halves. ``:122-125`` then divide each whole row
  by its own L2 norm (exactly ``sqrt(2)`` here) and ``:129`` is ``sims = all_queries.dot(all_imgs.T)``.
  Dividing a row of two unit halves by its norm and taking the dot product therefore equals
  ``(dot_local + dot_global) / 2``, i.e. the plain cosine ``score()`` computes. Measured, not
  asserted: on real rows from all six pickles the two expressions differ by at most
  **1.1920929e-07** (evidence ``raw/verify/87-clvc-composition.md`` §3);
* ``:134-135`` masks the reference row (``sims[i, ...] = -10e10``). That mask is **deliberately not
  inherited**: CIRCUS ranks over a reference-eligible full gallery, the same choice
  ``model_pair.py`` and ``model_encoder.py`` make for the identical upstream pattern. The parity
  evidence models the mask explicitly in both paths instead of hiding the difference.

Category is a real axis: ``train.py:156-157`` builds one local/global pair per FashionIQ category
and six files are staged, so ``category_scopes`` is ``("dress", "shirt", "toptee")`` and a query is
scored only against its own category's gallery matrix (registry ``category_model_policy:
independent``). A query's category is therefore an INPUT: it is never inferred from the gallery.

Divergence that cannot be removed from here: native query text is the templated pair
``"<BOS> {cap0} <AND> {cap1} <EOS>"`` (``datasets.py:174``), while the CIRCUS release supplies one
released sentence per query and the adapter must transport text verbatim. Feeding the release text
through unchanged (rather than re-templating it) is recorded in the evidence, not silently "fixed".

The pinned evaluator calls ``.cuda()`` unconditionally; ``cuda_census()`` counts those sites from
the pinned bytes, and the worker runs with CUDA available in the registered interpreter.
"""
from __future__ import annotations

import json
import os
import shutil
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
from workbench.composition.data import ANNOTATION_PIN
from workbench.composition.inputs import CONDITIONS
#: The routing contract is SHARED with LIMN, not reimplemented here: both models rank a query in
#: exactly one of the same three per-category spaces, both take the release loader's assignment as
#: final, and both fall back to the query's own FashionIQ caption annotation only when the loader
#: left the query unscoped. Reusing the one resolver is what keeps the two models scoring the SAME
#: query set; a second implementation would drift the first time either changed.
from workbench.composition.model_limn import (
    caption_scope_index,
    resolve_query_scope,
)
from workbench.replay.common import load_trusted_whole_model

#: ``model.py:127,199`` give each member a 1024-d feature; the evaluator concatenates the two.
HALF_DIMENSION = 1024
DIMENSION = 2 * HALF_DIMENSION

#: One local/global pair is staged per category; ``test.py`` scores a category against its own.
CATEGORIES = ("dress", "shirt", "toptee")

#: The four files the specification counts, and the two on the scoring path.
CUDA_FILES = ("model.py", "test.py", "text_model.py", "train.py")
SCORING_FILES = ("test.py", "text_model.py")

WORKER_MARKER = "CIRCUS_WORKER:"

#: Why ``image_only_empty_text`` cannot be encoded by this model, measured rather than assumed.
#:
#: ``TextLSTMModel.forward`` tokenises and then asserts on the FIRST token of the FIRST row
#: (``text_model.py:84-86``: ``type(x[0]) is list`` then ``type(x[0][0]) is int``). An empty edit
#: tokenises to ``[]``, so that indexing raises ``IndexError: list index out of range`` for every
#: row — measured through this adapter's own worker (evidence 87 §5). The text tower has no
#: zero-length path: ``nn.LSTM`` needs a sequence, and ``local_conv``/``global_conv`` sum over the
#: token axis. The edit is therefore REFUSED by name — a per-query error record the runner keeps —
#: instead of a traceback that fails the whole gallery job.
EMPTY_TEXT_REASON = (
    "CLVC-Net cannot encode an empty edit: TextLSTMModel.forward tokenises to zero tokens and "
    "indexes x[0][0] (text_model.py:84-86), which raises IndexError for every row, so "
    "image_only_empty_text is unrepresentable in this model's text tower rather than merely "
    "unsupported by the adapter")

#: The honest image-only path, as a NAMED condition distinct from ``image_only_empty_text``.
#:
#: ``image_only_empty_text`` stays unsupported for this model (``EMPTY_TEXT_REASON``): its text
#: tower cannot encode an empty edit at all. But CLVC-Net DOES have an image-only representation —
#: the same one its own retrieval pipeline already computes for a TARGET row, ``test.py:116``
#: ``cat(normalize(local.extract_img_feature(imgs)[1]),
#: normalize(global.extract_img_feature(imgs)[1]))`` — and the CLVC-Net paper names the baseline
#: it corresponds to: "Image_Only simply takes the reference image representation as the composed
#: query representation" (SIGIR'21, §4.2). So the ablation the paper reports is expressible with
#: THIS checkpoint, without touching the text tower; what is missing is a CONDITION NAME, not a
#: capability. ``IMAGE_ENCODER_ONLY`` is that name.
#:
#: It is NOT the paper's only difference: see ``IMAGE_ENCODER_ONLY_LEAKS_SCOPE`` for the routing
#: consequence, which is a separate declaration and not silently folded in.
IMAGE_ENCODER_ONLY = "clvc_image_encoder_only"
#: The worker OPERATION that expression is sent under. Distinct from ``"targets"`` only so a reply
#: can be attributed to the path that produced it; the row expression itself is the same.
IMAGE_ENCODER_ONLY_OPERATION = "image_only"

#: SCOPE LEAKAGE DECLARATION for ``clvc_image_encoder_only``: TRUE, and it is not a refinement.
#:
#: CLVC-Net scores a query against ONE category's gallery matrix, so every condition needs the
#: query's category. There are TWO routing inputs, and they are not the same input:
#:
#: * 5 884 of 6 003 queries: the RELEASE LOADER's assignment (``query.category``), computed from the
#:   reference image's stem against the authors' FashionIQ split/caption annotation. No caption text
#:   is read, and no image pixel is read. A dataset label, available independently of the pixels.
#: * 117 of 6 003 queries: the loader left them unscoped (their reference sits in two categories), so
#:   the shared resolver keys on ``(reference stem, normalize_query_text(edit caption))`` --
#:   ``model_limn.resolve_query_scope``'s own index key. For those 117 the EDIT CAPTION decides the
#:   category. The other 2 stay unscored.
#:
#: So under ``text_only_black_image`` and ``clvc_image_encoder_only`` the routing input is a dataset
#: label **plus, for those 117, the edit text that the condition otherwise removes**. Under the
#: image-only condition the caption is therefore not fully absent: it is absent from the EMBEDDING
#: (``composed_rows`` never receives it) and present in ROUTING for 117 queries. Under
#: ``text_only_black_image`` the same is true of the reference-derived label.
#:
#: Stated plainly rather than hidden. What it must never be read as is "text_only_black_image /
#: clvc_image_encoder_only see no reference information and no caption": ``text_only_black_image``
#: sees the row's category, and both see the edit caption for the 117. A run that needs the stricter
#: claim has to be a separate run identity that declares the exemption — see
#: ``category_routing_inputs`` in the evidence file — not a silent reuse of this one.
#:
#: The size of the effect on the published CLVC gap is MEASURED, not argued: over all 6 001 scored
#: queries it is 0.33061712371309204; over the 5 884 loader-scoped (caption-free) queries alone it is
#: 0.33400015944039774. Both come from the same rank table
#: (``runs/da0a2162be0e159a1a456cec/ranks``), via ``workbench/scripts/derive_routing_cohort_gap.py``.
IMAGE_ENCODER_ONLY_LEAKS_SCOPE = (
    "category routing reads the release loader's assignment (reference image stem) for 5884 "
    "queries and, for the 117 the loader left unscoped, (reference stem, edit caption text) "
    "resolved against the pinned caption annotation; under text_only_black_image and "
    "clvc_image_encoder_only the routing input is therefore a release DATASET LABEL plus, for "
    "those 117, the edit text that the condition otherwise removes")
#: One-line form for a run summary's `claim_narrowed` field.
IMAGE_ENCODER_ONLY_LEAKS_CAPTION = (
    "117 of 6001 scored queries had their category chosen from (reference stem, edit caption); a "
    "condition that removes the edit still used it for routing")

class Refused(RuntimeError):
    """This adapter's inputs or artifacts are not executable; no worker is launched."""

class CheckpointDigestMismatch(RuntimeError):
    """A staged member's bytes are not the authorised ones, so it is never unpickled."""

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

def _members() -> dict[str, tuple[str, str]]:
    """The six staged filenames per category, in ``(local, global)`` order."""
    return {category: (f"{category}_local_model.pt", f"{category}_global_model.pt")
            for category in CATEGORIES}

class Adapter(CompositionAdapter):
    """CLVC-Net's cosine scorer and encoder over per-category local/global pairs."""

    model_id = "clvc_net"
    reference_eligible = True
    score_dimension = DIMENSION
    category_scopes: tuple[str | None, ...] = CATEGORIES

    def __init__(self, config: Any = None, checkpoint_id: str | None = None) -> None:
        from workbench.backend.operator_config import resolve_config

        self.config = config if config is not None else resolve_config()
        self._model, self._checkpoint, self._interpreter = _resolve(self.config, checkpoint_id)
        super().__init__(self.config, self._checkpoint["checkpoint_id"])
        self.checkpoint = self._checkpoint
        self.checkpoint_path = Path(self._checkpoint["resolved_path"])
        self.source_root = Path(self.config.WORKBENCH_THIRD_PARTY_ROOT) / self._model["source_dir"]
        self.work_root: Path | None = None
        self._job_root: Path | None = None
        self._worker: subprocess.Popen[str] | None = None
        self._worker_log: Any | None = None
        self._worker_failed: str | None = None
        self._worker_counter = 0
        #: Every worker-reported PID, in reply order: the reuse proof (one PID for N batches).
        self.worker_pids: list[int] = []
        self._interpreter_verified = False
        #: Category routing (shared contract with LIMN; see the module-level import comment).
        self.dataset_root = Path(self.config.FASHIONIQ_ROOT)
        self._caption_scopes: Any = None
        self._scope_decisions: dict[str, dict[str, Any]] = {}

    #: Pin enforced on the caption annotation set, exactly as LIMN declares it. A fixture overrides
    #: this on the instance with ``{}`` ("synthetic bytes; row schema only").
    annotation_pin: Mapping[str, Mapping[str, Any]] = ANNOTATION_PIN

    def caption_scope_index(self) -> Any:
        """The annotation index, built once and shared by every routing call."""
        if getattr(self, "_caption_scopes", None) is None:
            self._caption_scopes = caption_scope_index(
                self.dataset_root, getattr(self, "annotation_pin", ANNOTATION_PIN))
        return self._caption_scopes

    def query_scope(self, query: Any,
                    candidate_names: Mapping[str, str] | None = None) -> str | None:
        """Route one released query to the checkpoint scope its own annotation names.

        The runner calls this only for the queries the release loader left unscoped (it treats a
        loader assignment as final), so the 5 884 already-scoped queries cannot move — those are
        routed from the loader's own reference-stem assignment and this method never sees them.

        For the remaining 119 the key IS ``(reference stem, normalized edit caption)``, taken from
        ``model_limn.resolve_query_scope``; the edit caption is therefore a routing input for those
        117 (2 stay unscored, their caption pair filed under both categories). ``None`` keeps a
        query unscored — never defaulted, never inferred from the target image or from a retrieval
        score. The caption's presence here is why
        ``IMAGE_ENCODER_ONLY_LEAKS_SCOPE`` says "dataset label plus, for those 117, the edit text".
        """
        index = self.caption_scope_index()
        record = resolve_query_scope(index, query, candidate_names)
        if record["scope"] is not None and record["scope"] not in self.category_scopes:
            record["reason"] = f"unresolved:scope_outside_declared:{record['scope']}"
            record["scope"] = None
        if not hasattr(self, "_scope_decisions"):
            self._scope_decisions = {}
        self._scope_decisions[str(record["qid"])] = record
        return record["scope"]

    def query_scope_resolution(self) -> dict[str, Any]:
        """``qid -> scope`` with one rationale per query, plus the annotation provenance."""
        decisions = [{"qid": qid, **{key: value for key, value in record.items() if key != "qid"}}
                     for qid, record in sorted(self._scope_decisions.items())]
        counts: dict[str, int] = {}
        for record in decisions:
            key = (f"{record['reason']}:{record['scope']}" if record["scope"] is not None
                   else str(record["reason"]))
            counts[key] = counts.get(key, 0) + 1
        return {"model_id": self.model_id,
                "resolver": f"{type(self).__module__}.{type(self).__qualname__}.query_scope",
                "annotation": self.caption_scope_index().report(),
                "counts_by_reason": dict(sorted(counts.items())),
                "unresolved": [record for record in decisions if record["scope"] is None],
                "query_count": len(decisions),
                "scope_by_qid": {record["qid"]: record["scope"] for record in decisions}}

    @classmethod
    def status(cls, config: Any = None) -> AdapterStatus:
        from workbench.backend.operator_config import resolve_config

        return _status(cls, config if config is not None else resolve_config())

    def mode_supported(self, condition: str) -> tuple[bool, str | None]:
        return _mode_support(condition, _structural_refusal(self))

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

    def encode_image_only_queries(self, image_paths: Sequence[str],
                                  categories: Sequence[str | None]) -> np.ndarray:
        """Query rows from the image encoder ALONE — the named ``clvc_image_encoder_only`` path.

        The row is ``test.py:116``'s target expression, applied to the reference image: the same
        ``extract_img_feature`` halves, the same transform, the same ``(local, global)``
        concatenation order and the same per-half ``F.normalize`` as the gallery matrix, so a
        reference image's row here differs from its own gallery row only by batch composition.

        No text argument exists on purpose: the edit caption is not an EMBEDDING input, so it cannot
        change this row, and ``compose_img_text``/``text_model`` are never reached. It is NOT a claim
        that the caption is absent from the query altogether: the query's CATEGORY may have been
        chosen from the caption by the resolver (117 of 6001 — ``IMAGE_ENCODER_ONLY_LEAKS_SCOPE``),
        which affects which gallery matrix the row is scored against, not the row. This is also NOT
        ``image_only_empty_text``: that condition drives the text tower with an empty string, which
        this model cannot encode at all (``EMPTY_TEXT_REASON``), and it stays unsupported.
        """
        if len(image_paths) != len(categories):
            raise ValueError("image_paths and categories must have equal lengths")
        return _run(self, IMAGE_ENCODER_ONLY_OPERATION, image_paths, (), categories)

    def score(self, queries: Any, targets: Any,
              categories: Sequence[str | None] | None = None) -> np.ndarray:
        """Cosine over 2048-d rows: ``test.py:122-129`` — no token max, no other normalization.

        Valid only for rows the encoders produced, i.e. two unit-norm 1024-d halves each: there the
        whole-row L2 of ``:125`` is exactly ``sqrt(2)``, so the upstream expression equals the plain
        cosine this inherits (measured max |Δ| 1.19e-07 over real rows of all six pickles). The
        upstream reference mask (``:134-135``) is deliberately absent — CIRCUS ranks over a
        reference-eligible gallery — and ``categories`` is forwarded because the base class needs it
        for scoped galleries.
        """
        return super().score(queries, targets, categories)

    def close(self) -> None:
        _stop_worker(self)
        if self._job_root is not None:
            shutil.rmtree(self._job_root, ignore_errors=True)
            self._job_root = None

    def restart(self) -> None:
        """Kill the worker (and its loaded checkpoints); reload on the next encode call."""
        self.close()

def _stop_worker(adapter: Adapter) -> None:
    worker = adapter._worker
    if worker is not None:
        if worker.stdin is not None:
            try:
                worker.stdin.write(json.dumps({"stop": True}) + "\n")
                worker.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            worker.terminate()
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
        adapter._worker = None
    if adapter._worker_log is not None:
        adapter._worker_log.close()
        adapter._worker_log = None

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

def _declared_digests(checkpoint: dict[str, Any]) -> dict[str, str]:
    """Filename -> author sha256, from the registry's own bundle declaration."""
    return {str(member["filename"]): str(member["expected_sha256"])
            for member in checkpoint.get("bundle_members") or [] if member.get("expected_sha256")}

def _blockers(model: dict[str, Any], checkpoint: dict[str, Any], path: Path,
              interpreter: str | None, config: Any) -> list[str]:
    """Structural reasons this model may not be scored, shared by status() and _run()."""
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
    digests = _declared_digests(checkpoint)
    missing = [name for pair in _members().values() for name in pair if name not in digests]
    if missing:
        reasons.append(f"checkpoint members carry no declared digest: {missing}")
    return reasons

def _mode_support(condition: str, reason: str | None) -> tuple[bool, str | None]:
    """One predicate for both `mode_supported` and the status row, so the two cannot disagree."""
    if condition == IMAGE_ENCODER_ONLY:
        # Supported by CONSTRUCTION, not by hope: the path is `extract_img_feature` on the same
        # pair from the same worker, and the CPU verification fixture executes it. It is a
        # DIFFERENT name from `image_only_empty_text` on purpose — that one stays unsupported.
        return (reason is None), reason
    if condition not in CONDITIONS:
        return False, f"unknown composition condition: {condition!r}"
    if condition == "image_only_empty_text":
        # Report the unsupported condition as unsupported: the runner asks `mode_supported` and
        # SKIPS a condition it answers False for (runner.py:1920-1925), recording it as
        # ABLATION_UNSUPPORTED instead of running a condition in which every query fails. The
        # cause is the MODEL's text tower, not this adapter: `_refuse_empty_edits` is the per-query
        # report of a behaviour measured in `text_model.py:84-86` (see EMPTY_TEXT_REASON). Saying
        # True here and then failing 6003 queries makes the plan lie.
        return False, "; ".join(item for item in (reason, EMPTY_TEXT_REASON) if item)
    return (reason is None), reason

def _structural_refusal(adapter: Adapter) -> str | None:
    return "; ".join(_blockers(adapter._model, adapter._checkpoint, adapter.checkpoint_path,
                               adapter._interpreter, adapter.config)) or None

def _job_root(adapter: Adapter, work_root: Path) -> Path:
    if adapter._job_root is None:
        adapter._job_root = work_root / adapter.model_id / f"worker-{uuid.uuid4().hex}"
        adapter._job_root.mkdir(parents=True)
    return adapter._job_root

def _query_categories(categories: Sequence[str | None]) -> None:
    """Refuse a query whose category is unresolved: nothing here can infer it."""
    for index, category in enumerate(categories):
        if category is None:
            raise Refused(
                "adapter_query_category_missing: model=clvc_net index="
                f"{index} category=None; a CLVC-Net query is encoded by ONE category's local/global "
                f"pair, so its category must be one of {CATEGORIES!r} and is never inferred from "
                "the gallery")
        if category not in CATEGORIES:
            raise Refused(
                f"adapter_query_category_unknown: model=clvc_net index={index} "
                f"category={category!r} not in {CATEGORIES!r}")

def _refuse_empty_edits(texts: Sequence[str]) -> None:
    """An empty edit tokenises to nothing and the text tower indexes token 0 (``text_model.py:86``).

    Measured through this worker: ``image_only_empty_text`` raises ``IndexError: list index out of
    range`` for every row. A named refusal is a reportable per-query error; a traceback from inside
    a gallery job is a failed run. Declaring the condition itself unsupported belongs to the
    registry/plan phase, not to this module's transport layer.
    """
    empty = [index for index, text in enumerate(texts) if not str(text).strip()]
    if empty:
        raise Refused(f"adapter_empty_edit_unrepresentable: model=clvc_net rows={empty[:10]} "
                      f"count={len(empty)}: {EMPTY_TEXT_REASON}")

def group_indices(categories: Sequence[str | None]) -> dict[str, list[int]]:
    """Category -> input positions, in first-appearance order (the worker's grouping)."""
    groups: dict[str, list[int]] = {}
    for index, category in enumerate(categories):
        groups.setdefault(str(category), []).append(index)
    return groups

def scatter_rows(rows: np.ndarray, categories: Sequence[str | None],
                 grouped: Mapping[str, np.ndarray]) -> np.ndarray:
    """Place each category's rows back at its INPUT positions.

    A category-grouped implementation that returned grouped order is the classic id-misalignment
    bug: the caller pairs row ``i`` with ``image_paths[i]``, so the scatter is asserted complete
    rather than assumed. ``rows`` is written in place and returned.
    """
    written = 0
    for category, positions in group_indices(categories).items():
        block = grouped[category]
        if block.shape[0] != len(positions):
            raise ValueError(f"clvc_worker_row_count: category={category} got {block.shape[0]} "
                             f"rows for {len(positions)} inputs")
        rows[np.asarray(positions, dtype=np.int64)] = block
        written += len(positions)
    if written != rows.shape[0]:
        raise AssertionError("clvc_worker_row_scatter_incomplete")
    return rows

def _run(adapter: Adapter, operation: str, image_paths: Sequence[str], texts: Sequence[str],
         categories: Sequence[str | None]) -> np.ndarray:
    """Send one job to the resident worker, starting it on first use."""
    if adapter.work_root is None:
        raise AdapterWorkRootError("clvc_net work_root was not set by runner")
    if not image_paths:
        return np.empty((0, DIMENSION), dtype=np.float32)
    # Inputs first: a caller that asks for something unrepresentable gets the specific reason, not
    # a structural verdict about a different problem.
    if operation == "queries":
        if len(texts) != len(image_paths):
            raise ValueError("CLVC-Net query image and text counts must match")
        _query_categories(categories)
        _refuse_empty_edits(texts)
    elif operation == IMAGE_ENCODER_ONLY_OPERATION:
        # The category is still an INPUT (the gallery is per category), but no text is transported
        # at all, so nothing here can reach the text tower.
        if texts:
            raise ValueError("clvc_image_encoder_only transports no text; got "
                             f"{len(texts)} texts for {len(image_paths)} images")
        _query_categories(categories)
    else:
        scopes = {category for category in categories}
        if len(scopes) != 1 or None in scopes:
            raise Refused("adapter_target_scope_invalid: model=clvc_net a gallery is encoded by "
                          f"exactly one category's pair; got {sorted(map(str, scopes))}")
    refusal = _structural_refusal(adapter)
    if refusal is not None:
        raise Refused(refusal)
    interpreter = adapter._interpreter
    if not interpreter or not Path(interpreter).is_file():
        raise Refused("no interpreter environment resolved for WORKBENCH_CLVC_PYTHON")
    if not adapter._interpreter_verified:
        # The shared gate runs before anything is launched, and raises on an unverified runtime.
        adapter.require_verified_interpreter()
        adapter._interpreter_verified = True
    if adapter._worker_failed is not None:
        raise AdapterWorkerError(adapter._worker_failed)

    work = _job_root(adapter, Path(adapter.work_root))
    adapter._worker_counter += 1
    job_path = work / f"job-{adapter._worker_counter}.json"
    output_path = work / f"output-{adapter._worker_counter}.npy"
    digests = _declared_digests(adapter._checkpoint)
    scope = ",".join(sorted({str(value) for value in categories}))
    job_path.write_text(json.dumps({
        "operation": operation,
        "image_paths": [str(Path(path).resolve()) for path in image_paths],
        "texts": list(texts),
        "categories": list(categories),
        "output_path": str(output_path),
        "checkpoint_root": str(adapter.checkpoint_path),
        "source_root": str(adapter.source_root),
        "members": {category: [{"path": str(Path(adapter.checkpoint_path) / name),
                                "sha256": digests[name], "member": member}
                               for member, name in zip(("local", "global"), _members()[category])]
                    for category in CATEGORIES},
        "batch_size": adapter.batch_size,
    }), encoding="utf-8")
    log_path = work / "worker.stderr.log"
    try:
        if adapter._worker is not None and adapter._worker.poll() is not None:
            status = adapter._worker.returncode
            _stop_worker(adapter)
            adapter._worker_failed = (
                f"adapter_worker_exited: model=clvc_net scope={scope} phase={operation} "
                f"exit_status={status} log={log_path}")
            raise AdapterWorkerError(adapter._worker_failed)
        if adapter._worker is None:
            adapter._worker_log = log_path.open("w", encoding="utf-8")
            adapter._worker = subprocess.Popen(
                [interpreter, "-m", "workbench.composition.model_clvc_net",
                 "--worker-job", str(job_path)], cwd=adapter.config.CIR_REPO_ROOT,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=adapter._worker_log,
                text=True, bufsize=1, env=worker_environment())
        else:
            assert adapter._worker.stdin is not None
            adapter._worker.stdin.write(json.dumps({"job_path": str(job_path)}) + "\n")
            adapter._worker.stdin.flush()
        assert adapter._worker.stdout is not None
        line = adapter._worker.stdout.readline()
        while line and not line.startswith(WORKER_MARKER):
            line = adapter._worker.stdout.readline()
        if not line:
            status = adapter._worker.poll() if adapter._worker is not None else None
            _stop_worker(adapter)
            adapter._worker_failed = (
                f"adapter_worker_exited: model=clvc_net scope={scope} phase={operation} "
                f"exit_status={status} log={log_path}")
            raise AdapterWorkerError(adapter._worker_failed)
        response = json.loads(line[len(WORKER_MARKER):])
        if not response.get("ok"):
            adapter._worker_failed = (
                f"adapter_worker_job_failed: model=clvc_net scope={scope} phase={operation} "
                f"error={response.get('error_type')}: {response.get('error')} log={log_path}")
            _stop_worker(adapter)
            raise AdapterWorkerError(adapter._worker_failed)
        if response.get("worker_pid") is not None:
            adapter.worker_pids.append(int(response["worker_pid"]))
        features = np.load(output_path, allow_pickle=False)
        if features.dtype != np.float32 or features.shape != (len(image_paths), DIMENSION):
            adapter._worker_failed = (
                f"adapter_worker_invalid_matrix: model=clvc_net scope={scope} phase={operation} "
                f"shape={features.shape}/{features.dtype} "
                f"expected=({len(image_paths)}, {DIMENSION})/float32 log={log_path}")
            _stop_worker(adapter)
            raise AdapterWorkerError(adapter._worker_failed)
        return features
    except OSError as error:
        status = adapter._worker.poll() if adapter._worker is not None else None
        _stop_worker(adapter)
        adapter._worker_failed = (
            f"adapter_worker_exited: model=clvc_net scope={scope} phase={operation} "
            f"exit_status={status}: {error} log={log_path}")
        raise AdapterWorkerError(adapter._worker_failed) from error
    finally:
        job_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)

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
    census = cuda_census(Path(config.WORKBENCH_THIRD_PARTY_ROOT) / model["source_dir"])
    notes = _cuda_summary(census) if census else (
        "pinned CLVC-Net source not found; its CUDA usage cannot be counted")
    if not reasons:
        from workbench.composition.adapters import _environment_gate_reason

        gate = _environment_gate_reason(model, config)
        reasons = [gate] if gate else []
    return AdapterStatus(cls.model_id, not reasons, "; ".join(reasons) or None,
                         tuple(str(item["checkpoint_id"]) for item in model["checkpoint_variants"]),
                         {condition: _mode_support(condition, "; ".join(reasons) or None)
                          for condition in CONDITIONS},
                         notes + ". Cosine over concatenated, per-half-normalized 2048-d rows; the "
                                  "upstream reference mask is NOT inherited. One resident worker per "
                                  "adapter, per-category checkpoints loaded lazily on first use.")

def image_row_builder(local_model: Any, global_model: Any, torch_module: Any,
                      functional: Any) -> Any:
    """The ONE image-only row expression, shared by the worker and the CPU fixture.

    ``test.py:116`` — ``cat(normalize(local.extract_img_feature(imgs)[1]),
    normalize(global.extract_img_feature(imgs)[1]))`` — applied to a query's reference image. The
    gallery matrix is produced by exactly this expression, so a reference image's image-only query
    row and its own gallery row differ only by batch composition, not by preprocessing, half order,
    normalisation or model.

    Torch and ``F`` are ARGUMENTS rather than module imports: this module is imported by the
    backend without torch loaded, and the row builder must remain the same object in the worker
    (GPU), in the CPU verification fixture, and in any test that stubs the text tower.
    """
    def build(images: Any) -> Any:
        return torch_module.cat([
            functional.normalize(local_model.extract_img_feature(images)[1]),
            functional.normalize(global_model.extract_img_feature(images)[1]),
        ], dim=1)
    return build

def composed_rows(operation: str, local_model: Any, global_model: Any, images: Any,
                  texts: Sequence[str], torch_module: Any, functional: Any) -> Any:
    """One forward batch → one 2048-d row per image. The only place an operation becomes a row.

    Pulled out of the worker loop so the CPU verification fixture can execute the REAL expression
    with the REAL weights on a machine without CUDA, instead of asserting a copy of it. The worker
    passes its loaded pair; the fixture passes the same two pickles loaded on CPU.
    """
    if operation == "queries":
        return torch_module.cat([
            functional.normalize(local_model.compose_img_text(images, list(texts))[1]),
            functional.normalize(global_model.compose_img_text(images, list(texts))[0]),
        ], dim=1)
    # ``targets`` and ``image_only`` share ONE expression (`test.py:116`), so the gallery row for an
    # image and its image-only query row cannot diverge in preprocessing, half order or norm.
    return image_row_builder(local_model, global_model, torch_module, functional)(images)

def load_category_pair(category: str, members: Sequence[dict[str, Any]],
                       torch_module: Any) -> tuple[Any, Any]:
    """Verify both members' digests, then unpickle them — in that order, per member.

    ``load_trusted_whole_model`` hashes before ``torch.load`` and refuses without a declared
    digest, so a substituted file is never executed as a pickle.
    """
    loaded: list[Any] = []
    for spec in members:
        path = Path(spec["path"])
        try:
            loaded.append(load_trusted_whole_model(path, spec.get("sha256"), torch_module))
        except (ValueError, FileNotFoundError) as error:
            raise CheckpointDigestMismatch(
                f"clvc_checkpoint_digest_mismatch: category={category} member={spec['member']} "
                f"path={path}: {error}") from error
    return loaded[0], loaded[1]

def _worker_main(job_path: Path) -> None:
    """Load authorised checkpoints lazily per category, serve one job per stdin line."""
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from PIL import Image

    def reply(ok: bool, **values: Any) -> None:
        print(WORKER_MARKER + json.dumps({"ok": ok, **values}), flush=True)

    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
        source_root = Path(job["source_root"])
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        # The NATIVE FashionIQ validation pipeline, `train.py:90-96` (fashioniq, split='val').
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        limit_torch_threads()
        workers = resolved_decode_workers()
        loaded: dict[str, tuple[Any, Any]] = {}
    except Exception as error:
        reply(False, error_type=type(error).__name__, error=str(error))
        return

    def pair_for(category: str) -> tuple[Any, Any]:
        """Load this category's local/global pair on first use, and keep it."""
        if category not in loaded:
            if category not in CATEGORIES:
                raise ValueError(f"clvc_worker_unknown_category: {category!r}")
            local, global_model = load_category_pair(category, job["members"][category], torch)
            loaded[category] = (local.cuda().eval(), global_model.cuda().eval())
        return loaded[category]

    def encode_group(category: str, operation: str, paths: Sequence[str],
                     texts: Sequence[str], batch_size: int) -> np.ndarray:
        local, global_model = pair_for(category)
        batches = [paths[start:start + batch_size] for start in range(0, len(paths), batch_size)]

        def decode(chunk: Sequence[str]) -> Any:
            images = []
            for path in chunk:
                with Image.open(path) as image:
                    images.append(transform(image.convert("RGB")))
            # ``test.py:79,98`` stack then ``.float()``; the transform already yields float32.
            return torch.stack(images).float()

        rows: list[np.ndarray] = []
        for index, images in enumerate(prefetch_rows(decode, batches, workers)):
            start = index * batch_size
            with torch.no_grad():
                images = images.cuda()
                composed = composed_rows(operation, local, global_model, images,
                                         texts[start:start + len(images)], torch, F)
            rows.append(composed.detach().float().cpu().numpy())
            del images, composed
        return np.concatenate(rows, axis=0) if rows else np.empty((0, DIMENSION), np.float32)

    def run_one(current: dict[str, Any]) -> None:
        try:
            paths, texts = current["image_paths"], current["texts"]
            categories = current["categories"]
            batch_size = max(1, int(current.get("batch_size") or 32))
            grouped: dict[str, np.ndarray] = {}
            for category, indices in group_indices(categories).items():
                grouped[category] = encode_group(
                    category, current["operation"],
                    [paths[index] for index in indices],
                    [texts[index] for index in indices] if texts else [], batch_size)
            # Scattered back to INPUT positions: a grouped-order reply would misalign every row.
            output = scatter_rows(np.empty((len(paths), DIMENSION), dtype=np.float32),
                                  categories, grouped)
            np.save(current["output_path"], output, allow_pickle=False)
            reply(True, worker_pid=os.getpid(), loaded=sorted(loaded))
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
            break

if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker-job":
        raise SystemExit(
            "usage: python -m workbench.composition.model_clvc_net --worker-job JOB.json")
    _worker_main(Path(sys.argv[2]))

ADAPTER = Adapter
