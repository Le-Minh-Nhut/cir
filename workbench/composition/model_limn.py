"""LIMN FashionIQ adapter and model-interpreter worker.

Runner sets ``work_root`` after construction, before encoding; encode methods refuse with
``AdapterWorkRootError`` when unset. Each instance owns a unique
``<work_root>/limn/worker-*/`` directory, and ``close()``/``restart()`` remove only that
directory, never the run root. A query scope selects both its whole-model checkpoint and the
matching embedding space for the entire released gallery. Target matrices from scopes are never
mixed, candidates are never category-filtered, and checkpoint choice never consults positive targets.
Persistent worker caches one model per scope.

A released query whose garment category the release loader could not pin down (a reference image
listed by two FashionIQ category splits) is still scoped here, from the author caption annotation:
LIMN's native protocol is ``fashioniq_val_split``, and the M-BEIR task-7 query text is the two
validation captions of its reference concatenated, so the caption row a query text came from names
the category the FashionIQ authors assigned it. Text that reconstructs under two category files is
left unscoped rather than defaulted.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from workbench.backend.model_settings import setting
from workbench.backend.operator_config import resolve_config
from workbench.backend.registry import (
    checkpoint_by_id,
    checkpoint_is_present,
    checkpoint_path,
    model_by_id,
)
from workbench.backend.runtime import resolve_model_interpreter
from workbench.composition.adapters import (
    AdapterStatus,
    AdapterWorkerError,
    CompositionAdapter,
    ExactRankUnavailable,
    limit_torch_threads,
    prefetch_rows,
    resolved_decode_workers,
    worker_environment,
)
from workbench.composition.inputs import CONDITIONS
from workbench.composition.data import ANNOTATION_PIN, inspect_annotation

BUNDLE_ID = "base_iter0_all_categories"

def normalize_query_text(text: str) -> str:
    """Casefold, strip punctuation, collapse whitespace: the release's own caption join."""
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", str(text).strip().lower()).split())


class CaptionScopeIndex(dict):
    """``(image stem, normalized text) -> frozenset(categories)`` plus the sources that were NOT read.

    A plain ``dict`` mapping keeps every existing reader working, while ``unreadable`` /
    ``malformed`` / ``empty`` / ``mismatch`` (all keyed by CATEGORY, e.g. ``"toptee"``) describe the
    sources that cannot be used as evidence. Without them, "the other category has no row for this
    pair" and "the other category was never opened, was gutted to ``[]``, or carries a garbled row"
    are the same state, and a resolver then routes an ambiguous query to the one category whose file
    happened to survive.

    ``incomplete`` is the single predicate a resolver must honour: true when any file is
    unreadable, malformed (any row that is not an object with a string ``candidate`` and a list of
    string ``captions``), empty, or off its committed pin (row count/digest disagreeing with
    ``data.ANNOTATION_PIN``).
    """

    unreadable: frozenset[str] = frozenset()
    malformed: Mapping[str, str] = {}
    empty: frozenset[str] = frozenset()
    mismatch: Mapping[str, str] = {}
    counts: Mapping[str, int | None] = {}
    digests: Mapping[str, str | None] = {}
    complete: bool = False

    @property
    def incomplete(self) -> bool:
        """True when this index cannot prove the absence of a row in any category."""
        return bool(self.unreadable or self.malformed or self.empty or self.mismatch)

    def report(self) -> dict[str, Any]:
        """Per-file provenance, in the shape a run summary and the inventory can read."""
        broken = set(self.unreadable) | set(self.empty) | set(self.malformed) | set(self.mismatch)
        problems = {**{name: "unreadable" for name in self.unreadable},
                    **{name: "empty" for name in self.empty},
                    **{name: problem for name, problem in self.malformed.items()},
                    **{name: problem for name, problem in self.mismatch.items()}}
        return {
            "complete": self.complete,
            "incomplete": self.incomplete,
            "unreadable": sorted(self.unreadable),
            "malformed": dict(sorted(self.malformed.items())),
            "empty": sorted(self.empty),
            "mismatch": dict(sorted(self.mismatch.items())),
            "files": {category: {"file": f"cap.{category}.val.json",
                                 "count": self.counts.get(category),
                                 "sha256": self.digests.get(category),
                                 "usable": category not in broken,
                                 "problem": problems.get(category)}
                      for category in sorted(self.counts)},
        }

def caption_scope_index(fashioniq_root: Path,
                        pin: Mapping[str, Mapping[str, Any]] | None = None) -> CaptionScopeIndex:
    """Map ``(image stem, joined validation caption text)`` to the annotating categories.

    M-BEIR task-7 query text is the reference image's two FashionIQ validation captions joined by
    " and ", so the row a query text came from names the category the authors annotated that image
    under. Keyed by image *and* text: two images in different categories can carry identical
    caption text, and the reference identity is what separates them. A ``(stem, text)`` pair that
    still appears under two categories is genuinely undecidable and maps to both, so callers can
    refuse instead of picking one.

    A file is only used when ``data.inspect_annotation`` accepts it: every row well formed, the
    file non-empty, and -- when ``pin`` names it -- its row count and digest matching the committed
    ``data.ANNOTATION_PIN``. JSON validity is not enough: ``[]`` parses, but a category gutted to
    ``[]`` (or truncated, or carrying one wrong-typed row) would otherwise look like a category with
    no row for a garment and turn a real ambiguity into a unique wrong answer. Such a file is named
    in the matching attribute, its rows are not indexed, and ``incomplete`` is true.

    ``pin`` is the mapping ``data.inspect_annotation`` checks: ``None`` means "this root carries no
    pinned bytes" -- the default because most callers here are hand-built fixtures -- and it still
    enforces the row schema plus non-emptiness. The ONE released-data caller, ``Adapter``, passes
    ``Adapter.annotation_pin`` (``data.ANNOTATION_PIN``), which is what makes live routing refuse a
    file whose rows no longer match the authors' bytes.
    """
    pin = {} if pin is None else pin
    index: dict[tuple[str, str], set[str]] = {}
    unreadable: set[str] = set()
    malformed: dict[str, str] = {}
    empty: set[str] = set()
    mismatch: dict[str, str] = {}
    counts: dict[str, int | None] = {}
    digests: dict[str, str | None] = {}
    for category in ("dress", "shirt", "toptee"):
        report = inspect_annotation(Path(fashioniq_root) / "captions" / f"cap.{category}.val.json",
                                    shape="captions", pin=pin)
        counts[category] = report.count
        digests[category] = report.actual_sha256
        if report.problem is None:
            for row in report.rows:
                text = normalize_query_text(
                    " and ".join(part for part in row["captions"] if part.strip()))
                if text:
                    index.setdefault((Path(row["candidate"]).stem, text), set()).add(category)
        elif report.problem == "unreadable":
            unreadable.add(category)
        elif report.problem == "malformed":
            malformed[category] = report.detail or "malformed"
        elif report.problem == "empty":
            empty.add(category)
        else:
            mismatch[category] = report.detail or str(report.problem)
    result = CaptionScopeIndex({key: frozenset(categories) for key, categories in index.items()})
    result.unreadable = frozenset(unreadable)
    result.malformed = dict(malformed)
    result.empty = frozenset(empty)
    result.mismatch = dict(mismatch)
    result.counts = dict(counts)
    result.digests = dict(digests)
    result.complete = not (unreadable or malformed or empty or mismatch)
    return result

def query_reference_stem(query: Any, candidate_names: Mapping[str, str] | None) -> str | None:
    """Return the reference image stem the release candidate table records for ``query``."""
    if not isinstance(candidate_names, Mapping):
        return None
    name = candidate_names.get(getattr(query, "reference_id", None))
    return Path(name).stem if isinstance(name, str) and name else None


def _index_complete(index: Mapping[tuple[str, str], frozenset[str]]) -> bool:
    """Whether ``index`` proves that a category with no row really has no row.

    A plain ``dict`` carries no provenance at all, so it cannot prove anything; only a
    ``CaptionScopeIndex`` whose ``complete`` flag is set can.
    """
    return isinstance(index, CaptionScopeIndex) and not index.incomplete


def resolve_query_scope(index: Mapping[tuple[str, str], frozenset[str]], query: Any,
                        candidate_names: Mapping[str, str] | None) -> dict[str, Any]:
    """One routing decision plus its rationale, before the caller checks it against its scopes.

    ``reason`` is ``loader`` (the release loader assigned ``query.category``),
    ``annotation:<category>`` (the loader left it unset and exactly one caption file names this
    reference/text pair) or ``unresolved:<why>``. ``provenance_incomplete`` is true when the
    decision cannot be verified against a complete annotation set: in particular a ``loader`` route
    taken while the index is incomplete is a route no annotation ever confirmed, so a reader must be
    able to tell it apart from a verified one.
    """
    category = getattr(query, "category", None)
    complete = _index_complete(index)
    record: dict[str, Any] = {"qid": getattr(query, "qid", None), "scope": None,
                              "reason": None, "provenance_incomplete": not complete,
                              "holding_categories": (), "annotation_complete": complete}
    if category is not None:
        record["scope"] = category
        record["reason"] = "loader"
        return record
    stem = query_reference_stem(query, candidate_names)
    if stem is None:
        record["reason"] = "unresolved:no_reference_identity"
        return record
    holders = index.get((stem, normalize_query_text(getattr(query, "query_txt", ""))), frozenset())
    record["holding_categories"] = tuple(sorted(holders))
    if not complete:
        record["reason"] = "unresolved:annotation_index_incomplete"
        return record
    if not holders:
        record["reason"] = "unresolved:no_annotation_row"
        return record
    if len(holders) > 1:
        record["reason"] = "unresolved:annotation_names_two_categories"
        return record
    resolved = next(iter(holders))
    record["scope"] = resolved
    record["reason"] = f"annotation:{resolved}"
    return record


def resolve_query_category(scopes: Sequence[str | None], index: Mapping[tuple[str, str], frozenset[str]],
                           query: Any, candidate_names: Mapping[str, str] | None) -> str | None:
    """Route one query to its category checkpoint, or ``None`` when the release does not say.

    A category the release loader already assigned is final: returned when it is one of ``scopes``
    and refused otherwise, because an out-of-scope loader assignment is an adapter gap, not an
    ambiguity to paper over with the annotation. Only an unscoped query falls through to its own
    caption annotation, which has to name exactly one scope; anything short of that stays ``None``
    and the query stays unscored.

    An annotation index that cannot prove it read every source -- unreadable, malformed, emptied or
    off-pin -- cannot prove uniqueness either, so every query the loader left unscoped returns
    ``None``: "the other file had no row" and "the other file was never opened / was gutted" must
    not produce the same answer.
    """
    record = resolve_query_scope(index, query, candidate_names)
    resolved = record["scope"]
    return resolved if resolved in scopes else None


class CategoryScopeError(ValueError):
    """Scope refusal with stable caller-batch indices and per-index missing/out-of-scope reasons."""

    def __init__(self, reason: str, batch_indices: Sequence[int], detail: str,
                 reasons_by_index: Mapping[int, str] | None = None) -> None:
        self.model_id = "limn"
        self.reason = reason
        self.batch_indices = tuple(batch_indices)
        self.reasons_by_index = dict(reasons_by_index or {})
        super().__init__(f"limn {reason} at batch indices {self.batch_indices}: {detail}")


class Adapter(CompositionAdapter):
    """Encode a full gallery separately in each of LIMN's three learned category spaces."""

    model_id = "limn"
    category_scopes = ("dress", "shirt", "toptee")
    reference_eligible = True
    #: Pin enforced on the caption annotation set. ``ANNOTATION_PIN`` is the default because an
    #: annotation file that silently stopped matching the authors' bytes must refuse routing, not
    #: route. A fixture overrides it on the instance with ``{}`` ("synthetic bytes; schema only").
    annotation_pin: Mapping[str, Mapping[str, Any]] = ANNOTATION_PIN

    def __init__(self, config: Any, checkpoint_id: str | None = None) -> None:
        config = config or resolve_config()
        super().__init__(config, checkpoint_id)
        # LIMN's whole model is 440M parameters; the pinned harness's literal batch 32 exhausts a
        # 4 GiB laptop card (OOM at 130 MiB on the first dress batch here). `replay.limn` already
        # owns this knob and documents that batch size changes only the forward-pass stack, never
        # the ranking, so the composition path reuses the same operator override instead of
        # inventing a second one.
        from workbench.replay.limn import resolved_batch_size

        self.batch_size = resolved_batch_size()
        self._model_record = model_by_id(self.model_id)
        self._bundle_id = checkpoint_id or BUNDLE_ID
        self._bundle = checkpoint_by_id(self._model_record, self._bundle_id)
        self._members = {member["checkpoint_id"]: member
                         for member in self._bundle.get("bundle_members", ())}
        variable = self._model_record["environment"]["python_env_var"]
        self.interpreter = resolve_model_interpreter(self._model_record, config) or setting(variable)
        self.source_root = Path(config.WORKBENCH_THIRD_PARTY_ROOT) / self._model_record["source_dir"]
        self.dataset_root = Path(config.FASHIONIQ_ROOT)
        self._caption_scopes: CaptionScopeIndex | None = None
        self._scope_decisions: dict[str, dict[str, Any]] = {}
        self._scope_map_written = False
        self._candidate_names: dict[str, str] | None = None
        self._counter = 0
        self._job_dir: Path | None = None
        self._worker: subprocess.Popen[str] | None = None
        self._worker_log: Any | None = None
        self._worker_failed: str | None = None
        self._interpreter_verified = False

    def candidate_names(self) -> dict[str, str]:
        """Lazily read the released candidate table: ``candidate id -> image path``."""
        if self._candidate_names is None:
            names: dict[str, str] = {}
            path = (Path(self.config.WORKBENCH_CIRCUS_RELEASE_ROOT)
                    / "mbeir_fashioniq_task7_cand_pool.jsonl")
            try:
                with path.open(encoding="utf-8") as stream:
                    for line in stream:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if isinstance(row, dict) and isinstance(row.get("did"), str) \
                                and isinstance(row.get("img_path"), str):
                            names[row["did"]] = row["img_path"]
            except (OSError, ValueError):
                names = {}
            self._candidate_names = names
        return self._candidate_names

    def _decisions(self) -> dict[str, dict[str, Any]]:
        """This adapter's routing decisions, created on first use.

        Tolerant of an instance built with ``object.__new__`` (the tests' seam for routing without a
        checkpoint or a worker), so recording a decision never needs a constructor.
        """
        if not hasattr(self, "_scope_decisions"):
            self._scope_decisions = {}
        return self._scope_decisions

    def query_scope(self, query: Any,
                    candidate_names: Mapping[str, str] | None = None) -> str | None:
        """Route one released query to the checkpoint scope its own annotation names.

        The runner calls this once per query where it used to read ``query.category`` directly.
        Returns the category the release loader already assigned when there is one, otherwise the
        category whose FashionIQ validation caption pair produced this query's text. ``None`` means
        the release does not determine a scope, so the query stays unscored instead of defaulting.

        Every decision is cached with its rationale by :meth:`query_scope_resolution`, which the
        runner-side consumers reach through the run artifact this adapter writes on ``close()``.
        """
        index = self.caption_scope_index()
        names = self.candidate_names() if candidate_names is None else candidate_names
        record = resolve_query_scope(index, query, names)
        record["batch_indices"] = (getattr(query, "index", None),)
        record["candidates"] = self._scope_diagnostics(index, query, names)
        if record["scope"] is not None and record["scope"] not in self.category_scopes:
            # An out-of-scope assignment is an adapter gap, not an ambiguity: name it as such
            # instead of reporting a bare `None` under a reason that claims a scope was found.
            record["reason"] = f"unresolved:scope_outside_declared:{record['scope']}"
            record["scope"] = None
        self._decisions()[str(record["qid"])] = record
        return record["scope"]

    def caption_scope_index(self) -> CaptionScopeIndex:
        """The annotation index, built once and shared by every routing call."""
        if getattr(self, "_caption_scopes", None) is None:
            self._caption_scopes = caption_scope_index(
                self.dataset_root, getattr(self, "annotation_pin", ANNOTATION_PIN))
        return self._caption_scopes

    @staticmethod
    def _scope_diagnostics(index: CaptionScopeIndex, query: Any,
                           candidate_names: Mapping[str, str] | None) -> dict[str, Any]:
        """What each candidate category file said about this query's reference and text.

        For an unresolved query this is the per-scope evidence a reader needs: the file, its row
        count, whether it could be used at all, and whether it holds this ``(stem, text)`` pair.
        """
        stem = query_reference_stem(query, candidate_names)
        text = normalize_query_text(getattr(query, "query_txt", ""))
        holders = index.get((stem, text), frozenset()) if stem is not None else frozenset()
        files = index.report()["files"]
        diagnostics: dict[str, Any] = {}
        for category, entry in files.items():
            diagnostics[category] = {
                "file": entry["file"],
                "count": entry["count"],
                "usable": entry["usable"],
                "problem": entry["problem"],
                "holds_pair": category in holders,
            }
        diagnostics["reference_stem"] = stem
        diagnostics["normalized_text"] = text
        return diagnostics

    def query_scope_resolution(self) -> dict[str, Any]:
        """``qid -> scope`` with one rationale per query, plus the annotation provenance.

        Written to ``<run>/scope/query_scope_<model>.json`` so the inventory and the run page can
        read the routing decisions of a finished run without re-running the resolver. Reasons are
        ``loader``, ``annotation:<category>`` and ``unresolved:<why>``; ``provenance_incomplete``
        marks a decision taken while the annotation set could not prove the absence of a row.
        """
        decisions = [{"qid": qid, **{key: value for key, value in record.items()
                                     if key not in {"qid", "candidates"}},
                      "candidates": record.get("candidates", {})}
                     for qid, record in sorted(self._decisions().items())]
        counts: dict[str, int] = {}
        for record in decisions:
            key = f"{record['reason']}:{record['scope']}" if record["scope"] is not None \
                else str(record["reason"])
            counts[key] = counts.get(key, 0) + 1
        return {
            "model_id": self.model_id,
            "resolver": f"{type(self).__module__}.{type(self).__qualname__}.query_scope",
            "annotation": self.caption_scope_index().report(),
            "counts_by_reason": dict(sorted(counts.items())),
            "unresolved": [record for record in decisions if record["scope"] is None],
            "provenance_incomplete_count": sum(bool(record["provenance_incomplete"])
                                               for record in decisions),
            "query_count": len(decisions),
            "scope_by_qid": {record["qid"]: record["scope"] for record in decisions},
            "decisions": decisions,
        }

    def _write_scope_artifact(self) -> Path | None:
        """Publish this run's routing decisions beside its rank artifact, if a run root exists.

        ``work_root`` is ``<artifacts>/composition/runs/<run_id>/work/<model>``, so the run
        directory is two levels up. Nothing here may fail a run: routing provenance is a record of
        what happened, not a scoring input.
        """
        decisions = self._decisions()
        if getattr(self, "_scope_map_written", False) or not decisions \
                or getattr(self, "work_root", None) is None:
            return None
        run_root = Path(self.work_root).parent.parent
        if run_root.parent.name != "runs":
            return None
        destination = run_root / "scope" / f"query_scope_{self.model_id}.json"
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(f".{uuid.uuid4().hex}.tmp")
            temporary.write_text(json.dumps(self.query_scope_resolution(), indent=2,
                                            sort_keys=True, default=str) + "\n", encoding="utf-8")
            temporary.replace(destination)
        except OSError:
            return None
        self._scope_map_written = True
        return destination

    @classmethod
    def status(cls, config: Any | None = None) -> AdapterStatus:
        """Check each bundle member, configured interpreter, its CUDA runtime, and bundle mapping."""
        config = config or resolve_config()
        model = model_by_id(cls.model_id)
        bundle = checkpoint_by_id(model, BUNDLE_ID)
        checkpoint_ids = tuple(bundle.get("required_checkpoint_ids", ()))
        variants = {item["checkpoint_id"]: item for item in model["checkpoint_variants"]}
        present: dict[str, bool] = {}
        for checkpoint_id in checkpoint_ids:
            member = checkpoint_by_id(model, checkpoint_id)
            path = checkpoint_path(cls.model_id, member, config.WORKBENCH_CHECKPOINT_ROOT)
            present[checkpoint_id] = checkpoint_is_present(
                path, member, model_id=cls.model_id, config=config)
        variable = model["environment"]["python_env_var"]
        setting_value = setting(variable)
        interpreter = resolve_model_interpreter(model, config) or setting_value
        interpreter_present = bool(interpreter and Path(interpreter).is_file())
        cuda_present = False
        if interpreter_present:
            try:
                probe = subprocess.run(
                    [str(interpreter), "-c",
                     "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"],
                    capture_output=True, text=True, check=True)
                cuda_present = probe.stdout.strip().split()[-1] == "True"
            except (OSError, subprocess.SubprocessError, IndexError):
                pass
        mapping = bundle.get("mapping_status")
        members_mapped = all(variants[item].get("checkpoint_mapping_status") == "VERIFIED_METADATA"
                             for item in checkpoint_ids)
        mapping_present = mapping == "VERIFIED_METADATA" and members_mapped
        checks = (f"checkpoints={'passed' if all(present.values()) and present else 'failed'} "
                  f"({present}); interpreter={'passed' if interpreter_present else 'failed'} "
                  f"({variable}={'set' if setting_value else 'unset'}); "
                  f"GPU={'passed' if cuda_present else 'failed'}; "
                  f"mapping={'passed' if mapping_present else 'failed'} ({mapping})")
        reason = None
        if not checkpoint_ids or not all(present.values()):
            missing = [item for item, is_present in present.items() if not is_present]
            reason = f"LIMN checkpoint bundle incomplete or integrity check failed: {missing or BUNDLE_ID}"
        elif not interpreter_present:
            reason = f"model interpreter unavailable: {variable} is {'set but missing' if setting_value else 'unset'}"
        elif not cuda_present:
            reason = "CUDA unavailable in LIMN model interpreter; native forward requires CUDA"
        elif not mapping_present:
            reason = f"checkpoint bundle mapping unresolved: {mapping}"
        if reason is None:
            # F1(c): the presence checks above all pass on an interpreter that the execution gate
            # would still refuse (wrong Python version, missing declared dependency). Ask the gate.
            from workbench.composition.adapters import _environment_gate_reason

            reason = _environment_gate_reason(model, config)
        return AdapterStatus(
            model_id=cls.model_id, ready=reason is None, reason=reason,
            checkpoint_ids=checkpoint_ids,
            per_mode={condition: (True, None) for condition in CONDITIONS}, notes=checks)

    def _run_job(self, phase: str, image_paths: Sequence[str | Path],
                 texts: Sequence[str] = (), categories: Sequence[str | None] = ()) -> np.ndarray:
        work_root = self.require_work_root()
        if not self.interpreter or not Path(self.interpreter).is_file():
            variable = self._model_record["environment"]["python_env_var"]
            raise RuntimeError(f"model interpreter unavailable: {variable} is unset or missing")
        if not self._interpreter_verified:
            # Same gate, same code path, as every other adapter: see
            # CompositionAdapter.require_verified_interpreter.
            self.require_verified_interpreter()
            self._interpreter_verified = True
        if self._worker_failed is not None:
            raise AdapterWorkerError(self._worker_failed)
        paths = [str(Path(path).expanduser().resolve()) for path in image_paths]
        missing = [path for path in paths if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"LIMN input image missing: {missing[0]}")
        adapter_root = work_root / self.model_id
        adapter_root.mkdir(parents=True, exist_ok=True)
        if self._job_dir is None:
            self._job_dir = adapter_root / f"worker-{uuid.uuid4().hex}"
            self._job_dir.mkdir()
        self._counter += 1
        job_path = self._job_dir / f"job-{self._counter}.json"
        output_path = self._job_dir / f"output-{self._counter}.npy"
        scope = ",".join(sorted({str(value) for value in categories})) or "global"
        payload = {
            "phase": phase, "image_paths": paths, "texts": list(texts),
            "categories": list(categories), "output_path": str(output_path),
            "checkpoint_root": str(self.config.WORKBENCH_CHECKPOINT_ROOT),
            "bundle_id": self._bundle_id, "source_root": str(self.source_root),
            "dataset_root": str(self.dataset_root), "batch_size": max(1, int(self.batch_size)),
        }
        job_path.write_text(json.dumps(payload), encoding="utf-8")
        if self._worker is not None and self._worker.poll() is not None:
            status = self._worker.returncode
            self._worker_failed = (
                f"adapter_worker_exited: model=limn scope={scope} phase={phase} exit_status={status}")
            self._stop_worker()
            raise AdapterWorkerError(self._worker_failed)
        if self._worker is None:
            self._worker_error = self._job_dir / "worker.stderr.log"
            self._worker_log = self._worker_error.open("w", encoding="utf-8")
            self._worker = subprocess.Popen(
                [self.interpreter, "-m", "workbench.composition.model_limn",
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
                f"adapter_worker_exited: model=limn scope={scope} phase={phase} exit_status={status}")
            raise AdapterWorkerError(self._worker_failed)
        response = json.loads(line[len("CIRCUS_WORKER:"):])
        if not response.get("ok"):
            job_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
            self._worker_failed = (
                f"adapter_worker_job_failed: model=limn scope={scope} phase={phase} "
                f"error={response.get('error_type')}: {response.get('error')}")
            self._stop_worker()
            raise AdapterWorkerError(self._worker_failed)
        result = np.load(output_path, allow_pickle=False).astype(np.float32, copy=False)
        if result.ndim != 2 or result.shape != (len(paths), 768):
            raise AdapterWorkerError(
                f"adapter_worker_invalid_matrix: model=limn scope={scope} phase={phase} "
                f"shape={result.shape} expected=({len(paths)}, 768)")
        job_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        return result

    def encode_queries(self, images: Sequence[str | Path], texts: Sequence[str],
                       categories: Sequence[str | None]) -> np.ndarray:
        self.require_work_root()
        if len(images) != len(texts) or len(images) != len(categories):
            raise ValueError("LIMN query image, text, and category counts must match")
        invalid = [(index, category, "missing_category" if category is None else "category_outside_scopes")
                   for index, category in enumerate(categories)
                   if category not in self.category_scopes]
        if invalid:
            reasons = {index: reason for index, _category, reason in invalid}
            raise CategoryScopeError(
                "category_scope_invalid", tuple(reasons),
                f"query categories missing or outside declared scopes: {invalid!r}", reasons)
        return self._run_job("queries", images, texts, categories)

    def encode_targets(self, images: Sequence[str | Path],
                       categories: Sequence[str | None]) -> np.ndarray:
        self.require_work_root()
        if len(images) != len(categories):
            raise CategoryScopeError("target_scope_count_mismatch", (),
                                     "LIMN target image and category counts must match")
        scopes = set(categories)
        if len(scopes) != 1:
            raise CategoryScopeError("mixed_target_scopes", (),
                                     f"LIMN target call requires one repeated scope; got {scopes!r}")
        category = next(iter(scopes))
        if category not in self.category_scopes:
            reason = "missing_category" if category is None else "category_outside_scopes"
            raise CategoryScopeError(reason, (), f"LIMN target scope {category!r} is not declared")
        return self._run_job("targets", images, categories=categories)

    def score(self, queries: Any, targets: Any,
              categories: Sequence[str | None] | None = None) -> np.ndarray:
        """Controller-side chunked normalized-dot scores in each query's matching space."""
        if not isinstance(targets, Mapping):
            raise ExactRankUnavailable("LIMN score requires scope-to-full-gallery matrices")
        if categories is None or len(categories) != len(queries):
            raise CategoryScopeError("missing_query_categories", tuple(range(len(queries))),
                                     "LIMN score requires one query scope per query")
        invalid = [(index, category, "missing_category" if category is None else "category_outside_scopes")
                   for index, category in enumerate(categories)
                   if category not in self.category_scopes or category not in targets]
        if invalid:
            reasons = {index: reason for index, _category, reason in invalid}
            raise CategoryScopeError(
                "category_scope_invalid", tuple(reasons),
                f"LIMN score query scopes lack matching gallery matrices: {invalid!r}", reasons)
        gallery_sizes = {np.shape(matrix)[0] for matrix in targets.values()}
        if len(gallery_sizes) != 1:
            raise CategoryScopeError("inconsistent_gallery_sizes", (),
                                     "LIMN scope matrices must each contain the same full gallery")
        gallery_size = next(iter(gallery_sizes), 0)
        scores = np.empty((len(queries), gallery_size), dtype=np.float32)
        for row, category in enumerate(categories):
            query = np.asarray(queries[row], dtype=np.float32)
            query_norm = np.linalg.norm(query)
            if query_norm:
                query = query / query_norm
            gallery = targets[category]
            for start in range(0, gallery_size, 8192):
                stop = min(start + 8192, gallery_size)
                scores[row, start:stop] = query @ np.asarray(gallery[start:stop]).T
        return scores

    def rank(self, queries: Any, targets: Any,
             target_positions: Sequence[Sequence[int] | None] | None = None,
             categories: Sequence[str | None] | None = None) -> list[int | None]:
        if target_positions is None:
            raise ExactRankUnavailable("LIMN exact rank requires positive target positions")
        scores = self.score(queries, targets, categories)
        if len(target_positions) != len(scores):
            raise ValueError("target position count does not match query count")
        ranks: list[int | None] = []
        for row, positions in zip(scores, target_positions):
            if not positions or not np.isfinite(row).all():
                ranks.append(None)
                continue
            valid = [position for position in positions if 0 <= position < row.size]
            if not valid:
                ranks.append(None)
                continue
            ranks.append(min(1 + int(np.count_nonzero(row > row[index]))
                             + int(np.count_nonzero(row[:index] == row[index])) for index in valid))
        return ranks

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
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
            self._worker = None
        if self._worker_log is not None:
            self._worker_log.close()
            self._worker_log = None

    def close(self) -> None:
        # Before any cleanup: the scope artifact lives beside the run, not in `work_root`, so the
        # runner's rmtree of the work directory cannot take it with it.
        self._write_scope_artifact()
        if self._worker is not None:
            if self._worker.stdin is not None:
                try:
                    self._worker.stdin.write(json.dumps({"stop": True}) + "\n")
                    self._worker.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
            try:
                self._worker.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._worker.terminate()
                try:
                    self._worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._worker.kill()
                    self._worker.wait()
            self._worker = None
        if self._worker_log is not None:
            self._worker_log.close()
            self._worker_log = None
        if self._job_dir is not None:
            # A latched failure leaves the job directory for the runner's preservation copy of
            # worker.stderr.log; the runner's rmtree removes it. Only a clean close deletes here.
            if self._worker_failed is None:
                shutil.rmtree(self._job_dir, ignore_errors=True)
            self._job_dir = None

    def restart(self) -> None:
        """Kill worker, unload cached scope models, and reload on next encode call."""
        self.close()

def _worker_main(job_path: Path) -> None:
    """Cache one verified LIMN whole model per scope while serving path-based jobs."""
    import importlib
    import sys

    import numpy as np
    import torch
    import open_clip
    from PIL import Image

    from workbench.backend.registry import checkpoint_by_id, checkpoint_path, model_by_id
    from workbench.replay.common import load_trusted_whole_model

    limit_torch_threads()
    workers = resolved_decode_workers()

    def reply(ok: bool, **values: Any) -> None:
        print("CIRCUS_WORKER:" + json.dumps({"ok": ok, **values}), flush=True)

    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
        model_record = model_by_id("limn")
        bundle = checkpoint_by_id(model_record, job["bundle_id"])
        members = {member["checkpoint_id"]: member for member in bundle.get("bundle_members", ())}
        source_path = Path(job["source_root"]) / "LIMN"
        if str(source_path) not in sys.path:
            sys.path.insert(0, str(source_path))
        native_model = importlib.import_module("model")
        if Path(native_model.__file__).resolve() != (source_path / "model.py").resolve():
            raise ImportError(f"refusing non-pinned LIMN model module: {native_model.__file__}")
    except Exception as error:
        reply(False, error_type=type(error).__name__, error=str(error))
        return

    models: dict[str, Any] = {}
    transforms: dict[str, Any] = {}
    dataset_ids = {"dress": 0, "shirt": 1, "toptee": 2}

    def load_scope(category: str, current: dict[str, Any]) -> None:
        if category in models:
            return
        checkpoint_id = f"base_iter0_{category}"
        checkpoint = checkpoint_by_id(model_record, checkpoint_id)
        member = members[checkpoint_id]
        member_path = checkpoint_path("limn", checkpoint, Path(current["checkpoint_root"]))
        model = load_trusted_whole_model(member_path, checkpoint["expected_sha256"], torch)
        if not isinstance(model, native_model.LIMN):
            raise TypeError(f"LIMN {category} checkpoint does not deserialize to pinned model.LIMN")
        model.cuda().eval()
        visual = model.backbone.clip.visual
        models[category] = model
        transforms[category] = open_clip.image_transform(
            visual.image_size, is_train=False, mean=visual.image_mean, std=visual.image_std)

    def run_one(current: dict[str, Any]) -> None:
        try:
            paths, texts, categories = current["image_paths"], current["texts"], current["categories"]
            for category in dict.fromkeys(categories):
                load_scope(category, current)
            batch_size = max(1, int(current["batch_size"]))
            outputs: list[np.ndarray | None] = [None] * len(paths)
            start = 0
            while start < len(paths):
                category = categories[start]
                stop = start + 1
                while stop < len(paths) and categories[stop] == category:
                    stop += 1
                model = models[category]
                category_transform = transforms[category]

                def decode(chunk: list[str], _transform: Any = category_transform) -> Any:
                    images = []
                    for path in chunk:
                        with Image.open(path) as image:
                            images.append(_transform(image.copy()))
                    return torch.stack(images)

                spans = [(batch_start, min(batch_start + batch_size, stop))
                         for batch_start in range(start, stop, batch_size)]
                # Same decode pool as the rest of the family. LIMN's per-category open_clip
                # transform is captured per group and used unchanged; only the serial
                # `Image.open` + transform loop is parallelised.
                for (batch_start, batch_stop), batch in zip(
                        spans, prefetch_rows(decode, [paths[a:b] for a, b in spans], workers)):
                    batch = batch.float().cuda()
                    dataset_type = torch.ones(batch_stop - batch_start).cuda() * dataset_ids[category]
                    with torch.no_grad():
                        if current["phase"] == "queries":
                            encoded = model.compose_query(
                                batch, texts[batch_start:batch_stop], dataset_type)[0]
                        else:
                            encoded = model.compose_target(batch, dataset_type)[0]
                    values = encoded.detach().float().cpu().numpy()
                    for offset, value in enumerate(values):
                        outputs[batch_start + offset] = value
                start = stop
            result = (np.stack(outputs).astype(np.float32, copy=False) if outputs
                      else np.empty((0, 768), dtype=np.float32))
            np.save(current["output_path"], result, allow_pickle=False)
            reply(True)
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
        raise SystemExit("usage: python -m workbench.composition.model_limn --worker-job JOB.json")
    _worker_main(Path(sys.argv[2]))


ADAPTER = Adapter
