from __future__ import annotations

import logging
import os

import csv
import io
import json
import sys
from datetime import datetime, timezone
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from workbench.backend.analysis import analysis_rows_from_maps, failure_jaccard_from_maps
from workbench.backend import runtime
from workbench.backend.errors import WorkbenchError
from workbench.backend.index import database_path, rebuild_index
from workbench.backend.jobs import DuplicateJobError, QueueShutdownError, SingleGpuQueue
from workbench.backend.operator_config import resolve_config
from workbench.backend.registry import checkpoint_availability, load_registry, sha256_file
from workbench.backend.schemas.results import Annotation, SavedCohort
from workbench.backend.storage import cohort_metrics, get_analysis_input, get_query, list_runs, query_page, validate_analysis_selection
from workbench.scripts.evaluate_models import guarded_plan

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
ANNOTATIONS_PATH = resolve_config().WORKBENCH_ARTIFACTS_ROOT / "annotations.json"
COHORTS_PATH = resolve_config().WORKBENCH_ARTIFACTS_ROOT / "cohorts.json"
LABELS = ["preservation_failure", "under_edit", "over_edit", "partial_edit", "wrong_attribute", "wrong_entity_binding", "reference_dominance", "text_dominance", "fine_grained_visual_confusion", "ambiguous_ground_truth", "possible_dataset_issue", "duplicate_or_near_duplicate", "other", "unknown"]
app = FastAPI(title="CIR Failure Analysis Workbench")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])
JOBS = SingleGpuQueue()

def _recover_interrupted_jobs() -> None:
    """A restart must never resurrect a phantom `running` job: records persisted by a
    previous process that died mid-flight become truthful `failed` entries at startup."""
    try:
        recovered = SingleGpuQueue.recover_interrupted(
            [resolve_config().WORKBENCH_ARTIFACTS_ROOT / "queue_jobs"])
        for record in recovered:
            logging.getLogger("workbench").warning(
                "recovered interrupted job %s (%s) as failed",
                record.get("job_id"), record.get("model_id"))
            # Surface the recovered record through the status API: an operator who missed
            # the startup log must still see the failed job in /api/jobs instead of an
            # empty queue.
            JOBS.adopt_recovered(record)
    except Exception:
        # Recovery must never block startup; the on-disk records stay available for
        # inspection.
        pass

_index_thread: threading.Thread | None = None
_index_build_error: Exception | None = None

def _ensure_index() -> None:
    """Ensure DuckDB index exists on startup, building it asynchronously if absent."""
    global _index_thread, _index_build_error
    config = resolve_config()
    db_path = database_path(config)

    # An existing index is left untouched: do not re-index on every boot to avoid
    # blocking startup on large result sets and clobbering warm query state.
    # Operators can trigger an explicit rebuild via POST /api/index/rebuild.
    if db_path.is_file():
        return

    if _index_thread is not None and _index_thread.is_alive():
        return

    results_dir = getattr(config, "WORKBENCH_RESULTS_ROOT", config.WORKBENCH_ARTIFACTS_ROOT / "results")

    def _build() -> None:
        global _index_build_error
        try:
            rebuild_index(results_root=results_dir, database_path=db_path)
        except Exception as exc:
            logging.getLogger("workbench").exception("Startup index build failed: %s", exc)
            _index_build_error = exc

    _index_build_error = None
    _index_thread = threading.Thread(target=_build, name="workbench-startup-index", daemon=True)
    _index_thread.start()

def wait_for_startup_index(timeout: float | None = 10.0) -> None:
    if _index_thread is not None and _index_thread.is_alive():
        _index_thread.join(timeout=timeout)

@app.on_event("startup")
def _startup() -> None:
    # Resolved when the server actually starts, not at import: a test importing this module
    # with a different artifacts root must not key recovery to the import-time root.
    _recover_interrupted_jobs()
    _ensure_index()

@app.on_event("shutdown")
def _shutdown() -> None:
    """Stop every live job's process group before the process exits.

    A job leads its own session (`start_new_session=True`), so the SIGTERM that `docker stop`
    sends to the backend reaches this process alone: without this hook the operator's evaluator
    child — the process that actually holds the GPU — survives the container and keeps holding
    the device. The queue persists an honest terminal status for each job it stops; if the
    backend is killed before the handshake runs, startup recovery still marks the record failed.
    """
    if _index_thread is not None and _index_thread.is_alive():
        _index_thread.join(timeout=2.0)
    survivors = JOBS.shutdown()
    if survivors:
        logging.getLogger("workbench").warning(
            "job process group(s) survived shutdown: %s", ", ".join(survivors))


def _job_launch_guard(job: object) -> None:
    """Refuse to dispatch a queued evaluation whose runtime is no longer launch-ready.

    The request handler checks readiness when the job is accepted, but the environment can
    change while a job waits (a GPU capability is invalidated by an interpreter or
    dependency change). The consumer therefore re-checks immediately before dispatch.
    """
    model_id = getattr(job, "model_id", None)
    if not model_id:
        return
    model = next((item for item in load_registry()["models"] if item["model_id"] == model_id), None)
    if model is None:
        # A model that is no longer registered cannot be shown to be ready: refuse it.
        raise RuntimeError(f"model is not registered: {model_id}")
    runtime.require_launch_ready(model)


JOBS.launch_guard = _job_launch_guard


@app.exception_handler(WorkbenchError)
async def workbench_error(_: Request, error: WorkbenchError) -> JSONResponse:
    # A missing serving index is the server's own state, not a bad request: 503 tells the
    # operator to rebuild, while 400 would blame the caller. Operator-input errors stay 400.
    status = 503 if error.code == "index_unavailable" else 400
    return JSONResponse(status_code=status, content=error.payload())


def read_json(path: Path) -> list[dict]:
    return json.loads(path.read_text()) if path.exists() else []


def write_json(path: Path, payload: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


@app.get("/api/models")
def models():
    registry = load_registry()
    out = []
    for model in registry["models"]:
        entry = {**model, "checkpoints": [checkpoint_availability(model, item) for item in model["checkpoint_variants"]]}
        # Bundles are first-class selections (LIMN needs all three category checkpoints for one
        # complete macro result), so a launcher or the UI must be able to see and pick them.
        bundles = model.get("checkpoint_bundles") or []
        if bundles:
            by_id = {c["checkpoint_id"]: c for c in model["checkpoint_variants"]}
            entry["checkpoint_bundles"] = [
                {**bundle,
                 "members": [checkpoint_availability(model, by_id[member])
                             for member in bundle.get("required_checkpoint_ids", []) if member in by_id]}
                for bundle in bundles
            ]
        out.append(entry)
    return out


def _check_index_ready(db_path: Path) -> None:
    if _index_thread is not None and _index_thread.is_alive():
        raise WorkbenchError(
            "index_unavailable",
            "DuckDB index is currently building at startup; please retry shortly.",
            {"database_path": str(db_path), "status": "building"},
        )
    if _index_build_error is not None and not db_path.is_file():
        cause = type(_index_build_error).__name__
        msg = f"Startup index build failed: {_index_build_error}"
        details: dict[str, Any] = {"database_path": str(db_path), "error": str(_index_build_error), "cause": cause}
        if isinstance(_index_build_error, WorkbenchError):
            details["inner_details"] = _index_build_error.details
        raise WorkbenchError("index_unavailable", msg, details)

@app.get("/api/runs")
def runs(protocol_id: str | None = None):
    config = resolve_config()
    db_path = database_path(config)
    _check_index_ready(db_path)
    return list_runs(protocol_id, database_path=db_path)

def _artifact_root(config) -> Path:
    """The artifacts root, resolving the fallback exactly as `evaluate_models.artifact_root`."""
    from workbench.scripts.evaluate_models import artifact_root

    return Path(artifact_root(config))

def _development_process() -> bool:
    """Whether this process is a development one, i.e. allowed to relax proof validation.

    Development is *explicit opt-in state*, never the absence of a deployment flag: the deployment
    itself sets neither variable, so treating "neither is set" as development is what made every
    deployed report uncertifiable. Either variable, when set to a non-empty value, says the
    operator declared this process a development one.
    """
    return any(os.environ.get(flag) for flag in ("PYTEST_CURRENT_TEST", "WORKBENCH_DEVELOPMENT_MODE"))

def _development_opt_in() -> bool:
    """``WORKBENCH_ALLOW_DEVELOPMENT_CERTIFICATES=1``: the operator's explicit development opt-in.

    It relaxes proof validation *in a development process only*; see ``_certification_mode``.
    """
    return str(os.environ.get("WORKBENCH_ALLOW_DEVELOPMENT_CERTIFICATES", "")).strip() == "1"

def _certification_mode(proof_exists: bool) -> str:
    """Which certification policy this process must apply to *this* report.

    * ``"deployment"`` -- the report has a completion proof, so it is certified through the
      authoritative verifier, in full (checksum and environment identity included). This is the
      real deployment case and the only one that may end in ``certified=true`` on a proof.
    * ``"development"`` -- a development process (pytest or ``WORKBENCH_DEVELOPMENT_MODE``) whose
      operator opted in with ``WORKBENCH_ALLOW_DEVELOPMENT_CERTIFICATES=1`` and whose report has no
      usable proof: a synthetic no-op evaluator writes no environment identity, so proof-level
      validation is impossible in principle, not merely slow. The receipt is ledger level and says
      so.
    * ``"proof_unavailable"`` -- neither applies: a completed, ledger-corroborated report whose
      completion proof is absent / pruned / lost. The execution is authenticated, but without
      the proof's environment identity it cannot be certified research evidence.
    """
    if proof_exists:
        return "deployment"
    if _development_process() and _development_opt_in():
        return "development"
    return "proof_unavailable"

def _no_proof_refusal() -> str:
    """Why a report with no usable completion proof cannot be certified here, by name."""
    if _development_process():
        return ("this process is in development and no completion proof exists for this report, so "
                "the authoritative verifier cannot validate it; set "
                "WORKBENCH_ALLOW_DEVELOPMENT_CERTIFICATES=1 to certify at ledger level in "
                "development only")
    return ("this report has no completion proof, so the authoritative verifier cannot validate "
            "its environment identity here; a deployment certifies only reports it produced "
            "itself, and WORKBENCH_ALLOW_DEVELOPMENT_CERTIFICATES=1 is honoured in a development "
            "process only")

#: Receipts are cached by (report digest, certification mode, ledger stamp, proof stamp,
#: checkpoint stamp, run-artifacts stamp). The key is the whole soundness argument of the cache, so it is a key and not a
#: convenience: the report digest fixes which bytes were verified, the mode which policy judged them,
#: the ledger stamp which ledger corroborated them, the proof stamp whether an evaluator-written
#: proof exists and which revision of it was validated, and the checkpoint stamp which revision of
#: the *artifact the certificate is about* those bytes describe. Neither the ledger read nor the
#: manifest validation (which re-hashes the checkpoint's multi-GB members) therefore runs on every
#: dashboard request -- only when a report, the process's certification mode, the ledger, a proof, or
#: the checkpoint artifact itself actually changed. Refusals are cached too: avoiding that cost per
#: request is the point, and every input that could turn a refusal into a certification is in the
#: key -- the mode included, so a development receipt is never re-served for a report that later
#: gained a proof (the proof stamp changes, and the mode with it).
#:
#: The checkpoint stamp is the one input that is neither authenticated nor cheap to hash, and
#: omitting it was a real defect: `verify_run` re-hashes the run's declared bundle members and
#: refuses a report whose members no longer match the manifest its proof recorded, so a cached
#: receipt whose key ignored the members kept serving `certified=true` for weights a later edit
#: replaced. See `_checkpoint_stamp` for why the token it contributes is sound.
_CERTIFICATION_CACHE: dict[tuple[str, str, str, str, str, str], dict] = {}

#: How many receipts the cache keeps. The key varies with the checkpoint's change token, so a
#: long-lived process watching an artifact be rewritten could otherwise grow this dict without
#: bound. Bounded and FIFO: a receipt is cheap to rebuild, and the entry evicted is a refusal at
#: worst, which the next request recomputes.
_CERTIFICATION_CACHE_LIMIT = 256

def _reports_root(config) -> Path:
    """Where the evaluator writes aggregate reports and completion proofs."""
    from workbench.scripts.evaluate_models import reports_root

    return reports_root(config)

def _proof_stamp(config, report: dict) -> str:
    """A change token for the completion proof `verify_run.verify` would validate, if any."""
    try:
        proof = (_reports_root(config) / (
            f"{report.get('model_id')}_{report.get('checkpoint_id')}_"
            f"{report.get('run_id')}_completion.json"))
        stat = proof.stat()
    except OSError:
        return "absent"
    # The proof's CONTENT, for the same reason the checkpoint and run-artifact tokens use content:
    # `size:mtime` is writable by whoever can edit the file, so a one-character edit with mtime
    # restored left this stamp frozen and a tampered proof kept serving `certified=true` while a
    # fresh verification refused it. The proof is small; hashing it is not a cost question.
    digest = _member_digest_cached(proof, stat)
    return f"absent-unreadable" if digest is None else digest

def _run_artifacts_stamp(config, report: dict) -> str:
    """A change token for the run's invocation record and logs, which the verifier also re-derives.

    `validate_run_manifest` re-reads `run_directory/command.json` (its independent witness that the
    execution really succeeded, and the file it cross-checks the proof's model/checkpoint/protocol
    against) and re-hashes `stdout.log` + `stderr.log` against the proof's `evidence_digest`. A
    receipt cache that ignored them served `certified=true` for a state the verifier refuses --
    repointed to another model's invocation record, or a log edited so the evidence digest no longer
    agrees -- because the *weights* were untouched.

    Content digests again, for the same reason as the checkpoint token: size and mtime are both
    writable by whoever can edit the file. These are small files, so hashing them is not the cost
    the checkpoint token has to avoid.
    """
    try:
        artifacts = getattr(config, "WORKBENCH_ARTIFACTS_ROOT", None)
        if artifacts is None:
            return "unresolved"
        run_dir = (Path(artifacts) / "logs"
                   / f"{report.get('model_id')}__{report.get('checkpoint_id')}__"
                     f"{report.get('run_id')}")
        names = ("command.json", "stdout.log", "stderr.log")
        tokens = []
        for name in names:
            path = run_dir / name
            try:
                stat = path.stat()
            except OSError:
                return f"missing:{name}"
            digest = _member_digest_cached(path, stat)
            if digest is None:
                return f"unreadable:{name}"
            tokens.append(f"{name}:{digest}")
        return "|".join(tokens)
    except Exception:
        return "unresolved"

def _ledger_path(config) -> Path:
    """The one ledger file both the evaluator and this process read."""
    from workbench.scripts.evaluate_models import evidence_ledger_path

    return evidence_ledger_path(config)

def _checkpoint_artifact(model_id: str, checkpoint_id: str) -> tuple[Path, list[str]] | None:
    """The directory an artifact's members live in, and their filenames. None when unresolved.

    The shape differs per artifact, and getting it wrong is not cosmetic -- both errors below were
    found by attack:

    * a **file** artifact's member is the file itself, living in its parent directory. Ignoring the
      `filename` field entirely made every single-file checkpoint (csmcir's `fashioniq`, LIMN's
      per-category `base_iter0_dress`) hash to the literal token `"unresolved"`, so its receipt
      cache key never moved and a rewritten weight kept serving a certificate.
    * a **directory** artifact's members live *inside* the directory, so the root is the checkpoint
      path itself. Using `.parent` made a run-directory checkpoint look for `config.json` beside
      the directory instead of inside it, so it always reported a missing member and never
      re-certified.

    Memoized on the pair *and the checkpoint root*: resolving it parses the registry YAML
    (~100 ms measured), and `/api/benchmark-runs` asks once per report. The root is part of the
    key because it comes from the operator's configuration -- a process whose configuration is
    redirected (a test, a second deployment in one interpreter) must not be served another root's
    paths, which is the same module-global leak that once made the environment planner answer from
    a previous test's cache. A deployment does not change its root while running; one that does
    restarts.
    """
    # The module-level `resolve_config`, not a fresh import of it: this module's configuration is
    # a seam its consumers (and the tests) patch, and re-importing would read the real deployment's
    # paths while every other path in this file honoured the patch.
    try:
        root = resolve_config().WORKBENCH_CHECKPOINT_ROOT
    except Exception:
        root = None
    key = (model_id, checkpoint_id, str(root))
    if key in _CHECKPOINT_ARTIFACTS:
        return _CHECKPOINT_ARTIFACTS[key]
    resolved_pair: tuple[Path, list[str]] | None = None
    try:
        from workbench.backend.registry import checkpoint_by_id, checkpoint_path, model_by_id

        model = model_by_id(model_id)
        variant = checkpoint_by_id(model, checkpoint_id)
        # `root` is this module's resolved checkpoint root (the patched seam); pass it explicitly
        # so the registry never falls back to its own resolution.
        path = checkpoint_path(model_id, variant, root)
        members = [str(name) for name in (variant.get("required_files") or [])]
        if not members:
            # A single-file artifact's member IS the file.
            if variant.get("filename"):
                resolved_pair = (path.parent, [str(variant["filename"])])
        else:
            # `checkpoint_path` already resolves a bundle to the directory its members share and a
            # directory artifact to the directory it names, so the members' root is that path --
            # whether or not it is installed yet, and no matter how its filename is spelled.
            resolved_pair = (path, members)
    except Exception:
        resolved_pair = None
    while len(_CHECKPOINT_ARTIFACTS) >= _CHECKPOINT_ARTIFACTS_LIMIT:
        del _CHECKPOINT_ARTIFACTS[next(iter(_CHECKPOINT_ARTIFACTS))]
    _CHECKPOINT_ARTIFACTS[key] = resolved_pair
    return resolved_pair

#: Memoized `_checkpoint_artifact` answers, bounded like the other caches.
_CHECKPOINT_ARTIFACTS: dict[tuple[str, str], tuple[Path, list[str]] | None] = {}
_CHECKPOINT_ARTIFACTS_LIMIT = 1024

def _checkpoint_stamp(report: dict) -> str:
    """A change token for the checkpoint artifact this report's certificate is *about*.

    The receipt answers "is this valid research evidence", and one of the things
    `validate_run_manifest` checks to answer it is that the run's declared bundle members still
    hash to the manifest its proof recorded. A receipt that ignored the members therefore stayed
    valid across an edit that the verifier refuses, which is a *false certificate* over weights
    nobody re-hashed -- reproduced before this existed: tamper one member, leave the report, the
    ledger and the proof untouched, and `/api/benchmark-runs` kept serving `certified=true` until
    the cache was cleared by hand.

    The first fix keyed the cache on each member's size and mtime, which is NOT sufficient: an
    in-place rewrite of the same length followed by `os.utime` moves neither field, and that attack
    still served a certificate (attacked and reproduced against that version). Size and mtime are
    both forgeable by the writer, so neither may decide a security question.

    The token is therefore the members' **content** digest, which is exactly the fact the verifier
    acts on, and the cost that made the first token size+mtime is paid only when a cheap,
    un-forgeable change detector moves. That detector is the inode change time (`st_ctime_ns`),
    which no userspace call can set: `os.utime` moves mtime and atime, while
    `os.chmod`/`os.chown`/`write` move ctime *forward*, so every content change is caught while an
    untouched file is not re-hashed per request. The digest is memoized per
    `(name, st_ctime_ns, st_size, st_ino)` -- a stale entry is impossible because the key cannot
    repeat without the file having been restored to an identical state, and the digest is the
    authority either way.

    A change that moves ctime without changing bytes (a chmod) costs one re-hash and then
    re-certifies identically: a wasted recomputation, never a wrong verdict.
    """
    resolved_pair = _checkpoint_artifact(str(report.get("model_id") or ""),
                                        str(report.get("checkpoint_id") or ""))
    if resolved_pair is None:
        return "unresolved"
    root, names = resolved_pair
    tokens = []
    for name in names:
        member = root / name
        try:
            stat = member.stat()
        except OSError:
            return f"missing:{name}"
        digest = _member_digest_cached(member, stat)
        if digest is None:
            return f"unreadable:{name}"
        tokens.append(f"{name}:{digest}")
    return "|".join(tokens)

#: Memoized content digests, keyed on the inode state that cannot repeat without the file being
#: in an identical state. Bounded like the receipt cache: an evicted entry costs one re-hash.
_MEMBER_DIGESTS: dict[tuple[str, int, int, int, int], str] = {}
_MEMBER_DIGESTS_LIMIT = 4096

def _member_digest_cached(path: Path, stat) -> str | None:
    """This member's content digest, re-read only when the inode's own state moved.

    The hasher is resolved from `workbench.backend.registry` at call time rather than through this
    module's own `sha256_file` binding: the token exists to predict what the *verifier* will
    compute, and the verifier hashes through the registry. This module's name is a seam other
    concerns patch -- the HTTP-layer tests replace it with a constant so fixtures are not hashed --
    and a security token reading a stubbed hasher computes a constant for every input, which is
    precisely how this fix first failed its own regression test.
    """
    # ctime changes when the file changes; mtime is in the key too so the one case ctime cannot
    # witness (a restore that rewinds mtime *and* reuses the same ctime and size) still misses.
    key = (str(path), stat.st_ctime_ns, stat.st_mtime_ns, stat.st_size, stat.st_ino)
    cached = _MEMBER_DIGESTS.get(key)
    if cached is not None:
        return cached
    try:
        from workbench.backend.registry import sha256_file as authoritative_sha256_file

        digest = authoritative_sha256_file(path)
    except (OSError, ValueError, ImportError):
        return None
    while len(_MEMBER_DIGESTS) >= _MEMBER_DIGESTS_LIMIT:
        del _MEMBER_DIGESTS[next(iter(_MEMBER_DIGESTS))]
    _MEMBER_DIGESTS[key] = digest
    return digest

def _ledger_stamp(config) -> str:
    """A change token for the ledger file: mtime plus size.

    The receipt cache is sound only because a *rewritten* ledger cannot pass
    ``read_evidence_ledger`` again (its HMAC chain stops verifying), so an in-place edit can only
    ever shrink the set of reports that are corroborated. A new entry is an append, and an append
    always moves both mtime and size. Either moving invalidates the cache; nothing else can turn a
    previously refused report into an acceptable one.
    """
    ledger = _ledger_path(config)
    try:
        stat = ledger.stat()
    except OSError:
        return "absent"
    # Content again. The ledger is append-only in normal operation, so its size and mtime do move
    # for every legitimate change -- but a REWRITE that preserves both (a truncation and rewrite of
    # equal length, with mtime restored) would leave this stamp frozen while `read_evidence_ledger`
    # answered differently. The digest is the fact; it is also what makes the cache key's claim
    # ("the ledger stamp which ledger corroborated them") true rather than assumed.
    digest = _member_digest_cached(ledger, stat)
    return "absent-unreadable" if digest is None else digest

def _verifier_verdict(config, report_path: Path, report: dict) -> dict | None:
    """The authoritative verdict for one report, or None when the verifier cannot resolve it.

    ``verify_run.verify`` is imported lazily so a backend that never serves a report pays nothing,
    and it is asked the *same* question ``pc.sh`` and the job queue ask. Nothing is re-implemented
    here: the verdict is the verifier's, verbatim.
    """
    from workbench.scripts import verify_run

    original_resolve_config = verify_run.resolve_config
    if config is not None:
        # The verifier resolves its own config from the process environment; the request already
        # has one, and handing it the same object is what keeps a deployment (or a test) verifying
        # the reports it is actually serving.
        verify_run.resolve_config = lambda: config
    try:
        return verify_run.verify(str(report.get("run_id") or ""), str(report.get("model_id") or ""),
                                 str(report.get("checkpoint_id") or ""),
                                 str(report.get("protocol_id") or ""))
    except Exception as error:                 # a verifier that cannot answer is not a pass
        print(f"workbench: could not verify {report_path}: {error}", file=sys.stderr)
        return None
    finally:
        verify_run.resolve_config = original_resolve_config

def _development_certification(config, report_path: Path, report: dict, digest: str) -> dict:
    """A ledger-level receipt for a development process, never for a deployment.

    Only the *execution* question is relaxed: the report bytes are the bytes the ledger records,
    under the identity it records, from a completed invocation -- all checked by the caller before
    this runs. The two research questions are answered by the verifier's own rules, called
    directly: ``metric_evidence_reasons`` (its extraction/category/macro contract) and
    ``_evidence_kind_reasons`` (its synthetic / not-an-experiment rule). What cannot run on this
    host is ``validate_run_manifest``, because there is no evaluator-written proof to validate; the
    receipt says so under ``development_mode`` and the fact is printed on stderr on every request,
    so it is never invisible to an operator or an auditor.
    """
    from workbench.scripts import verify_run

    metric_reasons, _ = verify_run.metric_evidence_reasons(
        str(report.get("model_id") or ""), str(report.get("checkpoint_id") or ""),
        str(report.get("protocol_id") or ""), report)
    reasons = [*metric_reasons, *verify_run._evidence_kind_reasons(report)]
    print(f"workbench: certifying {report_path} at ledger level under a development certificate "
          f"(report digest {digest}, run {report.get('run_id')!r}; the completion proof was not "
          "validated here)", file=sys.stderr)
    # `certified` answers the same question as `is_research_evidence` in every mode: the receipt is
    # cached by digest, so a receipt that claimed certification regardless of its own reasons would
    # hand every later request a `certified=true` for a report the verifier's rules refused.
    return {"report_digest": digest, "verified": not reasons, "is_research_evidence": not reasons,
            "reasons": [], "evidence_reasons": reasons, "certified": not reasons,
            "certification_mode": "development", "development_mode": True}

def _certify_report(config, report_path: Path, report: dict, digest: str, *,
                    mode: str | None = None) -> dict:
    """What the API owes a consumer about one report: four separate questions.

    (1) an aggregate artifact exists -- decided by the caller: a report on disk is served either
    way, because a historical or refused run is still a record of something that happened;
    (2) its execution is authenticated -- the report's bytes are the bytes the HMAC ledger records,
    under the identity it records, from a completed invocation (all checked before this runs);
    (3) it is VALID RESEARCH EVIDENCE -- the authoritative verifier's ``is_research_evidence``: the
    completion proof validates, the metric extraction really happened, the report's own numbers are
    usable, and its macro agrees with its own categories;
    (4) its paper-parity verdict -- ``parity_status`` and ``benchmark_gate``, served as before.

    ``mode`` is the policy ``_certification_mode`` already chose for this report; the caller passes
    it so the cache key and the receipt can never disagree about which policy judged the report.

    Only (3) is ever presented as certified evidence. A refused report keeps (1), (2) and (4) and
    carries ``certified=false`` with the verifier's own reasons, so a consumer can tell "an
    artifact exists" apart from "valid evidence" without reading English.
    """
    if mode is None:
        mode = _certification_mode(_proof_stamp(config, report) != "absent")
    if mode == "deployment":
        # The proof exists: the verifier is authoritative, in full, including the checksum and
        # environment identity of this run's own proof. A development opt-in cannot skip this: the
        # mode is chosen by the report's proof, not by the process's environment.
        verdict = _verifier_verdict(config, report_path, report)
        if verdict is None:
            return {"verified": False, "is_research_evidence": False, "reasons": [],
                    "evidence_reasons": ["the authoritative verifier could not resolve this run"],
                    "certified": False, "certification_mode": mode, "development_mode": False,
                    "state": "refused"}
        certified = bool(verdict.get("is_research_evidence"))
        # The receipt must carry the fields a consumer needs to attribute a per-query artifact to
        # THIS verification: the artifact digest the verifier's report records, plus the identity it
        # describes. Dropping them made `promote` unable to bind, so no per-query run could ever be
        # verified in production (found in review).
        return {"report_digest": digest,
                "per_query_artifact_digest": verdict.get("per_query_artifact_digest"),
                "model_id": verdict.get("model_id"),
                "checkpoint_id": verdict.get("checkpoint_id"),
                "verified": bool(verdict.get("verified")),
                "is_research_evidence": certified,
                "reasons": list(verdict.get("reasons") or []),
                "evidence_reasons": list(verdict.get("evidence_reasons") or []),
                "certified": certified,
                "certification_mode": mode,
                "development_mode": False,
                "state": "certified" if certified else "refused"}
    if mode == "development":
        receipt = _development_certification(config, report_path, report, digest)
        receipt["state"] = "certified" if receipt.get("certified") else "refused"
        return receipt
    return {"verified": False, "is_research_evidence": False, "reasons": [],
            "evidence_reasons": [_no_proof_refusal()], "certified": False,
            "certification_mode": mode, "development_mode": False,
            "state": "proof_unavailable" if mode == "proof_unavailable" else "refused"}

def _checkpoint_digest_or_refuse(path: Path, payload) -> str:
    """This submission's checkpoint digest, or a named refusal for an artifact it cannot frame."""
    try:
        return sha256_file(path)
    except (OSError, ValueError) as error:
        raise WorkbenchError(
            "checkpoint_unreadable",
            f"the checkpoint for this submission cannot be hashed: {error}",
            {"model_id": payload.model_id, "checkpoint_id": payload.checkpoint_id,
             "checkpoint_path": str(path)}) from error

def _aggregate_reports() -> list[dict]:
    """Read every official aggregate report the evaluator has written, ignoring malformed files.

    These reports live outside the DuckDB index by design: the index holds per-query `ResultRun`
    records, while an aggregate report can exist for a run whose per-query export was not saved.
    Reading them at request time keeps the API in sync with new evaluator output without a
    re-index.

    Serving one is not the same as certifying it. The report's *evidence* verdict is the
    authoritative verifier's (``workbench/scripts/verify_run.py``), attached to each report as a
    verification receipt: the same completion-proof, extraction, category, macro and synthetic
    contract ``pc.sh`` and the job queue enforce, so the API can no longer serve as research
    evidence a report the verifier would refuse. Only the cheap parts run per request -- a digest,
    the ledger read, and the receipt cache lookup; proof validation happens once per report per
    ledger revision.
    """
    from workbench.scripts.evaluate_models import (
        LedgerKeyError, ledger_hmac_key, read_evidence_ledger, reports_root,
        sha256_file as report_digest_of)

    config = resolve_config()
    if not reports_root(config).is_dir():
        return []
    # A ledger that cannot be *READ AT ALL* is a different answer from "no report is
    # corroborated", and the endpoints must not conflate them: this deployment carries both
    # `<CIR_REPO_ROOT>/.workbench_ledger_key` and `$XDG_CONFIG_HOME/cir/workbench_ledger_key`, and
    # when they differ every consumer used to answer `[]` with no reason stated anywhere. The
    # conflict is deliberately not resolved by preference -- whichever key this process picked
    # would be wrong for a ledger signed with the other -- so it is surfaced instead, naming both
    # paths, while reports are still not served: fail closed, never silently.
    #
    # A ledger that reads *back* empty is not this case: that is the ordinary "nothing is
    # corroborated" answer, and it is deliberately left alone.
    try:
        ledger = read_evidence_ledger(config)
    except LedgerKeyError as error:
        raise WorkbenchError(
            "ledger_key_unusable",
            f"the evidence ledger key is unusable, so no report can be corroborated: {error}",
            {"reason": str(error), "ledger": str(_ledger_path(config))})
    if ledger is None:
        # A missing ledger file is also a None answer, and a conflict that exists before the first
        # entry was ever written is still something the operator must fix (or no run can be
        # audited at all), so the key is asked directly here rather than left implicit.
        try:
            ledger_hmac_key(config)
        except LedgerKeyError as error:
            raise WorkbenchError(
                "ledger_key_unusable",
                f"the evidence ledger key is unusable, so no report can be corroborated: {error}",
                {"reason": str(error), "ledger": str(_ledger_path(config))})
        # With a usable key, None is a corrupt chain, not a missing one. That is a different
        # answer from "nothing is corroborated", and conflating them was a real defect: a tampered
        # or truncated ledger made every endpoint answer HTTP 200 with an empty list, which is
        # indistinguishable from an empty workspace and hid the compromise entirely (I02/I17).
        # A *missing* file stays a valid empty answer -- nothing has been recorded yet.
        if _ledger_path(config).is_file():
            raise WorkbenchError(
                "ledger_corrupt",
                "the evidence ledger exists but its HMAC chain does not verify, so no report can "
                "be corroborated and this is not an empty workspace",
                {"ledger": str(_ledger_path(config))})
    # A report is evidence only when the out-of-band ledger corroborates it. Any parseable JSON
    # in this directory could have been written by anyone with filesystem access; without the
    # ledger cross-check a forged report would be served as `passed=True`. A report whose digest,
    # identity fields, or completion status disagree with the ledger — or that has no ledger entry
    # at all — is isolated, never served.
    # Key the ledger by report digest, not run_id: two files claiming the same run_id must each
    # find the entry whose bytes they match, and a copied report must not inherit a different
    # run's corroboration.
    # Keyed on (digest, run) rather than digest alone. Two identical report files are legitimate --
    # a re-export of the same aggregate, or a byte-identical rerun -- and they share a digest, so a
    # digest-only index kept whichever entry came last and then failed the identity guard below for
    # the other, refusing a run the verifier accepts. The identity guard is the authority for
    # which run a witness belongs to, so the index must not destroy the candidate it needs.
    ledger_by_digest: dict[tuple, dict] = {}
    for _item in ledger or []:
        ledger_by_digest[(_item.get("report_digest"), _item.get("run_id"))] = _item
    ledger_stamp = _ledger_stamp(config)
    reports = []
    for path in sorted(reports_root(config).glob("*_aggregate.json")):
        try:
            report = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(report, dict) or report.get("artifact_type") != "official_aggregate_evaluation_report":
            continue
        if not _aggregate_report_is_well_formed(report, path):
            continue
        digest = report_digest_of(path)
        witness = ledger_by_digest.get((digest, report.get("run_id")))
        if witness is None:
            print(f"workbench: skipping aggregate report {path}: no evidence-ledger entry for its run", file=sys.stderr)
            continue
        corroborated = (
            witness.get("report_digest") == digest
            and str(witness.get("model_id")) == str(report.get("model_id"))
            and str(witness.get("checkpoint_id")) == str(report.get("checkpoint_id"))
            and str(witness.get("protocol_id")) == str(report.get("protocol_id"))
            # The ledger records how the run ended; only a run the evaluator child completed
            # successfully corroborates the report it wrote.
            and witness.get("return_code") == 0
            and witness.get("invocation_status") == "INVOCATION_COMPLETED"
        )
        if not corroborated:
            print(f"workbench: skipping aggregate report {path}: disagrees with its evidence-ledger entry", file=sys.stderr)
            continue
        if any(existing.get("run_id") == report.get("run_id") for existing in reports):
            # A second file claiming the same run_id is a copy: the endpoints index by
            # run_id, so serving both would present one run twice.
            print(f"workbench: skipping aggregate report {path}: duplicate run_id", file=sys.stderr)
            continue
        proof_stamp = _proof_stamp(config, report)
        mode = _certification_mode(proof_stamp != "absent")
        checkpoint_stamp = _checkpoint_stamp(report)
        run_stamp = _run_artifacts_stamp(config, report)
        # Six inputs, each the change token of one thing the receipt is a claim about: the
        # report's bytes, the policy that judged them, the ledger that corroborated them, the
        # proof that binds the run, the checkpoint the proof describes, and the run's own
        # invocation record and logs. Anything the verifier re-derives belongs in this key.
        cache_key = (digest, mode, ledger_stamp, proof_stamp, checkpoint_stamp, run_stamp)
        receipt = _CERTIFICATION_CACHE.get(cache_key)
        if receipt is None:
            receipt = _certify_report(config, path, report, digest, mode=mode)
            while len(_CERTIFICATION_CACHE) >= _CERTIFICATION_CACHE_LIMIT:
                del _CERTIFICATION_CACHE[next(iter(_CERTIFICATION_CACHE))]
            _CERTIFICATION_CACHE[cache_key] = receipt
        report["verification"] = {
            # The four questions, kept separate so a consumer can tell "an artifact exists" apart
            # from "valid evidence": (1) is answered by this report being served at all, (2) by
            # the ledger corroboration above, (3) by the verifier's receipt, (4) by the gate.
            "artifact_exists": True,
            "execution_authenticated": True,
            "is_research_evidence": bool(receipt["is_research_evidence"]),
            "certified": bool(receipt["certified"]),
            "development_mode": bool(receipt["development_mode"]),
            "certification_mode": receipt.get("certification_mode", mode),
            "state": receipt.get("state", "certified" if receipt["certified"] else ("proof_unavailable" if mode == "proof_unavailable" else "refused")),
            "report_digest": digest,
            "reasons": list(receipt["reasons"]),
            "evidence_reasons": list(receipt["evidence_reasons"]),
        }
        reports.append(report)
    return reports

#: Fields every aggregate report must carry for the API and the frontend to be meaningful. A file
#: that parses as JSON but lacks these was truncated or corrupted after writing; serving it would
#: crash the endpoints that index by `run_id`, so it is isolated instead.
_AGGREGATE_REQUIRED_FIELDS = ("run_id", "model_id", "checkpoint_id", "protocol_id")

def _aggregate_report_is_well_formed(report: dict, path: Path) -> bool:
    def reject(problem: str) -> bool:
        print(f"workbench: skipping malformed aggregate report {path}: {problem}", file=sys.stderr)
        return False
    missing = [key for key in _AGGREGATE_REQUIRED_FIELDS
               if not isinstance(report.get(key), str) or not report.get(key)]
    if missing:
        return reject(f"missing {missing}")
    gate = report.get("benchmark_gate")
    if gate is not None and not isinstance(gate, dict):
        return reject("benchmark_gate is not an object")
    return True

def _gate_fields(report: dict) -> dict:
    """The small, frontend-facing projection of an aggregate report.

    Only what the two warning labels and the threshold slider need. The gate verdict is kept
    whole -- recorded comparisons included -- so the slider can re-flag them without a second
    request; the frontend never recomputes a percentage. Every shape the frontend dereferences
    is normalized here, so a truncated report degrades to missing evidence instead of blanking
    the UI (the app has no error boundary).
    """
    gate = report.get("benchmark_gate")
    gate = gate if isinstance(gate, dict) else {}
    comparisons = gate.get("comparisons")
    gate = {**gate, "comparisons": [c for c in comparisons or [] if isinstance(c, dict)]
            if isinstance(comparisons, list) else []}
    verification = report.get("verification")
    return {
        "benchmark_gate": gate,
        "parity_status": report.get("parity_status"),
        # The receipt's first three answers, so a consumer can tell "an artifact exists" apart from
        # "valid evidence". The parity verdict stays its own field (`parity_status`, `benchmark_gate`
        # above), which keeps all four questions visible and separate. A report with no receipt --
        # only reachable if the report was mutated after it was read -- degrades to "not evidence",
        # never to "certified".
        "verification": (verification if isinstance(verification, dict) else {
            "artifact_exists": True, "execution_authenticated": False,
            "is_research_evidence": False, "certified": False, "development_mode": False,
            "certification_mode": "uncorroborated",
            "state": "uncorroborated",
            "report_digest": None, "reasons": ["no evidence ledger corroboration for this run"],
            "evidence_reasons": ["the report carries no verification receipt"],
        }),
        "variant_provenance": (report.get("variant_provenance")
                               if isinstance(report.get("variant_provenance"), dict) else {}),
        # The frontend iterates these with Object.entries, once at the top level and once per
        # category value; a truncated report degrades to null instead of a crashing shape.
        "category_metrics": (report.get("category_metrics")
                             if isinstance(report.get("category_metrics"), dict)
                             and all(isinstance(v, dict) for v in report["category_metrics"].values())
                             else None),
        "observed_metrics": (report.get("observed_metrics")
                             if isinstance(report.get("observed_metrics"), dict) else None),
        "execution_status": report.get("execution_status"),
        # A registry-declared synthetic-output model has no published output format; its numbers
        # are evidence of a different kind and the UI must be able to say so.
        "is_synthetic": bool(report.get("is_synthetic")),
    }

@app.get("/api/benchmark-runs")
def benchmark_runs():
    """Aggregate-report runs, including runs with no per-query export in the index.

    Every report carries the verifier's receipt: `verification.is_research_evidence` is the
    authoritative answer, and a report without it is an artifact whose execution is recorded but
    which is not certified research evidence.

    An unusable evidence-ledger key is not an empty result: it answers 400 `ledger_key_unusable`
    naming the cause and the remedy, because `[]` is otherwise indistinguishable from a stack
    that genuinely has no corroborated run.
    """
    return [{"run_id": report["run_id"], "model_id": report.get("model_id"),
             "checkpoint_id": report.get("checkpoint_id"), "protocol_id": report.get("protocol_id"),
             **_gate_fields(report)} for report in _aggregate_reports()]

@app.get("/api/runs/gate")
def run_gates():
    """Gate fields keyed by run_id, for joining onto `/api/runs` records that have a report."""
    return {report["run_id"]: _gate_fields(report) for report in _aggregate_reports()}


@app.get("/api/queries")
def queries(run_id: str, query_id: str | None = None, category: str | None = None, failed_at: int | None = None, min_rank: int | None = None, max_rank: int | None = None, order: str = "worst", limit: int = 50, offset: int = 0):
    return query_page(run_id, query_id=query_id, category=category, failed_at=failed_at, min_rank=min_rank, max_rank=max_rank, order=order, limit=limit, offset=offset)


@app.get("/api/compare/{query_id}")
def compare(query_id: str, run_ids: str, k: int = 10):
    selected = run_ids.split(",")
    runs, _ = validate_analysis_selection(selected, k=k)
    return [{"run": run, "query": {**(query := get_query(run["run_id"], query_id)), "top_results": query["top_results"][:k]}} for run in runs]


@app.get("/api/analysis")
def analysis(run_ids: str, k: int = 10):
    selected = run_ids.split(",")
    _, maps = get_analysis_input(selected, k)
    return analysis_rows_from_maps(selected, maps, k)


@app.get("/api/analysis/failure-jaccard")
def overlap(run_ids: str, k: int = 10):
    selected = run_ids.split(",")
    _, maps = validate_analysis_selection(selected, k=k, require_top_k=False)
    return failure_jaccard_from_maps(selected, maps, k)


class AnnotationInput(BaseModel):
    query_id: str
    protocol_id: str
    labels: list[str]
    note: str = ""


@app.get("/api/annotations")
def annotations(query_id: str | None = None, protocol_id: str | None = None):
    records = read_json(ANNOTATIONS_PATH)
    return [item for item in records if (query_id is None or item["query_id"] == query_id) and (protocol_id is None or item["protocol_id"] == protocol_id)]


@app.put("/api/annotations")
def save_annotation(payload: AnnotationInput):
    if invalid := sorted(set(payload.labels) - set(LABELS)):
        raise HTTPException(400, {"error": "invalid_annotation_labels", "message": "Unknown annotation labels.", "details": {"labels": invalid}})
    record = Annotation(**payload.model_dump(), updated_at=datetime.now(timezone.utc)).model_dump(mode="json")
    records = [item for item in read_json(ANNOTATIONS_PATH) if (item["query_id"], item["protocol_id"]) != (payload.query_id, payload.protocol_id)] + [record]
    write_json(ANNOTATIONS_PATH, records)
    return record


class CohortInput(BaseModel):
    cohort_id: str
    protocol_id: str
    query_ids: list[str] = Field(min_length=1)
    run_ids: list[str] = Field(min_length=1)
    definition: dict
    notes: str = ""


@app.get("/api/cohorts")
def cohorts():
    return read_json(COHORTS_PATH)


@app.put("/api/cohorts")
def save_cohort(payload: CohortInput):
    runs, maps = validate_analysis_selection(payload.run_ids)
    protocol = runs[0]["protocol_id"]
    if protocol != payload.protocol_id:
        raise WorkbenchError("cross_protocol", "Cohort protocol does not match selected runs.", {"cohort_protocol": payload.protocol_id, "run_protocol": protocol})
    unknown = sorted(set(payload.query_ids) - set(maps[payload.run_ids[0]]))
    if unknown:
        raise WorkbenchError("query_alignment_mismatch", "Cohort contains query IDs outside selected run universe.", {"unknown_query_ids": unknown[:20], "unknown_query_count": len(unknown)})
    record = SavedCohort(**payload.model_dump(), created_at=datetime.now(timezone.utc)).model_dump(mode="json")
    records = [item for item in read_json(COHORTS_PATH) if item["cohort_id"] != payload.cohort_id] + [record]
    write_json(COHORTS_PATH, records)
    return record


def _format_metric(val: Any) -> str:
    if val is None or (isinstance(val, float) and val != val):
        return "N/A"
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


@app.get("/api/cohorts/{cohort_id}/export/{format}")
def export_cohort(cohort_id: str, format: str):
    cohort = next((item for item in read_json(COHORTS_PATH) if item["cohort_id"] == cohort_id), None)
    if cohort is None:
        raise HTTPException(404, {"error": "cohort_not_found", "message": "Cohort not found.", "details": {"cohort_id": cohort_id}})
    metrics = cohort_metrics(cohort["run_ids"], cohort["query_ids"])
    if format == "json":
        return {**cohort, "metrics": metrics}
    if format == "csv":
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(metrics[0]) if metrics else ["run_id"])
        writer.writeheader()
        writer.writerows([{k: _format_metric(v) for k, v in item.items()} for item in metrics])
        return {"content": output.getvalue()}
    if format == "markdown":
        header = "| Run | N | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | Mean rank | Median rank |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        body = "\n".join(
            f"| {item['run_id']} | {_format_metric(item.get('query_count'))} | {_format_metric(item.get('r1'))} | {_format_metric(item.get('r5'))} | {_format_metric(item.get('r10'))} | {_format_metric(item.get('r20'))} | {_format_metric(item.get('r50'))} | {_format_metric(item.get('r100'))} | {_format_metric(item.get('mean_target_rank'))} | {_format_metric(item.get('median_target_rank'))} |"
            for item in metrics
        )
        return {"content": f"# {cohort_id}\n\nProtocol: {cohort['protocol_id']}\n\n{header}{body}\n"}
    raise HTTPException(400, {"error": "invalid_export_format", "message": "format must be json, csv, or markdown", "details": {"format": format}})


@app.post("/api/index/rebuild")
def index():
    global _index_build_error
    if _index_thread is not None and _index_thread.is_alive():
        _index_thread.join()
    config = resolve_config()
    db_path = database_path(config)
    results_dir = getattr(config, "WORKBENCH_RESULTS_ROOT", config.WORKBENCH_ARTIFACTS_ROOT / "results")
    count = rebuild_index(results_root=results_dir, database_path=db_path)
    _index_build_error = None
    return {"runs_indexed": count}


@app.get("/api/jobs")
def jobs():
    return JOBS.list()


class JobInput(BaseModel):
    model_id: str
    checkpoint_id: str
    protocol_id: str
    top_k: int = Field(default=200, ge=1, le=200)


def _duplicate_job_conflict(job_id: str, status: str) -> HTTPException:
    """The one 409 shape a duplicate submission gets, wherever the duplicate is detected."""
    return HTTPException(409, {"error": "duplicate_job",
                               "message": "An identical job is already queued or running.",
                               "details": {"job_id": job_id, "status": status}})

@app.post("/api/jobs")
def queue_job(payload: JobInput):
    registry = load_registry()
    model = next((item for item in registry["models"] if item["model_id"] == payload.model_id), None)
    if model is None:
        raise HTTPException(404, {"error": "model_not_found", "message": "Model is not registered.", "details": {"model_id": payload.model_id}})
    checkpoint = next((item for item in model["checkpoint_variants"] if item["checkpoint_id"] == payload.checkpoint_id), None)
    if checkpoint is None:
        # A declared bundle names a set of member checkpoints (LIMN's three category models);
        # selecting it is the *complete* benchmark, so the queue resolves it to the bundle
        # record instead of refusing a legitimate selection.
        #
        # Through `checkpoint_by_id`, not by rebuilding the record here. This site used to repeat
        # the fold as `{**anchor, "checkpoint_id": bundle_id, "bundle_id": bundle_id}` and drop
        # every member fact, which is a *different* record from the one the verifier, the planner
        # and `checkpoint_missing_paths` build: with no `bundle_members` and no `required_files`,
        # the presence check fell back to the single-file branch and reported the whole bundle
        # directory as missing. One resolver, so the queue cannot disagree with the evaluator about
        # what a bundle is.
        try:
            from workbench.backend.registry import checkpoint_by_id

            checkpoint = checkpoint_by_id(model, payload.checkpoint_id)
        except (KeyError, ValueError):
            checkpoint = None
    if checkpoint is None:
        raise HTTPException(404, {"error": "checkpoint_not_found", "message": "Checkpoint is not registered for model.", "details": {"model_id": payload.model_id, "checkpoint_id": payload.checkpoint_id}})
    config = resolve_config()
    dataset_root = config.WORKBENCH_THIRD_PARTY_ROOT / "CSMCIR" / "fashionIQ_dataset" if payload.model_id == "csmcir" else config.FASHIONIQ_ROOT
    # One identical evaluation at a time: an identical model+checkpoint+protocol job already
    # queued or in flight would duplicate expensive GPU work and double-book the single GPU
    # slot. Finished jobs do not block re-running.
    #
    # This loop is a fast path *only*: it reads outside the queue lock, so a concurrent POST
    # that appends in between is invisible here. The authoritative decision is the queue's
    # (see `refuse_duplicate=True` below), made atomically with the append.
    for existing in JOBS.list():
        if (existing.get("status") in {"queued", "preparing", "running"}
                and existing.get("model_id") == payload.model_id
                and existing.get("checkpoint_id") == payload.checkpoint_id
                and existing.get("protocol_id") == payload.protocol_id):
            raise _duplicate_job_conflict(existing["job_id"], existing["status"])
    plan, blockers = guarded_plan(model, checkpoint, payload.protocol_id, dataset_root, payload.top_k, config)
    if plan is None:
        raise HTTPException(409, {"error": "evaluation_blocked", "message": "Official evaluation is not ready.", "details": {"blockers": blockers}})
    # A plan may be built for a CUDA-declaring model whose GPU is unverified, but the queue
    # must not *start* one: the launch gate is enforced here exactly as it is in the CLI.
    try:
        runtime.require_launch_ready(model)
    except (runtime.GpuVerificationError, RuntimeError) as error:
        raise HTTPException(409, {"error": "gpu_verification_required",
                                  "message": "GPU runtime is not verified for this model.",
                                  "details": {"reason": str(error),
                                              "status": runtime.environment_status(model, probe=True)["status"]}})
    # The queue runs the *audited orchestrator*, not the adapter command directly. Going
    # straight to the model's evaluator bypassed the invocation ledger, so a result produced
    # through the API left no audit entry and could not be distinguished from an unrun one.
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{uuid.uuid4().hex[:8]}"
    argv = [sys.executable, str(SCRIPTS_DIR / "evaluate_models.py"),
            "--model", plan.model_id, "--checkpoint", plan.checkpoint_id,
            "--protocol", plan.protocol, "--dataset-root", str(plan.dataset_root),
            "--top-k", str(payload.top_k), "--run-id", run_id]
    # The queue owns `queue_jobs/<job_id>/`; the operator owns the run directory
    # `logs/<model>__<checkpoint>__<run_id>/` under exactly the name the operator computes for
    # itself. Handing the queue the operator's directory made two writers share one tree.
    job_directory = config.WORKBENCH_ARTIFACTS_ROOT / "queue_jobs" / uuid.uuid4().hex
    provenance = {
        "model_id": plan.model_id,
        "checkpoint_id": plan.checkpoint_id,
        "protocol_id": plan.protocol,
        "run_id": run_id,
        # Where the run's evidence will land: the operator writes its invocation verdict there,
        # and the queue reads that verdict instead of inferring completion from the exit code.
        "run_directory": str(config.WORKBENCH_ARTIFACTS_ROOT / "logs"
                              / f"{plan.model_id}__{plan.checkpoint_id}__{run_id}"),
        "upstream_expected_pin": plan.pin,
        "actual_source_commit": plan.actual_source_commit,
        # `sha256_file` refuses an artifact it cannot frame (a missing path, a non-regular
        # member, a symlink), and its ValueError would otherwise escape the handler and reach the
        # client as an unhandled 500 with no reason. A submission that cannot describe its own
        # artifact is a refusal with a name, not a server fault.
        "checkpoint_local_sha256": _checkpoint_digest_or_refuse(plan.checkpoint, payload),
        "dataset_root": str(plan.dataset_root),
    }
    try:
        # `refuse_duplicate=True` is what actually holds: the check rides the queue's own lock,
        # so two concurrent submissions of the same evaluation cannot both be appended.
        return JOBS.enqueue(plan.model_id, plan.checkpoint_id, plan.protocol, argv, ROOT,
                            job_directory, provenance, refuse_duplicate=True)
    except QueueShutdownError as error:
        # The queue is shutting down (or has): a client cannot be told its job was accepted,
        # so this is a server-state condition, not a bad request.
        raise HTTPException(503, {"error": "queue_unavailable",
                                  "message": "The evaluation queue is shutting down and cannot accept new jobs.",
                                  "details": {"reason": str(error)}}) from error
    except DuplicateJobError as error:
        # A duplicate that landed after the fast-path loop above is refused here, by the
        # existing job's identity, so the operator still learns which job holds the GPU.
        raise _duplicate_job_conflict(error.job_id, error.status) from error


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    try:
        return JOBS.cancel(job_id)
    except (KeyError, ValueError) as error:
        raise HTTPException(400, str(error)) from error


def resolve_image(category: str, image_id: str, root: Path | None = None) -> Path | None:
    if category not in {"dress", "shirt", "toptee"} or not image_id or Path(image_id).name != image_id:
        return None
    if root is None:
        from workbench.backend.operator_config import resolve_config

        root = resolve_config().FASHIONIQ_ROOT
    candidates = [root / category / f"{image_id}{extension}" for extension in (".png", ".jpg", ".jpeg")] + [root / "images" / f"{image_id}{extension}" for extension in (".png", ".jpg", ".jpeg")] + [root / "resized_image" / category / f"{image_id}{extension}" for extension in (".png", ".jpg", ".jpeg")]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


@app.get("/api/images/{category}/{image_id}")
def image(category: str, image_id: str):
    path = resolve_image(category, image_id)
    if path is None:
        raise HTTPException(404, {"error": "image_not_available", "message": "Image is unavailable locally or identifier is invalid.", "details": {"category": category, "image_id": image_id}})
    return FileResponse(path)
