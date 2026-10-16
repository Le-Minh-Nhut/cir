from __future__ import annotations

import logging

import csv
import io
import json
import sys
from datetime import datetime, timezone
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from workbench.backend.analysis import analysis_rows_from_maps, failure_jaccard_from_maps
from workbench.backend import runtime
from workbench.backend.errors import WorkbenchError
from workbench.backend.index import rebuild_index
from workbench.backend.jobs import DuplicateJobError, SingleGpuQueue
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

@app.on_event("startup")
def _startup() -> None:
    # Resolved when the server actually starts, not at import: a test importing this module
    # with a different artifacts root must not key recovery to the import-time root.
    _recover_interrupted_jobs()

@app.on_event("shutdown")
def _shutdown() -> None:
    """Stop every live job's process group before the process exits.

    A job leads its own session (`start_new_session=True`), so the SIGTERM that `docker stop`
    sends to the backend reaches this process alone: without this hook the operator's evaluator
    child — the process that actually holds the GPU — survives the container and keeps holding
    the device. The queue persists an honest terminal status for each job it stops; if the
    backend is killed before the handshake runs, startup recovery still marks the record failed.
    """
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


@app.get("/api/runs")
def runs(protocol_id: str | None = None):
    return list_runs(protocol_id)

def _aggregate_reports() -> list[dict]:
    """Read every official aggregate report the evaluator has written, ignoring malformed files.

    These reports live outside the DuckDB index by design: the index holds per-query `ResultRun`
    records, while an aggregate report can exist for a run whose per-query export was not saved.
    Reading them at request time keeps the API in sync with new evaluator output without a
    re-index.
    """
    from workbench.scripts.evaluate_models import reports_root

    config = resolve_config()
    if not reports_root(config).is_dir():
        return []
    # A report is evidence only when the out-of-band ledger corroborates it. Any parseable JSON
    # in this directory could have been written by anyone with filesystem access; without the
    # ledger cross-check a forged report would be served as `passed=True`. A report whose
    # digest, identity fields, or completion status disagree with the ledger — or that has no
    # ledger entry at all — is isolated, never served.
    from workbench.scripts.evaluate_models import read_evidence_ledger, sha256_file
    ledger = read_evidence_ledger(config)
    # Key the ledger by report digest, not run_id: two files claiming the same run_id must
    # each find the entry whose bytes they match, and a copied report must not inherit a
    # different run's corroboration.
    ledger_by_digest = {item.get("report_digest"): item for item in ledger or []}
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
        witness = ledger_by_digest.get(sha256_file(path))
        # A copied report must not inherit another run's corroboration: the witness entry's
        # identity fields must match the file's own claims.
        if witness is not None and witness.get("run_id") != report.get("run_id"):
            witness = None
        if witness is None:
            print(f"workbench: skipping aggregate report {path}: no evidence-ledger entry for its run", file=sys.stderr)
            continue
        corroborated = (
            witness.get("report_digest") == sha256_file(path)
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
    return {
        "benchmark_gate": gate,
        "parity_status": report.get("parity_status"),
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
    """Aggregate-report runs, including runs with no per-query export in the index."""
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
        writer.writerows(metrics)
        return {"content": output.getvalue()}
    if format == "markdown":
        header = "| Run | N | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | Mean rank | Median rank |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        body = "\n".join(f"| {item['run_id']} | {item['query_count']} | {item['r1']:.2f} | {item['r5']:.2f} | {item['r10']:.2f} | {item['r20']:.2f} | {item['r50']:.2f} | {item['r100']:.2f} | {item['mean_target_rank']:.2f} | {item['median_target_rank']:.2f} |" for item in metrics)
        return {"content": f"# {cohort_id}\n\nProtocol: {cohort['protocol_id']}\n\n{header}{body}\n"}
    raise HTTPException(400, {"error": "invalid_export_format", "message": "format must be json, csv, or markdown", "details": {"format": format}})


@app.post("/api/index/rebuild")
def index():
    return {"runs_indexed": rebuild_index()}


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
        # anchor instead of refusing a legitimate selection.
        bundle = next((b for b in model.get("checkpoint_bundles", [])
                       if b["bundle_id"] == payload.checkpoint_id), None)
        if bundle is not None:
            anchor_id = bundle["required_checkpoint_ids"][0]
            checkpoint = next((item for item in model["checkpoint_variants"]
                               if item["checkpoint_id"] == anchor_id), None)
            if checkpoint is not None:
                checkpoint = {**checkpoint, "checkpoint_id": bundle["bundle_id"],
                              "bundle_id": bundle["bundle_id"]}
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
        "checkpoint_local_sha256": sha256_file(plan.checkpoint),
        "dataset_root": str(plan.dataset_root),
    }
    try:
        # `refuse_duplicate=True` is what actually holds: the check rides the queue's own lock,
        # so two concurrent submissions of the same evaluation cannot both be appended.
        return JOBS.enqueue(plan.model_id, plan.checkpoint_id, plan.protocol, argv, ROOT,
                            job_directory, provenance, refuse_duplicate=True)
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
