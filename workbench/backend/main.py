from __future__ import annotations

import csv
import io
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from workbench.backend.analysis import aligned_queries, analysis_rows, cohort_metrics, failure_jaccard, require_same_protocol
from workbench.backend.errors import WorkbenchError
from workbench.backend.index import rebuild_index
from workbench.backend.jobs import SingleGpuQueue
from workbench.backend.registry import checkpoint_availability, load_registry
from workbench.backend.schemas.results import Annotation, SavedCohort
from workbench.backend.storage import indexed_runs

ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS_PATH = ROOT / "artifacts" / "annotations.json"
COHORTS_PATH = ROOT / "artifacts" / "cohorts.json"
LABELS = ["preservation_failure", "under_edit", "over_edit", "partial_edit", "wrong_attribute", "wrong_entity_binding", "reference_dominance", "text_dominance", "fine_grained_visual_confusion", "ambiguous_ground_truth", "possible_dataset_issue", "duplicate_or_near_duplicate", "other", "unknown"]
app = FastAPI(title="CIR Failure Analysis Workbench")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])
JOBS = SingleGpuQueue()


@app.exception_handler(WorkbenchError)
async def workbench_error(_: Request, error: WorkbenchError) -> JSONResponse:
    return JSONResponse(status_code=400, content=error.payload())


def runs_for_ids(run_ids: list[str]):
    runs = [run for run in indexed_runs() if run.run.run_id in run_ids]
    if len(runs) != len(run_ids):
        raise HTTPException(404, {"error": "run_not_found", "message": "One or more run IDs were not found.", "details": {"requested_run_ids": run_ids}})
    return [next(run for run in runs if run.run.run_id == run_id) for run_id in run_ids]


def read_json(path: Path) -> list[dict]:
    return json.loads(path.read_text()) if path.exists() else []


def write_json(path: Path, payload: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def page(items: list[dict], limit: int, offset: int) -> dict:
    if not 1 <= limit <= 200 or offset < 0:
        raise HTTPException(400, {"error": "invalid_pagination", "message": "limit must be 1..200 and offset must be non-negative.", "details": {"limit": limit, "offset": offset}})
    return {"items": items[offset:offset + limit], "total": len(items), "limit": limit, "offset": offset}


@app.get("/api/models")
def models():
    return [{**model, "checkpoints": [checkpoint_availability(model, item) for item in model["checkpoint_variants"]]} for model in load_registry()["models"]]


@app.get("/api/runs")
def runs(protocol_id: str | None = None):
    return [run.run.model_dump(mode="json") for run in indexed_runs() if protocol_id is None or run.run.protocol_id == protocol_id]


@app.get("/api/queries")
def queries(run_id: str, query_id: str | None = None, category: str | None = None, failed_at: int | None = None, order: str = "worst", limit: int = 50, offset: int = 0):
    run = runs_for_ids([run_id])[0]
    rows = [query.model_dump() for query in run.queries if (query_id is None or query.query_id == query_id) and (category is None or query.category == category)]
    if failed_at is not None:
        rows = [query for query in rows if query["target_rank"] > failed_at]
    return page(sorted(rows, key=lambda row: row["target_rank"], reverse=order != "best"), limit, offset)


@app.get("/api/compare/{query_id}")
def compare(query_id: str, run_ids: str):
    runs = runs_for_ids(run_ids.split(","))
    aligned_queries(runs)
    records = [{"run": run.run.model_dump(mode="json"), "query": next(query.model_dump() for query in run.queries if query.query_id == query_id)} for run in runs if any(query.query_id == query_id for query in run.queries)]
    if len(records) != len(runs):
        raise HTTPException(404, {"error": "query_not_found", "message": "Query not found in selected runs.", "details": {"query_id": query_id}})
    return records


@app.get("/api/analysis")
def analysis(run_ids: str, k: int = 10):
    return analysis_rows(runs_for_ids(run_ids.split(",")), k)


@app.get("/api/analysis/failure-jaccard")
def overlap(run_ids: str, k: int = 10):
    return failure_jaccard(runs_for_ids(run_ids.split(",")), k)


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
    record = Annotation(**payload.model_dump(), updated_at=datetime.now(UTC)).model_dump(mode="json")
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
    selected_runs = runs_for_ids(payload.run_ids)
    protocol = require_same_protocol(selected_runs)
    if protocol != payload.protocol_id:
        raise WorkbenchError("cross_protocol", "Cohort protocol does not match selected runs.", {"cohort_protocol": payload.protocol_id, "run_protocol": protocol})
    aligned = aligned_queries(selected_runs)
    unknown = sorted(set(payload.query_ids) - set(aligned))
    if unknown:
        raise WorkbenchError("query_alignment_mismatch", "Cohort contains query IDs outside selected run universe.", {"unknown_query_ids": unknown[:20], "unknown_query_count": len(unknown)})
    record = SavedCohort(**payload.model_dump(), created_at=datetime.now(UTC)).model_dump(mode="json")
    records = [item for item in read_json(COHORTS_PATH) if item["cohort_id"] != payload.cohort_id] + [record]
    write_json(COHORTS_PATH, records)
    return record


@app.get("/api/cohorts/{cohort_id}/export/{format}")
def export_cohort(cohort_id: str, format: str):
    cohort = next((item for item in read_json(COHORTS_PATH) if item["cohort_id"] == cohort_id), None)
    if cohort is None:
        raise HTTPException(404, {"error": "cohort_not_found", "message": "Cohort not found.", "details": {"cohort_id": cohort_id}})
    metrics = cohort_metrics(runs_for_ids(cohort["run_ids"]), cohort["query_ids"])
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


@app.post("/api/jobs")
def queue_job():
    raise HTTPException(403, "Evaluation execution is disabled in laptop development mode.")


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    try:
        return JOBS.cancel(job_id)
    except (KeyError, ValueError) as error:
        raise HTTPException(400, str(error)) from error


def resolve_image(category: str, image_id: str, root: Path | None = None) -> Path | None:
    if category not in {"dress", "shirt", "toptee"} or not image_id or Path(image_id).name != image_id:
        return None
    root = root or Path(os.environ.get("CIR_DATA_ROOT", ROOT.parent / "data")) / "FashionIQ"
    candidates = [root / category / f"{image_id}{extension}" for extension in (".png", ".jpg", ".jpeg")] + [root / "images" / f"{image_id}{extension}" for extension in (".png", ".jpg", ".jpeg")] + [root / "resized_image" / category / f"{image_id}{extension}" for extension in (".png", ".jpg", ".jpeg")]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


@app.get("/api/images/{category}/{image_id}")
def image(category: str, image_id: str):
    path = resolve_image(category, image_id)
    if path is None:
        raise HTTPException(404, {"error": "image_not_available", "message": "Image is unavailable locally or identifier is invalid.", "details": {"category": category, "image_id": image_id}})
    return FileResponse(path)
