from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from workbench.backend.analysis import analysis_rows, failure_jaccard, require_same_protocol
from workbench.backend.index import load_runs, rebuild_index
from workbench.backend.registry import checkpoint_availability, load_registry
from workbench.backend.schemas.results import Annotation, SavedCohort
from workbench.backend.jobs import SingleGpuQueue

ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS_PATH = ROOT / "artifacts" / "annotations.json"
COHORTS_PATH = ROOT / "artifacts" / "cohorts.json"
LABELS = ["preservation_failure", "under_edit", "over_edit", "partial_edit", "wrong_attribute", "wrong_entity_binding", "reference_dominance", "text_dominance", "fine_grained_visual_confusion", "ambiguous_ground_truth", "possible_dataset_issue", "duplicate_or_near_duplicate", "other", "unknown"]
app = FastAPI(title="CIR Failure Analysis Workbench")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])

JOBS = SingleGpuQueue()

def runs_for_ids(run_ids: list[str]):
    runs = [run for run in load_runs() if run.run.run_id in run_ids]
    if len(runs) != len(run_ids):
        raise HTTPException(404, "one or more run IDs not found")
    return runs


def read_json(path: Path) -> list[dict]:
    return json.loads(path.read_text()) if path.exists() else []


def write_json(path: Path, payload: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


@app.get("/api/models")
def models():
    records = []
    for model in load_registry()["models"]:
        records.append({**model, "checkpoints": [checkpoint_availability(model, item) for item in model["checkpoint_variants"]]})
    return records


@app.get("/api/runs")
def runs(protocol_id: str | None = None):
    return [run.model_dump(mode="json")["run"] for run in load_runs() if protocol_id is None or run.run.protocol_id == protocol_id]


@app.get("/api/queries")
def queries(run_id: str, category: str | None = None, failed_at: int | None = None, order: str = "worst"):
    run = runs_for_ids([run_id])[0]
    rows = [query.model_dump() for query in run.queries if category is None or query.category == category]
    if failed_at:
        rows = [query for query in rows if query["target_rank"] > failed_at]
    return sorted(rows, key=lambda row: row["target_rank"], reverse=order != "best")


@app.get("/api/compare/{query_id}")
def compare(query_id: str, run_ids: str):
    records = []
    for run in runs_for_ids(run_ids.split(",")):
        query = next((item for item in run.queries if item.query_id == query_id), None)
        if query:
            records.append({"run": run.run.model_dump(mode="json"), "query": query.model_dump()})
    if not records:
        raise HTTPException(404, "query not found")
    protocols = {item["run"]["protocol_id"] for item in records}
    if len(protocols) != 1:
        raise HTTPException(400, "Cross-protocol consensus statistics are invalid. Select runs from the same FashionIQ protocol.")
    return records


@app.get("/api/analysis")
def analysis(run_ids: str, k: int = 10):
    try:
        return analysis_rows(runs_for_ids(run_ids.split(",")), k)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@app.get("/api/analysis/failure-jaccard")
def overlap(run_ids: str, k: int = 10):
    try:
        return failure_jaccard(runs_for_ids(run_ids.split(",")), k)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


class AnnotationInput(BaseModel):
    query_id: str
    labels: list[str]
    note: str = ""


@app.get("/api/annotations")
def annotations():
    return read_json(ANNOTATIONS_PATH)


@app.put("/api/annotations")
def save_annotation(payload: AnnotationInput):
    if invalid := sorted(set(payload.labels) - set(LABELS)):
        raise HTTPException(400, f"invalid labels: {invalid}")
    record = Annotation(query_id=payload.query_id, labels=payload.labels, note=payload.note, updated_at=datetime.now(UTC)).model_dump(mode="json")
    records = [item for item in read_json(ANNOTATIONS_PATH) if item["query_id"] != payload.query_id] + [record]
    write_json(ANNOTATIONS_PATH, records)

class CohortInput(BaseModel):
    cohort_id: str
    protocol_id: str
    query_ids: list[str]
    model_ids: list[str]
    run_ids: list[str]
    definition: dict
    notes: str = ""


@app.get("/api/cohorts")
def cohorts():
    return read_json(COHORTS_PATH)


@app.put("/api/cohorts")
def save_cohort(payload: CohortInput):
    try:
        runs = runs_for_ids(payload.run_ids)
        if require_same_protocol(runs) != payload.protocol_id:
            raise ValueError("cohort protocol does not match selected runs")
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    record = SavedCohort(**payload.model_dump(), created_at=datetime.now(UTC)).model_dump(mode="json")
    records = [item for item in read_json(COHORTS_PATH) if item["cohort_id"] != payload.cohort_id] + [record]
    write_json(COHORTS_PATH, records)
    return record


@app.get("/api/cohorts/{cohort_id}/export/{format}")
def export_cohort(cohort_id: str, format: str):
    cohort = next((item for item in read_json(COHORTS_PATH) if item["cohort_id"] == cohort_id), None)
    if cohort is None:
        raise HTTPException(404, "cohort not found")
    if format == "json":
        return cohort
    rows = "\n".join(cohort["query_ids"])
    if format == "markdown":
        return {"content": f"# {cohort_id}\n\nProtocol: {cohort['protocol_id']}\n\n## Queries\n\n{rows}"}
    if format == "csv":
        return {"content": "query_id\n" + "\n".join(cohort["query_ids"])}
    raise HTTPException(400, "format must be json, csv, or markdown")


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


@app.get("/api/images/{category}/{image_id}")
def image(category: str, image_id: str):
    if category not in {"dress", "shirt", "toptee"} or "/" in image_id or "\\" in image_id:
        raise HTTPException(400, "invalid image identifier")
    root = Path(__import__("os").environ.get("CIR_DATA_ROOT", ROOT.parent / "data")) / "FashionIQ"
    candidates = [root / category / f"{image_id}.jpg", root / "images" / f"{image_id}.jpg"]
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        raise HTTPException(404, "image not available locally")
    return FileResponse(path)
