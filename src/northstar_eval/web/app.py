"""Web UI: a minimal FastAPI backend plus a single-page frontend.

The UI is a prototype surface over the same store, runner and
reporting code the CLI uses — there is no second implementation to
drift. Runs execute synchronously in the request: fine for the
sample dataset and the mock providers, and documented as a
limitation for slow real providers (see the design report).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

from ..compare import compare_runs
from ..dataset import dataset_summary, load_dataset, load_knowledge_base
from ..errors import DatasetError, NorthstarError, ProviderError, ReviewError, RunNotFoundError
from ..providers import ALL_PROVIDERS
from ..report import build_json_report, build_markdown_report
from ..review import get_review_queue, record_judgement
from ..runner import RunConfig, run_evaluation
from ..store import Store

HERE = Path(__file__).resolve().parent


class RunRequest(BaseModel):
    dataset: str
    provider: str = "mock-a"
    model: str = ""
    prompt_version: str = "v2"
    name: str = ""


class CompareRequest(BaseModel):
    run_a: str
    run_b: str


class JudgeRequest(BaseModel):
    reviewer: str
    judgement: str
    corrected_score: float | None = None
    note: str = ""


def create_app(db_path: str | Path = "northstar.db", upload_dir: str | Path = "uploads") -> FastAPI:
    """Build the FastAPI application bound to one SQLite store."""
    db_path = str(db_path)
    upload_path = Path(upload_dir)
    upload_path.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="Northstar Eval Platform")

    def store() -> Store:
        return Store(db_path)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(HERE / "index.html", media_type="text/html")

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "db": db_path}

    @app.post("/api/datasets/upload")
    async def upload_dataset(file: UploadFile = File(...)) -> dict:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in (".jsonl", ".csv"):
            raise HTTPException(400, "Only .jsonl and .csv datasets are accepted.")
        destination = upload_path / Path(file.filename).name
        with destination.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
        try:
            items = load_dataset(destination)
            summary = dataset_summary(items)
        except DatasetError as exc:
            destination.unlink(missing_ok=True)
            raise HTTPException(400, f"Dataset is invalid: {exc}") from exc
        kb_path = destination.parent / "knowledge_base.json"
        kb_docs = 0
        if kb_path.is_file():
            try:
                kb_docs = len(load_knowledge_base(kb_path))
            except DatasetError:
                kb_docs = 0
        return {
            "filename": destination.name,
            "path": str(destination),
            "total": summary["total"],
            "by_category": summary["by_category"],
            "hash": summary["hash"],
            "knowledge_base_docs": kb_docs,
        }

    @app.post("/api/knowledge-base/upload")
    async def upload_knowledge_base(file: UploadFile = File(...)) -> dict:
        # The knowledge base is the set of source documents the
        # assistant answers from and every citation is checked
        # against. Without an upload of its own, a user working
        # only through the UI could never run an evaluation, so it
        # gets the same upload-and-validate treatment as datasets.
        suffix = Path(file.filename or "").suffix.lower()
        if suffix != ".json":
            raise HTTPException(400, "Only a .json knowledge base is accepted.")
        destination = upload_path / "knowledge_base.json"
        with destination.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
        try:
            docs = load_knowledge_base(destination)
        except DatasetError as exc:
            destination.unlink(missing_ok=True)
            raise HTTPException(400, f"Knowledge base is invalid: {exc}") from exc
        return {
            "filename": "knowledge_base.json",
            "documents": len(docs),
            "doc_ids": sorted(docs),
        }

    @app.get("/api/datasets")
    def list_datasets() -> dict:
        files = []
        for path in sorted(upload_path.glob("*")):
            if path.suffix.lower() not in (".jsonl", ".csv"):
                continue
            try:
                summary = dataset_summary(load_dataset(path))
            except DatasetError:
                continue
            kb_path = path.parent / "knowledge_base.json"
            kb_docs = 0
            if kb_path.is_file():
                try:
                    kb_docs = len(load_knowledge_base(kb_path))
                except DatasetError:
                    kb_docs = 0
            files.append({
                "filename": path.name, "path": str(path),
                "knowledge_base_docs": kb_docs, **summary,
            })
        return {"datasets": files}

    @app.post("/api/runs")
    def start_run(request: RunRequest) -> dict:
        if request.provider not in ALL_PROVIDERS:
            raise HTTPException(400, f"Unknown provider {request.provider!r}.")
        dataset_path = Path(request.dataset)
        if not dataset_path.is_file():
            raise HTTPException(400, f"Dataset not found: {request.dataset}")
        try:
            summary = run_evaluation(RunConfig(
                dataset_path=dataset_path,
                provider_name=request.provider,
                model=request.model or request.provider,
                prompt_version=request.prompt_version,
                db_path=db_path,
                run_name=request.name,
            ))
        except (DatasetError, ProviderError, NorthstarError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return {
            "run_id": summary.run_id,
            "name": summary.name,
            "provider": summary.provider,
            "model": summary.model,
            "prompt_version": summary.prompt_version,
            "n_items": summary.n_items,
            "mean_score": summary.mean_score,
            "pass_rate": summary.pass_rate,
            "flagged": summary.flagged,
            "by_category": summary.by_category,
            "by_metric": summary.by_metric,
        }

    @app.get("/api/runs")
    def list_runs() -> dict:
        with store() as db:
            runs = db.list_runs()
        return {"runs": [
            {
                "id": r.id, "name": r.name, "dataset_name": r.dataset_name,
                "provider": r.provider, "model": r.model,
                "prompt_version": r.prompt_version, "created_at": r.created_at,
                "n_items": r.n_items, "mean_score": r.mean_score, "pass_rate": r.pass_rate,
            }
            for r in runs
        ]}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        try:
            with store() as db:
                run = db.get_run(run_id)
                results = db.get_results(run_id)
                stats = db.review_stats(run_id)
        except RunNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {
            "run": {
                "id": run.id, "name": run.name, "dataset_name": run.dataset_name,
                "dataset_hash": run.dataset_hash, "provider": run.provider,
                "model": run.model, "prompt_version": run.prompt_version,
                "scorer_version": run.scorer_version, "created_at": run.created_at,
                "n_items": run.n_items, "mean_score": run.mean_score,
                "pass_rate": run.pass_rate,
            },
            "results": [
                {
                    "id": r.id, "item_id": r.item_id, "category": r.category,
                    "question": r.question, "reference_answer": r.reference_answer,
                    "response": r.response, "metrics": r.metrics,
                    "mean_score": r.mean_score, "passed": r.passed,
                    "needs_review": r.needs_review, "flag_reasons": r.flag_reasons,
                    "confidence": r.confidence, "review": r.review,
                }
                for r in results
            ],
            "review_stats": stats,
        }

    @app.get("/api/runs/{run_id}/report")
    def get_report(run_id: str, format: str = "md"):
        if format not in ("md", "json"):
            raise HTTPException(400, "format must be 'md' or 'json'.")
        try:
            with store() as db:
                run = db.get_run(run_id)
                results = db.get_results(run_id)
                stats = db.review_stats(run_id)
        except RunNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        if format == "json":
            return JSONResponse(build_json_report(run, results, None, stats))
        return PlainTextResponse(
            build_markdown_report(run, results, None, stats), media_type="text/markdown"
        )

    @app.post("/api/compare")
    def compare(request: CompareRequest) -> dict:
        try:
            with store() as db:
                return compare_runs(db, request.run_a, request.run_b).to_dict()
        except RunNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except NorthstarError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/review/queue")
    def review_queue(run_id: str) -> dict:
        try:
            with store() as db:
                queue = get_review_queue(db, run_id)
                stats = db.review_stats(run_id)
        except RunNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {
            "run_id": run_id,
            "stats": stats,
            "queue": [
                {
                    "id": r.id, "item_id": r.item_id, "category": r.category,
                    "question": r.question, "reference_answer": r.reference_answer,
                    "response": r.response, "metrics": r.metrics,
                    "mean_score": r.mean_score, "flag_reasons": r.flag_reasons,
                }
                for r in queue
            ],
        }

    @app.post("/api/review/{result_id}")
    def judge(result_id: str, request: JudgeRequest) -> dict:
        try:
            with store() as db:
                return record_judgement(
                    db,
                    result_id=result_id,
                    reviewer=request.reviewer,
                    judgement=request.judgement,
                    corrected_score=request.corrected_score,
                    note=request.note,
                )
        except ReviewError as exc:
            raise HTTPException(400, str(exc)) from exc
        except NorthstarError as exc:
            raise HTTPException(404, str(exc)) from exc

    return app
