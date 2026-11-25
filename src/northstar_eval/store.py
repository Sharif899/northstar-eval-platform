"""SQLite persistence for runs, results and human reviews.

The prototype stores everything in one SQLite file (stdlib
``sqlite3`` — no extra dependency). That is a deliberate prototype
choice: a single file is easy to inspect, back up and ship, and the
schema is small. The store is the one module that knows about SQL;
everything else works with dataclasses. For production the same
interface would sit on PostgreSQL with a migrations table — see the
design report.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .errors import RunNotFoundError, StoreError
from .scorer import ItemScore

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    dataset_hash TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    scorer_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    n_items INTEGER NOT NULL,
    mean_score REAL NOT NULL,
    pass_rate REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs (id),
    item_id TEXT NOT NULL,
    category TEXT NOT NULL,
    question TEXT NOT NULL,
    reference_answer TEXT NOT NULL,
    response TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    mean_score REAL NOT NULL,
    passed INTEGER NOT NULL,
    needs_review INTEGER NOT NULL,
    flag_reasons_json TEXT NOT NULL,
    confidence REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_results_run ON results (run_id);
CREATE TABLE IF NOT EXISTS reviews (
    id TEXT PRIMARY KEY,
    result_id TEXT NOT NULL REFERENCES results (id),
    run_id TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    judgement TEXT NOT NULL,
    corrected_score REAL,
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reviews_result ON reviews (result_id);
"""


@dataclass
class RunRecord:
    """A completed evaluation run, as stored."""

    id: str
    name: str
    dataset_name: str
    dataset_hash: str
    provider: str
    model: str
    prompt_version: str
    scorer_version: str
    created_at: str
    n_items: int
    mean_score: float
    pass_rate: float


@dataclass
class ResultRecord:
    """One scored item, as stored, with its review if any."""

    id: str
    run_id: str
    item_id: str
    category: str
    question: str
    reference_answer: str
    response: str
    metrics: dict
    mean_score: float
    passed: bool
    needs_review: bool
    flag_reasons: list[str] = field(default_factory=list)
    confidence: float = 1.0
    review: dict | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    """Thin SQLite wrapper. Opens the database on construction."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.executescript(SCHEMA)
            self._conn.commit()
        except sqlite3.Error as exc:
            raise StoreError(f"Could not open the store at {self.path}: {exc}") from exc

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- runs ------------------------------------------------------
    def save_run(self, record: RunRecord, results: list[ItemScore]) -> None:
        try:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        record.id, record.name, record.dataset_name, record.dataset_hash,
                        record.provider, record.model, record.prompt_version,
                        record.scorer_version, record.created_at, record.n_items,
                        record.mean_score, record.pass_rate,
                    ),
                )
                for item in results:
                    self._conn.execute(
                        "INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            str(uuid.uuid4()), record.id, item.item_id, item.category,
                            item.question, item.reference_answer, item.response,
                            json.dumps(item.to_dict()["metrics"]),
                            item.mean_score, int(item.passed), int(item.needs_review),
                            json.dumps(item.flag_reasons), item.confidence,
                        ),
                    )
        except sqlite3.Error as exc:
            raise StoreError(f"Could not save run {record.id}: {exc}") from exc

    def get_run(self, run_id: str) -> RunRecord:
        row = self._conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise RunNotFoundError(f"No run with id {run_id!r} in {self.path}.")
        return RunRecord(**{key: row[key] for key in row.keys()})

    def list_runs(self) -> list[RunRecord]:
        rows = self._conn.execute("SELECT * FROM runs ORDER BY rowid DESC").fetchall()
        return [RunRecord(**{key: row[key] for key in row.keys()}) for row in rows]

    # -- results ---------------------------------------------------
    def get_results(self, run_id: str) -> list[ResultRecord]:
        self.get_run(run_id)  # Raises RunNotFoundError for unknown runs.
        rows = self._conn.execute(
            "SELECT * FROM results WHERE run_id = ? ORDER BY rowid", (run_id,)
        ).fetchall()
        return [self._row_to_result(row) for row in rows]

    def get_result(self, result_id: str) -> ResultRecord:
        row = self._conn.execute("SELECT * FROM results WHERE id = ?", (result_id,)).fetchone()
        if row is None:
            raise StoreError(f"No result with id {result_id!r} in {self.path}.")
        return self._row_to_result(row)

    def _row_to_result(self, row: sqlite3.Row) -> ResultRecord:
        review_row = self._conn.execute(
            "SELECT * FROM reviews WHERE result_id = ? ORDER BY created_at DESC LIMIT 1",
            (row["id"],),
        ).fetchone()
        review = dict(review_row) if review_row else None
        return ResultRecord(
            id=row["id"],
            run_id=row["run_id"],
            item_id=row["item_id"],
            category=row["category"],
            question=row["question"],
            reference_answer=row["reference_answer"],
            response=row["response"],
            metrics=json.loads(row["metrics_json"]),
            mean_score=row["mean_score"],
            passed=bool(row["passed"]),
            needs_review=bool(row["needs_review"]),
            flag_reasons=json.loads(row["flag_reasons_json"]),
            confidence=row["confidence"],
            review=review,
        )

    # -- reviews ---------------------------------------------------
    def save_review(
        self,
        result_id: str,
        reviewer: str,
        judgement: str,
        corrected_score: float | None,
        note: str,
    ) -> dict:
        result = self.get_result(result_id)
        review_id = str(uuid.uuid4())
        try:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO reviews VALUES (?,?,?,?,?,?,?,?)",
                    (
                        review_id, result_id, result.run_id, reviewer, judgement,
                        corrected_score, note, _now(),
                    ),
                )
        except sqlite3.Error as exc:
            raise StoreError(f"Could not save review for {result_id}: {exc}") from exc
        return {
            "id": review_id,
            "result_id": result_id,
            "run_id": result.run_id,
            "reviewer": reviewer,
            "judgement": judgement,
            "corrected_score": corrected_score,
            "note": note,
        }

    def review_stats(self, run_id: str) -> dict:
        self.get_run(run_id)
        flagged = self._conn.execute(
            "SELECT COUNT(*) FROM results WHERE run_id = ? AND needs_review = 1",
            (run_id,),
        ).fetchone()[0]
        reviewed = self._conn.execute(
            "SELECT COUNT(DISTINCT result_id) FROM reviews WHERE run_id = ?",
            (run_id,),
        ).fetchone()[0]
        flagged_reviewed = self._conn.execute(
            "SELECT COUNT(DISTINCT r.result_id) FROM reviews r "
            "JOIN results res ON res.id = r.result_id "
            "WHERE r.run_id = ? AND res.needs_review = 1",
            (run_id,),
        ).fetchone()[0]
        overturned = self._conn.execute(
            "SELECT COUNT(*) FROM ("
            "  SELECT result_id, MAX(created_at) AS latest FROM reviews "
            "  WHERE run_id = ? GROUP BY result_id"
            ") latest JOIN reviews r ON r.result_id = latest.result_id "
            "AND r.created_at = latest.latest WHERE r.judgement = 'disagree'",
            (run_id,),
        ).fetchone()[0]
        return {
            "flagged": flagged,
            "reviewed": reviewed,
            "flagged_reviewed": flagged_reviewed,
            "coverage": round(flagged_reviewed / flagged, 3) if flagged else 1.0,
            "overturned": overturned,
            "overturn_rate": round(overturned / reviewed, 3) if reviewed else 0.0,
        }
