"""Dataset and knowledge-base loading with strict validation.

A dataset is the contract between the client team and the platform:
every example must carry an id, a question, a reference answer, the
source documents it rests on, and the behaviour the assistant is
expected to show. Loading fails loudly on the first problem instead
of silently dropping rows, because a silently dropped row is a
regression the platform would never catch.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .errors import DatasetError

CATEGORIES = ("straightforward", "citation", "unanswerable", "unsafe", "instruction")

EXPECTED_BEHAVIOURS = (
    "answer_from_sources",
    "must_refuse",
    "must_say_unknown",
    "must_cite",
)

REQUIRED_FIELDS = ("id", "category", "question", "reference_answer", "expected_behaviour")


@dataclass(frozen=True)
class SourceDoc:
    """One knowledge-base document the assistant may draw on."""

    id: str
    title: str
    text: str


@dataclass(frozen=True)
class EvalItem:
    """One evaluation example.

    ``expected_behaviour`` is what a correct assistant must do:
    answer from the listed sources, refuse, say it does not know, or
    answer with a citation to a named document. The scorer checks
    exactly this, so the field is part of the test, not decoration.
    """

    id: str
    category: str
    question: str
    reference_answer: str
    source_doc_ids: tuple[str, ...] = ()
    expected_behaviour: str = "answer_from_sources"


def load_knowledge_base(path: str | Path) -> dict[str, SourceDoc]:
    """Load the knowledge base from a JSON file.

    The file is a list of objects with ``id``, ``title`` and ``text``.
    Duplicate ids or empty texts are errors.
    """
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"Knowledge base not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"Knowledge base {path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise DatasetError(f"Knowledge base {path} must be a non-empty JSON list of documents.")
    docs: dict[str, SourceDoc] = {}
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise DatasetError(f"Knowledge base {path}, entry {index}: expected an object.")
        for key in ("id", "title", "text"):
            if not entry.get(key) or not isinstance(entry[key], str):
                raise DatasetError(
                    f"Knowledge base {path}, entry {index}: missing or empty field {key!r}."
                )
        doc_id = entry["id"]
        if doc_id in docs:
            raise DatasetError(f"Knowledge base {path}: duplicate document id {doc_id!r}.")
        docs[doc_id] = SourceDoc(id=doc_id, title=entry["title"], text=entry["text"])
    return docs


def _validate_item(raw: dict, where: str) -> EvalItem:
    """Validate one raw example dict and build an EvalItem."""
    for key in REQUIRED_FIELDS:
        if key not in raw or raw[key] is None or str(raw[key]).strip() == "":
            raise DatasetError(f"{where}: missing or empty required field {key!r}.")
    item_id = str(raw["id"]).strip()
    category = str(raw["category"]).strip()
    if category not in CATEGORIES:
        raise DatasetError(
            f"{where}: unknown category {category!r}; expected one of {', '.join(CATEGORIES)}."
        )
    behaviour = str(raw["expected_behaviour"]).strip()
    if behaviour not in EXPECTED_BEHAVIOURS:
        raise DatasetError(
            f"{where}: unknown expected_behaviour {behaviour!r}; "
            f"expected one of {', '.join(EXPECTED_BEHAVIOURS)}."
        )
    doc_ids = raw.get("source_doc_ids", [])
    if isinstance(doc_ids, str):
        doc_ids = [d.strip() for d in doc_ids.split(";") if d.strip()]
    if not isinstance(doc_ids, list) or any(not isinstance(d, str) for d in doc_ids):
        raise DatasetError(f"{where}: source_doc_ids must be a list of document ids.")
    return EvalItem(
        id=item_id,
        category=category,
        question=str(raw["question"]).strip(),
        reference_answer=str(raw["reference_answer"]).strip(),
        source_doc_ids=tuple(str(d).strip() for d in doc_ids),
        expected_behaviour=behaviour,
    )


def load_dataset_jsonl(path: str | Path) -> list[EvalItem]:
    """Load a dataset from a JSONL file (one JSON object per line)."""
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"Dataset not found: {path}")
    items: list[EvalItem] = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path}, line {line_no}: not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise DatasetError(f"{path}, line {line_no}: expected a JSON object.")
        item = _validate_item(raw, f"{path}, line {line_no}")
        if item.id in seen:
            raise DatasetError(f"{path}, line {line_no}: duplicate item id {item.id!r}.")
        seen.add(item.id)
        items.append(item)
    if not items:
        raise DatasetError(f"Dataset {path} contains no examples.")
    return items


def load_dataset_csv(path: str | Path) -> list[EvalItem]:
    """Load a dataset from a CSV file.

    Columns: id, category, question, reference_answer,
    source_doc_ids (semicolon-separated), expected_behaviour.
    """
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"Dataset not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DatasetError(f"Dataset {path} is empty.")
        missing = [f for f in REQUIRED_FIELDS if f not in reader.fieldnames]
        if missing:
            raise DatasetError(f"Dataset {path}: missing columns: {', '.join(missing)}.")
        items: list[EvalItem] = []
        seen: set[str] = set()
        for row_no, row in enumerate(reader, start=2):
            raw = {key: (value or "") for key, value in row.items() if key is not None}
            item = _validate_item(raw, f"{path}, row {row_no}")
            if item.id in seen:
                raise DatasetError(f"{path}, row {row_no}: duplicate item id {item.id!r}.")
            seen.add(item.id)
            items.append(item)
    if not items:
        raise DatasetError(f"Dataset {path} contains no examples.")
    return items


def load_dataset(path: str | Path) -> list[EvalItem]:
    """Load a dataset, choosing the loader from the file suffix."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return load_dataset_jsonl(path)
    if path.suffix.lower() == ".csv":
        return load_dataset_csv(path)
    raise DatasetError(
        f"Cannot load {path}: unsupported suffix {path.suffix!r}; use .jsonl or .csv."
    )


def dataset_hash(items: list[EvalItem]) -> str:
    """SHA-256 of the canonical dataset contents.

    The hash is recorded in every run manifest, so a run can always
    be tied back to the exact dataset version that produced it. Two
    files with the same items in a different order hash the same,
    because the items are sorted by id first.
    """
    canonical = json.dumps(
        [
            {
                "id": item.id,
                "category": item.category,
                "question": item.question,
                "reference_answer": item.reference_answer,
                "source_doc_ids": sorted(item.source_doc_ids),
                "expected_behaviour": item.expected_behaviour,
            }
            for item in sorted(items, key=lambda i: i.id)
        ],
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def dataset_summary(items: list[EvalItem]) -> dict:
    """Counts by category and behaviour, for display and reports."""
    by_category: dict[str, int] = {c: 0 for c in CATEGORIES}
    by_behaviour: dict[str, int] = {b: 0 for b in EXPECTED_BEHAVIOURS}
    for item in items:
        by_category[item.category] += 1
        by_behaviour[item.expected_behaviour] += 1
    return {
        "total": len(items),
        "by_category": {k: v for k, v in by_category.items() if v},
        "by_behaviour": {k: v for k, v in by_behaviour.items() if v},
        "hash": dataset_hash(items),
    }
