"""Tests for dataset and knowledge-base loading and validation."""

import json

import pytest

from northstar_eval.dataset import (
    dataset_hash,
    dataset_summary,
    load_dataset,
    load_dataset_csv,
    load_dataset_jsonl,
    load_knowledge_base,
)
from northstar_eval.errors import DatasetError
from conftest import SAMPLE_CSV, SAMPLE_DATASET, SAMPLE_KB


def test_sample_dataset_jsonl_loads_28_items():
    items = load_dataset_jsonl(SAMPLE_DATASET)
    assert len(items) == 28
    assert {item.id for item in items} == {f"s-{i:02d}" for i in range(1, 13)} | {
        f"c-{i:02d}" for i in range(1, 6)} | {f"u-{i:02d}" for i in range(1, 5)} | {
        f"x-{i:02d}" for i in range(1, 5)} | {f"i-{i:02d}" for i in range(1, 4)}


def test_sample_dataset_csv_matches_jsonl():
    jsonl = load_dataset_jsonl(SAMPLE_DATASET)
    csv_items = load_dataset_csv(SAMPLE_CSV)
    assert [i.id for i in csv_items] == [i.id for i in jsonl]
    assert [i.reference_answer for i in csv_items] == [i.reference_answer for i in jsonl]
    assert [i.source_doc_ids for i in csv_items] == [i.source_doc_ids for i in jsonl]


def test_load_dataset_dispatches_on_suffix(tmp_path):
    assert len(load_dataset(SAMPLE_DATASET)) == 28
    assert len(load_dataset(SAMPLE_CSV)) == 28
    bad = tmp_path / "data.txt"
    bad.write_text("hello", encoding="utf-8")
    with pytest.raises(DatasetError, match="unsupported suffix"):
        load_dataset(bad)


def test_dataset_hash_is_order_independent_and_stable():
    items = load_dataset_jsonl(SAMPLE_DATASET)
    assert dataset_hash(items) == dataset_hash(list(reversed(items)))
    assert dataset_hash(items) == dataset_hash(load_dataset_jsonl(SAMPLE_DATASET))
    changed = list(items)
    changed[0] = items[0].__class__(**{**items[0].__dict__, "question": "Different?"})
    assert dataset_hash(changed) != dataset_hash(items)


def test_dataset_summary_counts():
    items = load_dataset_jsonl(SAMPLE_DATASET)
    summary = dataset_summary(items)
    assert summary["total"] == 28
    assert summary["by_category"]["straightforward"] == 12
    assert summary["by_category"]["citation"] == 5
    assert summary["by_behaviour"]["must_refuse"] == 4


def test_duplicate_id_is_an_error(tmp_path):
    path = tmp_path / "dup.jsonl"
    path.write_text(
        '{"id": "a", "category": "straightforward", "question": "Q?", '
        '"reference_answer": "A.", "expected_behaviour": "answer_from_sources"}\n'
        '{"id": "a", "category": "straightforward", "question": "Q2?", '
        '"reference_answer": "A2.", "expected_behaviour": "answer_from_sources"}\n',
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="duplicate item id"):
        load_dataset_jsonl(path)


def test_missing_field_is_an_error(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"id": "a", "category": "straightforward", "question": "Q?", '
        '"expected_behaviour": "answer_from_sources"}\n',
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="reference_answer"):
        load_dataset_jsonl(path)


def test_unknown_category_and_behaviour_are_errors(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"id": "a", "category": "mystery", "question": "Q?", "reference_answer": "A.", '
        '"expected_behaviour": "answer_from_sources"}\n',
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="unknown category"):
        load_dataset_jsonl(path)
    path.write_text(
        '{"id": "a", "category": "straightforward", "question": "Q?", "reference_answer": "A.", '
        '"expected_behaviour": "teleport"}\n',
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="unknown expected_behaviour"):
        load_dataset_jsonl(path)


def test_malformed_json_line_and_empty_file_are_errors(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="not valid JSON"):
        load_dataset_jsonl(bad)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(DatasetError, match="no examples"):
        load_dataset_jsonl(empty)


def test_csv_missing_column_is_an_error(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("id,question\nx,Why?\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="missing columns"):
        load_dataset_csv(path)


def test_csv_duplicate_id_is_an_error(tmp_path):
    path = tmp_path / "dup.csv"
    path.write_text(
        "id,category,question,reference_answer,source_doc_ids,expected_behaviour\n"
        "a,straightforward,Q?,A.,,answer_from_sources\n"
        "a,straightforward,Q2?,A2.,,answer_from_sources\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="duplicate item id"):
        load_dataset_csv(path)


def test_knowledge_base_loads_and_validates(tmp_path):
    docs = load_knowledge_base(SAMPLE_KB)
    assert len(docs) == 10
    assert docs["DOC-RETURNS"].title == "Returns and refunds"
    dup = tmp_path / "kb.json"
    dup.write_text(json.dumps([
        {"id": "D1", "title": "T", "text": "x"},
        {"id": "D1", "title": "T", "text": "y"},
    ]), encoding="utf-8")
    with pytest.raises(DatasetError, match="duplicate document id"):
        load_knowledge_base(dup)
    with pytest.raises(DatasetError, match="not found"):
        load_knowledge_base(tmp_path / "missing.json")
