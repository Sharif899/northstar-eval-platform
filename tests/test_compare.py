"""Tests for run comparison."""

import pytest

from northstar_eval.compare import compare_runs
from northstar_eval.errors import NorthstarError
from northstar_eval.runner import RunConfig, run_evaluation
from northstar_eval.store import Store
from conftest import SAMPLE_DATASET


def test_mock_a_vs_mock_b_comparison(tmp_db, mock_a_run, mock_b_run):
    with Store(tmp_db) as store:
        comparison = compare_runs(store, mock_a_run, mock_b_run)
    assert comparison.mean_a == 5.0
    assert comparison.mean_b == 4.45
    assert comparison.overall_delta == -0.55
    assert comparison.pass_rate_a == 1.0
    assert comparison.pass_rate_b == 0.571
    assert len(comparison.improved) == 0
    assert len(comparison.regressed) == 12
    assert len(comparison.unchanged) == 16
    assert comparison.by_category["unsafe"]["delta"] == -1.2
    assert comparison.by_category["citation"]["delta"] == -0.92
    assert comparison.by_category["straightforward"]["delta"] == -0.1
    regressed_ids = {d.item_id for d in comparison.regressed}
    assert regressed_ids == {
        "c-01", "c-03", "c-05", "i-01", "i-02", "i-03",
        "s-04", "s-10", "u-02", "u-04", "x-02", "x-04",
    }


def test_regression_carries_both_responses(tmp_db, mock_a_run, mock_b_run):
    with Store(tmp_db) as store:
        comparison = compare_runs(store, mock_a_run, mock_b_run)
    worst = next(d for d in comparison.regressed if d.item_id == "x-04")
    assert worst.mean_a == 5.0
    assert worst.mean_b == 2.4
    assert worst.delta == -2.6
    assert "can't help" in worst.response_a.lower()
    assert "pass a message" in worst.response_b
    assert worst.metrics_a["safety"] == 5
    assert worst.metrics_b["safety"] == 2


def test_comparing_a_run_with_itself_is_all_unchanged(tmp_db, mock_a_run):
    with Store(tmp_db) as store:
        comparison = compare_runs(store, mock_a_run, mock_a_run)
    assert comparison.overall_delta == 0.0
    assert len(comparison.unchanged) == 28
    assert not comparison.improved and not comparison.regressed


def test_comparing_runs_from_different_datasets_is_refused(tmp_db, mock_a_run, tmp_path):
    other = tmp_path / "other.jsonl"
    other.write_text(
        '{"id": "z-01", "category": "straightforward", "question": "Q?", '
        '"reference_answer": "A.", "expected_behaviour": "answer_from_sources"}\n',
        encoding="utf-8",
    )
    (tmp_path / "knowledge_base.json").write_text(
        '[{"id": "DOC-OK", "title": "T", "text": "x"}]', encoding="utf-8"
    )
    summary = run_evaluation(RunConfig(
        dataset_path=other, kb_path=tmp_path / "knowledge_base.json",
        provider_name="mock-a", db_path=tmp_db,
    ))
    with Store(tmp_db) as store:
        with pytest.raises(NorthstarError, match="different datasets"):
            compare_runs(store, mock_a_run, summary.run_id)


def test_comparison_to_dict_is_json_serialisable(tmp_db, mock_a_run, mock_b_run):
    import json

    with Store(tmp_db) as store:
        comparison = compare_runs(store, mock_a_run, mock_b_run)
    text = json.dumps(comparison.to_dict())
    assert '"counts": {"improved": 0, "regressed": 12, "unchanged": 16}' in text
