"""Tests for Markdown and JSON report generation."""

import json

from northstar_eval.compare import compare_runs
from northstar_eval.report import build_json_report, build_markdown_report, worst_items
from northstar_eval.store import Store


def test_worst_items_are_worst_first(tmp_db, mock_b_run):
    with Store(tmp_db) as store:
        results = store.get_results(mock_b_run)
    worst = worst_items(results)
    assert len(worst) == 5
    assert worst[0].item_id == "x-04"  # Mean 2.40, the lowest.
    assert worst[0].mean_score <= worst[-1].mean_score


def test_json_report_structure(tmp_db, mock_b_run):
    with Store(tmp_db) as store:
        run = store.get_run(mock_b_run)
        results = store.get_results(mock_b_run)
        stats = store.review_stats(mock_b_run)
    data = build_json_report(run, results, None, stats)
    assert data["manifest"]["dataset_hash"] == run.dataset_hash
    assert data["overall"]["n_items"] == 28
    assert data["overall"]["mean_score"] == 4.45
    assert data["overall"]["pass_rate"] == 0.571
    assert data["overall"]["flagged_for_review"] == 12
    assert data["by_category"]["unsafe"]["mean"] == 3.8
    assert data["by_metric"]["factual_accuracy"]["mean"] == 4.32
    assert len(data["failure_examples"]) == 5
    assert data["failure_examples"][0]["item_id"] == "x-04"
    assert data["review"]["flagged"] == 12
    assert len(data["results"]) == 28
    json.dumps(data)  # Must be JSON-serialisable.


def test_markdown_report_leads_with_failures(tmp_db, mock_b_run):
    with Store(tmp_db) as store:
        run = store.get_run(mock_b_run)
        results = store.get_results(mock_b_run)
        stats = store.review_stats(mock_b_run)
    text = build_markdown_report(run, results, None, stats)
    assert text.startswith("# Evaluation report — mock-b test run")
    assert "## Failure examples (worst items first)" in text
    assert "x-04" in text
    assert "## Human review" in text
    assert "Flagged: 12" in text
    # The report states its own limits.
    assert "deterministic rubric checks" in text


def test_markdown_report_with_comparison(tmp_db, mock_a_run, mock_b_run):
    with Store(tmp_db) as store:
        run_b = store.get_run(mock_b_run)
        results_b = store.get_results(mock_b_run)
        comparison = compare_runs(store, mock_a_run, mock_b_run)
    text = build_markdown_report(run_b, results_b, comparison)
    assert "## Comparison" in text
    assert "Regressions" in text
    assert "x-04" in text
    assert "Improved: **0**" in text
    assert "Regressed: **12**" in text
