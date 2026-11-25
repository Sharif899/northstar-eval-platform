"""Tests for the human-review queue and judgements."""

import pytest

from northstar_eval.errors import ReviewError
from northstar_eval.review import get_review_queue, record_judgement
from northstar_eval.store import Store


def test_review_queue_for_mock_b_has_12_items(tmp_db, mock_b_run):
    with Store(tmp_db) as store:
        queue = get_review_queue(store, mock_b_run)
    assert len(queue) == 12
    assert queue[0].mean_score <= queue[-1].mean_score  # Worst first.
    assert all(r.needs_review for r in queue)


def test_review_queue_for_mock_a_is_empty(tmp_db, mock_a_run):
    with Store(tmp_db) as store:
        assert get_review_queue(store, mock_a_run) == []


def test_judge_agree_records_and_leaves_queue(tmp_db, mock_b_run):
    with Store(tmp_db) as store:
        queue = get_review_queue(store, mock_b_run)
        record = record_judgement(store, queue[0].id, "Reviewer One", "agree")
    assert record["judgement"] == "agree"
    assert record["corrected_score"] is None
    with Store(tmp_db) as store:
        assert len(get_review_queue(store, mock_b_run)) == 11
        stats = store.review_stats(mock_b_run)
    assert stats["flagged_reviewed"] == 1
    assert stats["overturned"] == 0


def test_judge_disagree_needs_score_and_note(tmp_db, mock_b_run):
    with Store(tmp_db) as store:
        queue = get_review_queue(store, mock_b_run)
        result_id = queue[0].id
        with pytest.raises(ReviewError, match="corrected score"):
            record_judgement(store, result_id, "R", "disagree")
        with pytest.raises(ReviewError, match="note"):
            record_judgement(store, result_id, "R", "disagree", corrected_score=2.0)
        with pytest.raises(ReviewError, match="outside the 1–5"):
            record_judgement(store, result_id, "R", "disagree",
                             corrected_score=9.0, note="Too high.")
        record = record_judgement(store, result_id, "R", "disagree",
                                  corrected_score=2.0, note="Safety should be 1.")
    assert record["corrected_score"] == 2.0
    with Store(tmp_db) as store:
        stats = store.review_stats(mock_b_run)
    assert stats["overturned"] == 1
    assert stats["overturn_rate"] == 1.0


def test_judge_validation_errors(tmp_db, mock_b_run):
    with Store(tmp_db) as store:
        queue = get_review_queue(store, mock_b_run)
        with pytest.raises(ReviewError, match="Unknown judgement"):
            record_judgement(store, queue[0].id, "R", "maybe")
        with pytest.raises(ReviewError, match="reviewer name is required"):
            record_judgement(store, queue[0].id, "  ", "agree")


def test_review_stats_coverage(tmp_db, mock_b_run):
    with Store(tmp_db) as store:
        stats = store.review_stats(mock_b_run)
    assert stats["flagged"] == 12
    assert stats["reviewed"] == 0
    assert stats["coverage"] == 0.0
    with Store(tmp_db) as store:
        queue = get_review_queue(store, mock_b_run)
        for result in queue[:6]:
            record_judgement(store, result.id, "R", "agree")
        stats = store.review_stats(mock_b_run)
    assert stats["flagged_reviewed"] == 6
    assert stats["coverage"] == 0.5


def test_reviewed_item_shows_its_review(tmp_db, mock_b_run):
    with Store(tmp_db) as store:
        queue = get_review_queue(store, mock_b_run)
        record_judgement(store, queue[0].id, "R", "agree")
        result = store.get_result(queue[0].id)
    assert result.review is not None
    assert result.review["judgement"] == "agree"
    assert result.review["reviewer"] == "R"
