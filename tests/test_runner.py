"""Tests for running evaluations end to end and for reproducibility."""

import pytest

from northstar_eval.dataset import load_dataset_jsonl
from northstar_eval.errors import DatasetError, ProviderError
from northstar_eval.runner import RunConfig, manifest, run_evaluation
from northstar_eval.store import Store
from conftest import SAMPLE_CSV, SAMPLE_DATASET


def test_mock_a_run_is_perfect_on_sample_dataset(tmp_db, mock_a_run):
    with Store(tmp_db) as store:
        run = store.get_run(mock_a_run)
        results = store.get_results(mock_a_run)
    assert run.n_items == 28
    assert run.mean_score == 5.0
    assert run.pass_rate == 1.0
    assert len(results) == 28
    assert all(r.mean_score == 5.0 for r in results)


def test_mock_b_run_numbers(tmp_db, mock_b_run):
    with Store(tmp_db) as store:
        run = store.get_run(mock_b_run)
        results = store.get_results(mock_b_run)
    assert run.mean_score == 4.45
    assert run.pass_rate == 0.571
    assert sum(1 for r in results if r.needs_review) == 12


def test_runs_are_deterministic(tmp_db):
    first = run_evaluation(RunConfig(
        dataset_path=SAMPLE_DATASET, provider_name="mock-b", db_path=tmp_db,
    ))
    second = run_evaluation(RunConfig(
        dataset_path=SAMPLE_DATASET, provider_name="mock-b", db_path=tmp_db,
    ))
    assert first.run_id != second.run_id  # Distinct runs...
    assert first.mean_score == second.mean_score  # ...with identical scores.
    assert first.dataset_hash == second.dataset_hash


def test_manifest_records_reproducibility(tmp_db, mock_a_run):
    with Store(tmp_db) as store:
        run = store.get_run(mock_a_run)
    data = manifest(run)
    assert data["dataset_hash"] == run.dataset_hash
    assert data["provider"] == "mock-a"
    assert data["model"] == "mock-a"
    assert data["prompt_version"] == "v2"
    assert data["scorer_version"] == "1.0.0"
    assert data["determinism_seed"] == 42
    assert data["pass_threshold"] == 4.0


def test_csv_dataset_runs_the_same(tmp_db):
    summary = run_evaluation(RunConfig(
        dataset_path=SAMPLE_CSV, provider_name="mock-a", db_path=tmp_db,
    ))
    assert summary.mean_score == 5.0
    assert summary.n_items == 28


def test_run_fails_on_dataset_referencing_missing_docs(tmp_path, tmp_db):
    dataset = tmp_path / "bad.jsonl"
    dataset.write_text(
        '{"id": "a", "category": "straightforward", "question": "Q?", '
        '"reference_answer": "A.", "source_doc_ids": ["DOC-NOPE"], '
        '"expected_behaviour": "answer_from_sources"}\n',
        encoding="utf-8",
    )
    kb = tmp_path / "knowledge_base.json"
    kb.write_text('[{"id": "DOC-OK", "title": "T", "text": "x"}]', encoding="utf-8")
    with pytest.raises(ProviderError, match="DOC-NOPE"):
        run_evaluation(RunConfig(dataset_path=dataset, kb_path=kb, db_path=tmp_db))


def test_run_fails_on_unknown_provider(tmp_db):
    with pytest.raises(ProviderError, match="Unknown provider"):
        run_evaluation(RunConfig(
            dataset_path=SAMPLE_DATASET, provider_name="gpt-99", db_path=tmp_db,
        ))


def test_run_with_prompt_v1_records_version(tmp_db):
    summary = run_evaluation(RunConfig(
        dataset_path=SAMPLE_DATASET, provider_name="mock-a",
        prompt_version="v1", db_path=tmp_db,
    ))
    assert summary.prompt_version == "v1"
    with Store(tmp_db) as store:
        assert store.get_run(summary.run_id).prompt_version == "v1"


def test_unknown_run_raises(tmp_db, mock_a_run):
    with Store(tmp_db) as store:
        with pytest.raises(Exception, match="No run with id"):
            store.get_run("does-not-exist")
