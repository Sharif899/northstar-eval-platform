"""Tests for the web UI, via FastAPI's TestClient (no server needed)."""

import io
import json

import pytest
from fastapi.testclient import TestClient

from northstar_eval.web.app import create_app
from conftest import SAMPLE_DATASET, SAMPLE_KB


@pytest.fixture()
def client(tmp_path):
    app = create_app(tmp_path / "web.db", upload_dir=tmp_path / "uploads")
    return TestClient(app)


@pytest.fixture()
def uploaded_client(client, tmp_path):
    with open(SAMPLE_DATASET, "rb") as handle:
        response = client.post(
            "/api/datasets/upload",
            files={"file": ("sample_dataset.jsonl", handle, "application/json")},
        )
    assert response.status_code == 200, response.text
    # The knowledge base goes through its own upload, exactly as a
    # user of the UI would do it — no copying files behind the API.
    kb_response = client.post(
        "/api/knowledge-base/upload",
        files={"file": ("knowledge_base.json", SAMPLE_KB.read_bytes(), "application/json")},
    )
    assert kb_response.status_code == 200, kb_response.text
    return client


def test_health_and_index(client):
    assert client.get("/api/health").json()["status"] == "ok"
    response = client.get("/")
    assert response.status_code == 200
    assert "Northstar Eval Platform" in response.text
    assert "Start an evaluation" in response.text


def test_upload_valid_dataset(uploaded_client):
    data = uploaded_client.get("/api/datasets").json()["datasets"]
    assert len(data) == 1
    assert data[0]["total"] == 28
    assert data[0]["knowledge_base_docs"] == 10


def test_upload_valid_knowledge_base(client):
    response = client.post(
        "/api/knowledge-base/upload",
        files={"file": ("knowledge_base.json", SAMPLE_KB.read_bytes(), "application/json")},
    )
    assert response.status_code == 200
    assert response.json()["documents"] == 10
    assert response.json()["doc_ids"][0] == "DOC-CARE"
    assert response.json()["doc_ids"][-1] == "DOC-WARRANTY"


def test_upload_knowledge_base_wrong_suffix(client):
    response = client.post(
        "/api/knowledge-base/upload",
        files={"file": ("knowledge_base.txt", b"[]", "text/plain")},
    )
    assert response.status_code == 400


def test_upload_invalid_knowledge_base_rejected(client, tmp_path):
    # A document without text is not a usable source, and a rejected
    # upload must not leave a broken file behind.
    response = client.post(
        "/api/knowledge-base/upload",
        files={"file": ("knowledge_base.json",
                        b'[{"id": "DOC-X", "title": "No text here"}]',
                        "application/json")},
    )
    assert response.status_code == 400
    assert not (tmp_path / "uploads" / "knowledge_base.json").exists()


def test_upload_invalid_dataset_is_rejected(client):
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("bad.jsonl", io.BytesIO(b'{"id": "a"}\n'), "application/json")},
    )
    assert response.status_code == 400
    assert "invalid" in response.json()["detail"].lower()


def test_upload_wrong_suffix_is_rejected(client):
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert response.status_code == 400


def test_run_end_to_end(uploaded_client):
    datasets = uploaded_client.get("/api/datasets").json()["datasets"]
    path = datasets[0]["path"]
    response = uploaded_client.post("/api/runs", json={
        "dataset": path, "provider": "mock-b", "model": "mock-b",
        "prompt_version": "v2", "name": "web test run",
    })
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["mean_score"] == 4.45
    assert data["pass_rate"] == 0.571
    assert data["flagged"] == 12
    run_id = data["run_id"]

    detail = uploaded_client.get(f"/api/runs/{run_id}").json()
    assert len(detail["results"]) == 28
    assert detail["review_stats"]["flagged"] == 12

    md = uploaded_client.get(f"/api/runs/{run_id}/report?format=md")
    assert md.status_code == 200
    assert "Failure examples" in md.text
    js = uploaded_client.get(f"/api/runs/{run_id}/report?format=json")
    assert js.status_code == 200
    assert js.json()["overall"]["mean_score"] == 4.45


def test_run_unknown_provider_is_rejected(uploaded_client):
    datasets = uploaded_client.get("/api/datasets").json()["datasets"]
    response = uploaded_client.post("/api/runs", json={
        "dataset": datasets[0]["path"], "provider": "gpt-99",
    })
    assert response.status_code == 400


def test_compare_via_api(uploaded_client):
    datasets = uploaded_client.get("/api/datasets").json()["datasets"]
    path = datasets[0]["path"]
    a = uploaded_client.post("/api/runs", json={
        "dataset": path, "provider": "mock-a", "name": "a"}).json()["run_id"]
    b = uploaded_client.post("/api/runs", json={
        "dataset": path, "provider": "mock-b", "name": "b"}).json()["run_id"]
    response = uploaded_client.post("/api/compare", json={"run_a": a, "run_b": b})
    assert response.status_code == 200
    data = response.json()
    assert data["counts"] == {"improved": 0, "regressed": 12, "unchanged": 16}
    assert data["overall_delta"] == -0.55


def test_review_queue_and_judge_via_api(uploaded_client):
    datasets = uploaded_client.get("/api/datasets").json()["datasets"]
    run_id = uploaded_client.post("/api/runs", json={
        "dataset": datasets[0]["path"], "provider": "mock-b"}).json()["run_id"]
    queue = uploaded_client.get(f"/api/review/queue?run_id={run_id}").json()
    assert len(queue["queue"]) == 12
    first = queue["queue"][0]
    response = uploaded_client.post(f"/api/review/{first['id']}", json={
        "reviewer": "Web Tester", "judgement": "disagree",
        "corrected_score": 2.0, "note": "Safety should be 1.",
    })
    assert response.status_code == 200
    assert response.json()["judgement"] == "disagree"
    queue_after = uploaded_client.get(f"/api/review/queue?run_id={run_id}").json()
    assert len(queue_after["queue"]) == 11
    assert queue_after["stats"]["overturned"] == 1

    bad = uploaded_client.post(f"/api/review/{first['id']}", json={
        "reviewer": "Web Tester", "judgement": "disagree",
    })
    assert bad.status_code == 400


def test_unknown_run_is_404(client):
    assert client.get("/api/runs/nope").status_code == 404
    assert client.get("/api/review/queue?run_id=nope").status_code == 404
