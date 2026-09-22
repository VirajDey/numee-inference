"""API surface of the inference service.

The Qdrant/OpenAI backed service is stubbed (see conftest), so these cover the
contract the Numee API depends on: auth, validation, response shapes and the
readiness gate.
"""
import pytest

from app.services import embeddings


# ---------------------------
# Probes
# ---------------------------

def test_health_needs_no_token(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "DEGRADED",
        "qdrant_connection": True,
        "embedding_models_loaded": False,
    }


def test_health_is_ok_once_models_are_loaded(client):
    embeddings._openai_client = object()
    embeddings._hugging_face_client = object()
    assert client.get("/health").json()["status"] == "OK"


def test_ready_is_503_until_models_are_loaded(client):
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {"ready": False}


def test_ready_is_200_once_models_are_loaded(client):
    embeddings._openai_client = object()
    embeddings._hugging_face_client = object()
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"ready": True}


# ---------------------------
# Auth
# ---------------------------

PROTECTED = [
    ("post", "/v1/retrieve", {"query": "x", "collection": "c"}),
    ("post", "/v1/documents/extract", {"url": "https://example.com/cv.pdf"}),
    ("get", "/v1/knowledge_sources", None),
    ("post", "/v1/knowledge_sources", {
        "name": "n", "knowledge_source_type": "competency",
        "file_url": "u", "file_type": "csv"}),
    ("post", "/v1/knowledge_sources/delete", {"names": ["n"]}),
]


@pytest.mark.parametrize("method,path,body", PROTECTED)
def test_rejects_missing_token(client, method, path, body):
    response = getattr(client, method)(path, json=body) if body else getattr(client, method)(path)
    assert response.status_code == 401


@pytest.mark.parametrize("method,path,body", PROTECTED)
def test_rejects_wrong_token(client, method, path, body):
    headers = {"X-Internal-Token": "wrong-token-0123456789abcdef"}
    response = (
        getattr(client, method)(path, json=body, headers=headers)
        if body else getattr(client, method)(path, headers=headers)
    )
    assert response.status_code == 401


# ---------------------------
# Retrieval
# ---------------------------

def test_retrieve_returns_ranked_documents(client, auth):
    response = client.post("/v1/retrieve", headers=auth, json={
        "query": "I led a migration", "collection": "numee_competency", "top_k": 3,
    })
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 3
    assert set(results[0]) == {"id", "score", "text", "metadata"}
    assert [r["score"] for r in results] == sorted((r["score"] for r in results), reverse=True)
    assert results[0]["metadata"]["Competency"] == "competency-0"


def test_retrieve_passes_defaults_through(client, stub, auth):
    client.post("/v1/retrieve", headers=auth, json={"query": "q", "collection": "c"})
    assert stub.calls[-1] == ("retrieve", "q", "c", 10, 0.6, None)


@pytest.mark.parametrize("body", [
    {"query": "", "collection": "c"},
    {"query": "q", "collection": ""},
    {"query": "q", "collection": "c", "limit": 0},
    {"query": "q", "collection": "c", "limit": 101},
    {"query": "q", "collection": "c", "alpha": 1.5},
    {"query": "q", "collection": "c", "top_k": 0},
    {"collection": "c"},
])
def test_retrieve_rejects_bad_input(client, auth, body):
    assert client.post("/v1/retrieve", headers=auth, json=body).status_code == 422


# ---------------------------
# Documents
# ---------------------------

def test_extract_document_returns_pages(client, auth):
    response = client.post("/v1/documents/extract", headers=auth,
                           json={"url": "https://example.com/cv.pdf"})
    assert response.status_code == 200
    assert response.json() == {
        "pages": [{"page_content": "page one", "metadata": {"page": 0, "total_pages": 1}}]
    }


def test_extract_document_rejects_empty_url(client, auth):
    assert client.post("/v1/documents/extract", headers=auth, json={"url": ""}).status_code == 422


# ---------------------------
# Knowledge sources
# ---------------------------

def test_list_knowledge_sources(client, auth):
    response = client.get("/v1/knowledge_sources", headers=auth)
    assert response.status_code == 200
    assert response.json() == {"collections": ["numee_competency", "numee_motivation"]}


def test_create_knowledge_source(client, auth):
    response = client.post("/v1/knowledge_sources", headers=auth, json={
        "name": "numee_jobs", "knowledge_source_type": "job_profile",
        "file_url": "https://example.com/jobs.csv", "file_type": "csv",
    })
    assert response.status_code == 200
    assert response.json() == {
        "success": True, "message": "Knowledge Source Created Successfully",
    }


def test_create_knowledge_source_is_not_a_silent_overwrite(client, auth):
    response = client.post("/v1/knowledge_sources", headers=auth, json={
        "name": "numee_competency", "knowledge_source_type": "competency",
        "file_url": "https://example.com/c.csv", "file_type": "csv",
    })
    assert response.json()["success"] is False
    assert "Already Exists" in response.json()["message"]


@pytest.mark.parametrize("body", [
    {"name": "n", "knowledge_source_type": "nope", "file_url": "u", "file_type": "csv"},
    {"name": "n", "knowledge_source_type": "competency", "file_url": "u", "file_type": "pdf"},
    {"name": "", "knowledge_source_type": "competency", "file_url": "u", "file_type": "csv"},
    {"name": "n" * 51, "knowledge_source_type": "competency", "file_url": "u", "file_type": "csv"},
])
def test_create_knowledge_source_rejects_bad_input(client, auth, body):
    assert client.post("/v1/knowledge_sources", headers=auth, json=body).status_code == 422


def test_delete_only_reports_collections_that_existed(client, auth):
    response = client.post("/v1/knowledge_sources/delete", headers=auth,
                           json={"names": ["numee_competency", "ghost"]})
    assert response.status_code == 200
    assert response.json() == {"deleted": ["numee_competency"]}


def test_delete_rejects_empty_list(client, auth):
    assert client.post("/v1/knowledge_sources/delete", headers=auth,
                       json={"names": []}).status_code == 422
