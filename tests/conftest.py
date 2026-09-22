import os

## Settings are validated at import time, so the test environment has to be in
## place before anything under app/ is imported.
os.environ.setdefault("INFERENCE_SERVICE_TOKEN", "test-token-0123456789abcdef")
os.environ.setdefault("QDRANT_URL", "http://qdrant.invalid:6333")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.api.deps import get_service  # noqa: E402
from app.main import app  # noqa: E402
from app.services import embeddings  # noqa: E402

TOKEN = os.environ["INFERENCE_SERVICE_TOKEN"]


class StubService:
    """Stands in for the Qdrant and OpenAI backed service, so the API surface
    can be tested without either."""

    def __init__(self):
        self.collections = ["numee_competency", "numee_motivation"]
        self.calls = []

    async def health(self):
        return {"qdrant_connection": True, "embedding_models_loaded": embeddings.is_warm()}

    async def list_collections(self):
        return list(self.collections)

    async def delete_collections(self, names):
        deleted = [n for n in names if n in self.collections]
        self.collections = [c for c in self.collections if c not in deleted]
        return deleted

    async def create_knowledge_source(self, name, file_url, file_type, knowledge_source_type):
        self.calls.append(("create", name, knowledge_source_type, file_type))
        if name in self.collections:
            return {"success": False, "message": f"{name} Knowledge Source Already Exists"}
        self.collections.append(name)
        return {"success": True, "message": "Knowledge Source Created Successfully"}

    async def retrieve(self, query, collection, limit=10, alpha=0.6, top_k=None):
        self.calls.append(("retrieve", query, collection, limit, alpha, top_k))
        return [
            {
                "id": i,
                "score": round(0.9 - i / 10, 2),
                "text": f"Definition\nKey dims\nDescription/Keywords : \nalpha-{i}\nbeta\nomega\n",
                "metadata": {"Competency": f"competency-{i}", "Catagory": "Social"},
            }
            for i in range(top_k or limit)
        ]

    async def extract_document(self, url):
        self.calls.append(("extract", url))
        return [{"page_content": "page one", "metadata": {"page": 0, "total_pages": 1}}]


@pytest.fixture
def stub():
    return StubService()


@pytest.fixture
def client(stub):
    app.dependency_overrides[get_service] = lambda: stub
    ## Deliberately not used as a context manager: that would run lifespan and
    ## download the real embedding model. Skipping it also leaves the readiness
    ## gate cold, so both sides of it can be exercised.
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def auth():
    return {"X-Internal-Token": TOKEN}


@pytest.fixture(autouse=True)
def reset_warm_state():
    """Each test starts with the models cold."""
    embeddings._openai_client = None
    embeddings._hugging_face_client = None
    yield
    embeddings._openai_client = None
    embeddings._hugging_face_client = None
