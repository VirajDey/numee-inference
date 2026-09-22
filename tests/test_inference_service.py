"""Service internals: score fusion, collection handling and the concurrency bound."""
import asyncio
from types import SimpleNamespace

import pytest

from app.services.fusion import combine_results
from app.services.inference import InferenceService


def points(*pairs):
    """Build a Qdrant-shaped result set from (id, score) pairs."""
    return SimpleNamespace(points=[
        SimpleNamespace(id=pid, score=score, payload={"text": f"text-{pid}", "metadata": {"name": f"n-{pid}"}})
        for pid, score in pairs
    ])


# ---------------------------
# Fusion
# ---------------------------

def test_combine_normalises_each_space_before_blending():
    # Raw scores are on wildly different scales (0.7-0.9 against 12-30), so only
    # the within-space ranking may influence the result.
    openai = points((1, 0.90), (2, 0.80), (3, 0.70))          # normalised: 1.0, 0.5, 0.0
    transformers = points((1, 12.0), (2, 30.0), (3, 21.0))    # normalised: 0.0, 1.0, 0.5

    combined = combine_results(openai, transformers, alpha=0.6)

    # 2 wins on the blend despite ranking second in both spaces individually.
    assert [doc["id"] for doc in combined] == [2, 1, 3]
    assert combined[0]["score"] == 0.7   # 0.6*0.5 + 0.4*1.0
    assert combined[1]["score"] == 0.6   # 0.6*1.0 + 0.4*0.0
    assert combined[2]["score"] == 0.2   # 0.6*0.0 + 0.4*0.5


def test_alpha_shifts_the_winner_between_spaces():
    openai = points((1, 1.0), (2, 0.0))
    transformers = points((1, 0.0), (2, 1.0))

    assert combine_results(openai, transformers, alpha=1.0)[0]["id"] == 1
    assert combine_results(openai, transformers, alpha=0.0)[0]["id"] == 2


def test_combine_unions_documents_found_by_only_one_space():
    openai = points((1, 0.9), (2, 0.5))
    transformers = points((3, 0.9), (2, 0.5))

    combined = combine_results(openai, transformers)

    assert {doc["id"] for doc in combined} == {1, 2, 3}


def test_combine_flattens_the_payload():
    combined = combine_results(points((7, 0.9), (8, 0.1)), points((7, 0.9), (8, 0.1)))

    assert combined[0] == {"id": 7, "score": 1.0, "text": "text-7", "metadata": {"name": "n-7"}}


def test_combine_survives_identical_scores():
    # min == max, so normalisation divides by the epsilon rather than by zero.
    combined = combine_results(points((1, 0.5), (2, 0.5)), points((1, 0.5), (2, 0.5)))

    assert len(combined) == 2
    assert all(doc["score"] == 0.0 for doc in combined)


# ---------------------------
# Collections
# ---------------------------

class FakeStore:
    def __init__(self, collections=None, fail_on_upload=False):
        self.collections = list(collections or [])
        self.deleted = []
        self.created = []
        self.fail_on_upload = fail_on_upload

    def list_collections(self):
        return list(self.collections)

    def delete_collection(self, name):
        self.deleted.append(name)

    def create_collection(self, name):
        self.created.append(name)

    def to_points(self, documents, openai_embeds, hf_embeds):
        return list(zip(documents, openai_embeds, hf_embeds))

    def upload_points(self, points, name):
        if self.fail_on_upload:
            raise RuntimeError("qdrant is down")


def test_delete_collections_skips_unknown_names():
    store = FakeStore(["a", "b"])
    service = InferenceService(store=store)

    deleted = asyncio.run(service.delete_collections(["a", "ghost"]))

    assert deleted == ["a"]
    assert store.deleted == ["a"]


def test_create_knowledge_source_refuses_to_overwrite():
    store = FakeStore(["existing"])
    service = InferenceService(store=store)

    result = asyncio.run(service.create_knowledge_source(
        name="existing", file_url="u", file_type="csv", knowledge_source_type="competency"))

    assert result["success"] is False
    assert store.created == []


def test_create_knowledge_source_rolls_back_the_collection_it_made(monkeypatch):
    store = FakeStore(fail_on_upload=True)
    service = InferenceService(store=store)

    monkeypatch.setattr("app.services.inference.get_loader_by_type",
                        lambda _: SimpleNamespace(load_and_processed_excel=lambda filepath, file_type: "df"))
    monkeypatch.setattr("app.services.inference.dataframe_document_loader",
                        lambda df, col: [SimpleNamespace(page_content="a", metadata={})])
    monkeypatch.setattr("app.services.embeddings.embed_documents", lambda texts: ([[0.1]], [[0.2]]))

    with pytest.raises(Exception):
        asyncio.run(service.create_knowledge_source(
            name="new", file_url="u", file_type="csv", knowledge_source_type="competency"))

    assert store.created == ["new"]
    assert store.deleted == ["new"], "a collection created then failed must not be left behind"


def test_failure_before_creation_does_not_delete_anything(monkeypatch):
    store = FakeStore()
    service = InferenceService(store=store)

    def explode(filepath, file_type):
        raise RuntimeError("bad spreadsheet")

    monkeypatch.setattr("app.services.inference.get_loader_by_type",
                        lambda _: SimpleNamespace(load_and_processed_excel=explode))

    with pytest.raises(Exception):
        asyncio.run(service.create_knowledge_source(
            name="new", file_url="u", file_type="csv", knowledge_source_type="competency"))

    assert store.created == []
    assert store.deleted == [], "nothing was created, so nothing should be dropped"


# ---------------------------
# Concurrency bound
# ---------------------------

def test_cpu_work_is_capped_at_max_concurrency():
    """The whole point of the bound: a caller fanning out over many documents
    must not start one model invocation per document."""
    service = InferenceService(store=FakeStore(), max_concurrency=3)
    in_flight = 0
    peak = 0
    lock = __import__("threading").Lock()

    def slow_work():
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        __import__("time").sleep(0.05)
        with lock:
            in_flight -= 1
        return "done"

    async def drive():
        return await asyncio.gather(*[service._run(slow_work) for _ in range(12)])

    results = asyncio.run(drive())

    assert results == ["done"] * 12
    assert peak <= 3, f"ran {peak} concurrently, bound was 3"
