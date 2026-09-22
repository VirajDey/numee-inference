import asyncio

from app.core.config import settings
from app.core.exceptions import KnowledgeSourceError
from app.core.logging import get_logger
from app.services import embeddings
from app.services.documents import documents_to_payload, extract_document_content
from app.services.excel_loaders import dataframe_document_loader, get_loader_by_type
from app.services.fusion import combine_results
from app.services.qdrant_store import QdrantStore

logger = get_logger(__name__)


class InferenceService:
    """Everything this service does is blocking and CPU bound, so each call is
    pushed onto a thread and the pool is bounded - otherwise a caller fanning
    out over twenty resumes would start twenty model invocations at once.
    """

    def __init__(self, store: QdrantStore | None = None, max_concurrency: int | None = None):
        self.store = store or QdrantStore()
        self.semaphore = asyncio.Semaphore(max_concurrency or settings.inference_max_concurrency)

    async def _run(self, func, *args, **kwargs):
        async with self.semaphore:
            return await asyncio.to_thread(func, *args, **kwargs)

    async def health(self) -> dict:
        try:
            await asyncio.to_thread(self.store.list_collections)
            qdrant_connection = True
        except Exception:
            logger.warning("Qdrant health check failed")
            qdrant_connection = False

        return {
            "qdrant_connection": qdrant_connection,
            "embedding_models_loaded": embeddings.is_warm(),
        }

    async def list_collections(self) -> list[str]:
        return await asyncio.to_thread(self.store.list_collections)

    async def delete_collections(self, names: list[str]) -> list[str]:
        existing = await self.list_collections()
        deleted = []
        for name in names:
            if name in existing:
                await asyncio.to_thread(self.store.delete_collection, name)
                deleted.append(name)
        return deleted

    async def retrieve(
        self,
        query: str,
        collection: str,
        limit: int = 10,
        alpha: float = 0.6,
        top_k: int | None = None,
    ) -> list[dict]:
        openai_query_embedding, huggingface_query_embedding = await self._run(
            embeddings.embed_query, query
        )
        openai_results, sentence_transformers_results = await asyncio.to_thread(
            self.store.query,
            openai_query_embedding,
            huggingface_query_embedding,
            collection,
            limit,
        )
        combined = combine_results(openai_results, sentence_transformers_results, alpha)
        return combined[:top_k] if top_k else combined

    async def extract_document(self, url: str) -> list[dict]:
        documents = await self._run(extract_document_content, url)
        return documents_to_payload(documents)

    async def create_knowledge_source(
        self, name: str, file_url: str, file_type: str, knowledge_source_type: str
    ) -> dict:
        existing = await self.list_collections()
        if name in existing:
            return {"success": False, "message": f"{name} Knowledge Source Already Exists"}

        collection_created = False
        try:
            loader = get_loader_by_type(knowledge_source_type)

            df = await self._run(loader.load_and_processed_excel, filepath=file_url, file_type=file_type)
            documents = dataframe_document_loader(df, "page_content")

            openai_embeds, hf_embeds = await self._run(
                embeddings.embed_documents,
                [document.page_content for document in documents],
            )

            points = self.store.to_points(documents, openai_embeds, hf_embeds)

            await asyncio.to_thread(self.store.create_collection, name)
            collection_created = True

            await asyncio.to_thread(self.store.upload_points, points, name)

            logger.info("Knowledge source %s created with %d documents", name, len(documents))
            return {"success": True, "message": "Knowledge Source Created Successfully"}

        except Exception as e:
            logger.exception("Creating knowledge source %s failed", name)
            if collection_created:
                try:
                    await asyncio.to_thread(self.store.delete_collection, name)
                except Exception:
                    logger.exception("Rollback of collection %s failed", name)
            raise KnowledgeSourceError("Failed to create knowledge source") from e
