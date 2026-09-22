from qdrant_client import QdrantClient, models

from app.core.config import settings
from app.core.exceptions import InferenceError
from app.core.logging import get_logger

logger = get_logger(__name__)

qdrant_client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
    check_compatibility=False,
)

DEFAULT_COLLECTION = "numee_competency_multivector_representation"


class QdrantStore:
    def list_collections(self) -> list[str]:
        try:
            collections = qdrant_client.get_collections()
            return [collection.name for collection in collections.collections]
        except Exception as e:
            logger.exception("Listing Qdrant collections failed")
            raise InferenceError("Failed to list collections") from e

    def delete_collection(self, collection_name: str) -> None:
        try:
            qdrant_client.delete_collection(collection_name)
        except Exception as e:
            logger.exception("Deleting collection %s failed", collection_name)
            raise InferenceError("Failed to delete collection") from e

    def create_collection(self, collection_name: str = DEFAULT_COLLECTION) -> None:
        try:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "openai": models.VectorParams(
                        size=settings.openai_vector_size,
                        distance=models.Distance.COSINE,
                    ),
                    "sentence_transformers": models.VectorParams(
                        size=settings.sentence_transformer_vector_size,
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        ),
                        hnsw_config=models.HnswConfigDiff(m=0),
                    ),
                },
            )
        except Exception as e:
            logger.exception("Creating collection %s failed", collection_name)
            raise InferenceError("Failed to create collection") from e

    def to_points(self, documents, openai_embedded_texts, hugging_face_embedded_texts):
        try:
            return [
                models.PointStruct(
                    id=index,
                    vector={
                        "openai": openai_embedded_texts[index],
                        "sentence_transformers": hugging_face_embedded_texts[index],
                    },
                    payload={"text": data.page_content, "metadata": data.metadata},
                )
                for index, data in enumerate(documents)
            ]
        except Exception as e:
            logger.exception("Building Qdrant points failed")
            raise InferenceError("Failed to build points") from e

    def upload_points(self, points, collection_name: str = DEFAULT_COLLECTION) -> None:
        try:
            qdrant_client.upload_points(
                collection_name=collection_name,
                points=points,
                batch_size=8,
            )
        except Exception as e:
            logger.exception("Uploading points to %s failed", collection_name)
            raise InferenceError("Failed to upload points") from e

    def query(self, openai_embedded_query, huggingface_embedded_query, collection_name: str, limit: int = 10):
        """Both vector spaces are queried separately and fused afterwards."""
        try:
            openai_results = qdrant_client.query_points(
                collection_name=collection_name,
                query=openai_embedded_query,
                using="openai",
                limit=limit,
                with_payload=True,
            )
            sentence_transformers_results = qdrant_client.query_points(
                collection_name=collection_name,
                query=huggingface_embedded_query,
                using="sentence_transformers",
                limit=limit,
                with_payload=True,
            )
            return openai_results, sentence_transformers_results
        except Exception as e:
            logger.exception("Querying collection %s failed", collection_name)
            raise InferenceError("Failed to query collection") from e
