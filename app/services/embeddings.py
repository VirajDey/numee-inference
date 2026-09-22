import threading

from app.core.config import settings
from app.core.exceptions import InferenceError
from app.core.logging import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
_openai_client = None
_hugging_face_client = None


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        with _lock:
            if _openai_client is None:
                import openai
                _openai_client = openai.Client(api_key=settings.openai_api_key)
    return _openai_client


def get_hugging_face_client():
    """The sentence-transformers model is several hundred MB of resident memory,
    so it is loaded once per process and only when something needs it."""
    global _hugging_face_client
    if _hugging_face_client is None:
        with _lock:
            if _hugging_face_client is None:
                from langchain_huggingface import HuggingFaceEmbeddings
                logger.info("Loading embedding model %s", settings.huggingface_embedding_model)
                _hugging_face_client = HuggingFaceEmbeddings(
                    model_name=settings.huggingface_embedding_model
                )
                logger.info("Embedding model loaded")
    return _hugging_face_client


def warm_up() -> None:
    """Force both clients to initialise, so readiness reflects a model that is
    actually resident rather than one that will load on the first caller."""
    get_openai_client()
    get_hugging_face_client()


def is_warm() -> bool:
    return _openai_client is not None and _hugging_face_client is not None


def embed_documents(texts: list[str]):
    try:
        result = get_openai_client().embeddings.create(
            input=texts, model=settings.openai_embedding_model
        )
        openai_embedded_texts = [doc.embedding for doc in result.data]
        hugging_face_embedded_texts = get_hugging_face_client().embed_documents(texts)
        return openai_embedded_texts, hugging_face_embedded_texts
    except Exception as e:
        logger.exception("Embedding a document batch failed")
        raise InferenceError("Failed to embed documents") from e


def embed_query(query_text: str):
    try:
        openai_embedded_query = get_openai_client().embeddings.create(
            input=query_text, model=settings.openai_embedding_model
        ).data[0].embedding
        huggingface_embedded_query = get_hugging_face_client().embed_query(query_text)
        return openai_embedded_query, huggingface_embedded_query
    except Exception as e:
        logger.exception("Embedding a query failed")
        raise InferenceError("Failed to embed query") from e
