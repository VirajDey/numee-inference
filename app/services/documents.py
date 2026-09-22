from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.exceptions import DocumentExtractionError
from app.core.logging import get_logger

logger = get_logger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=3, multiplier=1))
def extract_document_content(file_url: str):
    """Download and parse a PDF. Blocking and CPU bound - callers run it in a
    thread rather than on the event loop."""
    try:
        return PyMuPDFLoader(file_url).load()
    except Exception as e:
        ## The url points at a candidate's CV, so it stays out of the message.
        logger.exception("PDF extraction failed")
        raise DocumentExtractionError("Failed to extract document") from e


def split_document_content(documents):
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            is_separator_regex=True,
            separators=["\n\n", "\n", r"(?<=\. )", " ", ""],
        )
        return splitter.split_documents(documents)
    except Exception as e:
        logger.exception("Splitting document content failed")
        raise DocumentExtractionError("Failed to split document") from e


def documents_to_payload(documents) -> list[dict]:
    """Documents cross the wire as plain JSON. PyMuPDF metadata can hold values
    json cannot encode, so anything unexpected is stringified."""
    payload = []
    for document in documents:
        metadata = {}
        for key, value in (document.metadata or {}).items():
            metadata[key] = value if isinstance(value, (str, int, float, bool, type(None))) else str(value)
        payload.append({"page_content": document.page_content, "metadata": metadata})
    return payload
