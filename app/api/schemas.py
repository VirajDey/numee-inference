from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    query: Annotated[str, Field(..., min_length=1, description="Text to embed and search with")]
    collection: Annotated[str, Field(..., min_length=1, description="Qdrant collection to search")]
    limit: Annotated[int, Field(default=10, ge=1, le=100, description="Points to pull from each vector space")]
    alpha: Annotated[float, Field(default=0.6, ge=0, le=1, description="Weight given to the openai vector space during fusion")]
    top_k: Annotated[Optional[int], Field(default=None, ge=1, description="Trim the fused result to this many documents")]


class RetrievedDocument(BaseModel):
    id: Any
    score: float
    text: Optional[str] = None
    metadata: Dict[str, Any] = {}


class RetrieveResponse(BaseModel):
    results: List[RetrievedDocument]


class ExtractDocumentRequest(BaseModel):
    url: Annotated[str, Field(..., min_length=1, description="Location of the PDF to parse")]


class DocumentPage(BaseModel):
    page_content: str
    metadata: Dict[str, Any] = {}


class ExtractDocumentResponse(BaseModel):
    pages: List[DocumentPage]


class CreateKnowledgeSourceRequest(BaseModel):
    name: Annotated[str, Field(..., min_length=1, max_length=50)]
    knowledge_source_type: Literal["motivation", "competency", "job_profile"]
    file_url: Annotated[str, Field(..., min_length=1)]
    file_type: Literal["csv", "excel"]


class CreateKnowledgeSourceResponse(BaseModel):
    success: bool
    message: Optional[str] = None


class ListKnowledgeSourcesResponse(BaseModel):
    collections: List[str]


class DeleteKnowledgeSourcesRequest(BaseModel):
    names: Annotated[List[str], Field(..., min_length=1)]


class DeleteKnowledgeSourcesResponse(BaseModel):
    deleted: List[str]


class HealthResponse(BaseModel):
    status: str
    qdrant_connection: bool
    embedding_models_loaded: bool


class ReadyResponse(BaseModel):
    ready: bool
