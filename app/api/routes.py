from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from app.api.deps import get_service
from app.api.schemas import (
    CreateKnowledgeSourceRequest,
    CreateKnowledgeSourceResponse,
    DeleteKnowledgeSourcesRequest,
    DeleteKnowledgeSourcesResponse,
    ExtractDocumentRequest,
    ExtractDocumentResponse,
    HealthResponse,
    ListKnowledgeSourcesResponse,
    ReadyResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from app.api.security import require_service_token
from app.services import embeddings
from app.services.inference import InferenceService

## Probes are unauthenticated so orchestrators can reach them without holding
## the service token.
probes = APIRouter(tags=["Probes"])

## Everything else requires the shared secret.
v1 = APIRouter(prefix="/v1", dependencies=[Depends(require_service_token)])


@probes.get("/health", response_model=HealthResponse)
async def health(service: InferenceService = Depends(get_service)):
    payload = await service.health()
    return {"status": "OK" if all(payload.values()) else "DEGRADED", **payload}


@probes.get("/ready", response_model=ReadyResponse)
async def ready():
    """Readiness gate for the load balancer - a replica whose model is still
    loading must not be sent traffic."""
    if not embeddings.is_warm():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"ready": False},
        )
    return {"ready": True}


@v1.post("/retrieve", response_model=RetrieveResponse, tags=["Retrieval"])
async def retrieve(data: RetrieveRequest, service: InferenceService = Depends(get_service)):
    results = await service.retrieve(
        query=data.query,
        collection=data.collection,
        limit=data.limit,
        alpha=data.alpha,
        top_k=data.top_k,
    )
    return {"results": results}


@v1.post("/documents/extract", response_model=ExtractDocumentResponse, tags=["Documents"])
async def extract_document(data: ExtractDocumentRequest, service: InferenceService = Depends(get_service)):
    return {"pages": await service.extract_document(data.url)}


@v1.get("/knowledge_sources", response_model=ListKnowledgeSourcesResponse, tags=["Knowledge Sources"])
async def list_knowledge_sources(service: InferenceService = Depends(get_service)):
    return {"collections": await service.list_collections()}


@v1.post("/knowledge_sources", response_model=CreateKnowledgeSourceResponse, tags=["Knowledge Sources"])
async def create_knowledge_source(data: CreateKnowledgeSourceRequest, service: InferenceService = Depends(get_service)):
    return await service.create_knowledge_source(
        name=data.name,
        file_url=data.file_url,
        file_type=data.file_type,
        knowledge_source_type=data.knowledge_source_type,
    )


@v1.post("/knowledge_sources/delete", response_model=DeleteKnowledgeSourcesResponse, tags=["Knowledge Sources"])
async def delete_knowledge_sources(data: DeleteKnowledgeSourcesRequest, service: InferenceService = Depends(get_service)):
    return {"deleted": await service.delete_collections(data.names)}
