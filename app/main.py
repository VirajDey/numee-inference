import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.api.routes import probes, v1
from app.core.exceptions import InferenceError
from app.core.logging import configure_logging, get_logger
from app.services import embeddings

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ## Load the models before the first request rather than on it - a cold
    ## replica would otherwise charge the first caller for the load.
    configure_logging()
    logger.info("Warming up embedding models")
    await asyncio.to_thread(embeddings.warm_up)
    logger.info("Inference service ready")
    yield


app = FastAPI(
    title="Numee Inference Service",
    description=(
        "Embeddings, vector retrieval and document extraction for the Numee "
        "agents API. Internal network only."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(probes)
app.include_router(v1)


@app.exception_handler(InferenceError)
async def inference_error_handler(request: Request, exc: InferenceError):
    """Failures are logged where they happen. The response stays generic: the
    inputs are CVs and job descriptions, and error detail is an easy way for
    personal data to leak back to a caller."""
    logger.error("%s failed: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)},
    )
