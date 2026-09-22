from functools import lru_cache

from app.services.inference import InferenceService


@lru_cache(maxsize=1)
def get_service() -> InferenceService:
    """Single instance per process, so the concurrency limiter is shared across
    requests. Injected as a dependency so tests can override it."""
    return InferenceService()
