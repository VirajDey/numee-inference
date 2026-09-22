import hmac

from fastapi import Header, HTTPException, status

from app.core.config import settings


def require_service_token(x_internal_token: str = Header(default="")):
    """This service is internal-network only and is never exposed publicly, but
    a shared secret keeps it from answering anything that reaches the host."""
    if not hmac.compare_digest(x_internal_token, settings.inference_service_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid service token",
        )
