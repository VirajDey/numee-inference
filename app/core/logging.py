import logging
import sys

from app.core.config import settings

_configured = False


def configure_logging() -> None:
    """Log to stdout. The service runs in a container, so the platform's log
    collector owns retention - writing files inside the container would only
    hide them."""
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s")
    )

    root = logging.getLogger("app")
    root.setLevel(settings.log_level.upper())
    root.handlers = [handler]
    root.propagate = False

    _configured = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f"app.{name}")
