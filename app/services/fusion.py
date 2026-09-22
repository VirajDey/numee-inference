from collections import defaultdict

import numpy as np

from app.core.exceptions import RetrievalError
from app.core.logging import get_logger

logger = get_logger(__name__)


def normalize(scores):
    s = np.array(scores)
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


def combine_results(openai_results, sentence_transformers_results, alpha: float = 0.6) -> list[dict]:
    """Weighted blend of the two vector spaces, min-max normalised within each.

    Runs here rather than in the caller so only the ranked result crosses the
    network instead of both raw result sets.
    """
    try:
        scores = defaultdict(lambda: {"openai": 0, "sentence_transformer": 0, "payload": None})

        openai_norm = normalize([p.score for p in openai_results.points])
        for p, s in zip(openai_results.points, openai_norm):
            scores[p.id]["openai"] = s
            scores[p.id]["payload"] = p.payload

        sentence_transformers_norm = normalize([p.score for p in sentence_transformers_results.points])
        for p, s in zip(sentence_transformers_results.points, sentence_transformers_norm):
            scores[p.id]["sentence_transformer"] = s
            scores[p.id]["payload"] = p.payload

        combined = sorted(
            [
                (pid, alpha * v["openai"] + (1 - alpha) * v["sentence_transformer"], v["payload"])
                for pid, v in scores.items()
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            {
                "id": pid,
                "score": round(float(score), 2),
                "text": (payload or {}).get("text"),
                "metadata": (payload or {}).get("metadata", {}),
            }
            for pid, score, payload in combined
        ]
    except Exception as e:
        logger.exception("Fusing retrieval results failed")
        raise RetrievalError("Failed to combine results") from e
