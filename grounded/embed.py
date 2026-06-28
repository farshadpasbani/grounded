"""Local embeddings. The model loads once, lazily; nothing leaves the machine."""
from __future__ import annotations

from functools import lru_cache

from . import config


@lru_cache(maxsize=1)
def _model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:  # the heavy ML dep is an optional extra
        raise ImportError(
            "Real embeddings need sentence-transformers. Install with: "
            "pip install -e '.[embed]'  (needs a torch-supported Python)."
        ) from e

    return SentenceTransformer(config.settings.embed_model)


def embed(texts: list[str]) -> list[list[float]]:
    # normalize_embeddings=True so a plain cosine in Qdrant is the similarity.
    vectors = _model().encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vectors]


def embed_one(text: str) -> list[float]:
    return embed([text])[0]
