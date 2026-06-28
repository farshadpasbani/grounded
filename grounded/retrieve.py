"""Embed the question, pull the top-k chunks."""
from __future__ import annotations

from . import embed, store
from .config import settings
from .store import Hit


def retrieve(question: str, k: int | None = None) -> list[Hit]:
    return store.search(embed.embed_one(question), k or settings.top_k)
