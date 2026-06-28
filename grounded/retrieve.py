"""Embed the question, pull the top-k chunks."""
from __future__ import annotations

from . import config, embed, store
from .store import Hit


def retrieve(question: str, k: int | None = None) -> list[Hit]:
    return store.search(embed.embed_one(question), k or config.settings.top_k)
