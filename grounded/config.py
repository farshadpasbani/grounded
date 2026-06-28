"""Settings, resolved from the environment with sensible defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _topics() -> list[str]:
    # Protected terms (employer IP, confidential codenames) must never appear in
    # output. Keep them OUT of source: set GROUNDED_PROTECTED_TOPICS (comma-
    # separated) in your environment. They are never committed; ip-guard filters
    # them at ingest and re-checks every answer.
    extra = os.environ.get("GROUNDED_PROTECTED_TOPICS", "")
    return [t.strip() for t in extra.split(",") if t.strip()]


@dataclass(frozen=True)
class Settings:
    # Vector store. Default to embedded Qdrant (a local directory, no Docker).
    # Set QDRANT_URL to point at a running server instead.
    qdrant_url: str = os.environ.get("QDRANT_URL", "")
    qdrant_path: str = os.environ.get("QDRANT_PATH", ".qdrant")
    collection: str = os.environ.get("GROUNDED_COLLECTION", "grounded")

    embed_model: str = os.environ.get("GROUNDED_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    embed_dim: int = 384  # bge-small-en-v1.5

    # Generation backend: "stub" needs no API key (deterministic, for offline
    # development and tests); "claude" is the production path.
    generator: str = os.environ.get("GROUNDED_GENERATOR", "stub")
    gen_model: str = os.environ.get("GROUNDED_GEN_MODEL", "claude-opus-4-8")

    # Chunking, in words (cheap, dependency-free; good enough for short docs).
    chunk_words: int = 200
    chunk_overlap: int = 40

    top_k: int = 5
    # Abstain when the best retrieved chunk is below this cosine score: the knob
    # that trades recall for honesty. Tuned against eval/questions.yaml with bge
    # over the current corpus: answerable questions score >=0.70, out-of-corpus
    # <=0.65 (the employer-IP probe is the closest, ~0.65), so 0.68 sits in the
    # gap with margin and gives 20/20 groundedness. Re-tune if you change the
    # embedder or the corpus.
    min_score: float = float(os.environ.get("GROUNDED_MIN_SCORE", "0.68"))

    protected_topics: list[str] = field(default_factory=_topics)


settings = Settings()
