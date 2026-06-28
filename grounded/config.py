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
    # Every env-driven field reads the environment at instantiation time, via
    # default_factory, so `config.Settings()` re-reads a changed environment
    # uniformly -- and a test that swaps in a fresh `config.settings` flips
    # every consumer at once. Pure constants (no env) stay as plain defaults.

    # Vector store. Default to embedded Qdrant (a local directory, no Docker).
    # Set QDRANT_URL to point at a running server instead.
    qdrant_url: str = field(default_factory=lambda: os.environ.get("QDRANT_URL", ""))
    qdrant_path: str = field(default_factory=lambda: os.environ.get("QDRANT_PATH", ".qdrant"))
    collection: str = field(default_factory=lambda: os.environ.get("GROUNDED_COLLECTION", "grounded"))

    embed_model: str = field(
        default_factory=lambda: os.environ.get("GROUNDED_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    )
    embed_dim: int = 384  # bge-small-en-v1.5

    # Generation backend: "stub" needs no API key (deterministic, for offline
    # development and tests); "claude" is the production path.
    generator: str = field(default_factory=lambda: os.environ.get("GROUNDED_GENERATOR", "stub"))
    gen_model: str = field(
        default_factory=lambda: os.environ.get("GROUNDED_GEN_MODEL", "claude-opus-4-8")
    )

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
    min_score: float = field(
        default_factory=lambda: float(os.environ.get("GROUNDED_MIN_SCORE", "0.68"))
    )

    protected_topics: list[str] = field(default_factory=_topics)

    # --- graph retrieval -----------------------------------------------------
    # Retrieval mode: "vector" (default, unchanged), "graph", or "hybrid".
    # Extractor: "rule" (deterministic, keyless, default) or "llm" (Claude).
    # GRAPH_PATH: ":memory:" -> pure-python in-memory seam (no kuzu); any other
    # value -> embedded KuzuDB at that directory. GRAPH_MAX_HOPS bounds traversal
    # so cyclic relationships can't run away.
    retrieval_mode: str = field(
        default_factory=lambda: os.environ.get("GROUNDED_RETRIEVAL_MODE", "vector")
    )
    extractor: str = field(
        default_factory=lambda: os.environ.get("GROUNDED_EXTRACTOR", "rule")
    )
    graph_path: str = field(
        default_factory=lambda: os.environ.get("GRAPH_PATH", ":memory:")
    )
    graph_max_hops: int = field(
        default_factory=lambda: int(os.environ.get("GRAPH_MAX_HOPS", "3"))
    )


settings = Settings()
