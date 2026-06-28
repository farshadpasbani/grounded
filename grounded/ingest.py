"""Build the index: read sources -> chunk -> ip-filter -> embed -> Qdrant."""
from __future__ import annotations

import glob
import os
from pathlib import Path

import yaml

from . import embed, ipguard, store
from .chunk import Chunk, chunk_markdown


def _load_sources(sources_path: str) -> list[Path]:
    spec = yaml.safe_load(Path(sources_path).read_text())
    paths: list[Path] = []
    for pattern in spec.get("globs", []):
        expanded = os.path.expanduser(pattern)
        paths += [Path(p) for p in glob.glob(expanded, recursive=True)]
    return sorted(set(paths))


def build(sources_path: str = "sources.yaml") -> dict:
    files = _load_sources(sources_path)

    chunks: list[Chunk] = []
    for f in files:
        chunks += chunk_markdown(str(f), f.read_text())

    # IP guard at ingest: a chunk that mentions a protected term never enters
    # the index, so it can never be retrieved or surfaced.
    kept = [c for c in chunks if ipguard.is_clean(c.text)]
    dropped = len(chunks) - len(kept)

    store.reset_collection()
    if kept:
        store.upsert(kept, embed.embed([c.text for c in kept]))

    return {"files": len(files), "chunks": len(kept), "ip_dropped": dropped}
