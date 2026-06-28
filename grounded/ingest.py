"""Build the index, dispatched by GROUNDED_RETRIEVAL_MODE.

  vector (default): read -> chunk -> ip-filter -> embed -> Qdrant. UNCHANGED.
  graph:            read -> rule-extract triples (ip-guarded) -> graph store.
  hybrid:           both.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import yaml

from . import config, embed, ipguard, store
from .chunk import Chunk, chunk_markdown
from .graph import extract
from .graph import store as gstore


def _load_sources(sources_path: str) -> list[Path]:
    spec = yaml.safe_load(Path(sources_path).read_text())
    paths: list[Path] = []
    for pattern in spec.get("globs", []):
        expanded = os.path.expanduser(pattern)
        paths += [Path(p) for p in glob.glob(expanded, recursive=True)]
    return sorted(set(paths))


def _project_name(path: Path, text: str) -> str:
    """Project node name: the first non-empty markdown heading, else the filename
    stem. An empty heading (`## ` / hashes-only) must not yield a "" node -- that
    would match every question's whole-word regex and silently ground junk."""
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            name = line.lstrip("# ").strip()
            if name:
                return name
    return path.stem


def _build_vector(files: list[Path]) -> dict:
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


def _build_graph(files: list[Path]) -> dict:
    docs: list[extract.Doc] = []
    for f in files:
        text = f.read_text()  # read once: feed both the name derivation and the body
        docs.append(extract.Doc(name=_project_name(f, text), text=text))
    triples = extract.extract(docs)  # ip-guarded inside the extractor

    gstore.reset()
    gstore.add_triples(triples)

    s = gstore.stats()
    return {"files": len(files), "nodes": s["nodes"], "edges": s["edges"]}


def build(sources_path: str = "sources.yaml") -> dict:
    files = _load_sources(sources_path)
    mode = config.settings.retrieval_mode

    if mode == "graph":
        return _build_graph(files)
    if mode == "hybrid":
        return {**_build_vector(files), **_build_graph(files)}
    return _build_vector(files)
