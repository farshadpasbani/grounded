"""Markdown chunking: split by heading, then window long sections by words."""
from __future__ import annotations

from dataclasses import dataclass

from . import config


@dataclass
class Chunk:
    source: str   # file path the chunk came from
    heading: str  # nearest heading, for citation context
    text: str


def _sections(markdown: str) -> list[tuple[str, str]]:
    """(heading, body) pairs. Text before the first heading gets an empty one."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in markdown.splitlines():
        if line.lstrip().startswith("#"):
            sections.append((line.lstrip("# ").strip(), []))
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(body).strip()) for h, body in sections if "\n".join(body).strip()]


def _window(words: list[str], size: int, overlap: int):
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        yield words[start : start + size]
        if start + size >= len(words):
            break


def chunk_markdown(source: str, markdown: str) -> list[Chunk]:
    out: list[Chunk] = []
    for heading, body in _sections(markdown):
        words = body.split()
        for window in _window(words, config.settings.chunk_words, config.settings.chunk_overlap):
            out.append(Chunk(source=source, heading=heading, text=" ".join(window)))
    return out
