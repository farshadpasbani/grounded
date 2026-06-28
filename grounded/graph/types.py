"""Neutral graph value types.

Lives below both the graph store and the vector store so either can describe a
traversal path without importing the other. No backend, no config, no I/O --
just the dataclasses a path is made of.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    src: str
    pred: str
    dst: str
    forward: bool  # True if the underlying edge is src -pred-> dst


@dataclass(frozen=True)
class GraphPath:
    steps: tuple[Step, ...]

    def nodes(self) -> list[str]:
        if not self.steps:
            return []
        return [self.steps[0].src] + [s.dst for s in self.steps]

    def entities(self) -> set[str]:
        return set(self.nodes())

    def render(self) -> str:
        if not self.steps:
            return ""
        out = self.steps[0].src
        for s in self.steps:
            arrow = f" -{s.pred}-> " if s.forward else f" <-{s.pred}- "
            out += arrow + s.dst
        return out
