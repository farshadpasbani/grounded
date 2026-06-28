"""Graph store behind one interface, two backends.

  - in-memory (pure-python adjacency): selected when GRAPH_PATH == ":memory:".
    No kuzu, no torch -- this is the seam the offline tests and CI run on.
  - embedded KuzuDB: selected for any other GRAPH_PATH (a directory). Cypher,
    no Docker. Imported lazily so the keyless path never needs the wheel
    (which has no Python 3.14 build yet).

Both build from triples and answer bounded traversals (shortest path / 1-hop
neighbourhood). max_hops bounds the walk so cyclic relationships can't run away.
"""
from __future__ import annotations

from collections import deque
from functools import lru_cache

from .. import config
from .extract import Triple
from .types import GraphPath, Step  # re-exported: `from .store import GraphPath` still resolves


class InMemoryGraph:
    """Pure-python adjacency. Edges are directed (subject -> object) but the
    traversal is undirected, so two projects sharing a technology are connected
    through it."""

    def __init__(self) -> None:
        self._types: dict[str, str] = {}
        # node -> list[(neighbour, predicate, forward)]
        self._adj: dict[str, list[tuple[str, str, bool]]] = {}

    def reset(self) -> None:
        self._types.clear()
        self._adj.clear()

    def add_triples(self, triples: list[Triple]) -> None:
        for t in triples:
            self._types[t.subj] = t.subj_type
            self._types[t.obj] = t.obj_type
            self._adj.setdefault(t.subj, []).append((t.obj, t.pred, True))
            self._adj.setdefault(t.obj, []).append((t.subj, t.pred, False))

    def has_node(self, name: str) -> bool:
        return name in self._types

    def node_names(self) -> set[str]:
        return set(self._types)

    def node_type(self, name: str) -> str | None:
        return self._types.get(name)

    def neighbourhood(self, name: str, hops: int = 1) -> list[GraphPath]:
        if name not in self._types or hops < 1:
            return []
        out: list[GraphPath] = []
        for dst, pred, fwd in self._adj.get(name, []):
            out.append(GraphPath((Step(name, pred, dst, fwd),)))
        return out

    def shortest_paths(self, a: str, b: str, max_hops: int) -> list[GraphPath]:
        if a not in self._types or b not in self._types or a == b:
            return []
        # BFS by hop count; collect every path of the first (minimal) length that
        # reaches b. Visited-by-level so cycles can't loop forever.
        frontier: list[tuple[str, tuple[Step, ...]]] = [(a, ())]
        seen = {a}
        for _hop in range(max_hops):
            nxt: list[tuple[str, tuple[Step, ...]]] = []
            found: list[GraphPath] = []
            level_seen: set[str] = set()
            for node, path in frontier:
                for dst, pred, fwd in self._adj.get(node, []):
                    if dst in seen:
                        continue
                    new_path = path + (Step(node, pred, dst, fwd),)
                    if dst == b:
                        found.append(GraphPath(new_path))
                    else:
                        nxt.append((dst, new_path))
                        level_seen.add(dst)
            if found:
                return found
            seen |= level_seen
            frontier = nxt
        return []

    def stats(self) -> dict:
        edges = sum(len(v) for v in self._adj.values()) // 2
        return {"nodes": len(self._types), "edges": edges}


@lru_cache(maxsize=1)
def _backend():
    if config.settings.graph_path == ":memory:":
        return InMemoryGraph()
    from .kuzu_store import KuzuGraph  # lazy: only when an embedded path is set

    return KuzuGraph(config.settings.graph_path)


def reset() -> None:
    _backend().reset()


def add_triples(triples: list[Triple]) -> None:
    _backend().add_triples(triples)


def has_node(name: str) -> bool:
    return _backend().has_node(name)


def node_names() -> set[str]:
    return _backend().node_names()


def node_type(name: str) -> str | None:
    return _backend().node_type(name)


def neighbourhood(name: str, hops: int = 1) -> list[GraphPath]:
    return _backend().neighbourhood(name, hops)


def shortest_paths(a: str, b: str, max_hops: int) -> list[GraphPath]:
    return _backend().shortest_paths(a, b, max_hops)


def stats() -> dict:
    return _backend().stats()
