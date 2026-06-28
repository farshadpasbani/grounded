"""Graph retrieval: match question entities -> traversal paths as evidence.

  >=2 known entities -> shortest path(s) between them.
  1 entity           -> its 1-hop neighbourhood.
  0 entities / no path within max hops -> empty evidence (the caller abstains).
"""
from __future__ import annotations

import re
from itertools import combinations

from .. import config
from . import store, terms
from .types import GraphPath


def match_entities(question: str) -> list[str]:
    """Node names present in the graph that the question refers to.

    Dictionary aliases resolve to their canonical Technology/Concept node;
    Project nodes are matched whole-word by name. Deterministic order, deduped.
    """
    nodes = store.node_names()
    out: list[str] = []

    # dictionary terms (alias-aware) that are actually in the graph
    for name, _typ in terms.match_terms(question):
        if name in nodes and name not in out:
            out.append(name)

    # project (and any other) node names matched whole-word
    for name in sorted(nodes):
        if name in out:
            continue
        if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", question, re.IGNORECASE):
            out.append(name)

    return out


def graph_evidence(question: str, max_hops: int | None = None) -> list[GraphPath]:
    hops = max_hops or config.settings.graph_max_hops
    entities = match_entities(question)
    if not entities:
        return []
    if len(entities) == 1:
        return store.neighbourhood(entities[0])
    paths: list[GraphPath] = []
    for a, b in combinations(entities, 2):
        paths += store.shortest_paths(a, b, hops)
    return paths
