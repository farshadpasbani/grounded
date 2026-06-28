"""Curated dictionary of public technology and concept terms.

The rule extractor scans corpus text against this dictionary to mint graph
nodes. Whole-word, case-insensitive, alias-aware. ONLY public tech/concept
terms live here -- no codenames, no employer IP. Protected terms are env-only
(GROUNDED_PROTECTED_TOPICS) and ip-guarded out at extraction.
"""
from __future__ import annotations

import re
from functools import lru_cache

# Predicate vocabulary. The rule extractor emits USES/APPLIES/RELATED_TO;
# MENTIONS is reserved for the (secondary) llm extractor's generic co-occurrence.
USES = "USES"           # Project -> Technology
APPLIES = "APPLIES"     # Project -> Concept
MENTIONS = "MENTIONS"   # generic co-occurrence (llm extractor)
RELATED_TO = "RELATED_TO"  # Project -> Project (shared Technology/Concept)

# canonical name -> list of lowercase aliases (the canonical name is also an alias)
TECHNOLOGIES: dict[str, list[str]] = {
    "Python": ["python"],
    "TypeScript": ["typescript"],
    "Qdrant": ["qdrant"],
    "KuzuDB": ["kuzudb", "kuzu"],
    "Claude": ["claude"],
    "PyTorch": ["pytorch"],
    "FastAPI": ["fastapi"],
    "Next.js": ["next.js", "nextjs"],
    "sentence-transformers": ["sentence-transformers", "sentence transformers"],
    "GitHub Actions": ["github actions"],
    "pytest": ["pytest"],
    "YAML": ["yaml"],
}

CONCEPTS: dict[str, list[str]] = {
    "deterministic gates": ["deterministic gates", "deterministic gate"],
    "grounding gate": ["grounding gate", "grounding-gate"],
    "RAG": ["rag", "retrieval-augmented generation", "retrieval augmented generation"],
    "knowledge graph": ["knowledge graph", "knowledge graphs"],
    "abstention": ["abstention", "abstain"],
    "agentic loop": ["agentic loop", "agentic loops"],
    "governance": ["governance"],
}

Technology = "Technology"
Concept = "Concept"
Project = "Project"


@lru_cache(maxsize=1)
def _alias_index() -> list[tuple[re.Pattern[str], str, str]]:
    """(compiled whole-word pattern, canonical name, node type), longest first.

    Cached: the dictionary is static, so the patterns are compiled once and
    reused across every scan. Tests that monkeypatch TECHNOLOGIES/CONCEPTS must
    call `_alias_index.cache_clear()` around the change.
    """
    entries: list[tuple[str, str, str]] = []
    for name, aliases in TECHNOLOGIES.items():
        for a in aliases:
            entries.append((a, name, Technology))
    for name, aliases in CONCEPTS.items():
        for a in aliases:
            entries.append((a, name, Concept))
    # longest alias first so multi-word terms win over their fragments
    entries.sort(key=lambda e: len(e[0]), reverse=True)
    out = []
    for alias, name, typ in entries:
        pat = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", re.IGNORECASE)
        out.append((pat, name, typ))
    return out


def match_terms(text: str) -> list[tuple[str, str]]:
    """Distinct (canonical name, node type) pairs found in `text`.

    Whole-word and case-insensitive. Order is deterministic: canonical names are
    returned in longest-alias-first order (the scan order of the alias index),
    not in order of first appearance in `text`.
    """
    found: dict[str, str] = {}
    order: list[str] = []
    for pat, name, typ in _alias_index():
        if name in found:
            continue
        if pat.search(text):
            found[name] = typ
            order.append(name)
    return [(name, found[name]) for name in order]
