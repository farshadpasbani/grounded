"""Triple extraction: corpus docs -> typed (subject, predicate, object) triples.

Pluggable backend, selected by GROUNDED_EXTRACTOR:
  - "rule":  deterministic, keyless, the default. Whole-word scan against the
             curated dictionary. ip-guarded so a protected term never becomes a
             node. This is the only path needed offline / in CI.
  - "llm":   Claude structured output -> triples. Wired behind the same
             interface, imported lazily so the keyless path never touches the SDK.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .. import config, ipguard
from . import terms


@dataclass(frozen=True)
class Doc:
    name: str   # Project node name (from doc title or path stem)
    text: str


@dataclass(frozen=True)
class Triple:
    subj: str
    subj_type: str
    pred: str
    obj: str
    obj_type: str


def _rule(docs: list[Doc]) -> list[Triple]:
    triples: list[Triple] = []
    # project name -> set of term names it touches (for RELATED_TO inference)
    project_terms: dict[str, set[str]] = {}

    for doc in docs:
        # ip-guard the Project node itself: a doc whose title is protected is
        # dropped whole, so no node and no edge can carry it.
        if not ipguard.is_clean(doc.name):
            continue
        project_terms.setdefault(doc.name, set())
        for name, typ in terms.match_terms(doc.text):
            if not ipguard.is_clean(name):
                continue  # protected candidate entity: never minted
            pred = terms.USES if typ == terms.Technology else terms.APPLIES
            triples.append(Triple(doc.name, terms.Project, pred, name, typ))
            project_terms[doc.name].add(name)

    # RELATED_TO: two projects that share any Technology/Concept are connected.
    for a, b in combinations(sorted(project_terms), 2):
        if project_terms[a] & project_terms[b]:
            triples.append(Triple(a, terms.Project, terms.RELATED_TO, b, terms.Project))

    # Dedupe (heading collisions across files can mint the same triple twice),
    # preserving order, so the in-memory and kuzu backends report identical stats.
    return list(dict.fromkeys(triples))


def _llm(docs: list[Doc]) -> list[Triple]:  # pragma: no cover - secondary path
    # Lazily import so the keyless path never imports anthropic.
    from .extract_llm import extract_llm

    return extract_llm(docs)


_BACKENDS = {"rule": _rule, "llm": _llm}


def extract(docs: list[Doc], backend: str | None = None) -> list[Triple]:
    name = backend or config.settings.extractor
    try:
        fn = _BACKENDS[name]
    except KeyError:
        raise ValueError(f"unknown GROUNDED_EXTRACTOR={name!r}; choose {list(_BACKENDS)}")
    return fn(docs)
