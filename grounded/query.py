"""End-to-end answer path, selected by GROUNDED_RETRIEVAL_MODE.

  vector (default): retrieve -> gate -> generate -> gate -> answer. UNCHANGED.
  graph:            match entities -> traverse -> generate over paths -> gate.
  hybrid:           union of vector hits + graph paths into one evidence set;
                    answers when either grounds it, abstains when neither does.

The gate verifies citations against the ids of the evidence actually returned,
so the same deterministic check covers chunks and paths alike.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config, gate, generate
from .graph import query as graph_query
from .graph.store import GraphPath
from .retrieve import retrieve
from .store import Hit

ABSTAIN = "I don't have that in my public notes."


@dataclass
class Answer:
    text: str
    grounded: bool
    citations: list[Hit] = field(default_factory=list)
    reason: str = ""
    paths: list[GraphPath] = field(default_factory=list)


def _path_to_hit(idx: int, path: GraphPath) -> Hit:
    """A traversal path as a citeable piece of evidence. score=1.0: a real path
    is a hard structural fact, not a similarity score."""
    rendered = path.render()
    return Hit(id=idx, score=1.0, source="graph", heading=rendered, text=rendered)


def ask(question: str) -> Answer:
    mode = config.settings.retrieval_mode
    if mode == "graph":
        return _ask_graph(question)
    if mode == "hybrid":
        return _ask_hybrid(question)
    return _ask_vector(question)


def _ask_vector(question: str) -> Answer:
    hits = retrieve(question)

    if gate.too_weak(hits):
        return Answer(ABSTAIN, grounded=False, reason="retrieval below confidence floor")

    result = generate.generate(question, hits)
    ok, reason = gate.verdict(result, retrieved_ids={h.id for h in hits})
    if not ok:
        return Answer(ABSTAIN, grounded=False, reason=reason)

    cited = [h for h in hits if h.id in set(result["citations"])]
    return Answer(result["answer"], grounded=True, citations=cited, reason=reason)


def _answer_from_evidence(question: str, evidence: list[Hit], paths: list[GraphPath]) -> Answer:
    """Generate over a fixed evidence set and gate the result."""
    if not evidence:
        return Answer(ABSTAIN, grounded=False, reason="no connecting path or chunk")

    result = generate.generate(question, evidence)
    ok, reason = gate.verdict(result, retrieved_ids={h.id for h in evidence})
    if not ok:
        return Answer(ABSTAIN, grounded=False, reason=reason)

    cited_ids = set(result["citations"])
    cited = [h for h in evidence if h.id in cited_ids]
    cited_paths = [p for h, p in zip(_graph_hits(evidence), paths) if h.id in cited_ids] if paths else []
    return Answer(result["answer"], grounded=True, citations=cited, reason=reason, paths=cited_paths)


def _graph_hits(evidence: list[Hit]) -> list[Hit]:
    return [h for h in evidence if h.source == "graph"]


def _ask_graph(question: str) -> Answer:
    paths = graph_query.graph_evidence(question)
    evidence = [_path_to_hit(i, p) for i, p in enumerate(paths)]
    return _answer_from_evidence(question, evidence, paths)


def _ask_hybrid(question: str) -> Answer:
    vhits = retrieve(question)
    vgood = [] if gate.too_weak(vhits) else list(vhits)
    paths = graph_query.graph_evidence(question)

    # Union into one evidence set with fresh, collision-free ids.
    evidence: list[Hit] = []
    for h in vgood:
        evidence.append(h)
    for p in paths:
        evidence.append(_path_to_hit(0, p))  # id reassigned below
    for i, h in enumerate(evidence):
        h.id = i
    return _answer_from_evidence(question, evidence, paths)
