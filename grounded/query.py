"""End-to-end answer path, selected by GROUNDED_RETRIEVAL_MODE.

  vector (default): retrieve -> gate -> generate -> gate -> answer. UNCHANGED.
  graph:            match entities -> traverse -> generate over paths -> gate.
  hybrid:           union of vector hits + graph paths into one evidence set;
                    answers when either grounds it, abstains when neither does.

The gate verifies citations against the ids of the evidence actually returned,
so the same deterministic check covers chunks and paths alike.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from . import config, gate, generate
from .graph import query as graph_query
from .graph.types import GraphPath
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


def _path_to_hit(path: GraphPath) -> Hit:
    """A traversal path as a citeable piece of evidence. score=1.0: a real path
    is a hard structural fact, not a similarity score. The id is a placeholder;
    `_assemble` owns id assignment."""
    rendered = path.render()
    return Hit(id=0, score=1.0, source="graph", heading=rendered, text=rendered, path=path)


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


def _answer_from_evidence(question: str, evidence: list[Hit]) -> Answer:
    """Generate over a fixed evidence set and gate the result.

    Evidence is self-describing: a graph hit carries its own `path`, so the
    cited paths fall straight out of the cited hits -- no parallel list to
    re-zip and keep aligned.
    """
    if not evidence:
        return Answer(ABSTAIN, grounded=False, reason="no connecting path or chunk")

    result = generate.generate(question, evidence)
    ok, reason = gate.verdict(result, retrieved_ids={h.id for h in evidence})
    if not ok:
        return Answer(ABSTAIN, grounded=False, reason=reason)

    cited_ids = set(result["citations"])
    cited = [h for h in evidence if h.id in cited_ids]
    cited_paths = [h.path for h in cited if h.path]
    return Answer(result["answer"], grounded=True, citations=cited, reason=reason, paths=cited_paths)


def _assemble(vector_hits: list[Hit], paths: list[GraphPath]) -> list[Hit]:
    """The one place ids are assigned. Union of vector hits and graph paths into
    a single evidence list with contiguous, collision-free ids -- so the gate's
    citation check sees one flat id space. Vector hits keep their fields but are
    reissued an id here; paths become path-carrying graph hits."""
    evidence = list(vector_hits) + [_path_to_hit(p) for p in paths]
    return [replace(h, id=i) for i, h in enumerate(evidence)]


def _ask_graph(question: str) -> Answer:
    paths = graph_query.graph_evidence(question)
    return _answer_from_evidence(question, _assemble([], paths))


def _ask_hybrid(question: str) -> Answer:
    vhits = retrieve(question)
    vgood = [] if gate.too_weak(vhits) else list(vhits)
    paths = graph_query.graph_evidence(question)
    return _answer_from_evidence(question, _assemble(vgood, paths))
