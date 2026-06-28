"""End-to-end answer path: retrieve -> gate -> generate -> gate -> answer."""
from __future__ import annotations

from dataclasses import dataclass, field

from . import gate, generate
from .retrieve import retrieve
from .store import Hit

ABSTAIN = "I don't have that in my public notes."


@dataclass
class Answer:
    text: str
    grounded: bool
    citations: list[Hit] = field(default_factory=list)
    reason: str = ""


def ask(question: str) -> Answer:
    hits = retrieve(question)

    if gate.too_weak(hits):
        return Answer(ABSTAIN, grounded=False, reason="retrieval below confidence floor")

    result = generate.generate(question, hits)
    ok, reason = gate.verdict(result, retrieved_ids={h.id for h in hits})
    if not ok:
        return Answer(ABSTAIN, grounded=False, reason=reason)

    cited = [h for h in hits if h.id in set(result["citations"])]
    return Answer(result["answer"], grounded=True, citations=cited, reason=reason)
