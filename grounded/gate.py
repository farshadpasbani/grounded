"""The grounding gate. Deterministic checks around the stochastic answer.

Two failure modes it closes:
  1. weak retrieval  -> abstain before spending a generation call
  2. ungrounded claim -> the model cited a chunk it was not given, or claimed it
                         could not answer; reject and abstain
"""
from __future__ import annotations

from . import config, ipguard
from .store import Hit


def too_weak(hits: list[Hit]) -> bool:
    """No hit, or the best one is below the confidence floor."""
    return not hits or hits[0].score < config.settings.min_score


def verdict(answer: dict, retrieved_ids: set[int]) -> tuple[bool, str]:
    """Is the generated answer allowed to ship?

    `answer` is the structured generation: {answer, citations, answered}.
    Returns (ok, reason). On not-ok the caller abstains.
    """
    if not answer.get("answered", False):
        return False, "model could not answer from the retrieved context"

    cited = set(answer.get("citations", []))
    if not cited:
        return False, "answer made claims with no citation"

    stray = cited - retrieved_ids
    if stray:
        return False, f"cited chunks not in the retrieved set: {sorted(stray)}"

    if not ipguard.is_clean(answer.get("answer", "")):
        return False, f"answer leaked protected topics: {ipguard.leaks(answer['answer'])}"

    return True, "grounded"
