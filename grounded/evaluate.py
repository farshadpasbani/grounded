"""Score the system against the labelled set.

Two numbers, the ones a reviewer actually asks for:
  recall@k     - for answerable questions, did the right source make the top-k?
  groundedness - did the system answer when it should and abstain when it should?
"""
from __future__ import annotations

from pathlib import Path

import yaml

from .config import settings
from .query import ask
from .retrieve import retrieve

EVAL_FILE = Path(__file__).resolve().parent.parent / "eval" / "questions.yaml"


def run() -> dict:
    items = yaml.safe_load(EVAL_FILE.read_text())

    recall_hits = recall_total = 0
    grounded_ok = 0

    for item in items:
        q = item["q"]
        answerable = item.get("answerable", True)

        # recall@k: is the expected source somewhere in the top-k?
        if answerable and item.get("source"):
            recall_total += 1
            hits = retrieve(q, settings.top_k)
            if any(item["source"] in f"{h.source} {h.heading}" for h in hits):
                recall_hits += 1

        # groundedness: answered-when-should, abstained-when-should
        a = ask(q)
        if a.grounded == answerable:
            grounded_ok += 1
        else:
            print(f"  MISS  answerable={answerable} grounded={a.grounded}  {q!r}  ({a.reason})")

    print(f"\nrecall@{settings.top_k}: {recall_hits}/{recall_total}")
    print(f"groundedness: {grounded_ok}/{len(items)}")
    return {
        "recall_at_k": (recall_hits, recall_total),
        "groundedness": (grounded_ok, len(items)),
    }
