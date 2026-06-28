"""Score the system against the labelled set.

Two numbers, the ones a reviewer actually asks for:
  recall@k     - for answerable questions, did the right source make the top-k?
  groundedness - did the system answer when it should and abstain when it should?
"""
from __future__ import annotations

from pathlib import Path

import yaml

from . import config
from .graph.query import graph_evidence
from .query import ask
from .retrieve import retrieve

EVAL_FILE = Path(__file__).resolve().parent.parent / "eval" / "questions.yaml"
GRAPH_EVAL_FILE = Path(__file__).resolve().parent.parent / "eval" / "graph_questions.yaml"


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
            hits = retrieve(q, config.settings.top_k)
            if any(item["source"] in f"{h.source} {h.heading}" for h in hits):
                recall_hits += 1

        # groundedness: answered-when-should, abstained-when-should
        a = ask(q)
        if a.grounded == answerable:
            grounded_ok += 1
        else:
            print(f"  MISS  answerable={answerable} grounded={a.grounded}  {q!r}  ({a.reason})")

    print(f"\nrecall@{config.settings.top_k}: {recall_hits}/{recall_total}")
    print(f"groundedness: {grounded_ok}/{len(items)}")
    return {
        "recall_at_k": (recall_hits, recall_total),
        "groundedness": (grounded_ok, len(items)),
    }


def run_graph() -> dict:
    """Score the graph against the labelled multi-hop set. Same bar as run():

      graph-recall - for answerable items, was a path covering the expected
                     entities actually traversed?
      groundedness - answered-when-a-path-exists, abstained-when-not.
    """
    items = yaml.safe_load(GRAPH_EVAL_FILE.read_text())

    recall_hits = recall_total = 0
    grounded_ok = 0

    for item in items:
        q = item["q"]
        answerable = item.get("answerable", True)

        # graph-recall: do the returned paths cover every expected entity?
        if answerable and item.get("entities"):
            recall_total += 1
            expected = set(item["entities"])
            covered = {n for p in graph_evidence(q) for n in p.nodes()}
            if expected <= covered:
                recall_hits += 1

        # groundedness: answered-when-should, abstained-when-should
        a = ask(q)
        if a.grounded == answerable:
            grounded_ok += 1
        else:
            print(f"  MISS  answerable={answerable} grounded={a.grounded}  {q!r}  ({a.reason})")

    print(f"\ngraph-recall: {recall_hits}/{recall_total}")
    print(f"groundedness: {grounded_ok}/{len(items)}")
    return {
        "graph_recall": (recall_hits, recall_total),
        "groundedness": (grounded_ok, len(items)),
    }
