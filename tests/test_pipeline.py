"""End-to-end spine test, fully offline.

Real chunker, real ip-guard, real Qdrant (embedded :memory:), real gate, stub
generator. Embeddings are faked deterministically so the test needs neither
torch nor a model download — the seam is `grounded.embed`, monkeypatched here.
"""
import math

import pytest

from grounded import config, embed, evaluate, gate, ingest, store
from grounded.config import settings
from grounded.query import ABSTAIN, ask
from grounded.retrieve import retrieve


def test_reassigning_config_settings_flips_a_consumer(monkeypatch):
    # Unified idiom: gate reads config.settings.min_score (not a stale import-time
    # binding), so swapping in a fresh Settings flips it without reimporting.
    hit = store.Hit(id=0, score=0.5, source="s", heading="h", text="t")
    monkeypatch.setattr(config, "settings", config.Settings(min_score=0.4))
    assert not gate.too_weak([hit])  # 0.5 >= 0.4
    monkeypatch.setattr(config, "settings", config.Settings(min_score=0.9))
    assert gate.too_weak([hit])  # 0.5 < 0.9


# --- deterministic fake embedding: word-set -> sparse normalized vector --------
def _fake_vec(text: str) -> list[float]:
    v = [0.0] * settings.embed_dim
    for word in text.lower().split():
        v[hash(word) % settings.embed_dim] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    monkeypatch.setattr(embed, "embed", lambda texts: [_fake_vec(t) for t in texts])
    monkeypatch.setattr(embed, "embed_one", lambda t: _fake_vec(t))
    # PYTHONHASHSEED is fixed by the runner so hash() is stable within the run.
    store._client.cache_clear()
    yield
    store._client.cache_clear()


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text(
        "# Agentic systems\nFarshad builds agentic loops and governance gates "
        "for coding agents, the interlock thesis applied to retrieval.\n"
    )
    # This file carries a protected term; ip-guard must keep it out of the index.
    (tmp_path / "b.md").write_text(
        "# Internal\nThe ACME-SECRET engine is confidential employer tooling.\n"
    )
    sources = tmp_path / "sources.yaml"
    sources.write_text(f"globs:\n  - '{tmp_path}/*.md'\n")
    return str(sources)


def test_ingest_filters_protected_terms(corpus):
    stats = ingest.build(corpus)
    assert stats["files"] == 2
    assert stats["chunks"] >= 1
    assert stats["ip_dropped"] >= 1  # b.md's protected chunk never enters the index


def test_retrieval_finds_the_right_source(corpus):
    ingest.build(corpus)
    hits = retrieve("what does Farshad build with agentic loops?")
    assert hits, "expected at least one hit"
    assert "a.md" in hits[0].source


def test_grounded_answer_is_cited_and_clean(corpus):
    ingest.build(corpus)
    ans = ask("what does Farshad build with agentic loops?")
    assert ans.grounded
    assert ans.citations
    assert ans.citations[0].id in {h.id for h in retrieve("what does Farshad build with agentic loops?")}
    assert "ACME-SECRET" not in ans.text  # never surfaces protected IP


def test_abstains_when_retrieval_is_weak(corpus):
    ingest.build(corpus)
    ans = ask("zxqw plooble grommet flarn")  # no overlap with the corpus
    assert not ans.grounded
    assert ans.text == ABSTAIN


# --- gate unit checks (pure, no store) ----------------------------------------
def test_gate_rejects_uncited_claim():
    ok, reason = gate.verdict({"answered": True, "answer": "x", "citations": []}, {1, 2})
    assert not ok and "citation" in reason


def test_gate_rejects_hallucinated_citation():
    ok, reason = gate.verdict({"answered": True, "answer": "x", "citations": [99]}, {1, 2})
    assert not ok and "not in the retrieved set" in reason


def test_gate_rejects_protected_leak():
    ok, reason = gate.verdict({"answered": True, "answer": "the ACME-SECRET engine", "citations": [1]}, {1})
    assert not ok and "protected" in reason


def test_gate_passes_grounded_clean_answer():
    ok, reason = gate.verdict({"answered": True, "answer": "agentic loops", "citations": [1]}, {1})
    assert ok and reason == "grounded"


# --- eval harness over the temp corpus ----------------------------------------
def test_eval_harness_runs(corpus, tmp_path, monkeypatch):
    ingest.build(corpus)
    eval_file = tmp_path / "questions.yaml"
    eval_file.write_text(
        "- q: 'what does Farshad build with agentic loops?'\n"
        "  source: 'a.md'\n"
        "  answerable: true\n"
        "- q: 'zxqw plooble grommet flarn'\n"
        "  answerable: false\n"
    )
    monkeypatch.setattr(evaluate, "EVAL_FILE", eval_file)
    metrics = evaluate.run()
    assert metrics["recall_at_k"] == (1, 1)
    assert metrics["groundedness"] == (2, 2)
