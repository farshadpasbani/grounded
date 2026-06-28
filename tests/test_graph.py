"""Keyless graph-retrieval tests: terms -> extract -> build -> traverse -> gate.

Everything here runs with no API key and no kuzu/torch: the rule extractor + the
in-memory graph seam (GRAPH_PATH=:memory:, set in conftest) + the stub generator.
"""
import pytest

from grounded.graph import extract, terms


# --- curated dictionary -------------------------------------------------------
def test_match_terms_finds_known_tech_and_concepts_whole_word():
    found = terms.match_terms("Built with Python and FastAPI, applying RAG and a grounding gate.")
    names = {name for name, _type in found}
    assert "Python" in names
    assert "FastAPI" in names
    assert "RAG" in names
    assert "grounding gate" in names


def test_match_terms_is_case_insensitive_and_uses_aliases():
    found = dict(terms.match_terms("we used kuzu and nextjs"))
    assert found.get("KuzuDB") == "Technology"
    assert found.get("Next.js") == "Technology"


def test_match_terms_does_not_match_substrings():
    # "pythonic" must not count as Python; whole-word only.
    found = {name for name, _ in terms.match_terms("a pythonic ragout")}
    assert "Python" not in found
    assert "RAG" not in found


# --- rule extractor -----------------------------------------------------------
def _docs():
    return [
        extract.Doc(name="grounded", text="grounded is RAG in Python with Qdrant and a grounding gate."),
        extract.Doc(name="tailored", text="tailored runs deterministic gates in Python over a CV."),
    ]


def test_rule_extractor_emits_uses_and_applies_triples():
    triples = extract.extract([_docs()[0]])
    uses = {(t.subj, t.obj) for t in triples if t.pred == terms.USES}
    applies = {(t.subj, t.obj) for t in triples if t.pred == terms.APPLIES}
    assert ("grounded", "Python") in uses
    assert ("grounded", "Qdrant") in uses
    assert ("grounded", "RAG") in applies
    assert ("grounded", "grounding gate") in applies


def test_rule_extractor_infers_related_to_on_shared_term():
    triples = extract.extract(_docs())
    related = {frozenset((t.subj, t.obj)) for t in triples if t.pred == terms.RELATED_TO}
    # both projects use Python -> they are RELATED_TO each other
    assert frozenset(("grounded", "tailored")) in related


def test_rule_extractor_ip_guards_protected_terms(monkeypatch):
    # A protected term that happens to be in the dictionary must never become a
    # node. We inject ACME-SECRET as a fake technology alias for this test only.
    monkeypatch.setitem(terms.TECHNOLOGIES, "ACME-SECRET", ["acme-secret"])
    doc = extract.Doc(name="leaky", text="leaky uses Python and the ACME-SECRET engine.")
    triples = extract.extract([doc])
    nodes = {t.subj for t in triples} | {t.obj for t in triples}
    assert "ACME-SECRET" not in nodes
    assert "Python" in nodes  # the clean term still lands


def test_rule_extractor_drops_project_whose_name_is_protected():
    doc = extract.Doc(name="ACME-SECRET", text="confidential uses Python.")
    triples = extract.extract([doc])
    assert triples == []
