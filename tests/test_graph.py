"""Keyless graph-retrieval tests: terms -> extract -> build -> traverse -> gate.

Everything here runs with no API key and no kuzu/torch: the rule extractor + the
in-memory graph seam (GRAPH_PATH=:memory:, set in conftest) + the stub generator.
"""
from grounded.graph import terms


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
