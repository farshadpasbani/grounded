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


# --- graph store + traversal (in-memory seam) ---------------------------------
from grounded.graph import store as gstore


@pytest.fixture
def graph(monkeypatch):
    gstore._backend.cache_clear()
    gstore.reset()
    yield gstore
    gstore._backend.cache_clear()


def _triple(s, p, o, st="Project", ot="Technology"):
    return extract.Triple(s, st, p, o, ot)


def test_store_builds_nodes_and_edges_from_triples(graph):
    graph.add_triples([_triple("grounded", terms.USES, "Python")])
    assert graph.has_node("grounded")
    assert graph.has_node("Python")
    assert not graph.has_node("nope")


def test_shortest_path_between_two_projects_via_shared_tech(graph):
    graph.add_triples([
        _triple("grounded", terms.USES, "Python"),
        _triple("tailored", terms.USES, "Python"),
    ])
    paths = graph.shortest_paths("grounded", "tailored", max_hops=3)
    assert paths, "expected a connecting path"
    nodes = paths[0].nodes()
    assert nodes[0] == "grounded" and nodes[-1] == "tailored"
    assert "Python" in nodes


def test_no_path_within_max_hops_returns_empty(graph):
    # a chain grounded-Python-tailored-Qdrant-other is 4 hops end-to-end
    graph.add_triples([
        _triple("grounded", terms.USES, "Python"),
        _triple("tailored", terms.USES, "Python"),
        _triple("tailored", terms.USES, "Qdrant"),
        _triple("other", terms.USES, "Qdrant"),
    ])
    assert graph.shortest_paths("grounded", "other", max_hops=2) == []
    assert graph.shortest_paths("grounded", "other", max_hops=4)


def test_single_entity_neighbourhood(graph):
    graph.add_triples([
        _triple("grounded", terms.USES, "Python"),
        _triple("grounded", terms.APPLIES, "RAG", ot="Concept"),
    ])
    nb = graph.neighbourhood("grounded")
    reached = {p.nodes()[-1] for p in nb}
    assert reached == {"Python", "RAG"}


def test_empty_graph_queries_do_not_crash(graph):
    assert graph.shortest_paths("a", "b", max_hops=3) == []
    assert graph.neighbourhood("a") == []
    assert not graph.has_node("a")


def test_cyclic_graph_traversal_is_bounded(graph):
    # a <-> b <-> c <-> a, all RELATED_TO; bounded hops must terminate
    graph.add_triples([
        _triple("a", terms.RELATED_TO, "b", ot="Project"),
        _triple("b", terms.RELATED_TO, "c", ot="Project"),
        _triple("c", terms.RELATED_TO, "a", ot="Project"),
    ])
    paths = graph.shortest_paths("a", "c", max_hops=3)
    assert paths
    assert min(len(p.steps) for p in paths) == 1  # a-c is direct


# --- graph query: entity matching -> evidence paths ---------------------------
from grounded.graph import query as gquery


@pytest.fixture
def built_graph(graph):
    graph.add_triples([
        extract.Triple("grounded", "Project", terms.USES, "Python", "Technology"),
        extract.Triple("tailored", "Project", terms.USES, "Python", "Technology"),
        extract.Triple("grounded", "Project", terms.APPLIES, "RAG", "Concept"),
        extract.Triple("lonely", "Project", terms.USES, "Qdrant", "Technology"),
    ])
    return graph


def test_graph_evidence_two_entities_returns_connecting_path(built_graph):
    paths = gquery.graph_evidence("how are grounded and tailored connected?")
    assert paths
    assert "Python" in paths[0].nodes()


def test_graph_evidence_single_entity_returns_neighbourhood(built_graph):
    paths = gquery.graph_evidence("tell me about grounded")
    reached = {n for p in paths for n in p.nodes()}
    assert "Python" in reached and "RAG" in reached


def test_graph_evidence_no_known_entity_is_empty(built_graph):
    assert gquery.graph_evidence("what is the capital of France?") == []


def test_graph_evidence_entities_present_but_no_path_is_empty(built_graph):
    # grounded and lonely share nothing
    assert gquery.graph_evidence("connect grounded and lonely") == []


def test_graph_evidence_matches_dictionary_alias(built_graph):
    # 'qdrant' alias resolves to the Qdrant node, single-entity neighbourhood
    paths = gquery.graph_evidence("anything using qdrant?")
    assert any("lonely" in p.nodes() for p in paths)
