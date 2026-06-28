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


# --- end-to-end: graph mode through query.ask ---------------------------------
from grounded import config
from grounded.query import ABSTAIN, ask


@pytest.fixture
def graph_mode(monkeypatch, built_graph):
    monkeypatch.setenv("GROUNDED_RETRIEVAL_MODE", "graph")
    monkeypatch.setattr(config, "settings", config.Settings())
    return built_graph


def test_graph_mode_multi_hop_answers_grounded_and_cited(graph_mode):
    ans = ask("how are grounded and tailored connected?")
    assert ans.grounded
    assert ans.citations
    assert "ACME-SECRET" not in ans.text


def test_graph_mode_no_path_abstains(graph_mode):
    ans = ask("connect grounded and lonely")
    assert not ans.grounded
    assert ans.text == ABSTAIN


def test_graph_mode_unknown_entity_abstains(graph_mode):
    ans = ask("what is the capital of France?")
    assert not ans.grounded
    assert ans.text == ABSTAIN


def test_graph_mode_empty_graph_abstains_gracefully(monkeypatch, graph):
    # no triples added: graph mode must abstain, not crash
    monkeypatch.setenv("GROUNDED_RETRIEVAL_MODE", "graph")
    monkeypatch.setattr(config, "settings", config.Settings())
    ans = ask("how are grounded and tailored connected?")
    assert not ans.grounded
    assert ans.text == ABSTAIN


# --- end-to-end: hybrid mode (vector hits UNION graph paths) ------------------
import math

from grounded import embed, ingest, store as vstore


def _fake_vec(text: str) -> list[float]:
    v = [0.0] * config.settings.embed_dim
    for word in text.lower().split():
        v[hash(word) % config.settings.embed_dim] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


@pytest.fixture
def hybrid_mode(monkeypatch, graph, tmp_path):
    monkeypatch.setattr(embed, "embed", lambda texts: [_fake_vec(t) for t in texts])
    monkeypatch.setattr(embed, "embed_one", lambda t: _fake_vec(t))
    vstore._client.cache_clear()
    # vector corpus: one doc about a topic with no graph entities
    (tmp_path / "weather.md").write_text(
        "# Weather notes\nThe rain in the valley falls on the quiet meadow each spring.\n"
    )
    src = tmp_path / "sources.yaml"
    src.write_text(f"globs:\n  - '{tmp_path}/*.md'\n")
    ingest.build(str(src))  # vector mode default for this build
    # graph: grounded + tailored share Python
    graph.add_triples([
        extract.Triple("grounded", "Project", terms.USES, "Python", "Technology"),
        extract.Triple("tailored", "Project", terms.USES, "Python", "Technology"),
    ])
    monkeypatch.setenv("GROUNDED_RETRIEVAL_MODE", "hybrid")
    monkeypatch.setattr(config, "settings", config.Settings())
    yield
    vstore._client.cache_clear()


def test_hybrid_answers_when_only_graph_grounds(hybrid_mode):
    # multi-hop question: no vector overlap, but a real graph path exists
    ans = ask("how are grounded and tailored connected?")
    assert ans.grounded
    assert ans.paths  # grounded by a path
    assert "ACME-SECRET" not in ans.text


def test_hybrid_answers_when_only_vector_grounds(hybrid_mode):
    # no known graph entity, but the vector corpus has the answer
    ans = ask("what falls on the quiet meadow each spring?")
    assert ans.grounded
    assert ans.citations


def test_hybrid_abstains_when_neither_grounds(hybrid_mode):
    ans = ask("zxqw plooble grommet flarn unrelated nonsense")
    assert not ans.grounded
    assert ans.text == ABSTAIN


# --- ingest builds the knowledge graph in graph mode --------------------------
def _graph_corpus(tmp_path):
    (tmp_path / "grounded.md").write_text("# grounded\ngrounded is RAG in Python with Qdrant.\n")
    (tmp_path / "tailored.md").write_text("# tailored\ntailored runs deterministic gates in Python.\n")
    (tmp_path / "secret.md").write_text("# ACME-SECRET\nthe ACME-SECRET tool uses Python.\n")
    src = tmp_path / "sources.yaml"
    src.write_text(f"globs:\n  - '{tmp_path}/*.md'\n")
    return str(src)


def test_graph_mode_ingest_builds_graph_and_ip_guards(monkeypatch, graph, tmp_path):
    src = _graph_corpus(tmp_path)
    monkeypatch.setenv("GROUNDED_RETRIEVAL_MODE", "graph")
    monkeypatch.setattr(config, "settings", config.Settings())
    stats = ingest.build(src)
    assert stats["nodes"] > 0 and stats["edges"] > 0
    names = graph.node_names()
    assert "grounded" in names and "tailored" in names and "Python" in names
    assert "ACME-SECRET" not in names  # protected project never enters the graph
    # the freshly built graph answers a multi-hop question
    ans = ask("how are grounded and tailored connected?")
    assert ans.grounded


# --- graph eval: graph-recall + groundedness ----------------------------------
from grounded import evaluate


def test_graph_eval_reports_recall_and_groundedness(monkeypatch, built_graph, tmp_path):
    monkeypatch.setenv("GROUNDED_RETRIEVAL_MODE", "graph")
    monkeypatch.setattr(config, "settings", config.Settings())
    eval_file = tmp_path / "graph_questions.yaml"
    eval_file.write_text(
        "- q: 'how are grounded and tailored connected?'\n"
        "  entities: ['grounded', 'tailored']\n"
        "  answerable: true\n"
        "- q: 'connect grounded and lonely'\n"
        "  answerable: false\n"
    )
    monkeypatch.setattr(evaluate, "GRAPH_EVAL_FILE", eval_file)
    metrics = evaluate.run_graph()
    assert metrics["graph_recall"] == (1, 1)
    assert metrics["groundedness"] == (2, 2)
