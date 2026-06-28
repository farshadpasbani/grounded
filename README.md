# grounded

![CI](https://github.com/farshadpasbani/grounded/actions/workflows/ci.yml/badge.svg)

A small RAG service over a public corpus, with an anti-hallucination gate.

Generic RAG retrieves chunks and trusts the model to use them. `grounded` does
the opposite: it answers only from what it retrieved, cites every claim back to a
source chunk, and abstains when retrieval is too weak to answer. Same thesis as
[interlock](https://github.com/farshadpasbani/interlock) and
[tailored](https://github.com/farshadpasbani/tailored): wrap a stochastic model
in deterministic gates so the output is something you can trust, not just demo.

The first corpus is my own public writing and project docs, so the live demo is
also the proof: ask it what I have built and every answer is cited and true, or
it says it does not know.

## Pipeline

```
ingest:  sources -> chunk -> ip-filter -> embed (local) -> Qdrant
query:   question -> embed -> retrieve top-k
                          -> gate: abstain if top score < threshold
         generate (Claude, structured: answer + cited chunk ids)
                          -> gate: every cited id must be in the retrieved set
                          -> ip-guard the final answer
         -> cited answer, or an honest "not in my sources"
```

Each stage maps to a thing a RAG system is judged on: chunking strategy, local
embeddings, a vector database (Qdrant), top-k retrieval, a grounding/abstention
gate, and a labelled eval set (recall@k + groundedness).

## Stack

- Python, FastAPI
- Qdrant (vector DB) — embedded by default (a local directory, no Docker); point
  `QDRANT_URL` at a server for production
- Local embeddings: `bge-small-en-v1.5` (sentence-transformers, CPU, nothing
  leaves the box at ingest time)
- Pluggable generation: `stub` (deterministic, no key — for development) or
  `claude` (`claude-opus-4-8`, structured outputs — production)

## Graph retrieval (vector + knowledge graph)

Vector search finds what is *semantically related*; a knowledge graph finds what
is *structurally connected* — the multi-hop questions a similarity search cannot
answer ("how are grounded and tailored connected?"). `grounded` runs both behind
the same grounding-gate thesis: an answer must map to a real graph path, or it
abstains. No fabricated relationships.

```
GROUNDED_RETRIEVAL_MODE = vector (default) | graph | hybrid
ingest (graph):  sources -> rule-extract typed triples (ip-guarded) -> graph store
query  (graph):  question -> match entities -> shortest path / neighbourhood
                          -> gate: abstain on no path; cite the traversed path
hybrid:          union vector chunks + graph paths; answer if EITHER grounds it
```

- **Extractor is pluggable**: `rule` (deterministic, keyless, default) scans the
  corpus against a curated public dictionary (`grounded/graph/terms.py`) to mint
  `Project` / `Technology` / `Concept` nodes and `USES` / `APPLIES` / `RELATED_TO`
  edges. `llm` (Claude structured output) is wired behind the same interface and
  imported lazily, so the keyless path never touches the SDK.
- **Store**: embedded KuzuDB (Cypher, no Docker) for production; a pure-python
  in-memory seam (`GRAPH_PATH=:memory:`) for the offline tests, so CI needs
  neither kuzu nor torch. `GRAPH_MAX_HOPS` bounds traversal (cycles can't run
  away). KuzuDB has no Python 3.14 wheel yet — install `'.[graph]'` under 3.13.

```sh
GROUNDED_RETRIEVAL_MODE=graph grounded ingest          # build the graph
GROUNDED_RETRIEVAL_MODE=graph grounded ask "how are grounded and tailored connected?"
GROUNDED_RETRIEVAL_MODE=graph grounded eval --graph    # graph-recall + groundedness
```

## Develop without an API key

Only generation talks to Claude. Embeddings are a local model and the vector DB
is embedded, so the whole pipeline — ingest, retrieve, the grounding gate,
eval — runs with **no key and no Docker** using the `stub` generator (the
default). Swap to real answers later by setting one env var.

```sh
pip install -e '.[embed]'              # adds local embeddings (needs torch)
# light/test-only install (no torch): pip install -e '.[dev]'

grounded ingest                        # stub generator, embedded Qdrant
grounded ask "what has Farshad built with agentic AI?"
grounded eval                          # recall@k + groundedness

# production answers, once you have a key:
export ANTHROPIC_API_KEY=sk-ant-...
export GROUNDED_GENERATOR=claude
grounded ask "..."
```

Run the offline test suite (no key, no Docker, no model download — embeddings
are faked in the tests):

```sh
PYTHONHASHSEED=0 pytest -q
```

### Config (env)

| var | default | meaning |
| --- | --- | --- |
| `GROUNDED_GENERATOR` | `stub` | `stub` (no key) or `claude` |
| `GROUNDED_RETRIEVAL_MODE` | `vector` | `vector`, `graph`, or `hybrid` |
| `GROUNDED_EXTRACTOR` | `rule` | `rule` (keyless) or `llm` (Claude) |
| `GRAPH_PATH` | `:memory:` | in-memory seam, or an embedded KuzuDB dir |
| `GRAPH_MAX_HOPS` | `3` | traversal bound (keeps cyclic graphs finite) |
| `QDRANT_URL` | _(unset)_ | set to use a Qdrant server instead of embedded |
| `QDRANT_PATH` | `.qdrant` | embedded store dir, or `:memory:` |
| `GROUNDED_MIN_SCORE` | `0.68` | abstention floor (tuned for `bge` on this corpus) |
| `GROUNDED_GEN_MODEL` | `claude-opus-4-8` | generation model |
| `GROUNDED_PROTECTED_TOPICS` | _(empty)_ | your private IP terms, comma-separated (never committed) |

## IP safety

Set `GROUNDED_PROTECTED_TOPICS` (comma-separated) to terms that must never
surface — employer IP, confidential codenames. They live only in your
environment, never in the repo, and ip-guard filters them at ingest **and**
re-checks every generated answer (the same idea tailored uses). The bot cannot
leak them even under a leading question.

## Status

Validated on real embeddings, no key required. Over a curated public corpus
(14 docs from grounded / interlock / tailored / feristal-site) with
real `bge`: **recall@5 = 15/15**, and the grounding gate tuned to **20/20** on a
20-question eval set (answerable answered, out-of-corpus abstained, including an
employer-IP probe). 9 offline tests green; CI runs them keyless; MIT licensed.

One production swap left: the Claude generator (`GROUNDED_GENERATOR=claude` + a
key) for real answers instead of the stub. Then the feristal demo surface.
