"""Generation step, behind a pluggable backend.

The contract is the same for every backend: given the question and the
retrieved chunks, return {answered, answer, citations:[chunk_ids]} so the gate
can verify groundedness deterministically.

- "stub":   no API key. Deterministic. Answers from the top chunk and cites it.
            Lets the whole pipeline (and the eval) run offline.
- "claude": production. Claude with structured outputs, constrained to answer
            only from the given chunks.
"""
from __future__ import annotations

from .config import settings
from .store import Hit

_SYSTEM = (
    "You answer questions strictly from the provided context chunks, which are "
    "drawn from Farshad Pasbani's public writing and project docs. Use only the "
    "chunks given. Do not use outside knowledge. If the context does not contain "
    "the answer, set answered=false and leave answer empty. When you do answer, "
    "cite the id of every chunk you used. Be concise and factual."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "answered": {"type": "boolean"},
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["answered", "answer", "citations"],
    "additionalProperties": False,
}


def _context_block(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[chunk {h.id}] (from {h.source} :: {h.heading or 'intro'})\n{h.text}"
        for h in hits
    )


def _stub(question: str, hits: list[Hit]) -> dict:
    """Deterministic stand-in for a real model. Quotes the top chunk verbatim
    and cites it, so the grounding gate exercises real ids and the eval runs
    without a key. It never invents content beyond what it retrieved."""
    if not hits:
        return {"answered": False, "answer": "", "citations": []}
    top = hits[0]
    snippet = " ".join(top.text.split()[:60])
    return {
        "answered": True,
        "answer": f"[stub] From {top.heading or 'intro'}: {snippet}",
        "citations": [top.id],
    }


def _claude(question: str, hits: list[Hit]) -> dict:
    import json

    import anthropic  # imported lazily so the stub path needs neither the SDK nor a key

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    msg = client.messages.create(
        model=settings.gen_model,  # swap to claude-haiku-4-5 to cut cost
        max_tokens=1024,
        system=_SYSTEM,
        output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
        messages=[
            {"role": "user", "content": f"Context:\n{_context_block(hits)}\n\nQuestion: {question}"}
        ],
    )
    text = next(b.text for b in msg.content if b.type == "text")
    return json.loads(text)


_BACKENDS = {"stub": _stub, "claude": _claude}


def generate(question: str, hits: list[Hit]) -> dict:
    try:
        backend = _BACKENDS[settings.generator]
    except KeyError:
        raise ValueError(f"unknown GROUNDED_GENERATOR={settings.generator!r}; choose {list(_BACKENDS)}")
    return backend(question, hits)
