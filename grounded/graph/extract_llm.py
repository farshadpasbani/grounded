"""LLM triple extractor: Claude structured output -> typed triples.

The secondary, non-default backend. Imported lazily by extract._llm so the
keyless path never imports anthropic. The rule extractor is the default and the
only one needed offline / in CI; this exists so the extractor is genuinely
pluggable and an LLM can mint richer edges (including MENTIONS) when a key is set.
"""
from __future__ import annotations

import json

from .. import config
from . import terms
from .extract import Doc, Triple

_SYSTEM = (
    "Extract a knowledge graph from the document as typed triples. Allowed node "
    "types: Project, Technology, Concept. Allowed predicates: USES "
    "(Project->Technology), APPLIES (Project->Concept), MENTIONS (generic "
    "co-occurrence), RELATED_TO (Project->Project). The document's project name is "
    "given; use it as the Project subject. Only emit triples grounded in the text. "
    "Do not invent technologies or concepts that are not mentioned."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "triples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subj": {"type": "string"},
                    "subj_type": {"type": "string", "enum": [terms.Project, terms.Technology, terms.Concept]},
                    "pred": {"type": "string", "enum": [terms.USES, terms.APPLIES, terms.MENTIONS, terms.RELATED_TO]},
                    "obj": {"type": "string"},
                    "obj_type": {"type": "string", "enum": [terms.Project, terms.Technology, terms.Concept]},
                },
                "required": ["subj", "subj_type", "pred", "obj", "obj_type"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["triples"],
    "additionalProperties": False,
}


def extract_llm(docs: list[Doc]) -> list[Triple]:
    import anthropic  # lazy: only when the llm backend is actually selected

    from .. import ipguard

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    out: list[Triple] = []
    for doc in docs:
        if not ipguard.is_clean(doc.name):
            continue
        msg = client.messages.create(
            model=config.settings.gen_model,
            max_tokens=2048,
            system=_SYSTEM,
            output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
            messages=[{"role": "user", "content": f"Project name: {doc.name}\n\n{doc.text}"}],
        )
        text = next(b.text for b in msg.content if b.type == "text")
        for t in json.loads(text)["triples"]:
            # ip-guard the LLM's output too: a protected term must never become a
            # node even if the model surfaces one.
            if ipguard.is_clean(t["subj"]) and ipguard.is_clean(t["obj"]):
                out.append(Triple(t["subj"], t["subj_type"], t["pred"], t["obj"], t["obj_type"]))
    return out
