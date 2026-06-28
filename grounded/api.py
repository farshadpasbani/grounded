"""Thin HTTP surface for the feristal demo. `uvicorn grounded.api:app`."""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .query import ask

app = FastAPI(title="grounded")


class Query(BaseModel):
    question: str


@app.post("/ask")
def ask_endpoint(q: Query):
    a = ask(q.question)
    return {
        "answer": a.text,
        "grounded": a.grounded,
        "sources": [
            {"source": h.source, "heading": h.heading, "score": round(h.score, 3)}
            for h in a.citations
        ],
    }
