"""IP guard: keep protected terms out of the index and out of every answer.

tailored's ip-guard, composed into the RAG path. Whole-word, case-insensitive.
"""
from __future__ import annotations

import re

from .config import settings


def _pattern(topics: list[str]) -> re.Pattern[str]:
    alts = "|".join(re.escape(t) for t in topics if t)
    return re.compile(rf"(?<!\w)(?:{alts})(?!\w)", re.IGNORECASE)


_PAT = _pattern(settings.protected_topics)


def leaks(text: str) -> list[str]:
    """Protected terms present in `text` (empty list = clean)."""
    return sorted({m.group(0) for m in _PAT.finditer(text)}) if settings.protected_topics else []


def is_clean(text: str) -> bool:
    return not leaks(text)
