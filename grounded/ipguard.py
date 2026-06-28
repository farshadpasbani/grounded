"""IP guard: keep protected terms out of the index and out of every answer.

tailored's ip-guard, composed into the RAG path. Whole-word, case-insensitive.
"""
from __future__ import annotations

import re

from . import config


def _pattern(topics: list[str]) -> re.Pattern[str]:
    alts = "|".join(re.escape(t) for t in topics if t)
    return re.compile(rf"(?<!\w)(?:{alts})(?!\w)", re.IGNORECASE)


# The protected set is fixed for a process (env-only), so compile the pattern
# once at import. The per-call read below still goes through config.settings.
_PAT = _pattern(config.settings.protected_topics)


def leaks(text: str) -> list[str]:
    """Protected terms present in `text` (empty list = clean)."""
    if not config.settings.protected_topics:
        return []
    return sorted({m.group(0) for m in _PAT.finditer(text)})


def is_clean(text: str) -> bool:
    return not leaks(text)
