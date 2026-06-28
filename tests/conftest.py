"""Test config: embedded in-memory Qdrant, stub generator. No key, no Docker."""
import os

# Must be set before grounded.config is imported anywhere.
os.environ.setdefault("QDRANT_URL", "")
os.environ["QDRANT_PATH"] = ":memory:"
os.environ["GROUNDED_GENERATOR"] = "stub"
# Fake test embeddings yield lower cosines than real bge; lower the floor so the
# floor still separates a real match (~0.15) from zero-overlap gibberish (0.0).
os.environ["GROUNDED_MIN_SCORE"] = "0.05"
# A fake protected term so the ip-guard tests exercise the mechanism without
# putting any real codename in the repo.
os.environ["GROUNDED_PROTECTED_TOPICS"] = "ACME-SECRET"
