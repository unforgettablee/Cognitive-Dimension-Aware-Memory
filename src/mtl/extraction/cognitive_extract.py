"""Cognitive memory extraction (v2: 5 LLM calls per task, 4x4 matrix).

Wraps harbor/experiments/utils/cognitive_memory_extract.py.
LLM settings are read from mtl.llm — call mtl.llm.configure() before the first extraction.
"""
import os
import sys

# Ensure harbor/experiments is importable
_HARBOR_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "harbor")
)
if _HARBOR_DIR not in sys.path:
    sys.path.insert(0, _HARBOR_DIR)


def extract_cognitive_memories(*args, **kwargs):
    """Import and call the real function on first use (after LLM config is set).

    Accepts optional keyword ``derive_traditional: bool`` (default True) to control
    whether traditional memories are derived from the cognitive matrix.
    """
    from experiments.utils.cognitive_memory_extract import extract_cognitive_memories as _real  # noqa: E402
    # Replace self in module globals so subsequent calls go directly to the real function
    globals()["extract_cognitive_memories"] = _real
    return _real(*args, **kwargs)


def _summarize_trajectory(*args, **kwargs):
    from experiments.utils.cognitive_memory_extract import _summarize_trajectory as _real  # noqa: E402
    globals()["_summarize_trajectory"] = _real
    return _real(*args, **kwargs)


def _extract_level(*args, **kwargs):
    from experiments.utils.cognitive_memory_extract import _extract_level as _real  # noqa: E402
    globals()["_extract_level"] = _real
    return _real(*args, **kwargs)


def _derive_traditional_memories(*args, **kwargs):
    from experiments.utils.cognitive_memory_extract import _derive_traditional_memories as _real  # noqa: E402
    globals()["_derive_traditional_memories"] = _real
    return _real(*args, **kwargs)


__all__ = [
    "extract_cognitive_memories",
    "_summarize_trajectory",
    "_extract_level",
    "_derive_traditional_memories",
]
