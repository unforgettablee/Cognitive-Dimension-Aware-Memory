"""Traditional memory extraction (trajectory, workflow, traj, summary, insight).

Wraps harbor/experiments/utils/memory_extract.py.
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


def _lazy_import(name: str):
    """Import and cache a function from the original module."""
    import importlib
    mod = importlib.import_module("experiments.utils.memory_extract")
    fn = getattr(mod, name)
    globals()[name] = fn  # cache for subsequent calls
    return fn


def extract_rawtraj_memory(*args, **kwargs):
    return _lazy_import("extract_rawtraj_memory")(*args, **kwargs)


def extract_workflow_memory(*args, **kwargs):
    return _lazy_import("extract_workflow_memory")(*args, **kwargs)


def extract_traj_memory(*args, **kwargs):
    return _lazy_import("extract_traj_memory")(*args, **kwargs)


def extract_summary_memory(*args, **kwargs):
    return _lazy_import("extract_summary_memory")(*args, **kwargs)


def extract_insight_memory(*args, **kwargs):
    return _lazy_import("extract_insight_memory")(*args, **kwargs)


def _call_llm_with_json(*args, **kwargs):
    return _lazy_import("_call_llm_with_json")(*args, **kwargs)


__all__ = [
    "extract_rawtraj_memory",
    "extract_workflow_memory",
    "extract_traj_memory",
    "extract_summary_memory",
    "extract_insight_memory",
    "_call_llm_with_json",
]
