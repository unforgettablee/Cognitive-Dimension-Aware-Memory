"""Prompt templates for memory extraction and cognitive analysis.

Bridges to original prompts in harbor/experiments/prompts/ with clean import paths.
"""
import os
import sys

_HARBOR_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "harbor")
)
if _HARBOR_DIR not in sys.path:
    sys.path.insert(0, _HARBOR_DIR)

from experiments.prompts.cognitive_memory import (  # noqa: E402
    TRAJECTORY_SUMMARY_PROMPT,
    COMBINED_MATRIX,
)
from experiments.prompts.memory import (  # noqa: E402
    WORKFLOW_CORRECT_EXTRACT_PROMPT,
    WORKFLOW_WRONG_EXTRACT_PROMPT,
    CODE_SPECIFIC_CORRECT_PROMPT,
    CODE_SPECIFIC_WRONG_PROMPT,
)

__all__ = [
    "TRAJECTORY_SUMMARY_PROMPT",
    "COMBINED_MATRIX",
    "WORKFLOW_CORRECT_EXTRACT_PROMPT",
    "WORKFLOW_WRONG_EXTRACT_PROMPT",
    "CODE_SPECIFIC_CORRECT_PROMPT",
    "CODE_SPECIFIC_WRONG_PROMPT",
]
