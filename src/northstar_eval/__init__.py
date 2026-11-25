"""Northstar Eval Platform — deterministic LLM evaluation with human review."""

from .scorer import PASS_THRESHOLD, SCORER_VERSION

__version__ = "0.1.0"
__all__ = ["__version__", "SCORER_VERSION", "PASS_THRESHOLD"]
