"""Errors raised by the Northstar evaluation platform.

Every error carries a plain-language message. Dataset errors name the
file, the line/row and the field at fault so a reviewer can fix the
data instead of guessing.
"""

from __future__ import annotations


class NorthstarError(Exception):
    """Base class for all platform errors."""


class DatasetError(NorthstarError):
    """The dataset file is missing, malformed or fails validation."""


class ProviderError(NorthstarError):
    """The LLM provider could not be configured or did not answer."""


class StoreError(NorthstarError):
    """The SQLite store could not be read or written."""


class RunNotFoundError(NorthstarError):
    """No run with that id exists in the store."""


class ReviewError(NorthstarError):
    """A review action is invalid (unknown result, bad judgement, ...)."""
