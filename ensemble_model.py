"""
Legacy compatibility module.

Use `recession_engine.ensemble_model.RecessionEnsembleModel` directly.
This file re-exports the canonical class to avoid duplicate implementations.
"""

from recession_engine.ensemble_model import RecessionEnsembleModel

__all__ = ["RecessionEnsembleModel"]
