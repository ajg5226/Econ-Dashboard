"""
Legacy compatibility module.

Use `recession_engine.data_acquisition.RecessionDataAcquisition` directly.
This file re-exports the canonical class to avoid duplicate implementations.
"""

from recession_engine.data_acquisition import RecessionDataAcquisition

__all__ = ["RecessionDataAcquisition"]
