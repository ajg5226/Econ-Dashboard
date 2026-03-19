"""
Recession Prediction Engine
Core modules for data acquisition and ensemble modeling
"""

from .data_acquisition import RecessionDataAcquisition
from .ensemble_model import RecessionEnsembleModel

__all__ = ['RecessionDataAcquisition', 'RecessionEnsembleModel']

