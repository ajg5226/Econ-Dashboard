"""
Recession Prediction Engine
Core modules for data acquisition and ensemble modeling
"""

from .data_acquisition import RecessionDataAcquisition
from .ensemble_model import RecessionEnsembleModel
from .glr_engine import GLRRegimeEngine

__all__ = ['RecessionDataAcquisition', 'RecessionEnsembleModel', 'GLRRegimeEngine']

