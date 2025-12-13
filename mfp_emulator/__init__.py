"""
MFP Emulator - Neural network emulator for mean free path predictions in reionization studies.
"""

from .emulator import MFPEmulator
from .reionization import (
    tanh_reionization_model,
    load_reionization_history,
    ReionizationHistory,
)
from .ndot import NdotCalculator
from .grid_search import GridSearch, TanhGridSearch
from .cosmology import setup_cosmology, get_deltaNL_interpolator
from .data import (
    load_chardin_data,
    get_observational_mfp_data,
    get_gamma_datasets,
)

__version__ = "0.1.0"
__all__ = [
    "MFPEmulator",
    "tanh_reionization_model",
    "load_reionization_history",
    "ReionizationHistory",
    "NdotCalculator",
    "GridSearch",
    "TanhGridSearch",
    "setup_cosmology",
    "get_deltaNL_interpolator",
    "load_chardin_data",
    "get_observational_mfp_data",
    "get_gamma_datasets",
]
