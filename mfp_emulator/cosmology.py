"""
Cosmology utilities and density field calculations.
"""

import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator
from functools import lru_cache
from typing import Optional, Tuple

# Try to import colossus, but allow graceful fallback
try:
    from colossus.cosmology import cosmology as colossus_cosmology
    HAS_COLOSSUS = True
except ImportError:
    HAS_COLOSSUS = False
    colossus_cosmology = None

# Module-level storage for cosmology and interpolator
_COSMO = None
_DELTANL_INTERPOLATOR = None

# Default cosmology parameters
DEFAULT_COSMOLOGY = {
    'flat': True,
    'H0': 68.0,
    'Om0': 0.305147,
    'Ob0': 0.0482266,
    'sigma8': 0.82033,
    'ns': 0.9667,
}


def setup_cosmology(
    H0: float = 68.0,
    Om0: float = 0.305147,
    Ob0: float = 0.0482266,
    sigma8: float = 0.82033,
    ns: float = 0.9667,
    name: str = 'mfp_cosmo',
) -> 'colossus_cosmology':
    """
    Set up the cosmology for calculations.
    
    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc
    Om0 : float
        Total matter density parameter
    Ob0 : float
        Baryon density parameter
    sigma8 : float
        Power spectrum normalization
    ns : float
        Spectral index
    name : str
        Name for the cosmology
        
    Returns
    -------
    cosmo : colossus cosmology object
    """
    global _COSMO
    
    if not HAS_COLOSSUS:
        raise ImportError(
            "colossus package required for cosmology calculations. "
            "Install with: pip install colossus"
        )
    
    params = {
        'flat': True,
        'H0': H0,
        'Om0': Om0,
        'Ob0': Ob0,
        'sigma8': sigma8,
        'ns': ns,
    }
    
    colossus_cosmology.addCosmology(name, params)
    _COSMO = colossus_cosmology.setCosmology(name)
    
    return _COSMO


def get_cosmology():
    """Get the current cosmology, setting up default if needed."""
    global _COSMO
    if _COSMO is None:
        _COSMO = setup_cosmology()
    return _COSMO


@lru_cache(maxsize=256)
def _sigma_cached(z_rounded: float, Rcell: float) -> float:
    """Calculate sigma(R,z) with caching."""
    cosmo = get_cosmology()
    z = float(z_rounded)
    R0 = Rcell * (3.0 / 4.0 / np.pi) ** (1.0 / 3.0)
    sigma0 = cosmo.sigma(R0, 0.0)
    d_of_z = cosmo.growthFactor(z)
    return sigma0 * d_of_z


def sigma(z: float, Rcell: float = 2.0) -> float:
    """
    Calculate sigma(R,z) - the rms density fluctuation smoothed on scale R at redshift z.
    
    Parameters
    ----------
    z : float
        Redshift
    Rcell : float
        Cell size in Mpc
        
    Returns
    -------
    sigma : float
        RMS density fluctuation
    """
    z_rounded = round(z * 20) / 20.0
    return _sigma_cached(z_rounded, Rcell)


def dlin_vectorized(deltaNL_array: np.ndarray, z: float, Rcell: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert non-linear density to linear density (vectorized).
    
    Uses the fitting formula from the spherical collapse model.
    
    Parameters
    ----------
    deltaNL_array : ndarray
        Non-linear density contrast values
    z : float
        Redshift
    Rcell : float
        Cell size in Mpc
        
    Returns
    -------
    deltalin : ndarray
        Linear density contrast
    delta_over_sigma : ndarray
        Linear density in units of sigma
    """
    cosmo = get_cosmology()
    R0 = Rcell * (3.0 / 4.0 / np.pi) ** (1.0 / 3.0)
    sigma0 = cosmo.sigma(R0, 0.0)
    d_of_z = cosmo.growthFactor(z)

    deltalin = (
        -1.35 * (1. + deltaNL_array) ** (-2./3.) +
        0.78785 * (1. + deltaNL_array) ** (-0.58661) -
        1.12431 * (1. + deltaNL_array) ** (-0.5) +
        1.68647
    )

    delta_over_sigma = deltalin / (d_of_z * sigma0)
    return deltalin, delta_over_sigma


def make_deltaNL_interpolator(Rcell: float = 2.0, verbose: bool = True) -> RegularGridInterpolator:
    """
    Create interpolation table for delta_over_sigma -> deltaNL conversion.
    
    This is expensive to compute but only needs to be done once.
    
    Parameters
    ----------
    Rcell : float
        Cell size in Mpc
    verbose : bool
        Print progress
        
    Returns
    -------
    interpolator : RegularGridInterpolator
        Interpolator that takes (z, delta_over_sigma) and returns deltaNL
    """
    if verbose:
        print("Creating deltaNL interpolation table...")

    n_z = 150
    n_delta = 150
    n_deltaNL = 2000

    deltaNL_array = np.logspace(-1, 3, n_deltaNL) - 1.0
    redshift = np.linspace(20, 3, n_z)
    delta_over_sigma = np.linspace(-3, 5, n_delta)

    deltaNL_grid = np.zeros((n_z, n_delta))

    for i, z in enumerate(redshift):
        if verbose and i % 30 == 0:
            print(f"  Progress: {i}/{n_z}")
        dlin_array, delta_over_sigma_array = dlin_vectorized(deltaNL_array, z, Rcell)
        func = interp1d(
            delta_over_sigma_array, deltaNL_array,
            fill_value='extrapolate', kind='linear'
        )
        deltaNL_grid[i, :] = func(delta_over_sigma)

    interpolator = RegularGridInterpolator(
        (redshift, delta_over_sigma),
        deltaNL_grid,
        method='linear',
        bounds_error=False,
        fill_value=None
    )

    if verbose:
        print("Done!")
    
    return interpolator


def get_deltaNL_interpolator(Rcell: float = 2.0) -> RegularGridInterpolator:
    """
    Get or create the deltaNL interpolator (cached).
    
    Parameters
    ----------
    Rcell : float
        Cell size in Mpc
        
    Returns
    -------
    interpolator : RegularGridInterpolator
    """
    global _DELTANL_INTERPOLATOR
    if _DELTANL_INTERPOLATOR is None:
        _DELTANL_INTERPOLATOR = make_deltaNL_interpolator(Rcell)
    return _DELTANL_INTERPOLATOR


def delta_over_sigma_to_deltaNL(delta_over_sigma: float, z: float, Rcell: float = 2.0) -> float:
    """
    Convert delta/sigma to deltaNL using interpolation.
    
    Parameters
    ----------
    delta_over_sigma : float
        Linear density in units of sigma
    z : float
        Redshift
    Rcell : float
        Cell size in Mpc
        
    Returns
    -------
    deltaNL : float
        Non-linear density contrast
    """
    interpolator = get_deltaNL_interpolator(Rcell)
    points = np.array([[z, delta_over_sigma]])
    return interpolator(points)[0]


@lru_cache(maxsize=256)
def _calculate_eta_cached(z_rounded: float, Rcell: float = 2.0) -> float:
    """Calculate eta with caching."""
    z = float(z_rounded)
    deltaNL_array = np.logspace(-1, 3, 2000) - 1.0
    sigma_z = sigma(z, Rcell)
    delta_lin_array, _ = dlin_vectorized(deltaNL_array, z, Rcell)

    integrand = (
        1.0 / (1.0 + deltaNL_array) *
        (1.0 / np.sqrt(2.0 * np.pi * sigma_z**2)) *
        np.exp(-delta_lin_array**2 / (2.0 * sigma_z**2))
    )

    integral_value = np.trapz(integrand, x=delta_lin_array)
    eta = 1.0 / integral_value if integral_value > 0 else 1.0
    return eta


def calculate_eta(z: float, Rcell: float = 2.0) -> float:
    """
    Calculate the eta correction factor for density averaging.
    
    Parameters
    ----------
    z : float
        Redshift
    Rcell : float
        Cell size in Mpc
        
    Returns
    -------
    eta : float
        Correction factor
    """
    z_rounded = round(z * 20) / 20.0
    return _calculate_eta_cached(z_rounded, Rcell)


# Physical constants
MPC_TO_CM = 3.08568e24  # cm per Mpc
H_PLANCK_EV = 4.135667696e-15  # Planck constant in eV·s
E_HI_IONIZATION = 13.6  # eV
E_HEII_IONIZATION = 54.4  # eV
NU_HI = E_HI_IONIZATION / H_PLANCK_EV  # Hz
NU_HEII = E_HEII_IONIZATION / H_PLANCK_EV  # Hz
