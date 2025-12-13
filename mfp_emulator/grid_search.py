"""
Grid search utilities for parameter fitting.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Union, Callable
import time

from .reionization import tanh_reionization_model, ReionizationHistory
from .data import get_observational_mfp_data, get_gamma_datasets, create_gamma_interpolator
from .cosmology import get_deltaNL_interpolator, calculate_eta


class GridSearch:
    """
    Grid search for fitting reionization parameters to MFP observations.
    
    Searches over gamma and z_re to find best-fit values by comparing
    emulator predictions to observational data.
    """
    
    def __init__(
        self,
        emulator,
        obs_data: Optional[Dict] = None,
        density: float = 0.0,
        energy_eV: float = 13.6,
    ):
        """
        Initialize grid search.
        
        Parameters
        ----------
        emulator : MFPEmulator
            Trained MFP emulator
        obs_data : dict, optional
            Observational MFP data. If None, loads default.
        density : float
            Density in sigma units for predictions
        energy_eV : float
            Photon energy in eV
        """
        self.emulator = emulator
        self.density = density
        self.energy_eV = energy_eV
        
        if obs_data is None:
            obs_data = get_observational_mfp_data(exclude_gaikwad=True)
        
        self.obs_data = obs_data
        self._prepare_observations()
        
        self.results: Optional[Dict] = None
    
    def _prepare_observations(self):
        """Flatten observational data for fitting."""
        self.obs_z = []
        self.obs_mfp = []
        self.obs_mfp_err = []
        
        for study, data in self.obs_data.items():
            self.obs_z.extend(data['z'])
            self.obs_mfp.extend(data['mfp'])
            avg_err = np.mean(data['mfp_err'], axis=1)
            self.obs_mfp_err.extend(avg_err)
        
        self.obs_z = np.array(self.obs_z)
        self.obs_mfp = np.array(self.obs_mfp)
        self.obs_mfp_err = np.array(self.obs_mfp_err)
    
    def predict_mfp(self, gamma: float, zre: float, z: float) -> float:
        """Predict MFP at given parameters."""
        try:
            return self.emulator.predict(
                z=z, z_re=zre, gamma=gamma,
                density=self.density, energy_eV=self.energy_eV
            )
        except Exception:
            return np.nan
    
    def calculate_chi_squared(self, gamma: float, zre: float) -> float:
        """Calculate chi-squared for given parameters."""
        predicted = np.array([self.predict_mfp(gamma, zre, z) for z in self.obs_z])
        valid = ~np.isnan(predicted) & (predicted > 0)
        
        if np.sum(valid) < len(self.obs_z) * 0.3:
            return np.inf
        
        obs_valid = self.obs_mfp[valid]
        err_valid = self.obs_mfp_err[valid]
        pred_valid = predicted[valid]
        err_valid[err_valid < 1e-9] = 1e-9
        
        return np.sum(((obs_valid - pred_valid) / err_valid)**2)
    
    def run(
        self,
        gamma_range: Tuple[float, float] = (0.03, 3.0),
        zre_range: Tuple[float, float] = (5.0, 8.0),
        n_gamma: int = 30,
        n_zre: int = 30,
        verbose: bool = True,
    ) -> Dict:
        """
        Run grid search.
        
        Parameters
        ----------
        gamma_range : tuple
            (min, max) for gamma_12 values
        zre_range : tuple
            (min, max) for z_re values
        n_gamma, n_zre : int
            Number of grid points
        verbose : bool
            Print progress
            
        Returns
        -------
        results : dict
            Dictionary with grids and best-fit values
        """
        if verbose:
            print(f"Grid search: Γ₁₂=[{gamma_range[0]}, {gamma_range[1]}], "
                  f"z_re=[{zre_range[0]}, {zre_range[1]}]")
        
        gamma_vals = np.logspace(np.log10(gamma_range[0]), np.log10(gamma_range[1]), n_gamma)
        zre_vals = np.linspace(zre_range[0], zre_range[1], n_zre)
        
        gamma_grid, zre_grid = np.meshgrid(gamma_vals, zre_vals)
        chi2_grid = np.zeros_like(gamma_grid)
        
        total = n_gamma * n_zre
        start_time = time.time()
        
        for i, zre in enumerate(zre_vals):
            for j, gamma in enumerate(gamma_vals):
                chi2_grid[i, j] = self.calculate_chi_squared(gamma, zre)
                
                if verbose and ((i * n_gamma + j + 1) % max(1, total // 10) == 0):
                    elapsed = time.time() - start_time
                    print(f"  Progress: {i * n_gamma + j + 1}/{total} ({elapsed:.1f}s)")
        
        min_idx = np.unravel_index(np.nanargmin(chi2_grid), chi2_grid.shape)
        
        self.results = {
            'gamma_grid': gamma_grid,
            'zre_grid': zre_grid,
            'chi2_grid': chi2_grid,
            'best_gamma': gamma_grid[min_idx],
            'best_zre': zre_grid[min_idx],
            'best_chi2': chi2_grid[min_idx],
            'obs_z': self.obs_z,
            'obs_mfp': self.obs_mfp,
            'obs_mfp_err': self.obs_mfp_err,
        }
        
        if verbose:
            print(f"\nBest fit: Γ₁₂={self.results['best_gamma']:.3f}, "
                  f"z_re={self.results['best_zre']:.2f}, "
                  f"χ²={self.results['best_chi2']:.2f}")
        
        return self.results


class TanhGridSearch:
    """
    Grid search for tanh reionization model parameters.
    
    Searches over z_re (midpoint) and Delta_z (duration) to find
    best-fit reionization history from MFP observations.
    """
    
    def __init__(
        self,
        emulator,
        obs_data: Optional[Dict] = None,
        gamma_data: Optional[Dict] = None,
        energy_eV: float = 13.6,
        Rcell: float = 2.0,
    ):
        """
        Initialize tanh grid search.
        
        Parameters
        ----------
        emulator : MFPEmulator
            Trained MFP emulator
        obs_data : dict, optional
            Observational MFP data
        gamma_data : dict, optional
            Gamma measurements for gamma(z) interpolation
        energy_eV : float
            Photon energy in eV
        Rcell : float
            Cell size for quadrature
        """
        self.emulator = emulator
        self.energy_eV = energy_eV
        self.Rcell = Rcell
        
        if obs_data is None:
            obs_data = get_observational_mfp_data(exclude_gaikwad=True, z_range=(4.8, 6.0))
        
        self.obs_data = obs_data
        self._prepare_observations()
        
        if gamma_data is None:
            gamma_data = get_gamma_datasets(priority_filter=True)
        
        self.gamma_interp, self.gamma_z_range, _ = create_gamma_interpolator(gamma_data)
        
        self._deltaNL_func = get_deltaNL_interpolator(Rcell)
        
        self.quad_points = np.array([-np.sqrt(3), 0, np.sqrt(3)])
        self.quad_weights = np.array([np.sqrt(np.pi)/6, 2*np.sqrt(np.pi)/3, np.sqrt(np.pi)/6])
        
        self.results: Optional[Dict] = None
    
    def _prepare_observations(self):
        """Flatten and filter observational data."""
        self.obs_z = []
        self.obs_mfp = []
        self.obs_mfp_err = []
        
        for study, data in self.obs_data.items():
            self.obs_z.extend(data['z'])
            self.obs_mfp.extend(data['mfp'])
            avg_err = np.mean(data['mfp_err'], axis=1)
            self.obs_mfp_err.extend(avg_err)
        
        self.obs_z = np.array(self.obs_z)
        self.obs_mfp = np.array(self.obs_mfp)
        self.obs_mfp_err = np.array(self.obs_mfp_err)
    
    def predict_mfp_tanh(
        self,
        z_obs: float,
        gamma_12: float,
        z_re: float,
        Delta_z: float,
    ) -> float:
        """
        Predict MFP using tanh model for z_re distribution.
        
        This marginalizes over the tanh-model P(z_re) distribution
        and integrates over density using Gaussian quadrature.
        """
        z_re_grid = np.linspace(z_obs + 0.1, 18.0, 100)
        Q_grid = tanh_reionization_model(z_re_grid, z_re, Delta_z)
        dQ_dz = np.gradient(Q_grid, z_re_grid)
        P_zre = -dQ_dz
        
        f_ion = tanh_reionization_model(z_obs, z_re, Delta_z)
        
        if f_ion <= 0.01:
            return np.nan
        
        mfp_at_zre = []
        for zre_val in z_re_grid:
            mfp = self._predict_mfp_quadrature(z_obs, zre_val, gamma_12)
            mfp_at_zre.append(mfp)
        
        mfp_at_zre = np.array(mfp_at_zre)
        valid = ~np.isnan(mfp_at_zre) & (mfp_at_zre > 0)
        
        if not np.any(valid):
            return np.nan
        
        kappa_integral = np.trapz(P_zre[valid] / mfp_at_zre[valid], z_re_grid[valid])
        
        if kappa_integral <= 0:
            return np.nan
        
        return f_ion / kappa_integral
    
    def _predict_mfp_quadrature(self, z_obs: float, z_re: float, gamma_12: float) -> float:
        """Predict MFP with Gaussian quadrature over density."""
        try:
            eta_val = calculate_eta(z_obs, self.Rcell)
            kappa_sum = 0.0
            valid_points = 0
            
            for sigma_pt, weight in zip(self.quad_points, self.quad_weights):
                mfp = self.emulator.predict(
                    z=z_obs, z_re=z_re, gamma=gamma_12,
                    density=sigma_pt, energy_eV=self.energy_eV
                )
                
                if mfp > 0 and mfp < 1e6:
                    delta_NL = self._deltaNL_func(np.array([[z_obs, sigma_pt]]))[0]
                    if delta_NL > -0.99:
                        kappa_sum += weight * (1.0 / mfp) / (1 + delta_NL)
                        valid_points += 1
            
            if valid_points > 0 and kappa_sum > 0:
                mean_kappa = (eta_val / np.sqrt(np.pi)) * kappa_sum
                return 1.0 / mean_kappa
            return np.nan
        except Exception:
            return np.nan
    
    def run(
        self,
        z_re_range: Tuple[float, float] = (5.5, 10.0),
        Delta_z_range: Tuple[float, float] = (0.5, 6.0),
        n_z_re: int = 40,
        n_Delta_z: int = 40,
        gamma_mode: str = 'per_redshift',
        verbose: bool = True,
    ) -> Dict:
        """
        Run tanh grid search.
        
        Parameters
        ----------
        z_re_range : tuple
            (min, max) for z_re (reionization midpoint)
        Delta_z_range : tuple
            (min, max) for Delta_z (duration parameter)
        n_z_re, n_Delta_z : int
            Number of grid points
        gamma_mode : str
            'per_redshift' - use gamma(z) at each observation z
            'median' - use median gamma across all z
            'representative' - use gamma at z=5.5
        verbose : bool
            Print progress
            
        Returns
        -------
        results : dict
            Dictionary with grids and best-fit values
        """
        if gamma_mode == 'per_redshift':
            gamma_at_z = np.array([float(self.gamma_interp(z)) * 1e12 for z in self.obs_z])
        elif gamma_mode == 'median':
            gamma_vals = np.array([float(self.gamma_interp(z)) for z in self.obs_z])
            gamma_single = np.median(gamma_vals) * 1e12
            gamma_at_z = np.full_like(self.obs_z, gamma_single)
        elif gamma_mode == 'representative':
            gamma_single = float(self.gamma_interp(5.5)) * 1e12
            gamma_at_z = np.full_like(self.obs_z, gamma_single)
        else:
            raise ValueError(f"Unknown gamma_mode: {gamma_mode}")
        
        if verbose:
            print(f"Tanh grid search: z_re=[{z_re_range[0]}, {z_re_range[1]}], "
                  f"Δz=[{Delta_z_range[0]}, {Delta_z_range[1]}]")
            print(f"Using {len(self.obs_z)} observations, gamma_mode='{gamma_mode}'")
        
        z_re_grid = np.linspace(z_re_range[0], z_re_range[1], n_z_re)
        Delta_z_grid = np.linspace(Delta_z_range[0], Delta_z_range[1], n_Delta_z)
        
        chi2_grid = np.zeros((n_z_re, n_Delta_z))
        n_valid_grid = np.zeros((n_z_re, n_Delta_z), dtype=int)
        
        total = n_z_re * n_Delta_z * len(self.obs_z)
        start_time = time.time()
        count = 0
        
        for i, z_re in enumerate(z_re_grid):
            for j, Delta_z in enumerate(Delta_z_grid):
                predicted = []
                
                for k, z_obs in enumerate(self.obs_z):
                    count += 1
                    mfp = self.predict_mfp_tanh(z_obs, gamma_at_z[k], z_re, Delta_z)
                    predicted.append(mfp)
                    
                    if verbose and count % 500 == 0:
                        elapsed = time.time() - start_time
                        rate = count / elapsed if elapsed > 0 else 1
                        eta = (total - count) / rate
                        print(f"  Progress: {count}/{total} ({100*count/total:.1f}%) - ETA: {eta/60:.1f}min")
                
                predicted = np.array(predicted)
                valid = ~np.isnan(predicted) & (predicted > 0)
                n_valid = np.sum(valid)
                n_valid_grid[i, j] = n_valid
                
                if n_valid < len(self.obs_z) * 0.5:
                    chi2_grid[i, j] = np.inf
                else:
                    chi2 = np.sum(((self.obs_mfp[valid] - predicted[valid]) / self.obs_mfp_err[valid])**2)
                    chi2_grid[i, j] = chi2
        
        elapsed = time.time() - start_time
        if verbose:
            print(f"Grid search complete in {elapsed/60:.1f} min")
        
        idx_best = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)
        z_re_best = z_re_grid[idx_best[0]]
        Delta_z_best = Delta_z_grid[idx_best[1]]
        chi2_best = chi2_grid[idx_best]
        n_valid_best = n_valid_grid[idx_best]
        
        dof = max(n_valid_best - 2, 1)
        reduced_chi2 = chi2_best / dof
        
        self.results = {
            'z_re_grid': z_re_grid,
            'Delta_z_grid': Delta_z_grid,
            'chi2_grid': chi2_grid,
            'n_valid_grid': n_valid_grid,
            'z_re_best': z_re_best,
            'Delta_z_best': Delta_z_best,
            'chi2_best': chi2_best,
            'reduced_chi2': reduced_chi2,
            'dof': dof,
            'gamma_mode': gamma_mode,
            'gamma_at_z': gamma_at_z,
            'obs_z': self.obs_z,
            'obs_mfp': self.obs_mfp,
            'obs_mfp_err': self.obs_mfp_err,
        }
        
        if verbose:
            print(f"\nBest fit: z_re={z_re_best:.2f}, Δz={Delta_z_best:.2f}")
            print(f"χ²={chi2_best:.2f}, reduced χ²={reduced_chi2:.2f}")
            print(f"Reionization: z_start≈{z_re_best+Delta_z_best:.1f}, z_end≈{z_re_best-Delta_z_best:.1f}")
        
        return self.results
