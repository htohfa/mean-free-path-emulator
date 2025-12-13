"""
Ionizing photon production rate (ndot) calculations.
"""

import numpy as np
from typing import Optional, Dict, Callable, Union
from scipy.interpolate import interp1d
import time

from .cosmology import (
    get_deltaNL_interpolator,
    calculate_eta,
    MPC_TO_CM,
    H_PLANCK_EV,
    E_HI_IONIZATION,
    E_HEII_IONIZATION,
    NU_HI,
    NU_HEII,
)
from .reionization import ReionizationHistory
from .data import get_gamma_datasets, create_gamma_interpolator


class NdotCalculator:
    """
    Calculator for ionizing photon production rate (ndot).
    
    This class implements the BH07 (Bolton & Haehnelt 2007) method
    and an emulator-based approach using MFP predictions.
    """
    
    def __init__(
        self,
        emulator,
        reionization_history: ReionizationHistory,
        Rcell: float = 2.0,
        alpha_s: float = 1.5,
    ):
        """
        Initialize the calculator.
        
        Parameters
        ----------
        emulator : MFPEmulator
            Trained MFP emulator
        reionization_history : ReionizationHistory
            Reionization history object
        Rcell : float
            Cell size in Mpc for density calculations
        alpha_s : float
            Spectral index of ionizing sources
        """
        self.emulator = emulator
        self.history = reionization_history
        self.Rcell = Rcell
        self.alpha_s = alpha_s
        
        # Ensure interpolator is ready
        self._deltaNL_func = get_deltaNL_interpolator(Rcell)
        
        # Gaussian quadrature points and weights for density integration
        self.quad_points_sigma = np.array([-np.sqrt(3), 0, np.sqrt(3)])
        self.quad_weights = np.array([
            np.sqrt(np.pi)/6,
            2*np.sqrt(np.pi)/3,
            np.sqrt(np.pi)/6
        ])
    
    def predict_mfp_gaussian_quadrature(
        self,
        z_obs: float,
        z_re: float,
        gamma_val: float,
        energy_eV: float = 13.6,
    ) -> float:
        """
        Predict MFP using Gaussian quadrature over density.
        
        Parameters
        ----------
        z_obs : float
            Observation redshift
        z_re : float
            Reionization redshift
        gamma_val : float
            Photoionization rate (gamma_12)
        energy_eV : float
            Photon energy in eV
            
        Returns
        -------
        mfp : float
            Mean free path in cMpc
        """
        try:
            eta_val = calculate_eta(z_obs, self.Rcell)
            
            kappa_sum = 0.0
            valid_points = 0
            
            for quad_sigma, quad_weight in zip(self.quad_points_sigma, self.quad_weights):
                try:
                    mfp_point = self.emulator.predict(
                        z=z_obs, z_re=z_re, gamma=gamma_val,
                        density=quad_sigma, energy_eV=energy_eV
                    )
                    
                    if mfp_point > 0 and mfp_point < 1e6:
                        points = np.array([[z_obs, quad_sigma]])
                        delta_NL = self._deltaNL_func(points)[0]
                        
                        if delta_NL > -0.99:
                            kappa_point = 1.0 / mfp_point
                            weighted_kappa = quad_weight * (kappa_point / (1 + delta_NL))
                            kappa_sum += weighted_kappa
                            valid_points += 1
                            
                except Exception:
                    continue
            
            if valid_points > 0 and kappa_sum > 0:
                mean_kappa = (eta_val / np.sqrt(np.pi)) * kappa_sum
                return 1.0 / mean_kappa
            else:
                return np.nan
                
        except Exception:
            return np.nan
    
    def predict_mfp_with_history(
        self,
        z_obs: float,
        gamma_val: float,
        energy_eV: float = 13.6,
        z_re_min: float = 5.0,
        z_re_max: float = 18.0,
        n_z_re: int = 130,
    ) -> float:
        """
        Predict MFP by marginalizing over reionization history.
        
        Parameters
        ----------
        z_obs : float
            Observation redshift
        gamma_val : float
            Photoionization rate (gamma_12)
        energy_eV : float
            Photon energy in eV
        z_re_min, z_re_max : float
            Range for z_re integration
        n_z_re : int
            Number of z_re grid points
            
        Returns
        -------
        mfp : float
            Mean free path in cMpc
        """
        z_re_grid = np.linspace(z_re_min, z_re_max, n_z_re)
        P_zre = self.history.P_zre(z_re_grid)
        f_ion = self.history.Q(z_obs)
        
        if f_ion <= 0.01:
            return np.nan
        
        mfp_predictions = np.array([
            self.predict_mfp_gaussian_quadrature(z_obs, z_re, gamma_val, energy_eV)
            for z_re in z_re_grid
        ])
        
        valid_mask = (z_re_grid > z_obs) & ~np.isnan(mfp_predictions) & (mfp_predictions > 0)
        
        if not np.any(valid_mask):
            return np.nan
        
        z_valid = z_re_grid[valid_mask]
        P_valid = P_zre[valid_mask]
        mfp_valid = mfp_predictions[valid_mask]
        
        total_prob_valid = np.trapz(P_valid, z_valid)
        
        if total_prob_valid < 0.01:
            return np.nan
        
        kappa_integral = np.trapz(P_valid / mfp_valid, z_valid)
        kappa_obs = kappa_integral / f_ion
        lambda_obs = 1.0 / kappa_obs
        
        return lambda_obs
    
    def compute_ndot_bh07(
        self,
        z: float,
        mfp: float,
        gamma: float,
        alpha_b: float = None,
        beta: float = 1.3,
    ) -> float:
        """
        Compute ndot using BH07 method.
        
        Parameters
        ----------
        z : float
            Redshift
        mfp : float
            Mean free path in cMpc
        gamma : float
            Photoionization rate in s^-1
        alpha_b : float, optional
            Background spectral index. If None, computed from beta.
        beta : float
            MFP scaling exponent (lambda ~ nu^beta)
            
        Returns
        -------
        ndot : float
            Ionizing photon production rate in s^-1 cMpc^-3
        """
        if alpha_b is None:
            alpha_b = 3.0 * beta - 0.5
        
        # Convert MFP to proper Mpc (cMpc -> pMpc)
        lambda_mfp_pMpc = mfp / (1 + z)
        lambda_mfp_cm = lambda_mfp_pMpc * MPC_TO_CM
        
        # Spectral integrals
        # I1 = integral of nu^(-alpha_s) from nu_HI to nu_HeII
        # I2 = integral of sigma(nu) * nu^(-alpha_s) from nu_HI to nu_HeII
        
        alpha = self.alpha_s
        
        # I1: integral of nu^(-alpha) from nu_HI to nu_HeII
        if alpha != 1.0:
            I1 = (NU_HEII**(1-alpha) - NU_HI**(1-alpha)) / (1 - alpha)
        else:
            I1 = np.log(NU_HEII / NU_HI)
        
        # I2: integral of sigma * nu^(-alpha) with sigma ~ nu^(-3)
        # So we need integral of nu^(-3-alpha)
        if alpha != -2.0:
            power = -3 - alpha
            I2_unnorm = (NU_HEII**(-2-alpha) - NU_HI**(-2-alpha)) / (-2 - alpha)
        else:
            I2_unnorm = np.log(NU_HEII / NU_HI)
        
        # sigma_HI at nu_HI threshold
        sigma_HI = 6.3e-18  # cm^2
        I2 = sigma_HI * NU_HI**3 * I2_unnorm
        
        # ndot = gamma * (I1 / I2) / lambda_mfp
        if I2 > 0 and lambda_mfp_cm > 0:
            ndot = gamma * (I1 / I2) / lambda_mfp_cm
            # Convert from cm^-3 to cMpc^-3
            ndot_cMpc = ndot * MPC_TO_CM**3
            return ndot_cMpc
        else:
            return np.nan
    
    def compute_ndot_emulator(
        self,
        z: float,
        gamma: float,
        energy_eV: float = 13.6,
    ) -> float:
        """
        Compute ndot using emulator-predicted MFP.
        
        Parameters
        ----------
        z : float
            Redshift
        gamma : float
            Photoionization rate in s^-1
        energy_eV : float
            Photon energy in eV
            
        Returns
        -------
        ndot : float
            Ionizing photon production rate in s^-1 cMpc^-3
        """
        gamma_12 = gamma * 1e12
        mfp = self.predict_mfp_with_history(z, gamma_12, energy_eV)
        
        if np.isnan(mfp) or mfp <= 0:
            return np.nan
        
        return self.compute_ndot_bh07(z, mfp, gamma)
    
    def run_analysis(
        self,
        z_values: np.ndarray = None,
        gamma_data: Dict = None,
        beta_values: list = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Run full ndot analysis over multiple redshifts.
        
        Parameters
        ----------
        z_values : ndarray, optional
            Redshifts to compute. Default: 4.8 to 6.0
        gamma_data : dict, optional
            Gamma measurements. Default: loads from get_gamma_datasets()
        beta_values : list, optional
            Beta values for BH07 comparison. Default: [1.0, 1.2, 1.3, 1.5, 1.7, 2.0]
        verbose : bool
            Print progress
            
        Returns
        -------
        results : dict
            Dictionary with ndot values for each method
        """
        if z_values is None:
            z_values = np.linspace(4.8, 6.0, 7)
        
        if gamma_data is None:
            gamma_data = get_gamma_datasets(priority_filter=True)
        
        if beta_values is None:
            beta_values = [1.0, 1.2, 1.3, 1.5, 1.7, 2.0]
        
        gamma_interp, _, _ = create_gamma_interpolator(gamma_data)
        
        results = {
            'z': list(z_values),
            'emulator': [],
            'emulator_err_lower': [],
            'emulator_err_upper': [],
        }
        
        for beta in beta_values:
            alpha_b = 3.0 * beta - 0.5
            results[f'BH07_beta_{beta}'] = []
            results[f'BH07_beta_{beta}_alpha_b'] = [alpha_b]
        
        start_time = time.time()
        
        for i, z in enumerate(z_values):
            if verbose:
                print(f"Processing z={z:.2f} ({i+1}/{len(z_values)})")
            
            gamma_val = float(gamma_interp(z))
            gamma_12 = gamma_val * 1e12
            
            # Emulator prediction
            mfp = self.predict_mfp_with_history(z, gamma_12)
            
            if not np.isnan(mfp) and mfp > 0:
                ndot = self.compute_ndot_bh07(z, mfp, gamma_val)
                results['emulator'].append(ndot)
                
                # Error propagation from gamma
                datasets = get_gamma_datasets()
                gamma_err = self._get_gamma_error_at_z(z, datasets)
                
                if gamma_err is not None:
                    gamma_lower = max(gamma_val - gamma_err[0], 1e-14)
                    gamma_upper = gamma_val + gamma_err[1]
                    
                    mfp_upper = self.predict_mfp_with_history(z, gamma_lower * 1e12)
                    mfp_lower = self.predict_mfp_with_history(z, gamma_upper * 1e12)
                    
                    ndot_lower = self.compute_ndot_bh07(z, mfp_lower, gamma_upper) if not np.isnan(mfp_lower) else np.nan
                    ndot_upper = self.compute_ndot_bh07(z, mfp_upper, gamma_lower) if not np.isnan(mfp_upper) else np.nan
                    
                    results['emulator_err_lower'].append(ndot - ndot_lower if not np.isnan(ndot_lower) else np.nan)
                    results['emulator_err_upper'].append(ndot_upper - ndot if not np.isnan(ndot_upper) else np.nan)
                else:
                    results['emulator_err_lower'].append(np.nan)
                    results['emulator_err_upper'].append(np.nan)
            else:
                results['emulator'].append(np.nan)
                results['emulator_err_lower'].append(np.nan)
                results['emulator_err_upper'].append(np.nan)
            
            # BH07 for each beta
            for beta in beta_values:
                if not np.isnan(mfp) and mfp > 0:
                    ndot_bh07 = self.compute_ndot_bh07(z, mfp, gamma_val, beta=beta)
                    results[f'BH07_beta_{beta}'].append(ndot_bh07)
                else:
                    results[f'BH07_beta_{beta}'].append(np.nan)
        
        elapsed = time.time() - start_time
        if verbose:
            print(f"Analysis complete in {elapsed:.1f}s")
        
        return results
    
    def _get_gamma_error_at_z(self, z: float, datasets: Dict) -> Optional[np.ndarray]:
        """Get gamma error at redshift z from nearest dataset point."""
        min_dist = float('inf')
        best_err = None
        
        for name, (z_arr, gamma_arr, err_arr) in datasets.items():
            for i, z_val in enumerate(z_arr):
                dist = abs(z - z_val)
                if dist < min_dist:
                    min_dist = dist
                    best_err = err_arr[i]
        
        if min_dist < 0.2:
            return best_err
        return None
