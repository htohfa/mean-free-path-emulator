"""
Reionization history models and utilities.
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import Optional, Callable, Union
from pathlib import Path


def tanh_reionization_model(z: Union[float, np.ndarray], z_re: float, Delta_z: float) -> Union[float, np.ndarray]:
    """
    Standard tanh model for reionization history Q(z).
    
    Q(z) = 0.5 * (1 - tanh((z - z_re) / Delta_z))
    
    Parameters
    ----------
    z : float or array
        Redshift(s)
    z_re : float
        Midpoint of reionization (where Q = 0.5)
    Delta_z : float
        Duration parameter (controls steepness)
        
    Returns
    -------
    Q : float or array
        Ionization fraction (0 = neutral, 1 = fully ionized)
    """
    return 0.5 * (1 - np.tanh((z - z_re) / Delta_z))


class ReionizationHistory:
    """
    Flexible reionization history class that can use either parametric models
    (like tanh) or tabulated data from files.
    """
    
    def __init__(
        self,
        model: str = "tanh",
        z_re: float = 7.0,
        Delta_z: float = 1.5,
        data_file: Optional[str] = None,
    ):
        """
        Initialize reionization history.
        
        Parameters
        ----------
        model : str
            Model type: 'tanh' for parametric, 'file' for tabulated data
        z_re : float
            Midpoint redshift (for tanh model)
        Delta_z : float
            Duration parameter (for tanh model)
        data_file : str, optional
            Path to file with columns [z, Q(z)] for 'file' model
        """
        self.model = model
        self.z_re = z_re
        self.Delta_z = Delta_z
        
        self._Q_interp: Optional[Callable] = None
        self._P_zre_interp: Optional[Callable] = None
        self._z_data: Optional[np.ndarray] = None
        self._Q_data: Optional[np.ndarray] = None
        
        if model == "file":
            if data_file is None:
                raise ValueError("data_file required for 'file' model")
            self._load_from_file(data_file)
        elif model == "tanh":
            self._setup_tanh()
        else:
            raise ValueError(f"Unknown model: {model}")
    
    def _load_from_file(self, path: str):
        """Load Q(z) from file and set up interpolators."""
        data = np.loadtxt(path)
        z_data = data[:, 0]
        Q_data = data[:, 1]
        
        # Ensure z is increasing
        if z_data[0] > z_data[-1]:
            z_data = z_data[::-1]
            Q_data = Q_data[::-1]
        
        self._z_data = z_data
        self._Q_data = Q_data
        
        self._Q_interp = interp1d(
            z_data, Q_data,
            bounds_error=False,
            fill_value=(1.0, 0.0),
            kind='linear'
        )
        
        # Compute P(z_re) = -dQ/dz
        dQ_dz = np.gradient(Q_data, z_data)
        P_zre = -dQ_dz
        
        self._P_zre_interp = interp1d(
            z_data, P_zre,
            bounds_error=False,
            fill_value=0.0,
            kind='linear'
        )
    
    def _setup_tanh(self):
        """Set up interpolators for tanh model over a grid."""
        z_grid = np.linspace(0, 20, 500)
        Q_grid = tanh_reionization_model(z_grid, self.z_re, self.Delta_z)
        
        self._z_data = z_grid
        self._Q_data = Q_grid
        
        self._Q_interp = lambda z: tanh_reionization_model(z, self.z_re, self.Delta_z)
        
        # P(z_re) for tanh: derivative is sech^2
        dQ_dz = -0.5 / self.Delta_z * (1 - np.tanh((z_grid - self.z_re) / self.Delta_z)**2)
        P_zre = -dQ_dz
        
        self._P_zre_interp = interp1d(
            z_grid, P_zre,
            bounds_error=False,
            fill_value=0.0,
            kind='linear'
        )
    
    def Q(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Get ionization fraction at redshift z.
        
        Parameters
        ----------
        z : float or array
            Redshift(s)
            
        Returns
        -------
        Q : float or array
            Ionization fraction
        """
        return self._Q_interp(z)
    
    def P_zre(self, z_re: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Get probability distribution P(z_re) = -dQ/dz.
        
        This gives the probability that a region was reionized at redshift z_re.
        
        Parameters
        ----------
        z_re : float or array
            Reionization redshift(s)
            
        Returns
        -------
        P : float or array
            Probability density
        """
        return self._P_zre_interp(z_re)
    
    def get_z_re_grid(self, z_min: float = 5.0, z_max: float = 18.0, n_points: int = 130) -> np.ndarray:
        """Get a grid of z_re values for integration."""
        return np.linspace(z_min, z_max, n_points)
    
    def update_params(self, z_re: Optional[float] = None, Delta_z: Optional[float] = None):
        """
        Update tanh model parameters.
        
        Only works for tanh model type.
        """
        if self.model != "tanh":
            raise ValueError("Can only update params for tanh model")
        
        if z_re is not None:
            self.z_re = z_re
        if Delta_z is not None:
            self.Delta_z = Delta_z
        
        self._setup_tanh()


def load_reionization_history(
    path: Optional[str] = None,
    model: str = "tanh",
    z_re: float = 7.0,
    Delta_z: float = 1.5,
) -> ReionizationHistory:
    """
    Convenience function to load a reionization history.
    
    Parameters
    ----------
    path : str, optional
        Path to data file (required if model='file')
    model : str
        'tanh' for parametric model, 'file' for tabulated data
    z_re : float
        Midpoint redshift for tanh model
    Delta_z : float
        Duration for tanh model
        
    Returns
    -------
    history : ReionizationHistory
        Configured reionization history object
    """
    return ReionizationHistory(
        model=model,
        z_re=z_re,
        Delta_z=Delta_z,
        data_file=path,
    )


def load_multiple_histories(history_files: dict) -> dict:
    """
    Load multiple reionization histories from files.
    
    Parameters
    ----------
    history_files : dict
        Dictionary mapping names to file paths, e.g.
        {'early_early': 'path/to/early_early.txt', ...}
        
    Returns
    -------
    histories : dict
        Dictionary mapping names to ReionizationHistory objects
    """
    histories = {}
    for name, path in history_files.items():
        histories[name] = load_reionization_history(path=path, model='file')
    return histories
