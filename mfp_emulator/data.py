"""
Data loading and preprocessing utilities.
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
from pathlib import Path



def get_observational_mfp_data(
    exclude_gaikwad: bool = True,
    z_range: Tuple[float, float] = (4.8, 6.0),
    priority_filter: bool = True,
) -> Dict[str, Dict]:
    """
    Get observational MFP measurements from various studies.
    
    Data is converted to comoving Mpc with h=0.68.
    
    Parameters
    ----------
    exclude_gaikwad : bool
        Whether to exclude Gaikwad+23 (model-dependent measurements)
    z_range : tuple
        (z_min, z_max) to filter data
    priority_filter : bool
        If True, keep only the most recent measurement at each redshift
        
    Returns
    -------
    data : dict
        Dictionary with study names as keys, containing 'z', 'mfp', 'mfp_err'
    """
    hh = 0.68
    
    datasets = {}
    
    # Gaikwad+23
    datasets['Gaikwad+23'] = {
        'z': np.array([4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0]),
        'mfp': np.array([50.12, 57.41, 46.45, 40.83, 34.04, 29.24, 28.90, 22.96, 16.59, 13.18, 10.47, 8.32]),
        'mfp_err': np.array([
            [19.91, 33.05], [21.104, 38.088], [18.909, 31.173], [15.713, 22.264],
            [12.163, 18.440], [10.187, 15.427], [11.124, 10.904], [7.826, 11.712],
            [7.476, 9.707], [5.769, 9.205], [4.976, 9.027], [4.052, 7.531]
        ]),
        'year': 2023,
    }
    
    # Worseck+14
    z_wor = np.array([4.56, 4.86, 5.16])
    mfp_wor = np.array([22.2, 15.1, 10.3]) * (1. + z_wor) * 0.7
    mfp_wor_err = np.array([[2.3, 2.3], [1.8, 1.8], [1.6, 1.6]]) * (1. + z_wor)[:, np.newaxis] * 0.7
    datasets['Worseck+14'] = {
        'z': z_wor,
        'mfp': mfp_wor,
        'mfp_err': mfp_wor_err,
        'year': 2014,
    }
    
    # Becker+21
    z_beck = np.array([5.1, 6.0])
    mfp_beck = np.array([9.09, 0.75]) * hh * (1 + z_beck)
    mfp_beck_err = np.array([[1.22, 1.62], [0.45, 0.65]]) * hh * (1 + z_beck)[:, np.newaxis]
    datasets['Becker+21'] = {
        'z': z_beck,
        'mfp': mfp_beck,
        'mfp_err': mfp_beck_err,
        'year': 2021,
    }
    
    # Zhu+23
    z_zhu = np.array([5.079545, 5.312231, 5.652356, 5.934156])
    mfp_zhu = np.array([9.333584, 5.396826, 3.310834, 0.807835]) * hh * (1 + z_zhu)
    mfp_zhu_err_lower = np.array([9.333584-7.531137, 5.396826-3.995292,
                                  3.310834-1.968358, 0.807835-0.324481]) * hh * (1 + z_zhu)
    mfp_zhu_err_upper = np.array([11.392222-9.333584, 6.868716-5.396826,
                                  6.052837-3.310834, 1.535472-0.807835]) * hh * (1 + z_zhu)
    datasets['Zhu+23'] = {
        'z': z_zhu,
        'mfp': mfp_zhu,
        'mfp_err': np.column_stack([mfp_zhu_err_lower, mfp_zhu_err_upper]),
        'year': 2023,
    }
    
    # Filter by exclusions and z range
    filtered = {}
    for name, data in datasets.items():
        if exclude_gaikwad and name == 'Gaikwad+23':
            continue
        
        mask = (data['z'] >= z_range[0]) & (data['z'] <= z_range[1])
        if np.any(mask):
            filtered[name] = {
                'z': data['z'][mask],
                'mfp': data['mfp'][mask],
                'mfp_err': data['mfp_err'][mask],
                'year': data['year'],
            }
    
    if not priority_filter:
        return filtered
    
    # Priority filter: keep most recent at each redshift
    all_z = []
    all_mfp = []
    all_err = []
    all_names = []
    
    for name, data in filtered.items():
        for i in range(len(data['z'])):
            all_z.append(data['z'][i])
            all_mfp.append(data['mfp'][i])
            all_err.append(data['mfp_err'][i])
            all_names.append((name, data['year']))
    
    # Group by rounded redshift
    z_to_best = {}
    for z, mfp, err, (name, year) in zip(all_z, all_mfp, all_err, all_names):
        z_key = round(z, 2)
        if z_key not in z_to_best or year > z_to_best[z_key][3]:
            z_to_best[z_key] = (z, mfp, err, year, name)
    
    # Reorganize by dataset
    result = {}
    for z_key, (z, mfp, err, year, name) in z_to_best.items():
        if name not in result:
            result[name] = {'z': [], 'mfp': [], 'mfp_err': [], 'year': year}
        result[name]['z'].append(z)
        result[name]['mfp'].append(mfp)
        result[name]['mfp_err'].append(err)
    
    # Convert to arrays
    for name in result:
        result[name]['z'] = np.array(result[name]['z'])
        result[name]['mfp'] = np.array(result[name]['mfp'])
        result[name]['mfp_err'] = np.array(result[name]['mfp_err'])
    
    return result


def get_gamma_datasets(
    exclude_gaikwad: bool = False,
    priority_filter: bool = True,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Get photoionization rate (gamma) measurements from various studies.
    
    Parameters
    ----------
    exclude_gaikwad : bool
        Whether to exclude Gaikwad+23 data
    priority_filter : bool
        Keep only most recent measurement at each redshift
        
    Returns
    -------
    datasets : dict
        Dictionary mapping study name to (z, gamma, gamma_err) tuples.
        gamma values are in s^-1 (not scaled).
    """
    # Gaikwad+23
    z_gaikwad = np.array([4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0])
    gamma_gaikwad = 1e-12 * np.array([0.501, 0.557, 0.508, 0.502, 0.404, 0.372,
                                       0.344, 0.319, 0.224, 0.178, 0.151, 0.145])
    gamma_gaikwad_err = 1e-12 * np.array([
        np.array([0.232, 0.275]), np.array([0.218, 0.376]), np.array([0.192, 0.324]),
        np.array([0.193, 0.292]), np.array([0.147, 0.272]), np.array([0.126, 0.217]),
        np.array([0.130, 0.219]), np.array([0.120, 0.194]), np.array([0.112, 0.223]),
        np.array([0.078, 0.194]), np.array([0.079, 0.151]), np.array([0.087, 0.157])
    ])

    # Calverley+11
    z_cav = np.array([5, 6])
    gamma_cav = np.array([10**-12.15, 10**-12.84])
    gamma_cav_err = np.array([
        10**-12.15 * np.array([1. - 10**-0.16, 10**0.16 - 1.]),
        10**-12.84 * np.array([1. - 10**-0.18, 10**0.18 - 1.])
    ])

    # Wyithe+11
    z_wyithe = np.array([5, 6])
    gamma_wyithe = np.array([0.47, 0.18]) * 1e-12
    gamma_wyithe_err = np.array([np.array([0.2, 0.3]), np.array([0.09, 0.18])]) * 1e-12

    # D'Aloisio+18
    z_da = np.array([4.8, 5.0, 5.2, 5.4, 5.6, 5.8])
    gamma_da = np.array([0.58, 0.53, 0.48, 0.47, 0.45, 0.29]) * 1e-12
    gamma_da_err = np.array([
        np.array([0.20, 0.08]), np.array([0.19, 0.09]), np.array([0.18, 0.10]),
        np.array([0.18, 0.12]), np.array([0.17, 0.14]), np.array([0.11, 0.11])
    ]) * 1e-12

    # Becker+21
    z_beck = np.array([5.1, 6])
    gamma_beck = np.array([7e-13, 3e-13])
    gamma_beck_err = np.array([
        np.array([7e-13 * (1. - 10**-0.15), 7e-13 * (10**0.15 - 1)]),
        np.array([3e-13 * (1. - 10**-0.15), 3e-13 * (10**0.15 - 1)])
    ])

    dataset_priority = {
        'Gaikwad+23': 2023,
        'Becker+21': 2021,
        'DAloisio+18': 2018,
        'Calverley+11': 2011,
        'Wyithe+11': 2011,
    }

    datasets = {
        'Gaikwad+23': (z_gaikwad, gamma_gaikwad, gamma_gaikwad_err),
        'Calverley+11': (z_cav, gamma_cav, gamma_cav_err),
        'Wyithe+11': (z_wyithe, gamma_wyithe, gamma_wyithe_err),
        'DAloisio+18': (z_da, gamma_da, gamma_da_err),
        'Becker+21': (z_beck, gamma_beck, gamma_beck_err),
    }
    
    if exclude_gaikwad:
        del datasets['Gaikwad+23']
        del dataset_priority['Gaikwad+23']
    
    if not priority_filter:
        return datasets
    
    # Priority filtering
    redshift_map = {}
    for dataset_name, (z_array, gamma_array, gamma_err_array) in datasets.items():
        for z_val, gamma_val, gamma_err_val in zip(z_array, gamma_array, gamma_err_array):
            if z_val not in redshift_map:
                redshift_map[z_val] = []
            redshift_map[z_val].append({
                'dataset': dataset_name,
                'gamma': gamma_val,
                'gamma_err': gamma_err_val,
                'year': dataset_priority[dataset_name]
            })

    filtered_datasets = {}
    for dataset_name, (z_array, gamma_array, gamma_err_array) in datasets.items():
        filtered_z = []
        filtered_gamma = []
        filtered_gamma_err = []

        for z_val, gamma_val, gamma_err_val in zip(z_array, gamma_array, gamma_err_array):
            measurements = redshift_map[z_val]
            highest_priority = max(measurements, key=lambda x: x['year'])
            
            if highest_priority['dataset'] == dataset_name:
                filtered_z.append(z_val)
                filtered_gamma.append(gamma_val)
                filtered_gamma_err.append(gamma_err_val)

        if len(filtered_z) > 0:
            filtered_datasets[dataset_name] = (
                np.array(filtered_z),
                np.array(filtered_gamma),
                np.array(filtered_gamma_err)
            )

    return filtered_datasets


def create_gamma_interpolator(
    datasets: Optional[Dict] = None,
) -> Tuple:
    """
    Create an interpolator for gamma(z) from observational data.
    
    Parameters
    ----------
    datasets : dict, optional
        Output from get_gamma_datasets(). If None, loads default.
        
    Returns
    -------
    interp : callable
        Interpolator function gamma(z)
    z_range : tuple
        (z_min, z_max) of available data
    datasets : dict
        The datasets used
    """
    from scipy.interpolate import interp1d
    
    if datasets is None:
        datasets = get_gamma_datasets(priority_filter=True)
    
    all_z = []
    all_gamma = []
    
    for name, (z, gamma, _) in datasets.items():
        all_z.extend(z)
        all_gamma.extend(gamma)
    
    # Sort by redshift
    sort_idx = np.argsort(all_z)
    all_z = np.array(all_z)[sort_idx]
    all_gamma = np.array(all_gamma)[sort_idx]
    
    # Remove duplicates by averaging
    unique_z = np.unique(all_z)
    unique_gamma = np.array([np.mean(all_gamma[all_z == z]) for z in unique_z])
    
    interp = interp1d(
        unique_z, unique_gamma,
        kind='linear',
        bounds_error=False,
        fill_value=(unique_gamma[0], unique_gamma[-1])
    )
    
    return interp, (unique_z.min(), unique_z.max()), datasets
