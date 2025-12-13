# MFP Emulator

A neural network emulator for predicting the mean free path (MFP) of ionizing photons during cosmic reionization. This package provides tools for training emulators on simulation data and using them to constrain reionization parameters from observations.

## Features

- **Neural Network Emulator**: Residual MLP architecture for fast, accurate MFP predictions as a function of redshift, reionization history, photoionization rate, density, and photon energy
- **Flexible Energy Support**: Train on single or multiple energy bins (default: 13.6-39.5 eV range)
- **Reionization Models**: Built-in tanh model and support for arbitrary tabulated Q(z) histories
- **Parameter Fitting**: Grid search routines for constraining reionization parameters (z_re, Δz, γ) from MFP observations
- **N-dot Calculations**: Compute ionizing photon production rates using the BH07 method
- **Observational Data**: Built-in catalogs from Gaikwad+23, Becker+21, Zhu+23, Worseck+14, and others

## Installation

```bash
# Basic installation
pip install mfp_emulator

# With cosmology support (requires colossus)
pip install mfp_emulator[cosmology]

# Full installation with plotting and dev tools
pip install mfp_emulator[all]

# From source
git clone https://github.com/yourusername/mfp_emulator.git
cd mfp_emulator
pip install -e ".[all]"
```

## Quick Start

### Training an Emulator

```python
from mfp_emulator import MFPEmulator

# Initialize and load data
emulator = MFPEmulator(data_path='path/to/mfp_data.txt')
emulator.load_data()

# Prepare train/val/test splits
emulator.prepare_data(test_size=0.1, validation_size=0.1)

# Train
emulator.train(
    hidden_dim=512,
    n_blocks=6,
    epochs=100,
    patience=25,
    verbose=True
)

# Evaluate
metrics = emulator.evaluate()

# Save
emulator.save('trained_emulator.pt')
```

### Making Predictions

```python
from mfp_emulator import MFPEmulator

# Load trained emulator
emulator = MFPEmulator()
emulator.load('trained_emulator.pt')

# Single prediction
mfp = emulator.predict(
    z=5.5,          # observation redshift
    z_re=7.0,       # reionization redshift
    gamma=0.5,      # gamma_12 (photoionization rate × 10^12)
    density=0.0,    # density in units of sigma (0 = mean)
    energy_eV=13.6  # photon energy
)
print(f"MFP = {mfp:.2f} cMpc")

# Batch predictions
import numpy as np
z = np.array([5.0, 5.5, 6.0])
z_re = np.array([7.0, 7.0, 7.0])
gamma = np.array([0.5, 0.4, 0.3])
density = np.zeros(3)

mfps = emulator.predict_batch(z, z_re, gamma, density, energy_eV=13.6)
```

### Reionization History

```python
from mfp_emulator import tanh_reionization_model, load_reionization_history

# Built-in tanh model
Q = tanh_reionization_model(z=5.5, z_re=7.0, Delta_z=1.5)
print(f"Ionization fraction at z=5.5: Q = {Q:.3f}")

# From file (columns: z, Q)
history = load_reionization_history(
    path='late_start_late_end.txt',
    model='file'
)
Q = history.Q(5.5)
P_zre = history.P_zre(7.0)  # probability of reionization at z_re=7

# Tanh model object for parameter updates
history = load_reionization_history(model='tanh', z_re=7.0, Delta_z=1.5)
history.update_params(z_re=8.0)  # change parameters on the fly
```

### Grid Search for Best-Fit Parameters

```python
from mfp_emulator import GridSearch, TanhGridSearch

# Simple gamma-zre grid search
search = GridSearch(emulator)
results = search.run(
    gamma_range=(0.03, 3.0),
    zre_range=(5.0, 8.0),
    n_gamma=30,
    n_zre=30
)
print(f"Best fit: gamma={results['best_gamma']:.2f}, z_re={results['best_zre']:.2f}")

# Tanh model grid search
tanh_search = TanhGridSearch(emulator)
results = tanh_search.run(
    z_re_range=(5.5, 10.0),
    Delta_z_range=(0.5, 6.0),
    n_z_re=40,
    n_Delta_z=40,
    gamma_mode='per_redshift'  # use gamma(z) at each observation
)
print(f"Best fit: z_re={results['z_re_best']:.2f}, Δz={results['Delta_z_best']:.2f}")
```

### N-dot (Ionizing Photon Production Rate)

```python
from mfp_emulator import NdotCalculator, load_reionization_history

history = load_reionization_history(path='late_start_late_end.txt', model='file')
calc = NdotCalculator(emulator, history, alpha_s=1.5)

# Full analysis
results = calc.run_analysis(
    z_values=np.linspace(4.8, 6.0, 7),
    beta_values=[1.0, 1.3, 1.5, 2.0]
)

# Single calculation
ndot = calc.compute_ndot_emulator(z=5.5, gamma=0.5e-12, energy_eV=13.6)
```

### Working with Observational Data

```python
from mfp_emulator import get_observational_mfp_data, get_gamma_datasets

# MFP observations
mfp_data = get_observational_mfp_data(
    exclude_gaikwad=True,      # exclude model-dependent data
    z_range=(4.8, 6.0),        # redshift range
    priority_filter=True       # keep most recent at each z
)

for study, data in mfp_data.items():
    print(f"{study}: {len(data['z'])} points at z={data['z']}")

# Gamma (photoionization rate) measurements
gamma_data = get_gamma_datasets(priority_filter=True)
```

## Data Format

The training data file should have the following columns:
```
z  z_re  gamma  density  MFP_E1  MFP_E2  MFP_E3  ...
```

Where:
- `z`: observation redshift
- `z_re`: reionization redshift of the region
- `gamma`: photoionization rate Γ₁₂ (in units of 10⁻¹² s⁻¹)
- `density`: density contrast in units of σ (0 = mean density)
- `MFP_E*`: mean free path at each energy bin in cMpc

Default energies: 13.6, 14.48, 16.7, 20.05, 25.5, 39.5 eV

## Command Line Interface

```bash
# Train a new emulator
mfp-emulator train data.txt -o emulator.pt --epochs 100

# Make a prediction
mfp-emulator predict emulator.pt --z 5.5 --z-re 7.0 --gamma 0.5

# Run tests from config file
mfp-emulator test tests.txt emulator.pt
```

Test config file format:
```
# Prediction tests
type=predict z=5.0 z_re=7.0 gamma=0.5 expected_mfp=25.0
type=predict z=5.5 z_re=7.0 gamma=0.3 density=-1.0

# Reionization tests
type=reionization model=tanh z_re=7.0 Delta_z=1.5 z=5.0 expected_Q=0.95

# N-dot tests
type=ndot z=5.5 z_re=7.0 Delta_z=1.5 gamma=5e-13
```

## Dependencies

**Required:**
- numpy >= 1.20
- scipy >= 1.7
- torch >= 1.9
- scikit-learn >= 0.24

**Optional:**
- colossus >= 1.3 (for cosmology calculations)
- matplotlib >= 3.4 (for plotting)

## Citation

If you use this package, please cite:
```
Tohfa et al. 2025
```


## Contributing

Contributions welcome! Please open an issue or submit a pull request.
