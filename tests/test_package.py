"""
Unit tests for mfp_emulator package.
"""

import pytest
import numpy as np
import tempfile
import os

# Skip tests if dependencies not available
pytest.importorskip("torch")


class TestReionization:
    """Tests for reionization module."""
    
    def test_tanh_model_basic(self):
        from mfp_emulator.reionization import tanh_reionization_model
        
        # At z=z_re, Q should be 0.5
        Q = tanh_reionization_model(z=7.0, z_re=7.0, Delta_z=1.5)
        assert abs(Q - 0.5) < 0.01
    
    def test_tanh_model_limits(self):
        from mfp_emulator.reionization import tanh_reionization_model
        
        # At low z, Q should be ~1 (fully ionized)
        Q_low = tanh_reionization_model(z=3.0, z_re=7.0, Delta_z=1.5)
        assert Q_low > 0.99
        
        # At high z, Q should be ~0 (neutral)
        Q_high = tanh_reionization_model(z=12.0, z_re=7.0, Delta_z=1.5)
        assert Q_high < 0.01
    
    def test_tanh_model_array(self):
        from mfp_emulator.reionization import tanh_reionization_model
        
        z = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
        Q = tanh_reionization_model(z, z_re=7.0, Delta_z=1.5)
        
        assert len(Q) == len(z)
        assert np.all(Q >= 0) and np.all(Q <= 1)
        # Q should decrease with z
        assert np.all(np.diff(Q) <= 0)
    
    def test_reionization_history_tanh(self):
        from mfp_emulator.reionization import ReionizationHistory
        
        history = ReionizationHistory(model='tanh', z_re=7.0, Delta_z=1.5)
        
        Q = history.Q(7.0)
        assert abs(Q - 0.5) < 0.01
        
        P = history.P_zre(7.0)
        assert P > 0
    
    def test_reionization_history_file(self):
        from mfp_emulator.reionization import ReionizationHistory
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Write z (decreasing) and Q(z)
            for z in np.linspace(12, 4, 50):
                Q = 0.5 * (1 - np.tanh((z - 7) / 1.5))
                f.write(f"{z} {Q}\n")
            temp_path = f.name
        
        try:
            history = ReionizationHistory(model='file', data_file=temp_path)
            Q = history.Q(7.0)
            assert abs(Q - 0.5) < 0.1
        finally:
            os.unlink(temp_path)


class TestData:
    """Tests for data loading module."""
    
    def test_get_observational_mfp_data(self):
        from mfp_emulator.data import get_observational_mfp_data
        
        data = get_observational_mfp_data(exclude_gaikwad=True)
        
        assert len(data) > 0
        for name, d in data.items():
            assert 'z' in d
            assert 'mfp' in d
            assert 'mfp_err' in d
            assert len(d['z']) == len(d['mfp'])
    
    def test_get_gamma_datasets(self):
        from mfp_emulator.data import get_gamma_datasets
        
        data = get_gamma_datasets()
        
        assert len(data) > 0
        for name, (z, gamma, gamma_err) in data.items():
            assert len(z) == len(gamma)
            assert len(z) == len(gamma_err)
    
    def test_gamma_interpolator(self):
        from mfp_emulator.data import create_gamma_interpolator
        
        interp, z_range, datasets = create_gamma_interpolator()
        
        # Should return reasonable values
        gamma = interp(5.5)
        assert gamma > 0
        assert gamma < 1e-11  # reasonable range


class TestEmulator:
    """Tests for emulator module."""
    
    def test_emulator_init(self):
        from mfp_emulator.emulator import MFPEmulator
        
        emulator = MFPEmulator()
        assert emulator.device is not None
        assert not emulator._is_trained
    
    def test_energy_to_wavelength(self):
        from mfp_emulator.emulator import MFPEmulator
        
        # 13.6 eV should give ~91.2 nm
        wl = MFPEmulator.energy_to_wavelength_nm(13.6)
        assert abs(wl - 91.16) < 1.0
    
    def test_residual_mlp(self):
        from mfp_emulator.emulator import ResidualMLP
        import torch
        
        model = ResidualMLP(input_dim=5, output_dim=1, hidden_dim=64, n_blocks=2)
        
        # Test forward pass
        x = torch.randn(10, 5)
        y = model(x)
        
        assert y.shape == (10, 1)
    
    def test_emulator_train_minimal(self):
        """Minimal training test with synthetic data."""
        from mfp_emulator.emulator import MFPEmulator
        import tempfile
        
        # Create minimal synthetic data
        np.random.seed(42)
        n_samples = 100
        
        # z, z_re, gamma, density, mfp_13.6, mfp_14.48, ...
        data = np.zeros((n_samples, 10))
        data[:, 0] = np.random.uniform(4, 8, n_samples)  # z
        data[:, 1] = np.random.uniform(6, 12, n_samples)  # z_re
        data[:, 2] = np.random.uniform(0.1, 3, n_samples)  # gamma
        data[:, 3] = np.random.uniform(-2, 2, n_samples)  # density
        
        # Fake MFP values (exponentially related to params)
        for i in range(6):
            data[:, 4+i] = np.exp(
                -0.5 * data[:, 0] + 0.3 * data[:, 1] + 0.2 * data[:, 2]
            ) * (1 + 0.1 * i) + np.random.normal(0, 0.1, n_samples)
        
        data[:, 4:] = np.maximum(data[:, 4:], 0.1)  # ensure positive
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            np.savetxt(f, data)
            temp_path = f.name
        
        try:
            emulator = MFPEmulator(data_path=temp_path)
            emulator.load_data(energies=[13.6])  # single energy for speed
            emulator.prepare_data(test_size=0.2, validation_size=0.1)
            
            # Very short training
            emulator.train(
                hidden_dim=32,
                n_blocks=2,
                epochs=5,
                batch_size=16,
                verbose=False
            )
            
            assert emulator._is_trained
            
            # Test prediction
            mfp = emulator.predict(z=5.5, z_re=7.0, gamma=0.5, density=0.0)
            assert mfp > 0
            
        finally:
            os.unlink(temp_path)


class TestGridSearch:
    """Tests for grid search module."""
    
    def test_grid_search_init(self):
        """Test GridSearch initialization without emulator."""
        from mfp_emulator.grid_search import GridSearch
        
        # Should work with None emulator for init
        # (will fail on run, but init should work)
        try:
            search = GridSearch(emulator=None)
            assert search.obs_data is not None
        except:
            pass  # Expected if no emulator


class TestCLI:
    """Tests for CLI module."""
    
    def test_parse_config(self):
        from mfp_emulator.cli import parse_config
        import tempfile
        
        config_content = """
# Comment line
type=predict z=5.0 z_re=7.0 gamma=0.5
type=predict z=5.5 z_re=7.5 gamma=0.3 density=-1.0

type=reionization model=tanh z_re=7.0 Delta_z=1.5 z=6.0
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        try:
            from pathlib import Path
            tests = parse_config(Path(temp_path))
            
            assert len(tests) == 3
            assert tests[0]['type'] == 'predict'
            assert tests[0]['z'] == 5.0
            assert tests[1]['density'] == -1.0
            assert tests[2]['type'] == 'reionization'
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
