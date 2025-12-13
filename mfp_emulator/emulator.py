"""
Neural network emulator for mean free path prediction.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from typing import Sequence, Optional, Union
from pathlib import Path


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.act = nn.Mish()
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor):
        residual = x
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.norm2(out)
        out = out + residual
        out = self.act(out)
        return out


class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 5,
        output_dim: int = 1,
        hidden_dim: Union[int, Sequence[int]] = 128,
        n_blocks: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim] * n_blocks
        else:
            assert len(hidden_dim) == n_blocks
            hidden_dims = list(hidden_dim)

        self.fc_in = nn.Linear(input_dim, hidden_dims[0])
        self.norm_in = nn.LayerNorm(hidden_dims[0])
        self.act = nn.Mish()
        self.dropout = nn.Dropout(dropout)
        blocks = [ResidualBlock(dim, dropout=dropout) for dim in hidden_dims]
        self.blocks = nn.Sequential(*blocks)
        self.fc_out = nn.Linear(hidden_dims[-1], output_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc_in.weight, gain=1.0)
        nn.init.constant_(self.fc_in.bias, 0.0)
        nn.init.xavier_uniform_(self.fc_out.weight, gain=1.0)
        nn.init.constant_(self.fc_out.bias, 0.0)

    def forward(self, x: torch.Tensor):
        x = self.fc_in(x)
        x = self.norm_in(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.fc_out(x)
        return x


class MFPEmulator:
    """
    Neural network emulator for predicting mean free path (MFP) as a function of
    redshift, reionization redshift, photoionization rate (gamma), density, and energy.
    
    The emulator can be trained on simulation data and then used for fast predictions.
    Supports multiple energy bins through wavelength as an input feature.
    """

    # Default energy bins (eV) and their column indices in standard data format
    DEFAULT_ENERGIES = {
        13.6: 4, 14.48: 5, 16.7: 6,
        20.05: 7, 25.5: 8, 39.5: 9
    }

    def __init__(
        self,
        data_path: Optional[str] = None,
        energies: Optional[dict] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the emulator.
        
        Parameters
        ----------
        data_path : str, optional
            Path to training data file. If None, must call load_data() before training.
        energies : dict, optional
            Dictionary mapping energy (eV) to column index in data file.
            Defaults to standard 6-energy setup.
        device : str, optional
            Device to use ('cuda' or 'cpu'). Auto-detects if None.
        """
        self.data_path = data_path
        self.energy_columns = energies if energies is not None else self.DEFAULT_ENERGIES
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model: Optional[nn.Module] = None
        self.input_scaler = RobustScaler()
        self.output_scaler = RobustScaler()
        
        self.X = None
        self.y_mfp = None
        self._is_trained = False

    @staticmethod
    def energy_to_wavelength_nm(energy_eV: float) -> float:
        """Convert photon energy (eV) to wavelength (nm)."""
        return 1239.8 / energy_eV

    def load_data(
        self,
        data_path: Optional[str] = None,
        energies: Optional[list] = None,
    ) -> tuple:
        """
        Load and preprocess training data.
        
        Parameters
        ----------
        data_path : str, optional
            Path to data file. Uses self.data_path if None.
        energies : list, optional
            List of energies (eV) to include. Uses all available if None.
            
        Returns
        -------
        X : ndarray
            Input features [z, z_re, gamma, density, wavelength/100]
        y : ndarray
            Target MFP values
        """
        path = data_path or self.data_path
        if path is None:
            raise ValueError("No data path provided")
        
        data = np.loadtxt(path)
        X_base = data[:, 0:4]  # z, z_re, gamma, density
        
        if energies is None:
            energies_to_use = list(self.energy_columns.keys())
        else:
            energies_to_use = energies
        
        X_expanded = []
        y_expanded = []
        
        for energy in energies_to_use:
            if energy not in self.energy_columns:
                raise ValueError(f"Energy {energy} eV not in available columns")
            col_idx = self.energy_columns[energy]
            wavelength_nm = self.energy_to_wavelength_nm(energy)
            wavelength_feature = np.full((X_base.shape[0], 1), wavelength_nm / 100.0)
            X_energy = np.hstack([X_base, wavelength_feature])
            y_energy = data[:, col_idx:col_idx+1]
            X_expanded.append(X_energy)
            y_expanded.append(y_energy)
        
        self.X = np.vstack(X_expanded)
        self.y_mfp = np.vstack(y_expanded)
        
        return self.X, self.y_mfp

    def compute_sample_weights(
        self,
        X: np.ndarray,
        weight_high_z: bool = True,
        weight_high_zre: bool = True,
        z_weight_power: float = 2.0,
        zre_weight_power: float = 1.5,
    ) -> np.ndarray:
        """
        Compute sample weights emphasizing high-z and near-reionization data.
        """
        z = X[:, 0]
        zre = X[:, 1]
        weights = np.ones(len(X))

        if weight_high_z:
            z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
            z_weights = 1.0 + z_norm ** z_weight_power * 3.0
            weights *= z_weights

        if weight_high_zre:
            zre_norm = (zre - zre.min()) / (zre.max() - zre.min() + 1e-8)
            zre_weights = 1.0 + zre_norm ** zre_weight_power * 2.0
            weights *= zre_weights

        z_ratio = z / zre
        boundary_weights = 1.0 + np.exp(5.0 * (z_ratio - 0.7))
        boundary_weights = np.clip(boundary_weights, 1.0, 10.0)
        weights *= boundary_weights

        weights = weights / weights.mean()
        return weights

    def prepare_data(
        self,
        test_size: float = 0.1,
        validation_size: float = 0.1,
        use_weighted_sampling: bool = True,
    ):
        """
        Split data and prepare for training.
        """
        if self.X is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.X, self.y_mfp, test_size=test_size, random_state=42, shuffle=True
        )
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, random_state=42
        )

        if use_weighted_sampling:
            self.train_weights = self.compute_sample_weights(X_train)
        else:
            self.train_weights = np.ones(len(X_train))

        eps = 1e-6
        y_train_log = np.log10(y_train + eps)
        y_val_log = np.log10(y_val + eps)
        y_test_log = np.log10(y_test + eps)

        self.X_train_scaled = self.input_scaler.fit_transform(X_train)
        self.X_val_scaled = self.input_scaler.transform(X_val)
        self.X_test_scaled = self.input_scaler.transform(X_test)
        self.y_train_scaled = self.output_scaler.fit_transform(y_train_log)
        self.y_val_scaled = self.output_scaler.transform(y_val_log)
        self.y_test_scaled = self.output_scaler.transform(y_test_log)

        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

    def train(
        self,
        hidden_dim: Union[int, Sequence[int]] = 512,
        n_blocks: int = 6,
        learning_rate: float = 0.0025,
        epochs: int = 100,
        patience: int = 25,
        batch_size: int = 64,
        dropout: float = 0.1,
        huber_delta: float = 1.0,
        use_weighted_loss: bool = True,
        verbose: bool = True,
    ):
        """
        Train the neural network.
        
        Parameters
        ----------
        hidden_dim : int or sequence
            Hidden layer dimension(s)
        n_blocks : int
            Number of residual blocks
        learning_rate : float
            Initial learning rate
        epochs : int
            Maximum training epochs
        patience : int
            Early stopping patience
        batch_size : int
            Training batch size
        dropout : float
            Dropout rate
        huber_delta : float
            Huber loss delta parameter
        use_weighted_loss : bool
            Whether to use sample weights in loss
        verbose : bool
            Print training progress
        """
        train_dataset = TensorDataset(
            torch.FloatTensor(self.X_train_scaled),
            torch.FloatTensor(self.y_train_scaled),
            torch.FloatTensor(self.train_weights.reshape(-1, 1)),
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(self.X_val_scaled),
            torch.FloatTensor(self.y_val_scaled),
        )

        sampler = WeightedRandomSampler(
            weights=self.train_weights,
            num_samples=len(self.train_weights),
            replacement=True,
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

        input_dim = self.X_train_scaled.shape[1]
        self.model = ResidualMLP(
            input_dim=input_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            dropout=dropout,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch, w_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                w_batch = w_batch.to(self.device)

                optimizer.zero_grad()
                pred = self.model(X_batch)
                
                if use_weighted_loss:
                    loss_raw = nn.functional.huber_loss(pred, y_batch, delta=huber_delta, reduction='none')
                    loss = (loss_raw * w_batch).mean()
                else:
                    loss = nn.functional.huber_loss(pred, y_batch, delta=huber_delta)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    pred = self.model(X_batch)
                    loss = nn.functional.huber_loss(pred, y_batch, delta=huber_delta)
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        self._is_trained = True

    def predict(
        self,
        z: float,
        z_re: float,
        gamma: float,
        density: float = 0.0,
        energy_eV: float = 13.6,
    ) -> float:
        """
        Predict MFP for given parameters.
        
        Parameters
        ----------
        z : float
            Redshift
        z_re : float
            Reionization redshift
        gamma : float
            Photoionization rate (×10^12 s^-1, i.e. gamma_12)
        density : float
            Density in units of sigma (0 = mean density)
        energy_eV : float
            Photon energy in eV
            
        Returns
        -------
        mfp : float
            Predicted mean free path in cMpc
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        wavelength_nm = self.energy_to_wavelength_nm(energy_eV)
        X_input = np.array([[z, z_re, gamma, density, wavelength_nm / 100.0]])
        X_scaled = self.input_scaler.transform(X_input)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            y_scaled = self.model(X_tensor).cpu().numpy()

        y_log = self.output_scaler.inverse_transform(y_scaled)
        mfp = 10**y_log - 1e-6
        return float(mfp[0, 0])

    def predict_batch(
        self,
        z: np.ndarray,
        z_re: np.ndarray,
        gamma: np.ndarray,
        density: np.ndarray,
        energy_eV: Union[float, np.ndarray] = 13.6,
    ) -> np.ndarray:
        """
        Batch prediction for arrays of parameters.
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        n = len(z)
        if isinstance(energy_eV, (int, float)):
            energy_eV = np.full(n, energy_eV)
        
        wavelengths = np.array([self.energy_to_wavelength_nm(e) / 100.0 for e in energy_eV])
        X_input = np.column_stack([z, z_re, gamma, density, wavelengths])
        X_scaled = self.input_scaler.transform(X_input)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            y_scaled = self.model(X_tensor).cpu().numpy()

        y_log = self.output_scaler.inverse_transform(y_scaled)
        mfp = 10**y_log - 1e-6
        return mfp.flatten()

    def save(self, path: str):
        """
        Save trained model and scalers.
        
        Saves model weights and scaler parameters separately to avoid
        pickle issues with sklearn objects in newer PyTorch versions.
        """
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained model")
        
        # Extract scaler parameters instead of pickling the objects
        save_dict = {
            'model_state': self.model.state_dict(),
            'input_scaler_params': {
                'center_': self.input_scaler.center_,
                'scale_': self.input_scaler.scale_,
            },
            'output_scaler_params': {
                'center_': self.output_scaler.center_,
                'scale_': self.output_scaler.scale_,
            },
            'energy_columns': self.energy_columns,
            'model_config': {
                'input_dim': self.model.fc_in.in_features,
                'hidden_dim': self.model.fc_in.out_features,
                'n_blocks': len(self.model.blocks),
            }
        }
        torch.save(save_dict, path)

    def load(self, path: str):
        """
        Load trained model and scalers.
        
        Compatible with PyTorch 2.6+ (handles weights_only default).
        """
        # Try weights_only=True first (PyTorch 2.6+), fall back if needed
        try:
            save_dict = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            # Older PyTorch without weights_only argument
            save_dict = torch.load(path, map_location=self.device)
        except Exception:
            # If weights_only fails (old format with pickled scalers), use weights_only=False
            save_dict = torch.load(path, map_location=self.device, weights_only=False)
        
        # Handle both old format (pickled scalers) and new format (scaler params)
        if 'input_scaler' in save_dict:
            # Old format - scalers were pickled directly
            self.input_scaler = save_dict['input_scaler']
            self.output_scaler = save_dict['output_scaler']
        else:
            # New format - reconstruct scalers from params
            self.input_scaler = RobustScaler()
            self.input_scaler.center_ = save_dict['input_scaler_params']['center_']
            self.input_scaler.scale_ = save_dict['input_scaler_params']['scale_']
            
            self.output_scaler = RobustScaler()
            self.output_scaler.center_ = save_dict['output_scaler_params']['center_']
            self.output_scaler.scale_ = save_dict['output_scaler_params']['scale_']
        
        self.energy_columns = save_dict['energy_columns']
        
        config = save_dict['model_config']
        self.model = ResidualMLP(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            n_blocks=config['n_blocks'],
        ).to(self.device)
        self.model.load_state_dict(save_dict['model_state'])
        self._is_trained = True

    def evaluate(self, verbose: bool = True) -> dict:
        """
        Evaluate model on test set.
        
        Returns
        -------
        metrics : dict
            Dictionary with MAE, RMSE, and R^2 scores
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(self.X_test_scaled).to(self.device)
            y_scaled_pred = self.model(X_tensor).cpu().numpy()
        
        y_log_pred = self.output_scaler.inverse_transform(y_scaled_pred)
        y_pred = 10**y_log_pred - 1e-6
        y_true = self.y_test
        
        mae = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(np.mean((y_pred - y_true)**2))
        
        log_mae = np.mean(np.abs(np.log10(y_pred + 1e-6) - np.log10(y_true + 1e-6)))
        
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1 - ss_res / ss_tot
        
        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'log_mae': float(log_mae),
            'r2': float(r2),
        }
        
        if verbose:
            print(f"Test metrics: MAE={mae:.3f}, RMSE={rmse:.3f}, log-MAE={log_mae:.4f}, R²={r2:.4f}")
        
        return metrics