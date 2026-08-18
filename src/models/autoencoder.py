"""
Autoencoder anomaly detector (PyTorch).

Selected as the 4th model because it learns a compressed, non-linear
representation of "normal" transaction behavior across all engineered
features jointly; transactions the network reconstructs poorly (high
reconstruction error) are, by construction, transactions that don't fit
the learned manifold of normal behavior — a signal the three shallower
models (which each rely on a single global notion of density/isolation/
boundary) can miss when anomalies are only visible in combinations of
features rather than any single one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.models.base import BaseAnomalyModel
from src.utils.helpers import ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when torch is absent
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for AutoencoderModel. Install it via "
            "`pip install torch` (see requirements.txt)."
        )


def _resolve_device(device_cfg: str) -> "torch.device":
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # "auto"


class _TorchAutoencoderNet(nn.Module if _TORCH_AVAILABLE else object):
    """Symmetric feed-forward autoencoder built dynamically from the
    layer-size lists in configs/model_config.yaml."""

    def __init__(
        self,
        n_features: int,
        encoder_layers: List[int],
        decoder_layers: List[int],
        dropout: float,
        batch_norm: bool,
        activation: str = "relu",
    ):
        super().__init__()
        act_fn = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh}.get(
            activation, nn.ReLU
        )

        def block(in_dim: int, out_dim: int) -> List[nn.Module]:
            layers: List[nn.Module] = [nn.Linear(in_dim, out_dim)]
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            return layers

        enc_dims = [n_features, *encoder_layers]
        encoder: List[nn.Module] = []
        for i in range(len(enc_dims) - 1):
            encoder.extend(block(enc_dims[i], enc_dims[i + 1]))
        self.encoder = nn.Sequential(*encoder)

        dec_dims = [enc_dims[-1], *decoder_layers, n_features]
        decoder: List[nn.Module] = []
        for i in range(len(dec_dims) - 2):
            decoder.extend(block(dec_dims[i], dec_dims[i + 1]))
        decoder.append(nn.Linear(dec_dims[-2], dec_dims[-1]))  # linear output, no activation
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoencoderModel(BaseAnomalyModel):
    """PyTorch autoencoder wrapper conforming to `BaseAnomalyModel`.

    Key hyperparameters:
    - `encoder_layers` / `decoder_layers`: bottleneck architecture; the
      narrowest layer forces the network to learn a compressed summary of
      "normal" behavior it must reconstruct from.
    - `learning_rate`, `epochs`, `batch_size`: standard training controls.
    - `early_stopping_patience`: epochs without validation-loss improvement
      before stopping, to avoid overfitting to noise in a small sample.
    - `anomaly_threshold.percentile`: reconstruction-error percentile above
      which a transaction is flagged (used instead of `contamination` here,
      since reconstruction error isn't a `contamination`-parameterized
      estimator the way sklearn's models are).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        _require_torch()
        cfg = dict(config or {})
        cfg.pop("enabled", None)
        arch_cfg = dict(cfg.get("architecture", {}))
        train_cfg = dict(cfg.get("training", {}))
        threshold_cfg = dict(cfg.get("anomaly_threshold", {}))
        random_state = cfg.get("random_state", 42)

        super().__init__(name="autoencoder", contamination=0.02, random_state=random_state)

        self.arch_cfg = arch_cfg
        self.train_cfg = train_cfg
        self.threshold_percentile = threshold_cfg.get("percentile", 98)
        self.device = _resolve_device(train_cfg.get("device", "auto"))
        self.net: Optional[_TorchAutoencoderNet] = None
        self.history_: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    def _build_net(self, n_features: int) -> "_TorchAutoencoderNet":
        return _TorchAutoencoderNet(
            n_features=n_features,
            encoder_layers=self.arch_cfg.get("encoder_layers", [64, 32, 16, 8]),
            decoder_layers=self.arch_cfg.get("decoder_layers", [16, 32, 64]),
            dropout=self.arch_cfg.get("dropout", 0.1),
            batch_norm=self.arch_cfg.get("batch_norm", True),
            activation=self.arch_cfg.get("activation", "relu"),
        ).to(self.device)

    def _fit_model(self, X: np.ndarray) -> None:
        torch.manual_seed(self.random_state)
        n_features = X.shape[1]
        self.net = self._build_net(n_features)

        val_split = self.train_cfg.get("validation_split", 0.1)
        n_val = max(1, int(len(X) * val_split)) if len(X) > 20 else 0
        rng = np.random.default_rng(self.random_state)
        perm = rng.permutation(len(X))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        X_train, X_val = X[train_idx], X[val_idx] if n_val else X[train_idx][:0]

        batch_size = min(self.train_cfg.get("batch_size", 512), max(1, len(X_train)))
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
            batch_size=batch_size, shuffle=True, drop_last=False,
        )

        optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.train_cfg.get("learning_rate", 0.001),
            weight_decay=self.train_cfg.get("weight_decay", 1e-5),
        )
        criterion = nn.MSELoss()

        epochs = self.train_cfg.get("epochs", 60)
        patience = self.train_cfg.get("early_stopping_patience", 8)
        min_delta = self.train_cfg.get("early_stopping_min_delta", 1e-4)

        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(epochs):
            self.net.train()
            epoch_loss, n_batches = 0.0, 0
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                recon = self.net(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            train_loss = epoch_loss / max(1, n_batches)
            self.history_["train_loss"].append(train_loss)

            if n_val:
                self.net.eval()
                with torch.no_grad():
                    val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
                    val_loss = criterion(self.net(val_tensor), val_tensor).item()
                self.history_["val_loss"].append(val_loss)

                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    logger.info(
                        "Autoencoder early stopping at epoch %d (best val_loss=%.6f).",
                        epoch + 1, best_val_loss,
                    )
                    break
            else:
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    "Autoencoder epoch %d/%d — train_loss=%.6f%s",
                    epoch + 1, epochs, train_loss,
                    f", val_loss={self.history_['val_loss'][-1]:.6f}" if n_val else "",
                )

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        self.net.eval()
        with torch.no_grad():
            tensor = torch.tensor(np.asarray(X, dtype=np.float32)).to(self.device)
            recon = self.net(tensor)
            # Per-row mean squared reconstruction error — already
            # "higher = more anomalous" by construction, no sign flip needed.
            errors = torch.mean((recon - tensor) ** 2, dim=1)
        return errors.cpu().numpy()

    def reconstruction_error_per_feature(self, X: np.ndarray) -> np.ndarray:
        """Per-sample, per-feature squared error — used by
        `src/explainability/explain.py` to show *which* features drove a
        given transaction's anomaly score."""
        self._check_fitted()
        self.net.eval()
        with torch.no_grad():
            tensor = torch.tensor(np.asarray(X, dtype=np.float32)).to(self.device)
            recon = self.net(tensor)
            errors = (recon - tensor) ** 2
        return errors.cpu().numpy()

    def fit(self, X: np.ndarray, threshold_percentile: Optional[float] = None) -> "AutoencoderModel":
        # Autoencoder uses a reconstruction-error percentile (not the
        # generic `contamination`-derived percentile) as its threshold,
        # per configs/model_config.yaml `autoencoder.anomaly_threshold`.
        pct = threshold_percentile if threshold_percentile is not None else self.threshold_percentile
        return super().fit(X, threshold_percentile=pct)  # type: ignore[return-value]

    def save(self, path: Path) -> None:
        path = Path(path)
        ensure_dir(path.parent)
        torch.save(
            {
                "state_dict": self.net.state_dict() if self.net is not None else None,
                "arch_cfg": self.arch_cfg,
                "train_cfg": self.train_cfg,
                "threshold_percentile": self.threshold_percentile,
                "threshold_": self.threshold_,
                "n_features_": self.n_features_,
                "train_time_seconds_": self.train_time_seconds_,
                "random_state": self.random_state,
                "history_": self.history_,
            },
            path,
        )
        logger.info("Saved Autoencoder checkpoint to %s", path)

    @classmethod
    def load(cls, path: Path) -> "AutoencoderModel":
        _require_torch()
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Autoencoder checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        model = cls({
            "architecture": checkpoint["arch_cfg"],
            "training": checkpoint["train_cfg"],
            "anomaly_threshold": {"percentile": checkpoint["threshold_percentile"]},
            "random_state": checkpoint["random_state"],
        })
        model.n_features_ = checkpoint["n_features_"]
        model.net = model._build_net(model.n_features_)
        if checkpoint["state_dict"] is not None:
            model.net.load_state_dict(checkpoint["state_dict"])
        model.net.eval()
        model.threshold_ = checkpoint["threshold_"]
        model.train_time_seconds_ = checkpoint["train_time_seconds_"]
        model.history_ = checkpoint.get("history_", {"train_loss": [], "val_loss": []})
        model.is_fitted = True
        return model