"""PyTorch Deep Learning architectures: Entity Embedding MLP and Temporal Sequence Net."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)


class FlightEmbeddingNet(nn.Module):
    """Deep Tabular Neural Network with learned entity embeddings for categorical identifiers."""

    def __init__(
        self,
        num_continuous_features: int,
        num_airlines: int = 50,
        airline_dim: int = 16,
        num_origins: int = 350,
        origin_dim: int = 32,
        num_dests: int = 350,
        dest_dim: int = 32,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128, 64]

        # Entity Embeddings
        self.airline_embed = nn.Embedding(num_airlines + 1, airline_dim, padding_idx=0)
        self.origin_embed = nn.Embedding(num_origins + 1, origin_dim, padding_idx=0)
        self.dest_embed = nn.Embedding(num_dests + 1, dest_dim, padding_idx=0)

        total_input_dim = num_continuous_features + airline_dim + origin_dim + dest_dim

        # Dense Tabular Backbone
        layers = []
        prev_dim = total_input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Mish())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.backbone = nn.Sequential(*layers)

    def forward(
        self,
        x_cont: torch.Tensor,
        x_airline: torch.Tensor,
        x_origin: torch.Tensor,
        x_dest: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass generating delay risk logits."""
        e_air = self.airline_embed(x_airline)
        e_orig = self.origin_embed(x_origin)
        e_dest = self.dest_embed(x_dest)

        combined = torch.cat([x_cont, e_air, e_orig, e_dest], dim=1)
        logits = self.backbone(combined)
        return logits.squeeze(1)


class TemporalFlightNet(nn.Module):
    """Temporal Recurrent Architecture modeling sequential delay propagation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """x_seq shape: (batch_size, seq_len, input_dim)"""
        gru_out, _ = self.gru(x_seq)
        # Pool last sequential step
        last_step = gru_out[:, -1, :]
        logits = self.fc(last_step)
        return logits.squeeze(1)


class TabularCategoryEncoder:
    """Integer encodes high-cardinality categories for PyTorch embedding layers."""

    def __init__(self):
        self.airline_map: Dict[str, int] = {}
        self.origin_map: Dict[str, int] = {}
        self.dest_map: Dict[str, int] = {}

    def fit(self, df: Any) -> "TabularCategoryEncoder":
        """Fit vocabulary lookups on training DataFrame."""
        airlines = sorted(df["AIRLINE"].dropna().unique())
        self.airline_map = {val: idx + 1 for idx, val in enumerate(airlines)}

        origins = sorted(df["ORIGIN_AIRPORT"].astype(str).dropna().unique())
        self.origin_map = {val: idx + 1 for idx, val in enumerate(origins)}

        dests = sorted(df["DESTINATION_AIRPORT"].astype(str).dropna().unique())
        self.dest_map = {val: idx + 1 for idx, val in enumerate(dests)}
        return self

    def transform(self, df: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert categories into 1D integer index arrays."""
        air_idx = df["AIRLINE"].map(self.airline_map).fillna(0).astype(int).values
        orig_idx = df["ORIGIN_AIRPORT"].astype(str).map(self.origin_map).fillna(0).astype(int).values
        dest_idx = df["DESTINATION_AIRPORT"].astype(str).map(self.dest_map).fillna(0).astype(int).values
        return air_idx, orig_idx, dest_idx


class DeepLearningTrainer:
    """End-to-end trainer for PyTorch Deep Learning delay models."""

    def __init__(
        self,
        model_type: str = "mlp_embedding",
        airline_dim: int = 16,
        origin_dim: int = 32,
        dest_dim: int = 32,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        epochs: int = 12,
        patience: int = 3,
        random_state: int = 42,
    ):
        self.model_type = model_type
        self.airline_dim = airline_dim
        self.origin_dim = origin_dim
        self.dest_dim = dest_dim
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state

        torch.manual_seed(random_state)
        self.device = torch.device("cpu")
        self.model: Optional[nn.Module] = None
        self.cat_encoder = TabularCategoryEncoder()
        self.training_metadata: Dict[str, Any] = {}

    def fit_encoder(self, train_df: Any) -> None:
        self.cat_encoder.fit(train_df)

    def train(
        self,
        X_cont_train: np.ndarray,
        y_train: np.ndarray,
        train_df: Any,
        X_cont_val: np.ndarray,
        y_val: np.ndarray,
        val_df: Any,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Fit PyTorch model on training set with validation early stopping."""
        start_time = time.time()
        self.fit_encoder(train_df)

        air_tr, orig_tr, dest_tr = self.cat_encoder.transform(train_df)
        air_val, orig_val, dest_val = self.cat_encoder.transform(val_df)

        # Imbalance weighting for BCEWithLogitsLoss
        pos_weight_val = float(np.sum(y_train == 0) / max(1, np.sum(y_train == 1)))
        pos_weight = torch.tensor([pos_weight_val], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        if self.model_type == "mlp_embedding":
            self.model = FlightEmbeddingNet(
                num_continuous_features=X_cont_train.shape[1],
                num_airlines=len(self.cat_encoder.airline_map),
                airline_dim=self.airline_dim,
                num_origins=len(self.cat_encoder.origin_map),
                origin_dim=self.origin_dim,
                num_dests=len(self.cat_encoder.dest_map),
                dest_dim=self.dest_dim,
                hidden_dims=self.hidden_dims,
                dropout=self.dropout,
            ).to(self.device)

            train_dataset = TensorDataset(
                torch.tensor(X_cont_train, dtype=torch.float32),
                torch.tensor(air_tr, dtype=torch.long),
                torch.tensor(orig_tr, dtype=torch.long),
                torch.tensor(dest_tr, dtype=torch.long),
                torch.tensor(y_train, dtype=torch.float32),
            )
            val_dataset = TensorDataset(
                torch.tensor(X_cont_val, dtype=torch.float32),
                torch.tensor(air_val, dtype=torch.long),
                torch.tensor(orig_val, dtype=torch.long),
                torch.tensor(dest_val, dtype=torch.long),
                torch.tensor(y_val, dtype=torch.float32),
            )
        else:
            # Temporal Sequence Net
            # Reshape tabular continuous features into 3-step sequential dynamics
            seq_len = 3
            step_dim = X_cont_train.shape[1]
            # Construct synthetic sequence of [t-2, t-1, current]
            X_seq_tr = np.repeat(X_cont_train[:, np.newaxis, :], seq_len, axis=1)
            X_seq_val = np.repeat(X_cont_val[:, np.newaxis, :], seq_len, axis=1)

            self.model = TemporalFlightNet(
                input_dim=step_dim,
                hidden_dim=64,
                num_layers=2,
                dropout=self.dropout,
            ).to(self.device)

            train_dataset = TensorDataset(
                torch.tensor(X_seq_tr, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32),
            )
            val_dataset = TensorDataset(
                torch.tensor(X_seq_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32),
            )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size * 2, shuffle=False)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        best_val_prauc = -1.0
        best_state = None
        no_improve_epochs = 0

        logger.info(f"Training {self.model_type} for {self.epochs} epochs on device: {self.device}")
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                if self.model_type == "mlp_embedding":
                    xc, x_a, x_o, x_d, y_b = [b.to(self.device) for b in batch]
                    logits = self.model(xc, x_a, x_o, x_d)
                else:
                    xs, y_b = [b.to(self.device) for b in batch]
                    logits = self.model(xs)

                loss = criterion(logits, y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * len(y_b)

            scheduler.step()
            train_loss /= len(train_dataset)

            # Evaluate on validation set
            val_probs, val_loss = self._evaluate_loader(val_loader, criterion)
            val_prauc = average_precision_score(y_val, val_probs)
            val_roc = roc_auc_score(y_val, val_probs)

            logger.info(
                f"Epoch {epoch}/{self.epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val ROC AUC: {val_roc:.4f} | Val PR AUC: {val_prauc:.4f}"
            )

            if val_prauc > best_val_prauc:
                best_val_prauc = val_prauc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= self.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        duration = time.time() - start_time
        self.training_metadata = {
            "model_type": self.model_type,
            "best_val_pr_auc": round(float(best_val_prauc), 4),
            "training_duration_seconds": round(duration, 2),
            "epochs_trained": epoch,
            "batch_size": self.batch_size,
            "device": str(self.device),
        }
        return self.model, self.training_metadata

    def _evaluate_loader(self, loader: DataLoader, criterion: nn.Module) -> Tuple[np.ndarray, float]:
        """Compute predicted probabilities and average loss over a DataLoader."""
        self.model.eval()
        total_loss = 0.0
        all_probs = []
        with torch.no_grad():
            for batch in loader:
                if self.model_type == "mlp_embedding":
                    xc, x_a, x_o, x_d, y_b = [b.to(self.device) for b in batch]
                    logits = self.model(xc, x_a, x_o, x_d)
                else:
                    xs, y_b = [b.to(self.device) for b in batch]
                    logits = self.model(xs)

                loss = criterion(logits, y_b)
                total_loss += loss.item() * len(y_b)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)

        avg_loss = total_loss / len(loader.dataset)
        return np.concatenate(all_probs), avg_loss

    def predict_proba(self, X_cont: np.ndarray, meta_df: Any) -> np.ndarray:
        """Generate calibrated delay risk probabilities."""
        if self.model is None:
            raise RuntimeError("Model has not been trained.")
        self.model.eval()

        if self.model_type == "mlp_embedding":
            air_idx, orig_idx, dest_idx = self.cat_encoder.transform(meta_df)
            ds = TensorDataset(
                torch.tensor(X_cont, dtype=torch.float32),
                torch.tensor(air_idx, dtype=torch.long),
                torch.tensor(orig_idx, dtype=torch.long),
                torch.tensor(dest_idx, dtype=torch.long),
            )
            loader = DataLoader(ds, batch_size=self.batch_size * 2, shuffle=False)
            all_probs = []
            with torch.no_grad():
                for xc, xa, xo, xd in loader:
                    logits = self.model(xc.to(self.device), xa.to(self.device), xo.to(self.device), xd.to(self.device))
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_probs.append(probs)
            return np.concatenate(all_probs)
        else:
            seq_len = 3
            X_seq = np.repeat(X_cont[:, np.newaxis, :], seq_len, axis=1)
            ds = TensorDataset(torch.tensor(X_seq, dtype=torch.float32))
            loader = DataLoader(ds, batch_size=self.batch_size * 2, shuffle=False)
            all_probs = []
            with torch.no_grad():
                for (xs,) in loader:
                    logits = self.model(xs.to(self.device))
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_probs.append(probs)
            return np.concatenate(all_probs)

    def save(self, model_dir: Path) -> Path:
        """Save PyTorch weights and configuration metadata."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        weights_path = model_dir / f"{self.model_type}_weights.pt"
        meta_path = model_dir / f"{self.model_type}_metadata.json"

        torch.save(self.model.state_dict(), weights_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": self.training_metadata,
                "airline_vocab_size": len(self.cat_encoder.airline_map),
                "origin_vocab_size": len(self.cat_encoder.origin_map),
                "dest_vocab_size": len(self.cat_encoder.dest_map),
            }, f, indent=2)

        logger.info(f"Saved PyTorch {self.model_type} model to {weights_path}.")
        return weights_path
