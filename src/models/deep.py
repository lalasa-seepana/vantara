"""Small CPU networks with deterministic training and validation early stopping."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset


class Network(nn.Module):
    """ANN, LSTM or autoencoder using the same configured hidden width."""

    def __init__(
        self, kind: str, width: int, cfg: dict[str, Any], categories: int = 7
    ) -> None:
        """Build the selected network without fitting it."""
        super().__init__()
        self.kind = kind
        h = cfg["hidden"]
        d = cfg["dropout"]
        if kind == "ann":
            self.layers = nn.Sequential(
                nn.Linear(width, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(d),
                nn.Linear(h, h // 2),
                nn.ReLU(),
                nn.Dropout(d),
                nn.Linear(h // 2, 1),
            )
        elif kind == "lstm":
            self.lstm = nn.LSTM(width - 1, h, batch_first=True)
            self.dropout = nn.Dropout(d)
            self.head = nn.Linear(h, 1 + categories)
        elif kind == "autoencoder":
            self.layers = nn.Sequential(
                nn.Linear(width, h),
                nn.ReLU(),
                nn.Linear(h, cfg["latent"]),
                nn.ReLU(),
                nn.Linear(cfg["latent"], h),
                nn.ReLU(),
                nn.Linear(h, width),
            )
        else:
            raise ValueError(f"Unknown network kind {kind}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict logits, reconstruction or timing/category outputs."""
        if self.kind == "lstm":
            lengths = x[:, :, 0].sum(1).long().clamp(min=1).cpu()
            packed = pack_padded_sequence(
                x[:, :, 1:], lengths, batch_first=True, enforce_sorted=False
            )
            _, (hidden, _) = self.lstm(packed)
            return self.head(self.dropout(hidden[-1]))
        return self.layers(x)


class NeuralModel:
    """Persistable estimator holding actual weights and loss history."""

    def __init__(
        self,
        kind: str,
        cfg: dict[str, Any],
        seed: int,
        horizon: int = 90,
        category_count: int = 7,
    ) -> None:
        """Store settings; training happens in fit."""
        self.kind, self.cfg, self.seed = kind, cfg, seed
        self.horizon, self.category_count = horizon, category_count

    def loss(
        self, output: torch.Tensor, target: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Use cross-entropy, reconstruction MSE or capped timing/category loss."""
        if self.kind == "ann":
            return nn.functional.binary_cross_entropy_with_logits(
                output[:, 0], target, pos_weight=self.positive_weight
            )
        if self.kind == "autoencoder":
            return nn.functional.mse_loss(output, x)
        timing = nn.functional.mse_loss(output[:, 0], target[:, 0])
        known = target[:, 1] >= 0
        category = (
            nn.functional.cross_entropy(output[known, 1:], target[known, 1].long())
            if known.any()
            else timing * 0
        )
        return timing + self.cfg.get("category_loss_weight", 0.2) * category

    def fit(
        self, x: np.ndarray, y: np.ndarray, xv: np.ndarray, yv: np.ndarray
    ) -> NeuralModel:
        """Train on x only and select the best epoch on xv."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.set_num_threads(self.cfg["threads"])
        torch.use_deterministic_algorithms(True)
        self.network = Network(self.kind, x.shape[-1], self.cfg, self.category_count)
        tensors = [
            torch.as_tensor(np.asarray(a), dtype=torch.float32) for a in [x, y, xv, yv]
        ]
        xt, yt, vt, vy = tensors
        self.positive_weight = (
            torch.tensor(float((1 - y).sum() / max(y.sum(), 1)))
            if self.kind == "ann"
            else torch.tensor(1.0)
        )
        loader = DataLoader(
            TensorDataset(xt, yt),
            batch_size=min(self.cfg["batch_size"], len(x)),
            shuffle=True,
            drop_last=self.kind == "ann",
            generator=torch.Generator().manual_seed(self.seed),
        )
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.cfg["learning_rate"],
            weight_decay=self.cfg["weight_decay"],
        )
        best, wait, best_state = float("inf"), 0, None
        self.history: list[dict[str, float]] = []
        for epoch in range(self.cfg["epochs"]):
            self.network.train()
            total, count = 0.0, 0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = self.loss(self.network(xb), yb, xb)
                penalty = self.cfg["l1"] * sum(
                    p.abs().sum() for p in self.network.parameters()
                )
                (loss + penalty).backward()
                optimizer.step()
                total += loss.item() * len(xb)
                count += len(xb)
            self.network.eval()
            with torch.no_grad():
                validation = self.loss(self.network(vt), vy, vt).item()
            self.history.append(
                {
                    "epoch": epoch + 1,
                    "training_loss": total / max(count, 1),
                    "validation_loss": validation,
                }
            )
            if validation < best:
                best, wait = validation, 0
                best_state = copy.deepcopy(self.network.state_dict())
            else:
                wait += 1
            if wait >= self.cfg["patience"]:
                break
        self.network.load_state_dict(best_state)
        self.network.eval()
        return self

    def raw(self, x: np.ndarray) -> np.ndarray:
        """Run CPU inference without updating batch normalization."""
        self.network.eval()
        with torch.no_grad():
            return self.network(
                torch.as_tensor(np.asarray(x), dtype=torch.float32)
            ).numpy()

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return binary ANN probabilities."""
        logits = np.clip(self.raw(x)[:, 0], -40, 40)
        p = 1 / (1 + np.exp(-logits))
        return np.column_stack([1 - p, p])

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return capped waiting days for the LSTM or autoencoder errors."""
        raw = self.raw(x)
        if self.kind == "lstm":
            return np.clip(
                np.expm1(np.clip(raw[:, 0], 0, 1) * np.log1p(self.horizon)),
                0,
                self.horizon,
            )
        if self.kind == "autoencoder":
            return np.mean((raw - x) ** 2, axis=1)
        return self.predict_proba(x)[:, 1]


def scale_sequences(
    train: np.ndarray, *others: np.ndarray
) -> tuple[list[np.ndarray], dict[str, list[float]]]:
    """Fit amount/gap scaling on valid training events, excluding padding."""
    events = train[:, :, 1:3][train[:, :, 0] == 1]
    mean = events.mean(axis=0)
    std = events.std(axis=0)
    std[std == 0] = 1
    params = {"mean": mean.tolist(), "std": std.tolist()}
    return [transform_sequences(a, params) for a in (train, *others)], params


def transform_sequences(x: np.ndarray, params: dict[str, list[float]]) -> np.ndarray:
    """Apply a saved sequence scaler and preserve zero padding."""
    result = x.copy()
    result[:, :, 1:3] = (result[:, :, 1:3] - np.array(params["mean"])) / np.array(
        params["std"]
    )
    result[:, :, 1:3] *= result[:, :, :1]
    return result
