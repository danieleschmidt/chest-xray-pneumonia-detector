"""Training pipeline for PneumoniaCNN.

Usage:
    python -m src.train [--epochs 10] [--samples 500] [--batch-size 32]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from .dataset import SyntheticXRayDataset
from .model import PneumoniaCNN


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    n_samples: int = 500
    img_size: int = 64
    batch_size: int = 32
    epochs: int = 5
    lr: float = 1e-3
    dropout: float = 0.5
    seed: int = 42
    device: str = "cpu"
    val_frac: float = 0.15
    test_frac: float = 0.15


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _split_indices(
    n: int, val_frac: float, test_frac: float, seed: int
) -> Tuple[List[int], List[int], List[int]]:
    indices = list(range(n))
    train_idx, temp_idx = train_test_split(
        indices, test_size=val_frac + test_frac, random_state=seed
    )
    rel_test = test_frac / (val_frac + test_frac)
    val_idx, test_idx = train_test_split(temp_idx, test_size=rel_test, random_state=seed)
    return train_idx, val_idx, test_idx


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) >= 0.5).long().squeeze(1)
    return (preds == targets.long()).float().mean().item()


# ──────────────────────────────────────────────────────────────────────────────
# Core loop
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg: TrainConfig) -> Dict[str, List[float]]:
    """Train PneumoniaCNN and return history dict.

    Returns
    -------
    dict with keys: train_loss, train_acc, val_loss, val_acc
    """
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    # ── Data ─────────────────────────────────────────────────────────
    full_ds = SyntheticXRayDataset(n_samples=cfg.n_samples, size=cfg.img_size, seed=cfg.seed)
    train_idx, val_idx, test_idx = _split_indices(
        len(full_ds), cfg.val_frac, cfg.test_frac, cfg.seed
    )
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)
    test_ds = Subset(full_ds, test_idx)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    # ── Model ─────────────────────────────────────────────────────────
    model = PneumoniaCNN(dropout=cfg.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [],
        "val_loss": [],   "val_acc": [],
    }

    for epoch in range(1, cfg.epochs + 1):
        # ── Train ────────────────────────────────────────────────────
        model.train()
        t_loss, t_acc = 0.0, 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(imgs).squeeze(1)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            t_acc += _accuracy(logits.unsqueeze(1), lbls)
        t_loss /= len(train_loader)
        t_acc /= len(train_loader)

        # ── Validate ─────────────────────────────────────────────────
        model.eval()
        v_loss, v_acc = 0.0, 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                logits = model(imgs).squeeze(1)
                v_loss += criterion(logits, lbls).item()
                v_acc += _accuracy(logits.unsqueeze(1), lbls)
        v_loss /= len(val_loader)
        v_acc /= len(val_loader)

        scheduler.step()

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        print(
            f"Epoch {epoch:02d}/{cfg.epochs}  "
            f"train_loss={t_loss:.4f}  train_acc={t_acc:.3f}  "
            f"val_loss={v_loss:.4f}  val_acc={v_acc:.3f}"
        )

    # ── Test set evaluation ───────────────────────────────────────────
    model.eval()
    test_acc = 0.0
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits = model(imgs).squeeze(1)
            test_acc += _accuracy(logits.unsqueeze(1), lbls)
    test_acc /= len(test_loader)
    print(f"\nTest accuracy: {test_acc:.3f}")
    history["test_acc"] = [test_acc]

    return history, model  # type: ignore[return-value]


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train PneumoniaCNN on synthetic data")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cpu")
    a = p.parse_args()
    return TrainConfig(
        n_samples=a.samples,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        device=a.device,
    )


if __name__ == "__main__":
    cfg = _parse_args()
    train(cfg)
