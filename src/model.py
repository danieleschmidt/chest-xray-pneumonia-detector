"""PneumoniaCNN — lightweight CNN for binary X-ray classification.

Architecture:
    3 convolutional blocks (conv → ReLU → maxpool)
    Adaptive average pool → flatten
    2 fully-connected layers with dropout
    Binary output (sigmoid) for NORMAL vs PNEUMONIA
"""
import torch
import torch.nn as nn


class PneumoniaCNN(nn.Module):
    """Binary CNN classifier for chest X-ray pneumonia detection.

    Parameters
    ----------
    dropout: float
        Dropout probability applied between the two FC layers (default 0.5).
    """

    def __init__(self, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 1×64×64 → 32×32×32
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 32×32×32 → 64×16×16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 64×16×16 → 128×8×8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # → 128×4×4 = 2048
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),  # binary logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probability of PNEUMONIA (class 1)."""
        with torch.no_grad():
            return torch.sigmoid(self(x))

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return 0/1 predictions."""
        return (self.predict_proba(x) >= threshold).long().squeeze(1)
