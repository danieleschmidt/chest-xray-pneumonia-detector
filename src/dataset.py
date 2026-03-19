"""SyntheticXRayDataset — generates grayscale 64×64 synthetic chest X-rays.

NORMAL images
    • Gaussian noise baseline
    • Two soft elliptical "lung" masks (bright regions)

PNEUMONIA images
    • Same as NORMAL
    • Additional bright blotchy patches (simulating airspace consolidation)

No real patient data required — suitable for unit tests, demos, and
smoke-testing the full training pipeline.
"""
from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _draw_ellipse(
    img: np.ndarray,
    cx: int,
    cy: int,
    rx: int,
    ry: int,
    intensity: float,
) -> np.ndarray:
    """Paint a soft ellipse onto *img* (in-place, returns img)."""
    H, W = img.shape
    y_idx, x_idx = np.ogrid[:H, :W]
    dist = ((x_idx - cx) / rx) ** 2 + ((y_idx - cy) / ry) ** 2
    mask = np.clip(1.0 - dist, 0, 1) ** 2
    img += intensity * mask
    return img


def make_xray(label: int, size: int = 64, rng: random.Random | None = None) -> np.ndarray:
    """Create a single synthetic grayscale X-ray image.

    Parameters
    ----------
    label: int
        0 = NORMAL, 1 = PNEUMONIA
    size: int
        Width and height in pixels (default 64)
    rng: random.Random | None
        Optional seeded RNG for reproducibility.

    Returns
    -------
    np.ndarray of shape (1, size, size), dtype float32, values in [0, 1]
    """
    rng = rng or random
    img = np.zeros((size, size), dtype=np.float32)

    # ── Background tissue noise ──────────────────────────────────────
    img += np.random.normal(0.15, 0.04, (size, size)).astype(np.float32)

    # ── Lung fields (bilateral ovals) ────────────────────────────────
    mid = size // 2
    offset = size // 8
    lung_rx, lung_ry = size // 5, size // 3
    _draw_ellipse(img, mid - offset, mid, lung_rx, lung_ry, 0.30)
    _draw_ellipse(img, mid + offset, mid, lung_rx, lung_ry, 0.30)

    # ── Heart silhouette (darker central oval) ───────────────────────
    _draw_ellipse(img, mid, mid + size // 8, size // 10, size // 8, -0.10)

    if label == 1:
        # ── Pneumonia consolidation (1–3 bright patches) ─────────────
        n_patches = rng.randint(1, 3)
        for _ in range(n_patches):
            # bias patches toward the lung fields
            side = rng.choice([-1, 1])
            px = int(rng.gauss(mid + side * offset, size / 10))
            py = int(rng.gauss(mid, size / 8))
            pr = rng.randint(size // 12, size // 7)
            _draw_ellipse(img, px, py, pr, pr, rng.uniform(0.25, 0.45))

    img = np.clip(img, 0.0, 1.0)
    return img[np.newaxis, :]  # (1, H, W)


class SyntheticXRayDataset(Dataset):
    """PyTorch Dataset of synthetic chest X-ray images.

    Parameters
    ----------
    n_samples: int
        Total number of images (split ~50/50 between classes).
    size: int
        Image spatial dimension (default 64).
    seed: int | None
        Random seed for reproducibility.
    """

    CLASSES = ["NORMAL", "PNEUMONIA"]

    def __init__(
        self,
        n_samples: int = 500,
        size: int = 64,
        seed: int | None = 42,
    ) -> None:
        super().__init__()
        self.size = size
        rng = random.Random(seed)
        np_rng_state = np.random.get_state()
        if seed is not None:
            np.random.seed(seed)

        labels = [i % 2 for i in range(n_samples)]  # alternating 0/1
        rng.shuffle(labels)

        self.images: list[np.ndarray] = []
        self.labels: list[int] = []
        for lbl in labels:
            self.images.append(make_xray(lbl, size=size, rng=rng))
            self.labels.append(lbl)

        np.random.set_state(np_rng_state)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = torch.from_numpy(self.images[idx])       # float32 (1, H, W)
        lbl = torch.tensor(self.labels[idx], dtype=torch.float32)  # scalar
        return img, lbl
