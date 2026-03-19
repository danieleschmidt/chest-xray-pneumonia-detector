"""Tests for SyntheticXRayDataset and make_xray."""
import numpy as np
import torch
import pytest
from src.dataset import SyntheticXRayDataset, make_xray


class TestMakeXray:
    def test_shape(self):
        img = make_xray(0)
        assert img.shape == (1, 64, 64)

    def test_custom_size(self):
        img = make_xray(1, size=32)
        assert img.shape == (1, 32, 32)

    def test_values_in_range(self):
        for lbl in [0, 1]:
            img = make_xray(lbl)
            assert img.min() >= 0.0
            assert img.max() <= 1.0

    def test_pneumonia_brighter_on_average(self):
        # Averaged over many samples, pneumonia should be slightly brighter
        normals = np.array([make_xray(0).mean() for _ in range(30)])
        pneums = np.array([make_xray(1).mean() for _ in range(30)])
        assert pneums.mean() > normals.mean()

    def test_dtype(self):
        img = make_xray(0)
        assert img.dtype == np.float32


class TestSyntheticXRayDataset:
    def test_length(self):
        ds = SyntheticXRayDataset(n_samples=100)
        assert len(ds) == 100

    def test_item_shape(self):
        ds = SyntheticXRayDataset(n_samples=10)
        img, lbl = ds[0]
        assert img.shape == (1, 64, 64)
        assert img.dtype == torch.float32

    def test_label_type(self):
        ds = SyntheticXRayDataset(n_samples=10)
        _, lbl = ds[0]
        assert lbl.dtype == torch.float32

    def test_label_values(self):
        ds = SyntheticXRayDataset(n_samples=20)
        labels = [ds[i][1].item() for i in range(len(ds))]
        assert set(labels).issubset({0.0, 1.0})

    def test_roughly_balanced(self):
        ds = SyntheticXRayDataset(n_samples=200)
        labels = [ds[i][1].item() for i in range(len(ds))]
        n_pneu = sum(1 for l in labels if l == 1.0)
        # Should be roughly 50/50 ± 10%
        assert 80 <= n_pneu <= 120

    def test_reproducible_with_seed(self):
        ds1 = SyntheticXRayDataset(n_samples=10, seed=7)
        ds2 = SyntheticXRayDataset(n_samples=10, seed=7)
        img1, lbl1 = ds1[3]
        img2, lbl2 = ds2[3]
        assert torch.allclose(img1, img2)
        assert lbl1 == lbl2

    def test_different_seeds_differ(self):
        ds1 = SyntheticXRayDataset(n_samples=10, seed=1)
        ds2 = SyntheticXRayDataset(n_samples=10, seed=2)
        img1, _ = ds1[0]
        img2, _ = ds2[0]
        # Very unlikely to be identical
        assert not torch.allclose(img1, img2)
