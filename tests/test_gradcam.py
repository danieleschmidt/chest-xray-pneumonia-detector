"""Tests for GradCAM."""
import numpy as np
import torch
import pytest
from src.model import PneumoniaCNN
from src.gradcam import GradCAM


@pytest.fixture
def trained_model():
    """A tiny model with one training step so gradients are non-trivial."""
    model = PneumoniaCNN()
    x = torch.randn(4, 1, 64, 64)
    y = torch.ones(4)
    import torch.nn as nn
    opt = torch.optim.Adam(model.parameters())
    logits = model(x).squeeze(1)
    loss = nn.BCEWithLogitsLoss()(logits, y)
    loss.backward()
    opt.step()
    model.eval()
    return model


def test_heatmap_shape(trained_model):
    cam = GradCAM(trained_model)
    img = torch.randn(1, 1, 64, 64)
    heatmap = cam(img)
    assert heatmap.shape == (64, 64)


def test_heatmap_values_normalised(trained_model):
    cam = GradCAM(trained_model)
    img = torch.randn(1, 1, 64, 64)
    heatmap = cam(img)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_heatmap_dtype(trained_model):
    cam = GradCAM(trained_model)
    img = torch.randn(1, 1, 64, 64)
    heatmap = cam(img)
    assert heatmap.dtype == np.float32


def test_heatmap_3d_input(trained_model):
    """Accept (C, H, W) without explicit batch dim."""
    cam = GradCAM(trained_model)
    img = torch.randn(1, 64, 64)
    heatmap = cam(img)
    assert heatmap.shape == (64, 64)


def test_describe_heatmap_returns_string(trained_model):
    cam = GradCAM(trained_model)
    img = torch.randn(1, 1, 64, 64)
    heatmap = cam(img)
    desc = cam.describe_heatmap(heatmap)
    assert isinstance(desc, str)
    assert "GradCAM" in desc
    assert "Dominant region" in desc


def test_auto_layer_selection():
    """GradCAM should auto-find last Conv2d if no target_layer supplied."""
    model = PneumoniaCNN()
    model.eval()
    cam = GradCAM(model)  # no target_layer
    img = torch.randn(1, 1, 64, 64)
    heatmap = cam(img)
    assert heatmap.shape == (64, 64)
