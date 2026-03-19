"""Tests for PneumoniaCNN."""
import torch
import pytest
from src.model import PneumoniaCNN


def test_output_shape():
    model = PneumoniaCNN()
    x = torch.randn(4, 1, 64, 64)
    out = model(x)
    assert out.shape == (4, 1), f"Expected (4,1), got {out.shape}"


def test_output_shape_single():
    model = PneumoniaCNN()
    x = torch.randn(1, 1, 64, 64)
    out = model(x)
    assert out.shape == (1, 1)


def test_predict_proba_bounds():
    model = PneumoniaCNN()
    x = torch.randn(8, 1, 64, 64)
    proba = model.predict_proba(x)
    assert proba.min().item() >= 0.0
    assert proba.max().item() <= 1.0


def test_predict_binary():
    model = PneumoniaCNN()
    x = torch.randn(8, 1, 64, 64)
    preds = model.predict(x)
    assert set(preds.tolist()).issubset({0, 1})


def test_gradient_flows():
    model = PneumoniaCNN()
    x = torch.randn(2, 1, 64, 64)
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    # Check at least one parameter has a gradient
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_different_batch_sizes():
    model = PneumoniaCNN()
    for bs in [1, 8, 16]:
        x = torch.randn(bs, 1, 64, 64)
        out = model(x)
        assert out.shape == (bs, 1)
