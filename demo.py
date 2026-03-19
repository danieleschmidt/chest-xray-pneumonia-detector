#!/usr/bin/env python3
"""Demo: train PneumoniaCNN for 5 epochs on 500 synthetic images,
then display a GradCAM heatmap description for a test sample.

Run:
    python demo.py
"""
from __future__ import annotations

import sys
import torch
from src.train import TrainConfig, train
from src.gradcam import GradCAM
from src.dataset import SyntheticXRayDataset


def main() -> None:
    print("=" * 60)
    print("  Chest X-Ray Pneumonia Detector — Demo")
    print("=" * 60)
    print()

    cfg = TrainConfig(
        n_samples=500,
        img_size=64,
        batch_size=32,
        epochs=5,
        lr=1e-3,
        device="cpu",
    )

    print(f"Training on {cfg.n_samples} synthetic images for {cfg.epochs} epochs…\n")
    history, model = train(cfg)

    print()
    print("Training complete.")
    print(f"  Final train acc : {history['train_acc'][-1]:.3f}")
    print(f"  Final val acc   : {history['val_acc'][-1]:.3f}")
    print(f"  Test accuracy   : {history['test_acc'][0]:.3f}")
    print()

    # ── GradCAM demo ──────────────────────────────────────────────────
    print("─" * 60)
    print("GradCAM interpretability demo")
    print("─" * 60)

    ds = SyntheticXRayDataset(n_samples=10, size=64, seed=99)
    # Find a PNEUMONIA sample
    pneu_img, pneu_lbl = None, None
    for img, lbl in ds:
        if lbl.item() == 1:
            pneu_img, pneu_lbl = img, lbl
            break

    if pneu_img is None:
        print("No PNEUMONIA sample found in demo batch — using first sample.")
        pneu_img, pneu_lbl = ds[0]

    model.eval()
    cam = GradCAM(model)
    heatmap = cam(pneu_img)
    print(cam.describe_heatmap(heatmap))
    print()
    print("A higher activation in a region indicates the model")
    print("attended there when predicting PNEUMONIA.")
    print()
    print("Done ✓")


if __name__ == "__main__":
    main()
