"""GradCAM implementation for PneumoniaCNN.

Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017).
Highlights which regions of the input X-ray most strongly influenced
the model's PNEUMONIA prediction.

Usage:
    cam = GradCAM(model, target_layer=model.features[-3])  # last conv block
    heatmap = cam(image_tensor)   # shape (H, W), values in [0, 1]
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Compute GradCAM heatmaps for a PneumoniaCNN (or any CNN).

    Parameters
    ----------
    model: nn.Module
        Trained model in eval mode.
    target_layer: nn.Module | None
        The convolutional layer to hook. Defaults to the last Conv2d
        found in ``model.features``.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
    ) -> None:
        self.model = model
        self.model.eval()

        if target_layer is None:
            # Auto-select last Conv2d in model.features
            target_layer = self._find_last_conv(model)
        self.target_layer = target_layer

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._register_hooks()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _find_last_conv(model: nn.Module) -> nn.Module:
        last_conv = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is None:
            raise ValueError("No Conv2d layer found in model.")
        return last_conv

    def _register_hooks(self) -> None:
        def fwd_hook(module, input, output):
            self._activations = output.detach()

        def bwd_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, img: torch.Tensor, class_idx: int = 1) -> np.ndarray:
        """Compute GradCAM heatmap for *img*.

        Parameters
        ----------
        img: torch.Tensor of shape (1, C, H, W) or (C, H, W)
            Preprocessed image tensor on the same device as the model.
        class_idx: int
            Class index to explain (1 = PNEUMONIA, 0 = NORMAL).

        Returns
        -------
        np.ndarray of shape (H, W) with values in [0, 1].
        """
        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = img.requires_grad_(True)

        # Forward pass
        logit = self.model(img)  # (1, 1)
        score = logit[:, 0] if class_idx == 1 else -logit[:, 0]

        # Backward
        self.model.zero_grad()
        score.sum().backward()

        # Pool gradients over spatial dims → importance weights
        # _gradients: (1, C, H', W')
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)

        # Upsample to input resolution
        H, W = img.shape[-2], img.shape[-1]
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam.astype(np.float32)

    def describe_heatmap(self, heatmap: np.ndarray) -> str:
        """Return a human-readable summary of the GradCAM heatmap.

        Divides the image into quadrants and reports which regions carry
        the highest activation — useful for logging without a display.
        """
        H, W = heatmap.shape
        mid_h, mid_w = H // 2, W // 2
        quadrants = {
            "upper-left":  heatmap[:mid_h, :mid_w],
            "upper-right": heatmap[:mid_h, mid_w:],
            "lower-left":  heatmap[mid_h:, :mid_w],
            "lower-right": heatmap[mid_h:, mid_w:],
        }
        scores = {name: arr.mean() for name, arr in quadrants.items()}
        dominant = max(scores, key=scores.__getitem__)
        overall = heatmap.mean()
        peak = heatmap.max()

        lines = [
            f"GradCAM summary (PNEUMONIA class):",
            f"  Overall mean activation : {overall:.3f}",
            f"  Peak activation         : {peak:.3f}",
            f"  Dominant region         : {dominant} (mean={scores[dominant]:.3f})",
            "  Region breakdown:",
        ]
        for name, score in sorted(scores.items(), key=lambda x: -x[1]):
            bar = "█" * int(score * 20)
            lines.append(f"    {name:<14}: {score:.3f}  {bar}")
        return "\n".join(lines)
