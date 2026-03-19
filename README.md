# Chest X-Ray Pneumonia Detector

A CNN-based binary classifier for detecting pneumonia in chest X-rays,
built with **PyTorch**. Includes a synthetic data generator, GradCAM
interpretability, and a clean training pipeline — no real patient data
required to run the demo or tests.

> **⚠️ Medical Disclaimer**
> This project is for **educational and research purposes only**.
> It is **not** validated for clinical use and must **not** be used to
> diagnose, treat, or inform medical decisions for any patient.
> Always consult qualified medical professionals for clinical interpretation
> of radiological findings.

---

## Architecture

```
PneumoniaCNN
├── Conv Block 1  — Conv2d(1→32)  + BN + ReLU + MaxPool2d   → 32×32×32
├── Conv Block 2  — Conv2d(32→64) + BN + ReLU + MaxPool2d   → 64×16×16
├── Conv Block 3  — Conv2d(64→128)+ BN + ReLU + MaxPool2d   → 128×8×8
├── AdaptiveAvgPool2d(4×4)                                   → 128×4×4
└── Classifier
    ├── Linear(2048 → 256) + ReLU + Dropout(0.5)
    └── Linear(256 → 1)    (binary logit → sigmoid)
```

Binary labels: `0 = NORMAL`, `1 = PNEUMONIA`

---

## Project Structure

```
chest-xray-pneumonia-detector/
├── src/
│   ├── model.py       — PneumoniaCNN definition
│   ├── dataset.py     — SyntheticXRayDataset + make_xray()
│   ├── train.py       — Training pipeline (Adam + cosine LR, BCE loss)
│   └── gradcam.py     — GradCAM interpretability
├── tests/
│   ├── test_model.py
│   ├── test_dataset.py
│   ├── test_gradcam.py
│   └── test_train.py
├── demo.py            — 5-epoch demo with GradCAM output
└── requirements.txt
```

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
# or, with dev/test deps:
pip install -r requirements-dev.txt
```

### 2. Run the demo

```bash
python demo.py
```

Trains for 5 epochs on 500 synthetic images, prints per-epoch metrics,
test accuracy, and a text description of the GradCAM heatmap.

### 3. Run tests

```bash
pytest tests/ -v
```

---

## Synthetic Data

`SyntheticXRayDataset` generates grayscale 64×64 "X-ray" images:

| Class      | Construction |
|------------|-------------|
| NORMAL     | Gaussian noise + bilateral oval lung masks + dark cardiac shadow |
| PNEUMONIA  | Same as NORMAL + 1–3 bright consolidation patches in lung fields |

No real patient data is downloaded or required. All images are
generated deterministically from a random seed.

---

## GradCAM

`GradCAM` hooks the last convolutional layer and back-propagates
the PNEUMONIA logit to produce a spatial importance map (Selvaraju et al., 2017).

```python
from src.gradcam import GradCAM

cam = GradCAM(model)
heatmap = cam(image_tensor)          # np.ndarray (H, W), values in [0, 1]
print(cam.describe_heatmap(heatmap)) # text summary of hot regions
```

---

## Real Data

To train on the [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
dataset, replace `SyntheticXRayDataset` with a `torchvision.datasets.ImageFolder`
loader pointing at the `train/` directory:

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
ds = datasets.ImageFolder("data/chest_xray/train", transform=transform)
```

---

## References

- Selvaraju, R.R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks
  via Gradient-based Localization.* ICCV.
- Wang, X. et al. (2017). *ChestX-ray8: Hospital-scale Chest X-Ray Database and Benchmarks.*
  CVPR.

---

## License

MIT
