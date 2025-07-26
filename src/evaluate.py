"""Utility script to compute metrics and plot a confusion matrix from predictions."""

import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def evaluate_predictions(
    pred_csv: str,
    label_csv: str | None = None,
    output_png: str = "reports/confusion_matrix.png",
    normalize_cm: bool = False,
    threshold: float = 0.5,
    metrics_csv: str | None = None,
    num_classes: int = 1,
):
    """Evaluate predictions stored in ``pred_csv``.

    Parameters
    ----------
    pred_csv:
        CSV with columns ``filepath`` and ``prediction``.
    label_csv:
        Optional CSV with columns ``filepath`` and ``label``. If not provided,
        ``pred_csv`` must already contain a ``label`` column.
    output_png:
        Path to save the confusion matrix plot.
    normalize_cm:
        Whether to normalize the confusion matrix by true labels.
    threshold:
        Probability threshold for converting predictions to binary labels.
        Must be between 0 and 1 (inclusive).
    metrics_csv:
        Optional path to save the computed metrics as a CSV file.
    num_classes:
        Number of classes in the predictions. Set >1 for multi-class CSVs.

    Raises
    ------
    ValueError
        If threshold is not between 0 and 1 (inclusive), or if threshold is NaN.
    """
    # Input validation
    import math

    if not (0 <= threshold <= 1) or math.isnan(threshold):
        raise ValueError(
            f"Threshold must be between 0 and 1 (inclusive), got {threshold}"
        )

    df = pd.read_csv(pred_csv)
    if "label" not in df.columns:
        if label_csv is None:
            raise ValueError("No ground truth labels provided")
        labels = pd.read_csv(label_csv)
        df = df.merge(labels, on="filepath")

    # Check for empty dataset
    if df.empty:
        raise ValueError(
            "Cannot evaluate empty dataset. Check input CSV file contents."
        )

    y_true = df["label"].values
    if num_classes == 1:
        y_probs = df["prediction"].values
        y_pred = (y_probs > threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_probs)
    else:
        prob_cols = [f"prob_{i}" for i in range(num_classes)]
        y_probs = df[prob_cols].values
        y_pred = df["prediction"].astype(int).values
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        y_true_cat = pd.get_dummies(y_true, drop_first=False).values
        try:
            roc_auc = roc_auc_score(y_true_cat, y_probs, multi_class="ovr")
        except (ValueError, ZeroDivisionError) as e:
            # Handle cases where ROC-AUC cannot be computed (e.g., all probabilities are zero)
            import warnings

            warnings.warn(f"ROC-AUC calculation failed: {e}. Setting ROC-AUC to NaN.")
            roc_auc = float("nan")

    cm = confusion_matrix(y_true, y_pred, normalize="true" if normalize_cm else None)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }

    if metrics_csv:
        pd.DataFrame([metrics]).to_csv(metrics_csv, index=False)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prediction CSV")
    parser.add_argument("--pred_csv", required=True, help="CSV with predictions")
    parser.add_argument("--label_csv", help="CSV with ground truth labels")
    parser.add_argument(
        "--img_size",
        type=int,
        nargs=2,
        default=[150, 150],
        metavar=("HEIGHT", "WIDTH"),
        help="Unused. For backward compatibility",
    )
    parser.add_argument(
        "--output_png",
        default="reports/confusion_matrix.png",
        help="Output confusion matrix image",
    )
    parser.add_argument(
        "--normalize_cm",
        action="store_true",
        help="Normalize the confusion matrix by true labels",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for positive class",
    )
    parser.add_argument(
        "--metrics_csv",
        default=None,
        help="Optional path to save the computed metrics as CSV",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1,
        help="Number of classes in the predictions CSV",
    )
    args = parser.parse_args()

    metrics = evaluate_predictions(
        args.pred_csv,
        args.label_csv,
        args.output_png,
        normalize_cm=args.normalize_cm,
        threshold=args.threshold,
        metrics_csv=args.metrics_csv,
        num_classes=args.num_classes,
    )
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
