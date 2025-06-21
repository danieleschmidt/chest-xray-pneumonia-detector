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


def evaluate_predictions(pred_csv: str, label_csv: str | None = None, output_png: str = "reports/confusion_matrix.png"):
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
    """
    df = pd.read_csv(pred_csv)
    if "label" not in df.columns:
        if label_csv is None:
            raise ValueError("No ground truth labels provided")
        labels = pd.read_csv(label_csv)
        df = df.merge(labels, on="filepath")

    y_true = df["label"].values
    y_probs = df["prediction"].values
    y_pred = (y_probs > 0.5).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_probs)

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


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
        "--output_png", default="reports/confusion_matrix.png", help="Output confusion matrix image"
    )
    args = parser.parse_args()

    metrics = evaluate_predictions(args.pred_csv, args.label_csv, args.output_png)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
