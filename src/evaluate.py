"""Metrics and plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (classification_report, f1_score, mean_squared_error,
                             r2_score, roc_auc_score, roc_curve)

CLASS_NAMES = ["Default", "Paid Off"]


def best_cutoff(y_true, proba, grid=np.arange(0.05, 0.96, 0.01)) -> float:
    """Probability cut-off for 'Paid Off' that maximises macro-F1.

    Chosen on VALIDATION data. A fixed 0.5 cut-off on imbalanced data makes models
    predict "paid" for almost everything and hides how well they rank defaults.
    """
    best, best_f1 = 0.5, -1.0
    for c in grid:
        f1 = f1_score(y_true, (proba >= c).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best, best_f1 = float(c), f1
    return best


def classification_row(model_name: str, split: str, y_true, y_score, cutoff: float) -> dict:
    y_pred = (np.asarray(y_score) >= cutoff).astype(int)
    rep = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                output_dict=True, zero_division=0)
    return {
        "model": model_name, "split": split, "cutoff": cutoff,
        "precision_default": rep["Default"]["precision"], "recall_default": rep["Default"]["recall"],
        "f1_default": rep["Default"]["f1-score"],
        "precision_paid": rep["Paid Off"]["precision"], "recall_paid": rep["Paid Off"]["recall"],
        "f1_paid": rep["Paid Off"]["f1-score"],
        "precision_weighted": rep["weighted avg"]["precision"],
        "recall_weighted": rep["weighted avg"]["recall"],
        "f1_weighted": rep["weighted avg"]["f1-score"],
        "f1_macro": rep["macro avg"]["f1-score"],
        "roc_auc": roc_auc_score(y_true, y_score),
    }


def regression_row(model_name: str, y_tr, p_tr, y_va, p_va, y_te, p_te) -> dict:
    return {
        "model": model_name,
        "train_mse": mean_squared_error(y_tr, p_tr), "val_mse": mean_squared_error(y_va, p_va),
        "test_mse": mean_squared_error(y_te, p_te),
        "train_r2": r2_score(y_tr, p_tr), "val_r2": r2_score(y_va, p_va), "test_r2": r2_score(y_te, p_te),
    }


def plot_confusion_matrices(matrices: dict[str, np.ndarray], path: Path) -> None:
    n = len(matrices)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4 * nrows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, (name, cm) in zip(axes.ravel(), matrices.items()):
        ax.imshow(cm, cmap="Blues")
        ax.set(title=f"{name} (test)", xticks=[0, 1], yticks=[0, 1],
               xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
               xlabel="Predicted", ylabel="True")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(scores: dict[str, tuple], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (y_true, y_score) in scores.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, y_score):.3f})")
    ax.plot([0, 1], [0, 1], "r--", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="ROC curves (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
