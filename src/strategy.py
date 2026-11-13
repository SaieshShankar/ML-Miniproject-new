"""Threshold-based loan selection strategy built on predicted NAR."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def investment_curve(pred: np.ndarray, actual_nar: np.ndarray):
    """Rank loans by predicted NAR; return % invested and cumulative mean actual NAR."""
    order = np.argsort(-pred)
    n = len(pred)
    cum_return = np.cumsum(actual_nar[order]) / np.arange(1, n + 1)
    return np.arange(1, n + 1) / n, cum_return, pred[order]


def find_threshold(pred: np.ndarray, actual_nar: np.ndarray,
                   target_return: float = 0.15, min_loans: int = 100) -> float:
    """Loosest threshold M whose selected loans still average >= target_return.

    Mirrors the report: at least ``min_loans`` loans must be invested in.
    """
    _, cum_return, sorted_pred = investment_curve(pred, actual_nar)
    k_min = min(min_loans, len(pred))
    ok = np.where(cum_return[k_min - 1:] >= target_return)[0]
    k = (ok[-1] + k_min) if len(ok) else k_min
    return float(sorted_pred[k - 1])


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """95 % bootstrap confidence interval for the mean return of the selected loans."""
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def simulate(pred: np.ndarray, actual_nar: np.ndarray, threshold: float) -> dict:
    """Invest (equal $ amounts) in every loan with predicted NAR >= threshold."""
    mask = pred >= threshold
    n_sel = int(mask.sum())
    lo, hi = bootstrap_ci(actual_nar[mask])
    return {
        "threshold": threshold,
        "n_selected": n_sel,
        "pct_selected": 100.0 * n_sel / len(pred),
        "avg_annualised_return": float(actual_nar[mask].mean()) if n_sel else float("nan"),
        "ci95_low": lo, "ci95_high": hi,
        "baseline_all_loans_return": float(actual_nar.mean()),
    }


def plot_curves(val: tuple, test: tuple, threshold: float, path: Path) -> None:
    """``val`` / ``test`` are (predicted NAR, actual NAR) arrays."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (title, (pred, nar)) in zip(axes, (("Validation set", val), ("Test set", test))):
        pct, cum, _ = investment_curve(pred, nar)
        ax.plot(100 * pct, 100 * cum, lw=2)
        sel = simulate(pred, nar, threshold)
        ax.axvline(sel["pct_selected"], color="r", ls="--", lw=1,
                   label=f"M={threshold:.3f}: {sel['pct_selected']:.2f}% of loans")
        ax.set(title=f"Investment return trend - {title}",
               xlabel="% of loans invested (ranked by predicted NAR)")
        ax.legend()
    axes[0].set_ylabel("Annualised return (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
