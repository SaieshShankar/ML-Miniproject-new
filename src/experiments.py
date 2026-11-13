"""Experiment stages: classification, regression and loan-selection simulation."""
from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix

from . import config as C
from . import evaluate as E
from . import strategy as S
from .explain import TREE_MODELS
from .models import get_classifiers, get_regressors

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def run_classification(X_tr, y_tr, X_va, y_va, X_te, y_te, seed=C.RANDOM_STATE, tuned=None,
                       results_dir: Path = C.RESULTS_DIR, models_dir: Path = C.MODELS_DIR):
    """Train the classifiers; cut-offs and model selection use VALIDATION data only."""
    rows, matrices, roc_inputs, fitted, cutoffs = [], {}, {}, {}, {}
    splits = (("train", X_tr, y_tr), ("val", X_va, y_va), ("test", X_te, y_te))
    for name, model in get_classifiers(X_tr.shape[1], seed, tuned).items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        proba = {s: model.predict_proba(X)[:, 1] for s, X, _ in splits}
        cutoff = E.best_cutoff(y_va, proba["val"])
        for s, _, y in splits:
            rows.append(E.classification_row(name, s, y, proba[s], cutoff))
        matrices[name] = confusion_matrix(y_te, (proba["test"] >= cutoff).astype(int))
        roc_inputs[name] = (y_te, proba["test"])
        fitted[name], cutoffs[name] = model, cutoff
        t = rows[-1]
        log.info("[clf] %-20s test: AUC=%.3f | default recall=%.2f precision=%.2f | "
                 "weighted F1=%.3f | cut-off=%.2f (%.0fs)", name, t["roc_auc"], t["recall_default"],
                 t["precision_default"], t["f1_weighted"], cutoff, time.time() - t0)

    table = pd.DataFrame(rows)
    table.to_csv(results_dir / "classification_results.csv", index=False)
    E.plot_confusion_matrices(matrices, results_dir / "confusion_matrices.png")
    E.plot_roc(roc_inputs, results_dir / "roc_curves.png")

    val = table[table["split"] == "val"].set_index("model")
    best = val["roc_auc"].idxmax()                      # selected on validation AUC
    joblib.dump(fitted[best], models_dir / "best_classifier.joblib", compress=3)
    log.info("[clf] best model (validation AUC): %s", best)

    # SHAP needs a tree model: use the best one if it is a tree model, else the best tree model
    trees = [m for m in val.index if m in TREE_MODELS]
    explain_name = best if best in TREE_MODELS else (val.loc[trees, "roc_auc"].idxmax() if trees else None)
    explain_file = "best_classifier.joblib"
    if explain_name and explain_name != best:
        joblib.dump(fitted[explain_name], models_dir / "shap_model.joblib", compress=3)
        explain_file = "shap_model.joblib"
    return table, best, cutoffs[best], {"name": explain_name, "file": explain_file}


def run_regression(X_tr, nar_tr, X_va, nar_va, X_te, nar_te, seed=C.RANDOM_STATE, tuned=None,
                   results_dir: Path = C.RESULTS_DIR, models_dir: Path = C.MODELS_DIR):
    """Train the NAR regressors; best model is chosen on validation R^2."""
    rows, fitted = [], {}
    for name, model in get_regressors(seed, tuned).items():
        t0 = time.time()
        model.fit(X_tr, nar_tr)
        p_tr, p_va, p_te = model.predict(X_tr), model.predict(X_va), model.predict(X_te)
        rows.append(E.regression_row(name, nar_tr, p_tr, nar_va, p_va, nar_te, p_te))
        fitted[name] = (model, p_va, p_te)
        log.info("[reg] %-28s test MSE=%.4f R2=%.3f (val R2=%.3f) (%.0fs)",
                 name, rows[-1]["test_mse"], rows[-1]["test_r2"], rows[-1]["val_r2"], time.time() - t0)

    table = pd.DataFrame(rows)
    table.to_csv(results_dir / "regression_results.csv", index=False)
    best = table.loc[table["val_r2"].idxmax(), "model"]   # selected on validation R^2
    model, p_va, p_te = fitted[best]
    joblib.dump(model, models_dir / "best_regressor.joblib", compress=3)
    log.info("[reg] best model (validation R2): %s", best)
    return table, best, p_va, p_te


def run_strategy(pred_va, nar_va, pred_te, nar_te, target_return=0.05, min_loans=100,
                 results_dir: Path = C.RESULTS_DIR) -> dict:
    """Pick threshold M on VALIDATION loans, then evaluate it once on the test set."""
    threshold = S.find_threshold(pred_va, nar_va, target_return, min_loans)
    result = {"val": S.simulate(pred_va, nar_va, threshold),
              "test": S.simulate(pred_te, nar_te, threshold)}
    S.plot_curves((pred_va, nar_va), (pred_te, nar_te), threshold,
                  results_dir / "investment_curves.png")
    pd.DataFrame(result).T.to_csv(results_dir / "strategy_results.csv")
    for split in ("val", "test"):
        r = result[split]
        log.info("[strategy] %-4s M=%.3f -> %d loans (%.2f%%), avg annualised return %.1f%% "
                 "[95%% CI %.1f%% to %.1f%%] (all-loans baseline %.1f%%)", split, r["threshold"],
                 r["n_selected"], r["pct_selected"], 100 * r["avg_annualised_return"],
                 100 * r["ci95_low"], 100 * r["ci95_high"], 100 * r["baseline_all_loans_return"])
    return result
