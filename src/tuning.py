"""Hyper-parameter tuning (Optuna) for LightGBM and CatBoost.

Trials are scored on the VALIDATION set (ROC-AUC for classification, R^2 for regression);
early stopping also uses validation data. The test set is never touched here.
The number of boosting rounds is found by early stopping and stored with the best parameters.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from sklearn.metrics import r2_score, roc_auc_score

from . import config as C
from .models import make_catboost, make_lightgbm

log = logging.getLogger(__name__)


def _score(task, model, X_va, y_va) -> float:
    if task == "clf":
        return float(roc_auc_score(y_va, model.predict_proba(X_va)[:, 1]))
    return float(r2_score(y_va, model.predict(X_va)))


def tune_lightgbm(task, X_tr, y_tr, X_va, y_va, n_trials=25, seed=C.RANDOM_STATE) -> dict:
    import lightgbm as lgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            num_leaves=trial.suggest_int("num_leaves", 8, 128, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 300, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 0.8),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 50.0, log=True),
        )
        model = make_lightgbm(task, params, n_estimators=1500, seed=seed)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  eval_metric="auc" if task == "clf" else "l2",
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        trial.set_user_attr("n_estimators", int(model.best_iteration_ or 1500))
        return _score(task, model, X_va, y_va)

    return _run(objective, n_trials, seed, "LightGBM", task)


def tune_catboost(task, X_tr, y_tr, X_va, y_va, n_trials=15, seed=C.RANDOM_STATE) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            depth=trial.suggest_int("depth", 4, 8),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        )
        model = make_catboost(task, params, iterations=1000, seed=seed)
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, use_best_model=True)
        trial.set_user_attr("n_estimators", int(model.get_best_iteration() + 1))
        return _score(task, model, X_va, y_va)

    return _run(objective, n_trials, seed, "CatBoost", task)


def _run(objective, n_trials, seed, name, task) -> dict:
    import optuna
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_trial
    metric = "val ROC-AUC" if task == "clf" else "val R2"
    log.info("[tune] %-8s %-3s best %s=%.4f after %d trials | params=%s | rounds=%d",
             name, task, metric, best.value, n_trials,
             {k: round(v, 4) if isinstance(v, float) else v for k, v in best.params.items()},
             best.user_attrs["n_estimators"])
    return {"params": best.params, "n_estimators": best.user_attrs["n_estimators"],
            "val_score": best.value, "n_trials": n_trials}


def tune_all(task, X_tr, y_tr, X_va, y_va, n_trials, seed=C.RANDOM_STATE) -> dict:
    """Tune whichever of LightGBM / CatBoost is installed."""
    out = {}
    try:
        import lightgbm  # noqa: F401
        out["LightGBM"] = tune_lightgbm(task, X_tr, y_tr, X_va, y_va, n_trials, seed)
    except ImportError:
        log.warning("lightgbm not installed - skipping")
    try:
        import catboost  # noqa: F401
        out["CatBoost"] = tune_catboost(task, X_tr, y_tr, X_va, y_va, max(5, n_trials // 2), seed)
    except ImportError:
        log.warning("catboost not installed - skipping")
    return out


def save_params(tuned: dict, path: Path) -> None:
    path.write_text(json.dumps(tuned, indent=2, default=float))
