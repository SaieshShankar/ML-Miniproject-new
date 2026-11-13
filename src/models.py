"""Model factories. Paper models + a few documented improvements (marked 'EXTENSION')."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor

from . import config as C


def _has(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def make_lightgbm(task: str, params: dict | None = None, n_estimators: int = 300,
                  seed: int = C.RANDOM_STATE):
    """LightGBM classifier ('clf') or regressor ('reg'). Defaults are sensible, untuned values."""
    import lightgbm as lgb
    base = dict(learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0)
    base.update(params or {})
    common = dict(n_estimators=n_estimators, random_state=seed, n_jobs=-1, verbose=-1, **base)
    if task == "clf":
        return lgb.LGBMClassifier(class_weight="balanced", **common)
    return lgb.LGBMRegressor(**common)


def make_catboost(task: str, params: dict | None = None, iterations: int = 500,
                  seed: int = C.RANDOM_STATE):
    """CatBoost classifier ('clf') or regressor ('reg')."""
    from catboost import CatBoostClassifier, CatBoostRegressor
    base = dict(learning_rate=0.05, depth=6, l2_leaf_reg=3.0)
    base.update(params or {})
    common = dict(iterations=iterations, random_seed=seed, thread_count=-1, verbose=0,
                  allow_writing_files=False, **base)
    if task == "clf":
        return CatBoostClassifier(auto_class_weights="Balanced", **common)
    return CatBoostRegressor(loss_function="RMSE", **common)


def _boosters(task: str, seed: int, tuned: dict | None) -> dict:
    """Default + (optionally) tuned LightGBM / CatBoost models, skipping uninstalled libraries."""
    out, tuned = {}, tuned or {}
    if _has("lightgbm"):
        out["LightGBM"] = make_lightgbm(task, seed=seed)
        if "LightGBM" in tuned:
            t = tuned["LightGBM"]
            out["LightGBM (tuned)"] = make_lightgbm(task, t["params"], t["n_estimators"], seed)
    if _has("catboost"):
        out["CatBoost"] = make_catboost(task, seed=seed)
        if "CatBoost" in tuned:
            t = tuned["CatBoost"]
            out["CatBoost (tuned)"] = make_catboost(task, t["params"], t["n_estimators"], seed)
    return out


class BalancedResampler(BaseEstimator, ClassifierMixin):
    """Random-oversamples the minority class before fitting ``estimator``.

    sklearn's MLPClassifier has no ``class_weight``; without balancing, the network in
    the first run collapsed to predicting "Paid Off" for every loan.
    """

    def __init__(self, estimator, random_state: int = 0):
        self.estimator = estimator
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        y = np.asarray(y)
        groups = sorted([np.where(y == 0)[0], np.where(y == 1)[0]], key=len)
        minority, majority = groups
        extra = rng.choice(minority, size=len(majority) - len(minority), replace=True)
        idx = np.concatenate([majority, minority, extra])
        rng.shuffle(idx)
        self.estimator_ = clone(self.estimator).fit(X[idx], y[idx])
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.estimator_.predict(X)


def get_classifiers(n_features: int, seed: int = C.RANDOM_STATE, tuned: dict | None = None) -> dict:
    models = {
        "LogisticRegression": LogisticRegression(          # paper: L2, balanced weights
            class_weight="balanced", max_iter=1000),
        "NeuralNetwork": BalancedResampler(                # paper architecture, + oversampling
            MLPClassifier(hidden_layer_sizes=(10, 10, 5, 3), activation="relu", batch_size=200,
                          learning_rate_init=0.001, alpha=1e-4, max_iter=40, random_state=seed),
            random_state=seed),
        "RandomForest": RandomForestClassifier(            # paper: 200 trees, Gini, <=50 features
            n_estimators=200, criterion="gini", max_features=min(50, n_features),
            min_samples_leaf=20, class_weight="balanced_subsample",   # EXTENSION: class balance
            n_jobs=-1, random_state=seed),
        "GradientBoosting": HistGradientBoostingClassifier(   # EXTENSION (not in the paper)
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            class_weight="balanced", early_stopping=True, random_state=seed),
    }
    models.update(_boosters("clf", seed, tuned))       # EXTENSION: LightGBM / CatBoost
    return models


def get_regressors(seed: int = C.RANDOM_STATE, tuned: dict | None = None) -> dict:
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge(alpha=1)": Ridge(alpha=1.0),
        "NeuralNetwork": MLPRegressor(                     # paper: ReLU + Adam, squared loss
            hidden_layer_sizes=(20, 10, 5, 3), activation="relu", solver="adam",
            batch_size=200, alpha=1e-4, early_stopping=True, random_state=seed),
    }
    for depth in (4, 8, 10):
        models[f"RandomForest(depth={depth})"] = RandomForestRegressor(
            n_estimators=100, max_depth=depth, max_features="sqrt", n_jobs=-1, random_state=seed)
    models["GradientBoosting"] = HistGradientBoostingRegressor(   # EXTENSION (not in the paper)
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
        early_stopping=True, random_state=seed)
    models.update(_boosters("reg", seed, tuned))       # EXTENSION: LightGBM / CatBoost
    return models
