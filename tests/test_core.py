import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_sample_data import generate            # noqa: E402
from src import strategy as S                                # noqa: E402
from src.data import compute_nar, feature_columns, prepare_dataset, split  # noqa: E402
from src.features import MaxFactorImputer, build_preprocessor  # noqa: E402


def test_nar_fully_paid_one_year():
    assert compute_nar([1100], [1000], [365])[0] == pytest.approx(0.10)


def test_nar_total_loss_and_missing_days():
    out = compute_nar([0, 500], [1000, 1000], [200, np.nan])
    assert out[0] == -1.0 and -1.0 <= out[1] < 0


def test_max_factor_imputer():
    X = np.array([[1.0], [4.0], [np.nan]])
    out = MaxFactorImputer(factor=2.0).fit(X).transform(X)
    assert out[2, 0] == 8.0


def test_threshold_respects_min_loans_and_target():
    rng = np.random.default_rng(0)
    pred = rng.normal(size=2000)
    actual = pred * 0.1 + rng.normal(0, 0.02, 2000)
    m = S.find_threshold(pred, actual, target_return=0.1, min_loans=50)
    res = S.simulate(pred, actual, m)
    assert res["n_selected"] >= 50 and res["avg_annualised_return"] >= 0.1


def test_pipeline_end_to_end_no_nans():
    df = prepare_dataset(generate(1500, seed=1), years=(2012, 2015))
    assert set(df["label"].unique()) == {0, 1}
    cols = feature_columns(df)
    train, val, test = split(df)
    assert len(train) + len(val) + len(test) == len(df)
    pre = build_preprocessor(cols)
    X_tr, X_te = pre.fit_transform(train[cols]), pre.transform(test[cols])
    assert X_tr.shape[1] == X_te.shape[1] and not np.isnan(X_tr).any() and not np.isnan(X_te).any()


def test_balanced_resampler_does_not_collapse():
    from sklearn.linear_model import LogisticRegression
    from src.models import BalancedResampler
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 3))
    y = (X[:, 0] + rng.normal(0, 1, 2000) > 1.2).astype(int)          # ~12 % positives
    model = BalancedResampler(LogisticRegression()).fit(X, y)
    assert model.predict(X).mean() > 0.2                               # predicts the minority too


def test_best_cutoff_and_bootstrap():
    from src.evaluate import best_cutoff
    rng = np.random.default_rng(1)
    y = (rng.random(3000) < 0.2).astype(int)
    score = np.clip(0.15 + 0.5 * y + rng.normal(0, 0.1, 3000), 0, 1)
    assert 0.25 < best_cutoff(y, score) < 0.55
    lo, hi = S.bootstrap_ci(np.array([0.1, 0.2, 0.3, 0.2]))
    assert lo <= 0.2 <= hi


def test_boosters_train_and_predict():
    pytest.importorskip("lightgbm")
    pytest.importorskip("catboost")
    from src.models import make_catboost, make_lightgbm
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 5)).astype(np.float32)
    y = (X[:, 0] + rng.normal(0, 1, 600) > 0).astype(int)
    for make in (make_lightgbm, make_catboost):
        proba = make("clf").fit(X, y).predict_proba(X)[:, 1]
        assert proba.shape == (600,) and 0 <= proba.min() <= proba.max() <= 1
        assert make("reg").fit(X, X[:, 0]).predict(X).shape == (600,)


def test_tuning_smoke():
    pytest.importorskip("lightgbm")
    pytest.importorskip("catboost")
    pytest.importorskip("optuna")
    from src.tuning import tune_all
    rng = np.random.default_rng(1)
    X = rng.normal(size=(800, 6)).astype(np.float32)
    y = (X[:, 0] + rng.normal(0, 1, 800) > 0).astype(int)
    out = tune_all("clf", X[:500], y[:500], X[500:], y[500:], n_trials=2)
    assert set(out) == {"LightGBM", "CatBoost"} and out["LightGBM"]["val_score"] > 0.5
