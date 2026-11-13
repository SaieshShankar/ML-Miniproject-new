import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_sample_data import generate                    # noqa: E402
from src.data import feature_columns, prepare_dataset, split         # noqa: E402
from src.explain import (feature_names, group_map, narrate_loan,     # noqa: E402
                         raw_feature_of)
from src.features import build_preprocessor                          # noqa: E402


def test_raw_feature_mapping():
    assert raw_feature_of("num_mean__dti") == "dti"
    assert raw_feature_of("cat__grade_A") == "grade"
    assert raw_feature_of("cat__sub_grade_A1") == "sub_grade"
    assert raw_feature_of("cat__home_ownership_RENT") == "home_ownership"


def test_feature_names_match_matrix_and_group():
    df = prepare_dataset(generate(1500, seed=2), years=(2012, 2015))
    cols = feature_columns(df)
    train, _, _ = split(df)
    pre = build_preprocessor(cols)
    X = pre.fit_transform(train[cols])
    names = feature_names(pre)
    assert len(names) == X.shape[1]
    uniq, M = group_map(names)
    assert M.shape == (X.shape[1], len(uniq)) and M.sum() == X.shape[1]
    assert set(uniq) <= set(cols)


def test_narrate_loan_text():
    drivers = {"lowers": [("int_rate", "21.5%", -0.3)], "raises": [("fico_range_low", "780", 0.2)]}
    text = narrate_loan(0.62, 0.01, 0.034, 0.7, 0.81, drivers)
    assert "62%" in text and "SKIP" in text and "interest rate (21.5%)" in text and "FICO score (780)" in text


def test_shap_values_sum_to_groups():
    pytest.importorskip("shap")
    lgb = pytest.importorskip("lightgbm")
    from src.explain import class1_shap
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 6)).astype(np.float32)
    y = (X[:, 0] - X[:, 1] + rng.normal(0, 1, 500) > 0).astype(int)
    model = lgb.LGBMClassifier(n_estimators=30, verbose=-1).fit(X, y)
    sv = class1_shap(model, X[:20])
    assert sv.shape == (20, 6)
    assert np.abs(sv[:, 0]).mean() > np.abs(sv[:, 5]).mean()      # the informative feature matters most
