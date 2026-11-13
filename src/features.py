"""Feature preprocessing pipeline (fit on the training split only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config as C


class MaxFactorImputer(BaseEstimator, TransformerMixin):
    """'Max-set' imputation: missing -> ``factor`` x column maximum.

    Used for 'months since last ...' fields, where a missing value means the
    event never happened, so the applicant should not be penalised.
    """

    def __init__(self, factor: float = C.MAX_FACTOR):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        with np.errstate(all="ignore"):
            fill = self.factor * np.nanmax(X, axis=0) if X.size else np.zeros(X.shape[1])
        self.fill_ = np.where(np.isnan(fill), 0.0, fill)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        rows, cols = np.where(np.isnan(X))
        X[rows, cols] = self.fill_[cols]
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def _to_days(X) -> np.ndarray:
    """'Mar-2004' style dates -> days since 1970-01-01 (NaN if unparseable)."""
    X = pd.DataFrame(X)
    out = X.apply(lambda c: (pd.to_datetime(c, format="%b-%Y", errors="coerce")
                             - pd.Timestamp("1970-01-01")).dt.days)
    return out.to_numpy(dtype=float)


def _to_float32(X) -> np.ndarray:
    return np.asarray(X, dtype=np.float32)


def build_preprocessor(columns: list[str]) -> Pipeline:
    """Impute -> one-hot -> drop constant columns -> standardise (mean 0, std 1)."""
    pick = lambda group: [c for c in group if c in columns]  # noqa: E731
    column_transformer = ColumnTransformer(
        [
            ("num_mean", SimpleImputer(strategy="mean"), pick(C.NUMERIC_MEAN)),
            ("num_zero", SimpleImputer(strategy="constant", fill_value=0.0), pick(C.NUMERIC_ZERO)),
            ("num_max", MaxFactorImputer(), pick(C.NUMERIC_MAX)),
            ("date", Pipeline([("days", FunctionTransformer(_to_days, feature_names_out="one-to-one")),
                               ("impute", SimpleImputer(strategy="mean"))]), pick(C.DATE_FEATURES)),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
             pick(C.CATEGORICAL)),
        ],
        sparse_threshold=0.0,
    )
    return Pipeline([
        ("columns", column_transformer),
        ("drop_constant", VarianceThreshold(0.0)),
        ("scale", StandardScaler()),
        ("float32", FunctionTransformer(_to_float32)),
    ])
