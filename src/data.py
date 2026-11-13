"""Loading, cleaning, labelling and splitting the LendingClub data."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C

log = logging.getLogger(__name__)
DATE_FMT = "%b-%Y"   # e.g. "Dec-2014"


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Feature columns actually present in ``df`` (datasets vary slightly by vintage)."""
    wanted = C.NUMERIC_MEAN + C.NUMERIC_ZERO + C.NUMERIC_MAX + C.CATEGORICAL + C.DATE_FEATURES
    return [c for c in wanted if c in df.columns]


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read only the columns we need (the full CSV has ~150 columns / 2M rows)."""
    path = Path(path)
    header = pd.read_csv(path, nrows=0).columns
    wanted = set(C.NUMERIC_MEAN + C.NUMERIC_ZERO + C.NUMERIC_MAX + C.CATEGORICAL
                 + C.DATE_FEATURES + C.LABEL_SOURCE_COLUMNS)
    usecols = [c for c in header if c in wanted]
    missing = set(C.LABEL_SOURCE_COLUMNS) - set(usecols)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    log.info("Reading %d columns from %s", len(usecols), path.name)
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def compute_nar(total_payment, loan_amount, days,
                min_days: int = C.NAR_MIN_DAYS, clip=C.NAR_CLIP) -> np.ndarray:
    """Net annualised return: (total_payment / loan_amount) ** (365 / days) - 1."""
    days = np.asarray(days, dtype=float)
    days = np.where(np.isnan(days), min_days, np.maximum(days, min_days))
    ratio = np.asarray(total_payment, dtype=float) / np.asarray(loan_amount, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        nar = np.power(ratio, 365.0 / days) - 1.0
    nar = np.nan_to_num(nar, nan=clip[0], posinf=clip[1], neginf=clip[0])
    return np.clip(nar, *clip)


def prepare_dataset(raw: pd.DataFrame, years=C.ISSUE_YEARS,
                    sample_size: int | None = None, seed: int = C.RANDOM_STATE) -> pd.DataFrame:
    """Keep finalised loans, create ``label`` (1 = Fully Paid) and ``nar`` columns."""
    df = raw.copy()
    status = df["loan_status"].astype(str).str.replace(C.CREDIT_POLICY_PREFIX, "", regex=False).str.strip()
    final = status.isin([C.POSITIVE_STATUS, *C.NEGATIVE_STATUSES])
    df, status = df[final].copy(), status[final]
    df["label"] = (status == C.POSITIVE_STATUS).astype(int).to_numpy()

    issue = pd.to_datetime(df["issue_d"], format=DATE_FMT, errors="coerce")
    in_range = issue.dt.year.between(*years)
    df, issue = df[in_range].copy(), issue[in_range]
    log.info("Finalised loans issued %d-%d: %d (%.1f%% negative)",
             *years, len(df), 100 * (1 - df["label"].mean()))

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed)
        issue = issue.loc[df.index]
        log.info("Sub-sampled to %d loans", len(df))

    last = pd.to_datetime(df["last_pymnt_d"], format=DATE_FMT, errors="coerce")
    days = (last - issue).dt.days
    df["nar"] = compute_nar(df["total_pymnt"], df["loan_amnt"], days)

    for col in C.PERCENT_STRING_COLUMNS:           # older files store "13.5%"
        if col in df.columns and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col].astype(str).str.rstrip("% "), errors="coerce")
    for col in C.CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].astype(object).where(df[col].notna(), "missing").astype(str)

    return df.drop(columns=["loan_status", "issue_d", "total_pymnt", "last_pymnt_d"]).reset_index(drop=True)


def split(df: pd.DataFrame, test_size: float = C.TEST_SIZE, val_size: float = C.VAL_SIZE,
          seed: int = C.RANDOM_STATE):
    """Stratified train / validation / test split.

    Test = 30 % of all loans (as in the report). Validation = 15 % of the remaining 70 %.
    Validation data is used for model selection, probability cut-offs and the strategy
    threshold, so the test set is only used for the final, honest evaluation.
    """
    train_val, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)
    train, val = train_test_split(train_val, test_size=val_size,
                                  stratify=train_val["label"], random_state=seed)
    return train, val, test
