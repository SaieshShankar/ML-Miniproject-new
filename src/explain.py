"""Light-weight explainability: grouped SHAP values and plain-English explanations.

SHAP values are computed on tree models (Random Forest / LightGBM / CatBoost). The ~1,000
one-hot columns are summed back into the ~30 original loan fields so results are readable.
SHAP shows what the model *uses*, not what *causes* default.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

log = logging.getLogger(__name__)

TREE_MODELS = {"RandomForest", "LightGBM", "LightGBM (tuned)", "CatBoost", "CatBoost (tuned)"}

FRIENDLY = {
    "int_rate": "interest rate", "grade": "LendingClub grade", "sub_grade": "LendingClub sub-grade",
    "dti": "debt-to-income ratio", "annual_inc": "annual income", "loan_amnt": "loan amount",
    "funded_amnt": "funded amount", "installment": "monthly instalment", "term": "loan term",
    "fico_range_low": "FICO score", "fico_range_high": "FICO score (upper)",
    "revol_util": "revolving credit utilisation", "revol_bal": "revolving balance",
    "open_acc": "open credit lines", "total_acc": "total credit lines", "inq_last_6mths": "recent credit enquiries",
    "delinq_2yrs": "recent delinquencies", "pub_rec": "public records", "emp_length": "employment length",
    "home_ownership": "home ownership", "verification_status": "income verification",
    "purpose": "loan purpose", "addr_state": "state", "zip_code": "zip code",
    "earliest_cr_line": "length of credit history", "mths_since_last_delinq": "months since last delinquency",
    "initial_list_status": "listing status", "application_type": "application type",
}


def friendly(feature: str) -> str:
    return FRIENDLY.get(feature, feature.replace("_", " "))


def _fmt(feature: str, value) -> str:
    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
        if feature in ("int_rate", "revol_util"):
            return f"{value:.1f}%"
        if feature in ("loan_amnt", "annual_inc", "funded_amnt", "revol_bal", "installment"):
            return f"${value:,.0f}"
        return f"{value:g}"
    return str(value).strip()


# --------------------------------------------------------------------------- features
def feature_names(pre) -> np.ndarray:
    """Names of the columns the fitted preprocessor outputs."""
    names = pre.named_steps["columns"].get_feature_names_out()
    return np.asarray(names)[pre.named_steps["drop_constant"].get_support()]


def raw_feature_of(name: str) -> str:
    """'num_mean__dti' -> 'dti';  'cat__grade_A' -> 'grade'."""
    rest = name.split("__", 1)[1]
    if name.startswith("cat__"):
        for col in sorted(C.CATEGORICAL, key=len, reverse=True):
            if rest.startswith(col + "_"):
                return col
    return rest


def group_map(names) -> tuple[list[str], np.ndarray]:
    """Indicator matrix (n_encoded x n_raw) that sums encoded columns into raw loan fields."""
    raws = [raw_feature_of(n) for n in names]
    uniq = list(dict.fromkeys(raws))
    M = np.zeros((len(raws), len(uniq)))
    M[np.arange(len(raws)), [uniq.index(r) for r in raws]] = 1.0
    return uniq, M


# --------------------------------------------------------------------------- SHAP
def class1_shap(model, X) -> np.ndarray:
    """SHAP values for the 'Paid Off' class. Positive = raises chance of repayment."""
    import shap
    sv = shap.TreeExplainer(model).shap_values(X, check_additivity=False)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv


def run_explain(pre, model, model_name: str, X_test, n_samples: int = 1000,
                seed: int = C.RANDOM_STATE, results_dir: Path = C.RESULTS_DIR) -> pd.DataFrame:
    """Global importance: mean |SHAP| per original loan field, saved as a plot and a CSV."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_test), size=min(n_samples, len(X_test)), replace=False)
    sv = class1_shap(model, X_test[idx])
    uniq, M = group_map(feature_names(pre))
    grouped = sv @ M
    table = (pd.DataFrame({"feature": uniq, "mean_abs_shap": np.abs(grouped).mean(axis=0)})
             .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    table["label"] = table["feature"].map(friendly)
    table.to_csv(results_dir / "shap_top_features.csv", index=False)

    top = table.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["label"], top["mean_abs_shap"], color="#1f77b4")
    ax.set(xlabel="Mean |SHAP value| (average impact on the model's output)",
           title=f"What drives default risk? ({model_name}, {len(idx)} test loans)")
    fig.tight_layout()
    fig.savefig(results_dir / "shap_summary.png", dpi=150)
    plt.close(fig)
    log.info("[explain] top drivers: %s", ", ".join(table["label"].head(5)))
    return table


def explain_loan(model, pre, X_row, raw_row: pd.DataFrame, only: set | None = None, top: int = 3) -> dict:
    """Top fields raising / lowering the repayment probability for one loan."""
    uniq, M = group_map(feature_names(pre))
    g = (class1_shap(model, X_row) @ M)[0]
    items = [(f, g[i]) for i, f in enumerate(uniq) if only is None or f in only]
    pack = lambda xs: [(f, _fmt(f, raw_row[f].iloc[0]) if f in raw_row else "n/a", float(v)) for f, v in xs]  # noqa: E731
    lowers = sorted([x for x in items if x[1] < 0], key=lambda x: x[1])[:top]
    raises = sorted([x for x in items if x[1] > 0], key=lambda x: -x[1])[:top]
    return {"lowers": pack(lowers), "raises": pack(raises)}


# --------------------------------------------------------------------------- plain English
def _list(drivers) -> str:
    return ", ".join(f"{friendly(f)} ({v})" for f, v, _ in drivers)


def narrate_loan(p_paid: float, nar: float, threshold: float | None, cutoff: float,
                 base_rate: float, drivers: dict) -> str:
    """One short paragraph explaining a single loan's result."""
    rel = "above" if p_paid >= base_rate else "below"
    text = (f"The model estimates a {p_paid:.0%} chance that this loan is fully repaid, {rel} the "
            f"{base_rate:.0%} average in the training data. ")
    text += ("It is **not flagged** as high default risk. " if p_paid >= cutoff
             else "It is **flagged as high default risk**. ")
    text += f"The predicted annualised return is {nar:.1%}"
    if threshold is not None:
        text += (f", {'above' if nar >= threshold else 'below'} the strategy threshold of {threshold:.1%}, "
                 f"so the strategy says **{'INVEST' if nar >= threshold else 'SKIP'}**")
    text += ". "
    if drivers["lowers"]:
        text += f"Factors pushing the repayment chance down the most: {_list(drivers['lowers'])}. "
    if drivers["raises"]:
        text += f"Factors pushing it up the most: {_list(drivers['raises'])}. "
    return text.strip()


def build_summary(meta: dict, n_split: tuple, base_rate: float, clf_table=None,
                  reg_table=None, strategy=None, shap_table=None) -> str:
    """Plain-English summary of a full run (saved to results/summary.txt)."""
    lines = [f"Run summary: {n_split[0]:,} training, {n_split[1]:,} validation and {n_split[2]:,} test loans; "
             f"{1 - base_rate:.1%} of loans defaulted."]
    if clf_table is not None and meta.get("best_classifier"):
        t = clf_table[(clf_table["model"] == meta["best_classifier"]) & (clf_table["split"] == "test")].iloc[0]
        lift = t["precision_default"] / (1 - base_rate)
        lines.append(
            f"Classification: the best model ({meta['best_classifier']}) reaches a test ROC-AUC of {t['roc_auc']:.3f}. "
            f"It catches {t['recall_default']:.0%} of defaulting loans; when it flags a loan, "
            f"{t['precision_default']:.0%} actually default, about {lift:.1f}x the overall rate.")
    if reg_table is not None and meta.get("best_regressor"):
        r = reg_table[reg_table["model"] == meta["best_regressor"]].iloc[0]
        lines.append(
            f"Return prediction: the best regressor ({meta['best_regressor']}) has a test R2 of {r['test_r2']:.3f}, "
            "so it explains only a small share of the variation in returns; its value is in ranking loans, not exact forecasts.")
    if strategy:
        s, v = strategy["test"], strategy["val"]
        lines.append(
            f"Strategy: investing in loans with predicted return >= {s['threshold']:.3f} selects {s['pct_selected']:.1f}% "
            f"of test loans ({s['n_selected']:,}) and earned {s['avg_annualised_return']:.1%} a year "
            f"(95% CI {s['ci95_low']:.1%} to {s['ci95_high']:.1%}) versus {s['baseline_all_loans_return']:.1%} for the average loan. "
            f"On validation loans the same rule earned {v['avg_annualised_return']:.1%}.")
    if shap_table is not None and len(shap_table):
        lines.append("Main drivers of predicted default risk (SHAP): "
                     + ", ".join(shap_table["label"].head(5)) + ".")
    lines.append("Caveat: SHAP shows what the model relies on, not what causes default; returns are equal-weighted "
                 "averages of per-loan annualised returns, not a portfolio return.")
    return "\n\n".join(lines)
