#!/usr/bin/env python
"""LendingClub loan default & profitability prediction - command line entry point.

Examples
--------
python main.py --data data/accepted_2007_to_2018Q4.csv.gz
python main.py --data data/sample_loans.csv --stages classify regress strategy
"""
from __future__ import annotations

import argparse
import json
import logging

import joblib
import numpy as np

from src import config as C
from src.data import feature_columns, load_raw, prepare_dataset, split
from src.experiments import run_classification, run_regression, run_strategy
from src.explain import build_summary, run_explain
from src.features import build_preprocessor
from src.tuning import save_params, tune_all


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="Path to LendingClub accepted-loans CSV (.csv or .csv.gz)")
    p.add_argument("--stages", nargs="+", default=["classify", "regress", "strategy", "explain"],
                   choices=["classify", "regress", "strategy", "explain"])
    p.add_argument("--sample-size", type=int, default=100_000,
                   help="Random sub-sample of finalised loans (0 = use everything; needs lots of RAM)")
    p.add_argument("--years", type=int, nargs=2, default=list(C.ISSUE_YEARS), metavar=("FROM", "TO"))
    p.add_argument("--target-return", type=float, default=0.05, help="Target annualised return for threshold M")
    p.add_argument("--min-pct", type=float, default=1.0,
                   help="Invest in at least this %% of loans (keeps the strategy statistically meaningful)")
    p.add_argument("--tune", action="store_true",
                   help="Tune LightGBM / CatBoost with Optuna on the validation set (adds time)")
    p.add_argument("--trials", type=int, default=25,
                   help="Optuna trials for LightGBM (CatBoost gets half); only used with --tune")
    p.add_argument("--shap-samples", type=int, default=1000,
                   help="Number of test loans used for the SHAP summary (explain stage)")
    p.add_argument("--seed", type=int, default=C.RANDOM_STATE)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("main")
    for d in (C.MODELS_DIR, C.RESULTS_DIR):
        d.mkdir(exist_ok=True)

    # 1. data ---------------------------------------------------------------
    df = prepare_dataset(load_raw(args.data), tuple(args.years), args.sample_size or None, args.seed)
    cols = feature_columns(df)
    train_df, val_df, test_df = split(df, seed=args.seed)
    log.info("Train: %d | Validation: %d | Test: %d loans | raw features: %d",
             len(train_df), len(val_df), len(test_df), len(cols))

    # 2. features (fit on train only -> no leakage) --------------------------
    pre = build_preprocessor(cols)
    X_tr = pre.fit_transform(train_df[cols])
    X_va = pre.transform(val_df[cols])
    X_te = pre.transform(test_df[cols])
    log.info("Feature matrix after preprocessing: %d features", X_tr.shape[1])
    joblib.dump(pre, C.MODELS_DIR / "preprocessor.joblib", compress=3)

    y_tr, y_va, y_te = (d["label"].to_numpy() for d in (train_df, val_df, test_df))
    nar_tr, nar_va, nar_te = (d["nar"].to_numpy() for d in (train_df, val_df, test_df))
    meta = {"feature_columns": cols, "n_features": int(X_tr.shape[1])}

    # 3. stages -------------------------------------------------------------
    tuned_clf = tuned_reg = None
    if args.tune:
        if "classify" in args.stages:
            tuned_clf = tune_all("clf", X_tr, y_tr, X_va, y_va, args.trials, args.seed)
        if "regress" in args.stages:
            tuned_reg = tune_all("reg", X_tr, nar_tr, X_va, nar_va, args.trials, args.seed)
        save_params({"classification": tuned_clf, "regression": tuned_reg},
                    C.RESULTS_DIR / "tuning_best_params.json")

    clf_table = reg_table = strategy_result = shap_table = None
    meta["base_rate"] = float(y_tr.mean())          # share of loans that were fully repaid

    if "classify" in args.stages:
        clf_table, best_clf, cutoff, shap_info = run_classification(
            X_tr, y_tr, X_va, y_va, X_te, y_te, args.seed, tuned_clf)
        meta["best_classifier"], meta["classifier_cutoff"] = best_clf, cutoff
        meta["shap_model"], meta["shap_model_file"] = shap_info["name"], shap_info["file"]

    p_va = p_te = None
    if "regress" in args.stages:
        reg_table, best_reg, p_va, p_te = run_regression(X_tr, nar_tr, X_va, nar_va, X_te, nar_te, args.seed, tuned_reg)
        meta["best_regressor"] = best_reg

    if "strategy" in args.stages:
        if p_va is None:                                   # reuse a previously saved regressor
            reg = joblib.load(C.MODELS_DIR / "best_regressor.joblib")
            p_va, p_te = reg.predict(X_va), reg.predict(X_te)
        min_loans = max(50, int(len(val_df) * args.min_pct / 100))
        strategy_result = result = run_strategy(p_va, nar_va, p_te, nar_te, args.target_return, min_loans)
        meta["threshold"] = result["val"]["threshold"]
        meta["strategy"] = result

    meta_path = C.MODELS_DIR / "meta.json"
    old = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    if "explain" in args.stages:
        info = {**old, **meta}
        if not info.get("shap_model"):
            log.warning("[explain] no tree model available (run the classify stage first) - skipping SHAP")
        else:
            try:
                model = joblib.load(C.MODELS_DIR / info["shap_model_file"])
                shap_table = run_explain(pre, model, info["shap_model"], X_te, args.shap_samples, args.seed)
            except ImportError:
                log.warning("[explain] the 'shap' package is not installed (pip install shap) - skipping")

    summary = build_summary({**old, **meta}, (len(train_df), len(val_df), len(test_df)), meta["base_rate"],
                            clf_table, reg_table, strategy_result, shap_table)
    (C.RESULTS_DIR / "summary.txt").write_text(summary, encoding="utf-8")
    log.info("\n%s", summary)
    meta_path.write_text(json.dumps({**old, **meta}, indent=2, default=float))
    log.info("Done. Tables and plots are in %s", C.RESULTS_DIR)


if __name__ == "__main__":
    main()
