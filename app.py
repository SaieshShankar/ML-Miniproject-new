"""Streamlit demo: score a single loan and explain the result in plain English.
Run:  streamlit run app.py
(requires `python main.py ...` to have been run first so that models/ is populated)
"""
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src import config as C
from src.explain import explain_loan, narrate_loan

st.set_page_config(page_title="LendingClub Credit Risk", page_icon="💳")
st.title("💳 LendingClub Loan Risk Scorer")
st.caption("Classifier: probability the loan is fully repaid. Regressor: expected net annualised return (NAR). "
           "SHAP: which loan details pushed the result up or down.")


@st.cache_resource
def load_artifacts():
    meta = json.loads((C.MODELS_DIR / "meta.json").read_text())
    pre = joblib.load(C.MODELS_DIR / "preprocessor.joblib")
    clf = joblib.load(C.MODELS_DIR / "best_classifier.joblib")
    reg = joblib.load(C.MODELS_DIR / "best_regressor.joblib")
    shap_file = meta.get("shap_model_file")
    shap_model = joblib.load(C.MODELS_DIR / shap_file) if shap_file and (C.MODELS_DIR / shap_file).exists() else None
    return pre, clf, reg, shap_model, meta


try:
    pre, clf, reg, shap_model, meta = load_artifacts()
except FileNotFoundError:
    st.error("Trained models not found. Run `python main.py --data <csv>` first.")
    st.stop()

left, right = st.columns(2)
with left:
    loan_amnt = st.number_input("Loan amount ($)", 1000, 40000, 12000, 500)
    term = st.selectbox("Term", [" 36 months", " 60 months"])
    int_rate = st.slider("Interest rate (%)", 5.0, 30.0, 13.5, 0.1)
    grade = st.selectbox("LendingClub grade", list("ABCDEFG"), index=2)
    purpose = st.selectbox("Purpose", ["debt_consolidation", "credit_card", "home_improvement",
                                       "small_business", "other"])
with right:
    annual_inc = st.number_input("Annual income ($)", 10000, 500000, 65000, 1000)
    dti = st.slider("Debt-to-income ratio", 0.0, 40.0, 17.0, 0.5)
    fico = st.slider("FICO score", 660, 845, 700, 5)
    home = st.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN"])
    revol_util = st.slider("Revolving utilisation (%)", 0.0, 120.0, 50.0, 1.0)

if st.button("Score loan", type="primary"):
    # Unspecified fields stay missing -> handled by the same imputers used in training
    row = pd.DataFrame({c: [np.nan] for c in meta["feature_columns"]})
    for c in C.CATEGORICAL:
        if c in row:
            row[c] = "missing"
    values = {"loan_amnt": loan_amnt, "funded_amnt": loan_amnt, "term": term, "int_rate": int_rate,
              "grade": grade, "purpose": purpose, "annual_inc": annual_inc, "dti": dti,
              "fico_range_low": fico, "fico_range_high": fico + 4, "home_ownership": home,
              "revol_util": revol_util}
    for k, v in values.items():
        if k in row:
            row[k] = v
    X = pre.transform(row)
    p_paid = float(clf.predict_proba(X)[0, 1])
    nar = float(reg.predict(X)[0])
    threshold = meta.get("threshold")
    cutoff = meta.get("classifier_cutoff", 0.5)

    c1, c2 = st.columns(2)
    c1.metric("P(fully paid)", f"{p_paid:.1%}",
              delta="Low default risk" if p_paid >= cutoff else "High default risk",
              delta_color="normal" if p_paid >= cutoff else "inverse")
    c2.metric("Predicted NAR", f"{nar:.1%}")
    if threshold is not None:
        if nar >= threshold:
            st.success(f"✅ INVEST - predicted NAR is above the strategy threshold M = {threshold:.3f}")
        else:
            st.warning(f"⛔ SKIP - predicted NAR is below the strategy threshold M = {threshold:.3f}")

    # --- plain-English explanation (SHAP) -----------------------------------------
    st.subheader("Why this result?")
    drivers = {"lowers": [], "raises": []}
    if shap_model is not None:
        try:
            # only explain fields the user actually entered (the rest are imputed)
            drivers = explain_loan(shap_model, pre, X, row, only=set(values) - {"funded_amnt", "fico_range_high"})
        except Exception as exc:                      # e.g. shap not installed
            st.caption(f"SHAP explanation unavailable ({exc}).")
    st.markdown(narrate_loan(p_paid, nar, threshold, cutoff, meta.get("base_rate", 0.81), drivers))
    st.caption("SHAP shows which inputs the model relies on for this loan - not what causes default. "
               "Fields you didn't enter are treated as missing.")

summary_png = C.RESULTS_DIR / "shap_summary.png"
if summary_png.exists():
    with st.expander("Overall: what drives default risk in the model?"):
        st.image(str(summary_png))
