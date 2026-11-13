#!/usr/bin/env python
"""Generate a small SYNTHETIC LendingClub-like CSV for smoke tests and offline demos.

The data only mimics the schema and rough statistics of the real dataset; results on it
are NOT meaningful. Use the real LendingClub file for the actual project results.

    python scripts/generate_sample_data.py --rows 30000 --out data/sample_loans.csv
"""
import argparse

import numpy as np
import pandas as pd


def generate(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    risk = rng.normal(size=n)                                        # latent credit risk
    grade_idx = np.clip(np.round(3 + 1.3 * risk + rng.normal(0, 0.7, n)), 0, 6).astype(int)
    grade = np.array(list("ABCDEFG"))[grade_idx]
    term_m = rng.choice([36, 60], n, p=[0.75, 0.25])
    loan = np.clip(np.round(rng.lognormal(9.4, 0.6, n), -2), 1000, 40000)
    int_rate = np.clip(6 + 2.6 * grade_idx + rng.normal(0, 0.8, n), 5.3, 30.0)
    fico = np.clip(np.round((700 - 20 * risk + rng.normal(0, 20, n)) / 5) * 5, 660, 845)
    dti = np.clip(rng.normal(17 + 2 * risk, 7), 0, 40)

    issue = (pd.Timestamp("2012-01-01") + pd.to_timedelta(rng.integers(0, 1461, n), unit="D"))
    issue = pd.Series(issue).dt.to_period("M").dt.to_timestamp()
    earliest = issue - pd.to_timedelta(rng.integers(3 * 365, 30 * 365, n), unit="D")

    p_default = 1 / (1 + np.exp(-(-1.7 + 0.9 * risk + 0.03 * (dti - 17))))
    paid = rng.random(n) > p_default
    months = np.where(paid, np.round(term_m * rng.uniform(0.4, 1.0, n)),
                      np.array([rng.integers(3, max(4, int(t * 0.7))) for t in term_m]))
    simple_interest = int_rate / 100 * months / 12
    total_pymnt = np.where(paid, loan * (1 + 0.55 * simple_interest),
                           loan * (months / term_m) * rng.uniform(0.5, 1.0, n) * (1 + simple_interest))
    last = (issue + pd.to_timedelta(months * 30.4, unit="D")).dt.strftime("%b-%Y")

    status = np.where(paid, "Fully Paid", rng.choice(["Charged Off", "Default"], n, p=[0.97, 0.03]))
    status = np.where(rng.random(n) < 0.05, "Current", status)       # exercise the status filter

    nan_if = lambda a, frac: np.where(rng.random(n) < frac, np.nan, a)  # noqa: E731
    return pd.DataFrame({
        "loan_amnt": loan, "funded_amnt": loan, "term": [f" {t} months" for t in term_m],
        "int_rate": np.round(int_rate, 2), "installment": np.round(loan * (1 + int_rate / 100) / term_m, 2),
        "grade": grade, "sub_grade": [g + str(i) for g, i in zip(grade, rng.integers(1, 6, n))],
        "emp_length": rng.choice(["< 1 year", "1 year", "5 years", "10+ years", np.nan], n),
        "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN"], n, p=[0.45, 0.4, 0.15]),
        "annual_inc": np.round(rng.lognormal(11.1, 0.5, n)),
        "verification_status": rng.choice(["Verified", "Source Verified", "Not Verified"], n),
        "issue_d": issue.dt.strftime("%b-%Y"),
        "loan_status": status,
        "purpose": rng.choice(["debt_consolidation", "credit_card", "home_improvement", "small_business", "other"], n),
        "zip_code": [f"{z}xx" for z in rng.integers(100, 999, n)],
        "addr_state": rng.choice(["CA", "NY", "TX", "FL", "IL", "TN", "WA"], n),
        "dti": np.round(dti, 2), "delinq_2yrs": nan_if(rng.poisson(0.3, n), 0.02),
        "earliest_cr_line": earliest.dt.strftime("%b-%Y"),
        "fico_range_low": fico, "fico_range_high": fico + 4,
        "inq_last_6mths": rng.poisson(0.8, n), "mths_since_last_delinq": nan_if(rng.integers(1, 120, n), 0.6),
        "mths_since_last_record": nan_if(rng.integers(1, 120, n), 0.85),
        "open_acc": rng.integers(2, 30, n), "pub_rec": rng.poisson(0.15, n),
        "revol_bal": np.round(rng.lognormal(9.3, 1.0, n)), "revol_util": np.round(np.clip(rng.normal(50 + 5 * risk, 25), 0, 120), 1),
        "total_acc": rng.integers(4, 60, n), "initial_list_status": rng.choice(["w", "f"], n),
        "application_type": "Individual",
        "total_pymnt": np.round(total_pymnt, 2), "last_pymnt_d": last,
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=30000)
    ap.add_argument("--out", default="data/sample_loans.csv")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    generate(a.rows, a.seed).to_csv(a.out, index=False)
    print(f"Wrote {a.rows} synthetic loans to {a.out}")
