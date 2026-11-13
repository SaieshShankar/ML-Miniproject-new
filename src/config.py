"""Central configuration: paths, label definitions and feature groups."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

RANDOM_STATE = 42
TEST_SIZE = 0.3                 # 70 / 30 split, as in the reference report
ISSUE_YEARS = (2012, 2015)      # loans old enough to have reached a final status

# --- Labels -----------------------------------------------------------------
POSITIVE_STATUS = "Fully Paid"                 # label 1
NEGATIVE_STATUSES = ["Charged Off", "Default"]  # label 0
CREDIT_POLICY_PREFIX = "Does not meet the credit policy. Status:"

# --- Net annualised return (NAR) --------------------------------------------
NAR_MIN_DAYS = 60               # avoids exploding exponents for very short loans
NAR_CLIP = (-1.0, 1.0)          # bounds the regression target

# --- Feature groups (only application-time fields -> no label leakage) -------
# Missing values: mean-set / zero-set / max-set (see report, Sec. III-B)
NUMERIC_MEAN = [
    "loan_amnt", "funded_amnt", "int_rate", "installment", "annual_inc", "dti",
    "fico_range_low", "fico_range_high", "open_acc", "total_acc",
    "revol_bal", "revol_util",
]
NUMERIC_ZERO = [
    "delinq_2yrs", "inq_last_6mths", "pub_rec", "pub_rec_bankruptcies",
    "collections_12_mths_ex_med",
]
NUMERIC_MAX = [
    "mths_since_last_delinq", "mths_since_last_record", "mths_since_last_major_derog",
]
MAX_FACTOR = 2.0                # max-set fill = MAX_FACTOR * column max
CATEGORICAL = [
    "term", "grade", "sub_grade", "emp_length", "home_ownership",
    "verification_status", "purpose", "addr_state", "zip_code",
    "application_type", "initial_list_status",
]
DATE_FEATURES = ["earliest_cr_line"]   # converted to days since epoch
PERCENT_STRING_COLUMNS = ["int_rate", "revol_util"]

# Columns needed only to build labels / targets (never used as features)
LABEL_SOURCE_COLUMNS = ["loan_status", "issue_d", "total_pymnt", "last_pymnt_d"]

# --- Validation split (paper's "future work": a dev set for tuning / selection)
VAL_SIZE = 0.15                 # fraction of the training portion held out for validation
