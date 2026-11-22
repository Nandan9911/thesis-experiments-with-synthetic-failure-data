"""Reading the Scania APS data files, the misclassification cost, and the data audit.

Part of the common evaluation protocol (experiments/common). The experiments load the data through `load_training_set`
and `load_test_set`, so that the raw files are parsed in one way, and compute the cost with `misclassification_cost` or,
in the few scripts that count the errors themselves, with the two cost constants defined here. The exceptions are the
scripts in rq1_transformer_2026_reproduction/authors_code that repeat the authors' preprocessing: they read the data
files with pandas in the way the authors' code does.

The data are the two files of the UCI dataset "APS Failure at Scania Trucks", collected from heavy Scania trucks in
everyday use: 60,000 training rows (1,000 failures) and 16,000 test rows (375 failures), each with a class label and
170 anonymised numeric attributes (single counters and the bins of histograms). The positive class consists of
failures of one component of the Air Pressure System (APS); the negative class consists of trucks with failures in
components not related to the APS. About the files:

* both begin with a 20-line licence notice (Scania CV AB, GNU General Public License version 3) above the header
  row; the reader looks for the header row, the line that starts with "class", and skips everything above it, so a
  copy with the notice removed is read in the same way;
* missing values are written as "na" and become NaN (about 8 % of all cells; failure rows have about twice the
  missing rate of healthy rows);
* the class column holds "pos" (APS failure) or "neg" (other trucks), coded here as 1 and 0.

The cost of a set of decisions is the one that comes with the dataset (it was the metric of the IDA 2016 Industrial
Challenge):

    cost = 500 x (missed failures, i.e. false negatives) + 10 x (false alarms, i.e. false positives)

500 is Scania's estimate of the cost of missing a faulty truck, which may then break down, and 10 the cost of an
unnecessary check by a mechanic at a workshop.

Run as a script, the module checks that the data files are the ones used in the thesis:

    python -m experiments.common.data_loading

It reads data/aps_failure_training_set.csv and data/aps_failure_test_set.csv, prints the audit and writes it to
results/data_audit.json: the SHA-256 digests of both files, row and failure counts, missing-value rates, constant
attributes, the number of rows that occur in both files, and the cost of the two trivial classifiers (flag no truck,
flag every truck) as a check of the cost function.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common.paths import DATA_AUDIT_FILE, TEST_CSV, TRAINING_CSV

COST_OF_MISSED_FAILURE = 500   # false negative: a faulty truck that is not checked (Scania's estimate)
COST_OF_FALSE_ALARM = 10       # false positive: an unnecessary check at a workshop (Scania's estimate)

# Rows of the official test set. Out-of-fold costs, which are counted over the 60,000 training rows, are also
# reported multiplied by TEST_SET_ROWS / 60,000 so that they can be read on the scale of a test cost.
TEST_SET_ROWS = 16000

# The one attribute that has the same value in every row of the training file (constant_features in
# results/data_audit.json). It carries no information, and most experiment scripts drop it before fitting anything
# (the preprocessing pipeline of Beikmohammadi et al., repeated in rq1_transformer_2026_reproduction/authors_code, keeps
# it).
CONSTANT_ATTRIBUTE = "cd_000"


def file_sha256(path: Path) -> str:
    """SHA-256 digest of a file (read in blocks of 1 MiB)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_uci_file(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Attributes (DataFrame of float32, one column per attribute) and labels (int8, 1 = APS failure) of one file."""
    with open(path, "r") as handle:
        first_line = handle.readline()
    lines_above_header = 0
    if not first_line.startswith("class"):
        with open(path, "r") as handle:
            for line_number, line in enumerate(handle):
                if line.startswith("class"):
                    lines_above_header = line_number
                    break
    table = pd.read_csv(path, skiprows=lines_above_header, na_values=["na"], low_memory=False)
    y = (table["class"].astype(str).str.strip() == "pos").astype(np.int8).to_numpy()
    X = table.drop(columns=["class"]).astype(np.float32)
    return X, y


def load_training_set() -> tuple[pd.DataFrame, np.ndarray]:
    """The 60,000 training rows: attributes X (170 float32 columns, NaN where missing) and labels y (1 = failure)."""
    return _read_uci_file(TRAINING_CSV)


def load_test_set() -> tuple[pd.DataFrame, np.ndarray]:
    """The 16,000 rows of the official test set, in the same form as load_training_set."""
    return _read_uci_file(TEST_CSV)


def misclassification_cost(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Challenge cost of the decisions y_pred (1 = flag the truck for a check) against the true labels y_true.

    Returns a dict with the cost, the four cells of the confusion matrix (tp, fn, fp, tn), recall = tp / (tp + fn)
    and precision = tp / (tp + fp); recall and precision are 0 when their denominator is 0.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return {
        "cost": COST_OF_MISSED_FAILURE * fn + COST_OF_FALSE_ALARM * fp,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
    }


def audit_data_files() -> dict:
    """Integrity checks of the two data files, under the key names of results/data_audit.json.

    Missing-value rates and constant attributes refer to the training file. duplicate_rows_train_test counts the
    pairs of identical training and test rows, compared with missing values set to -1 and values rounded to three
    decimals. cost_check_all_neg and cost_check_all_pos are the test costs of flagging no truck and every truck.
    """
    X_train, y_train = load_training_set()
    X_test, y_test = load_test_set()
    missing_rate_per_attribute = X_train.isna().mean()
    return {
        "train_sha256": file_sha256(TRAINING_CSV),
        "test_sha256": file_sha256(TEST_CSV),
        "train_rows": int(len(X_train)), "train_pos": int(y_train.sum()),
        "test_rows": int(len(X_test)), "test_pos": int(y_test.sum()),
        "n_features": int(X_train.shape[1]),
        "features_identical": list(X_train.columns) == list(X_test.columns),
        "train_missing_cells_frac": float(X_train.isna().to_numpy().mean()),
        "test_missing_cells_frac": float(X_test.isna().to_numpy().mean()),
        "features_over_50pct_missing": int((missing_rate_per_attribute > 0.5).sum()),
        "features_over_70pct_missing": int((missing_rate_per_attribute > 0.7).sum()),
        "pos_row_missing_frac": float(X_train[y_train == 1].isna().to_numpy().mean()),
        "neg_row_missing_frac": float(X_train[y_train == 0].isna().to_numpy().mean()),
        "constant_features": [column for column in X_train.columns if X_train[column].nunique(dropna=True) <= 1],
        "duplicate_rows_train_test": int(pd.merge(X_train.fillna(-1).round(3), X_test.fillna(-1).round(3),
                                                  how="inner").shape[0]),
        "cost_check_all_neg": misclassification_cost(y_test, np.zeros_like(y_test))["cost"],
        "cost_check_all_pos": misclassification_cost(y_test, np.ones_like(y_test))["cost"],
    }


if __name__ == "__main__":
    audit = audit_data_files()
    print(json.dumps(audit, indent=2))
    DATA_AUDIT_FILE.parent.mkdir(exist_ok=True)
    with open(DATA_AUDIT_FILE, "w") as handle:
        json.dump(audit, handle, indent=2)
