"""Score the five models released by Beikmohammadi et al. on the official test set at the published 0.5 cut-off (RQ1b).

The authors' repository contains five trained Keras models (Model/...run1 to run5; one per seed of their published
design) and reports a mean test cost of 3,440. This script loads each model and scores the 16,000 test rows (and,
for run 1, the 60,000 training rows) after the preprocessing pipeline of Beikmohammadi et al.: the imputed tables come
from the work folder, and a min-max scaler is fitted on the table named by the positional argument and applied to the
imputed training and test tables. Two sets of tables exist (--tables):

  five_round  the first approximation (impute_five_rounds.py, resample_five_round_table.py): the imputer capped at
              five rounds instead of up to 100;
  exact       the authors' inputs regenerated exactly (prepare_authors_inputs.py): 100 imputer rounds.

The published pipeline fits the scaler on the resampled table (train_resampled.csv); the first re-scoring fitted it
on the imputed training table (train_imputed.csv), the default of the positional argument. The four combinations
were run as follows (from the repository root):

    module=experiments.rq1_transformer_2026_reproduction.authors_code.evaluate_shipped_models
    python -m $module --tables five_round                       # the first re-scoring
    python -m $module --tables exact train_resampled.csv        # the exact re-scoring
    python -m $module --tables exact train_imputed.csv          # attribution (i)
    python -m $module --tables five_round train_resampled.csv   # attribution (ii)

The last two cross the two deviations of the first re-scoring (run_scaler_attribution.sh runs them). Each prints, per
model, the cost, missed failures (FN) and false alarms (FP) at the 0.5 cut-off (a row is flagged when its probability
exceeds 0.5, as in the authors' code), the test AUC, and a few distribution counts. It also prints and stores a
sweep of thresholds on the test set and the cost at 10/510 = 0.0196, the cut-off that would minimise the cost if the
probabilities were calibrated; both use the test labels (diagnostic).

Writes a table with one row per model (columns run, test_cost, test_FN, test_FP, train_cost, train_FN, train_FP,
test_AUC, best_thr_on_test, best_thr_cost, cost_thr0p0196, secs; the training columns only for run 1) and each
model's test probabilities, run<k>_test_probabilities.npy, to the locations in OUTPUTS below. The console output was
kept as logs/evaluate_shipped_models_<...>.log (see the README of this folder).
"""
import argparse
import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from experiments.common.data_loading import COST_OF_FALSE_ALARM, COST_OF_MISSED_FAILURE
from experiments.rq1_transformer_2026_reproduction.authors_code.file_locations import (
    FIVE_ROUND_WORK_DIR, LOGS_DIR, SHIPPED_MODELS_EXACT_INPUTS_DIR, SHIPPED_MODELS_FIVE_ROUND_TABLES_DIR, WORK_DIR,
    shipped_model_folders)

TABLES = {"five_round": FIVE_ROUND_WORK_DIR, "exact": WORK_DIR}
# (tables, scaler source) -> (folder for the probability vectors, file for the table of results). The probabilities
# of the two attribution runs were not kept as results; they stay in the work folder.
OUTPUTS = {
    ("five_round", "train_imputed.csv"): (SHIPPED_MODELS_FIVE_ROUND_TABLES_DIR,
                                          LOGS_DIR / "evaluate_shipped_models_five_round_tables.csv"),
    ("exact", "train_resampled.csv"): (SHIPPED_MODELS_EXACT_INPUTS_DIR,
                                       SHIPPED_MODELS_EXACT_INPUTS_DIR / "evaluation_scaler_on_resampled.csv"),
    ("exact", "train_imputed.csv"): (WORK_DIR, SHIPPED_MODELS_EXACT_INPUTS_DIR / "evaluation_scaler_on_imputed.csv"),
    ("five_round", "train_resampled.csv"): (
        FIVE_ROUND_WORK_DIR, SHIPPED_MODELS_EXACT_INPUTS_DIR / "evaluation_five_round_tables_scaler_on_resampled.csv"),
}
TEST_THRESHOLD_SWEEP = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]


def cost(y, probabilities, threshold=0.5):
    """(cost, FN, FP) of flagging the rows whose probability exceeds threshold."""
    decisions = (probabilities > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, decisions, labels=[0, 1]).ravel()
    return COST_OF_MISSED_FAILURE * fn + COST_OF_FALSE_ALARM * fp, fn, fp


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", required=True, choices=list(TABLES),
                        help="five_round: the first approximation; exact: the exactly regenerated inputs")
    parser.add_argument("scaler_source", nargs="?", default="train_imputed.csv",
                        choices=["train_imputed.csv", "train_resampled.csv"],
                        help="table of the work folder on which the min-max scaler is fitted")
    return parser.parse_args()


def main():
    args = parse_arguments()
    import tf_keras as keras
    work_dir = TABLES[args.tables]
    probabilities_dir, results_table_file = OUTPUTS[(args.tables, args.scaler_source)]
    X_train = pd.read_csv(work_dir / "train_imputed.csv")
    X_test = pd.read_csv(work_dir / "test_imputed.csv")
    y_train = pd.read_csv(work_dir / "train_label.csv")["class"].values
    y_test = pd.read_csv(work_dir / "test_label.csv")["class"].values
    scaler_table = pd.read_csv(work_dir / args.scaler_source)
    print("scaler fitted on", args.scaler_source, scaler_table.shape, flush=True)
    scaler = MinMaxScaler().fit(scaler_table.values)
    X_train_scaled = scaler.transform(X_train.values).astype(np.float32)
    X_test_scaled = scaler.transform(X_test.values).astype(np.float32)
    print("train", X_train_scaled.shape, "test", X_test_scaled.shape, "positives in test", y_test.sum(), flush=True)
    probabilities_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_folder in shipped_model_folders():
        run = model_folder.name[-4:]
        started = time.time()
        model = keras.models.load_model(str(model_folder), compile=False)

        def predict(X):
            # the network expects one channel per attribute: (rows, attributes, 1)
            return model.predict(X.reshape(-1, X.shape[1], 1), batch_size=256, verbose=0).ravel()

        test_probabilities = predict(X_test_scaled)
        np.save(probabilities_dir / f"{run}_test_probabilities.npy", test_probabilities)
        test_cost, test_fn, test_fp = cost(y_test, test_probabilities)
        if run == "run1":
            train_cost, train_fn, train_fp = cost(y_train, predict(X_train_scaled))
        else:
            train_cost, train_fn, train_fp = (np.nan, np.nan, np.nan)
        auc = roc_auc_score(y_test, test_probabilities)
        # thresholds tried on the test labels (diagnostic of the probability scale)
        sweep = {threshold: cost(y_test, test_probabilities, threshold) for threshold in TEST_THRESHOLD_SWEEP}
        best_threshold_on_test = min(sweep, key=lambda threshold: sweep[threshold][0])
        rows.append(dict(run=run, test_cost=test_cost, test_FN=test_fn, test_FP=test_fp, train_cost=train_cost,
                         train_FN=train_fn, train_FP=train_fp, test_AUC=round(auc, 5),
                         best_thr_on_test=best_threshold_on_test, best_thr_cost=sweep[best_threshold_on_test][0],
                         cost_thr0p0196=cost(y_test, test_probabilities, 10 / 510)[0],
                         secs=round(time.time() - started, 1)))
        print(rows[-1], flush=True)
        print("  sweep (threshold: cost, FN, FP):", sweep, flush=True)
        positive_probabilities = test_probabilities[y_test == 1]
        negative_probabilities = test_probabilities[y_test == 0]
        print("  test failures with probability <= 0.5:", (positive_probabilities <= 0.5).sum(),
              " < 0.1:", (positive_probabilities < 0.1).sum(), " < 0.02:", (positive_probabilities < 0.02).sum(),
              "| healthy test trucks with probability > 0.5:", (negative_probabilities > 0.5).sum(),
              " > 0.1:", (negative_probabilities > 0.1).sum(), " > 0.02:", (negative_probabilities > 0.02).sum(),
              flush=True)
    results = pd.DataFrame(rows)
    print(results.to_string())
    print("MEAN test cost", results.test_cost.mean(), "FN", results.test_FN.mean(), "FP", results.test_FP.mean(),
          "| train cost", results.train_cost.mean(), "FN", results.train_FN.mean(), "FP", results.train_FP.mean())
    results_table_file.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_table_file, index=False)


if __name__ == "__main__":
    main()
