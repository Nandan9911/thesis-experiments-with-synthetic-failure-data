"""Run 1 of the models released by Beikmohammadi et al., with a min-max scaling taken from the raw training columns
(RQ1b, an early check).

Part of the first re-scoring of the released models, on the five-round tables, before the authors' inputs were
regenerated exactly (see the README of this folder). The models missed most failures on those tables, and one
possible cause was the scaler: imputation can create values far outside an attribute's observed range, and a scaler
fitted on the imputed table then compresses the real values. Here the minimum and maximum of every attribute are
taken from the raw, unimputed training file instead, as if imputation had created no new extremes, and the five-round
imputed test table is scaled with them (values outside [0, 1] are left as they are). Only run 1 is scored.

    python -m experiments.rq1_transformer_2026_reproduction.authors_code.evaluate_shipped_models_raw_scale

Reads the five-round test table and labels from cache/authors_code_work/five_round_tables/ (impute_five_rounds.py)
and the raw training file from the authors' repository. Prints the cost, FN and FP at the 0.5 cut-off and at a few
other cut-offs (diagnostics on the test labels), the AUC and the average precision, and writes the probabilities to
results/rq1_transformer_2026_reproduction/authors_code/shipped_models_five_round_tables/ as
run1_test_probabilities_raw_scale.npy.
"""
import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from experiments.common.data_loading import COST_OF_FALSE_ALARM, COST_OF_MISSED_FAILURE
from experiments.rq1_transformer_2026_reproduction.authors_code.file_locations import (
    AUTHORS_REPOSITORY, FIVE_ROUND_WORK_DIR, SHIPPED_MODELS_FIVE_ROUND_TABLES_DIR)

OUTPUT_FILE = SHIPPED_MODELS_FIVE_ROUND_TABLES_DIR / "run1_test_probabilities_raw_scale.npy"


def cost(y, probabilities, threshold=0.5):
    """(cost, FN, FP) of flagging the rows whose probability exceeds threshold."""
    tn, fp, fn, tp = confusion_matrix(y, (probabilities > threshold).astype(int), labels=[0, 1]).ravel()
    return COST_OF_MISSED_FAILURE * fn + COST_OF_FALSE_ALARM * fp, fn, fp


def main():
    import tf_keras as keras
    X_test = pd.read_csv(FIVE_ROUND_WORK_DIR / "test_imputed.csv")
    y_test = pd.read_csv(FIVE_ROUND_WORK_DIR / "test_label.csv")["class"].values
    # the authors' copy of the training file has no licence header, so it is read directly
    raw_training = pd.read_csv(AUTHORS_REPOSITORY / "aps_failure_training_set.csv",
                               na_values="na").drop(columns=["class"])[X_test.columns]
    column_minimum = raw_training.min().values
    column_maximum = raw_training.max().values
    column_range = np.where(column_maximum - column_minimum == 0, 1, column_maximum - column_minimum)
    X_test_scaled = ((X_test.values - column_minimum) / column_range).astype(np.float32)
    print("raw-range scaled test table: min %.3f max %.3f; fraction of cells outside [0,1]: %.5f"
          % (X_test_scaled.min(), X_test_scaled.max(), ((X_test_scaled < 0) | (X_test_scaled > 1)).mean()),
          flush=True)
    for model_folder in sorted(AUTHORS_REPOSITORY.glob("Model/*run1")):
        model = keras.models.load_model(str(model_folder), compile=False)
        started = time.time()
        probabilities = model.predict(X_test_scaled.reshape(-1, 142, 1), batch_size=256, verbose=0).ravel()
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        np.save(OUTPUT_FILE, probabilities)
        print("run1 / raw-range scaler: threshold 0.5 cost, FN, FP =", cost(y_test, probabilities),
              "| AUC %.5f AP %.4f | secs %.0f" % (roc_auc_score(y_test, probabilities),
                                                  average_precision_score(y_test, probabilities),
                                                  time.time() - started),
              flush=True)
        for threshold in [0.0196, 0.1, 0.3, 0.7]:
            print("   threshold", threshold, "->", cost(y_test, probabilities, threshold), flush=True)
        positive_probabilities = probabilities[y_test == 1]
        negative_probabilities = probabilities[y_test == 0]
        print("   failures with probability <= 0.5:", int((positive_probabilities <= 0.5).sum()),
              "; healthy trucks with probability > 0.5:", int((negative_probabilities > 0.5).sum()),
              "; with probability > 0.0196:", int((negative_probabilities > 0.0196).sum()), flush=True)


if __name__ == "__main__":
    main()
