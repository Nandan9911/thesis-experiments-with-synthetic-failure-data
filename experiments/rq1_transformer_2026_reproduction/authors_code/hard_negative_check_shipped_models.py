"""The hard-negative comparison: which healthy trucks the five models released by Beikmohammadi et al. flag (RQ1b, a
diagnostic).

On the exactly regenerated inputs the five released models cost 3,440 on the official test set on average (6.8 missed
failures and 4.0 false alarms among 15,625 healthy trucks). Model 1, the one also scored on the training rows, makes
4 false alarms on the test set but 416 among the 59,000 healthy training trucks (cost 12,660 on the 60,000 training
rows). The published training table is the resampled one, and RENN (repeated edited nearest neighbours, explained in
preprocessing_2026.py) removed the healthy rows that lie close to failure rows, so the models never trained on
exactly the healthy trucks that look most like failures. Such rows are called hard negatives here.

The test set contains hard negatives of the same kind. Applying RENN's rule once ("the one-shot rule": a negative is
hard unless all three of its nearest neighbours in the published resampled table are healthy) to every test negative
marks the test negatives that RENN would have removed had they been in the training file. This script compares how
often each released model flags these hard test negatives and the hard training negatives that RENN removed; neither
group is in the published training table. It scores, with each released model, the groups

  training negatives kept by RENN (seen in training)    training negatives removed by RENN (never seen)
  test negatives that are hard (never seen)             the other test negatives
  training failures                                     test failures

and reports the fraction flagged at the published 0.5 cut-off and summaries of the predicted probability. Rows are
identified as kept by RENN by exact matching between the scaled training table and the scaled resampled table (RENN
copies real rows unchanged; only failure rows are synthesised).

The comparison (a diagnostic) uses the test labels only to define the groups. The same comparison for models
trained with a known share of the test rows comes from the test-inclusion controls of
train_authors_model_with_resume.py (evaluated by hard_negative_check_retrained_models.py in the parent folder). The
follow-up (hard_negative_check_shipped_models_followup.py) splits the training negatives further, and
hard_negative_check_tree_control.py applies the same groups to the thesis's own tree models.

    python -m experiments.rq1_transformer_2026_reproduction.authors_code.hard_negative_check_shipped_models

Needs the exact inputs in the authors' repository (prepare_authors_inputs.py) and loads the five released models.
Writes results/rq1_transformer_2026_reproduction/authors_code/shipped_models_exact_inputs/hard_negative_check.json
(per group: n, and per run flag_rate_0.5, mean_p, median_p, p90, p99) and prints a table; the console output of the
reported run is hard_negative_check.log in the same folder. It also saves the three groups of negatives it
computes (removed, hard_tr, hard_te; see file_locations.py) as hard_negative_masks.npy there, which the other
hard-negative scripts read.
"""
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from experiments.rq1_transformer_2026_reproduction.authors_code.file_locations import (
    AUTHORS_REPOSITORY, HARD_NEGATIVE_MASKS_FILE, RESAMPLED_LABELS, RESAMPLED_TABLE_SCALED,
    SHIPPED_MODELS_EXACT_INPUTS_DIR, TEST_LABELS, TEST_TABLE_SCALED, TRAINING_LABELS, TRAINING_TABLE_SCALED,
    shipped_model_folders)

OUTPUT_FILE = SHIPPED_MODELS_EXACT_INPUTS_DIR / "hard_negative_check.json"


def row_keys(table):
    """One hash per row of the table, rounded to six decimals, for exact row matching between two tables."""
    return pd.util.hash_pandas_object(pd.DataFrame(np.round(table.astype(np.float64), 6)), index=False).to_numpy()


def hard_negative_masks(X_train, y_train, X_resampled, y_resampled, X_test, y_test):
    """The groups of negatives: (kept_by_renn over all training rows, hard_test over the test negatives,
    hard_train over the training negatives)."""
    kept_by_renn = np.isin(row_keys(X_train), row_keys(X_resampled))
    training_negative = y_train == 0
    print(f"training negatives: {int(training_negative.sum())}, "
          f"kept by RENN: {int((kept_by_renn & training_negative).sum())}, "
          f"removed: {int((~kept_by_renn & training_negative).sum())}; training positives matched in the resampled "
          f"table: {int((kept_by_renn & ~training_negative).sum())} of {int((~training_negative).sum())}", flush=True)
    # RENN's rule, applied once to the test negatives against the resampled table (3 neighbours, kind_sel "all")
    started = time.time()
    neighbours = NearestNeighbors(n_neighbors=3).fit(X_resampled)
    _, neighbour_index = neighbours.kneighbors(X_test[y_test == 0])
    hard_test = ~(y_resampled[neighbour_index] == 0).all(axis=1)
    print(f"test negatives: {int((y_test == 0).sum())}, 'hard' (RENN would remove them): {int(hard_test.sum())}  "
          f"[{time.time() - started:.0f}s]", flush=True)
    _, neighbour_index = neighbours.kneighbors(X_train[training_negative])
    hard_train = ~(y_resampled[neighbour_index] == 0).all(axis=1)   # the same rule on the training negatives
    print(f"training negatives the same rule calls hard: {int(hard_train.sum())} "
          f"(removed by RENN: {int((~kept_by_renn & training_negative).sum())})", flush=True)
    return kept_by_renn, hard_test, hard_train


def main():
    import tf_keras as keras
    X_train = pd.read_csv(AUTHORS_REPOSITORY / TRAINING_TABLE_SCALED).to_numpy(np.float32)
    y_train = pd.read_csv(AUTHORS_REPOSITORY / TRAINING_LABELS)["class"].to_numpy()
    X_resampled = pd.read_csv(AUTHORS_REPOSITORY / RESAMPLED_TABLE_SCALED).to_numpy(np.float32)
    y_resampled = pd.read_csv(AUTHORS_REPOSITORY / RESAMPLED_LABELS)["class"].to_numpy()
    X_test = pd.read_csv(AUTHORS_REPOSITORY / TEST_TABLE_SCALED).to_numpy(np.float32)
    y_test = pd.read_csv(AUTHORS_REPOSITORY / TEST_LABELS)["class"].to_numpy()
    print("train", X_train.shape, "resampled", X_resampled.shape, "test", X_test.shape, flush=True)
    kept_by_renn, hard_test, hard_train = hard_negative_masks(X_train, y_train, X_resampled, y_resampled,
                                                              X_test, y_test)
    training_negative = y_train == 0
    # The other hard-negative scripts read the three groups from this file instead of recomputing the neighbours.
    HARD_NEGATIVE_MASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.save(HARD_NEGATIVE_MASKS_FILE, {"removed": ~kept_by_renn[training_negative], "hard_tr": hard_train,
                                       "hard_te": hard_test})
    # the group names are the keys of the reported file
    groups = {
        "train negatives kept by RENN (seen)": X_train[training_negative & kept_by_renn],
        "train negatives removed by RENN (never seen)": X_train[training_negative & ~kept_by_renn],
        "test negatives, hard (never seen)": X_test[y_test == 0][hard_test],
        "test negatives, rest": X_test[y_test == 0][~hard_test],
        "train positives": X_train[~training_negative],
        "test positives": X_test[y_test == 1],
    }
    summary = {group: {"n": int(len(rows))} for group, rows in groups.items()}
    for model_folder in shipped_model_folders():
        run = model_folder.name[-4:]
        model = keras.models.load_model(str(model_folder), compile=False)
        for group, rows in groups.items():
            probabilities = model.predict(rows.reshape(-1, rows.shape[1], 1), batch_size=256, verbose=0).ravel()
            summary[group][run] = {"flag_rate_0.5": float((probabilities > 0.5).mean()),
                                   "mean_p": float(probabilities.mean()), "median_p": float(np.median(probabilities)),
                                   "p90": float(np.quantile(probabilities, 0.9)),
                                   "p99": float(np.quantile(probabilities, 0.99))}
        print(run, {group: round(summary[group][run]["flag_rate_0.5"], 4) for group in groups}, flush=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as handle:
        json.dump(summary, handle, indent=1)
    print("\nfraction flagged at 0.5 (mean over the five models):")
    for group in groups:
        runs = [key for key in summary[group] if key.startswith("run")]
        flag_rates = [summary[group][run]["flag_rate_0.5"] for run in runs]
        mean_probability = np.mean([summary[group][run]["mean_p"] for run in runs])
        print(f"  {group:48s} n={summary[group]['n']:6d}  flagged {np.mean(flag_rates):.4f}  "
              f"mean probability {mean_probability:.4f}")


if __name__ == "__main__":
    main()
