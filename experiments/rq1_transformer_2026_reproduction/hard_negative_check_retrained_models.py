"""The hard-negative comparison repeated on the models retrained with the released code (RQ1b, a diagnostic).

authors_code/hard_negative_check_shipped_models.py defines the hard negatives (the healthy trucks that the RENN step
of the preprocessing pipeline of Beikmohammadi et al. removed from the training table, and the test negatives that its
rule, applied once as "the one-shot rule", would have removed) and finds that the five released models flag the two
groups at different rates (37 % of the hard removed training negatives, 0.9 % of the hard test negatives).

This script makes the same comparison for the models of authors_code/train_authors_model_with_resume.py, retrained
from scratch on the exactly regenerated inputs:

* the retraining runs (the authors' seeds 0 to 4), with no test rows in the training table;
* the test-inclusion controls, in which a stratified half (seeds 0, 1, 2) or all (seed 0) of the official test rows
  were added to the training table before the authors' validation split, so that they took part in training like
  any other row. Their test negatives are split into the ones that were added (seen) and the rest (unseen).

The controls show what the comparison looks like for a model whose training data included a known share of the test
rows, and the retraining runs what it looks like for one whose training data included none. For each run the
final model's probabilities for the 60,000 training rows and the 16,000 test rows, saved by the run, are cut at the
published 0.5 (a row is flagged at or above it), and the fraction flagged is computed for: training negatives kept
by RENN (trained on), removed by RENN (never seen), and the removed ones that are hard by the one-shot rule; hard
test negatives and the other test negatives (split into seen and unseen for the controls). The final model is the
one at the run's last full evaluation: epoch 8,000 for the retraining run of seed 0 and for seed 0 of the control runs
with half of the test rows in the training data, epoch 1,000 for the other runs (the retraining run of seed 1 included,
whose extension was stopped before its next evaluation).

The test labels are used only to define the groups and to report.

    python -m experiments.rq1_transformer_2026_reproduction.hard_negative_check_retrained_models

Reads the labels of the two data files, the masks in
results/rq1_transformer_2026_reproduction/authors_code/shipped_models_exact_inputs/hard_negative_masks.npy and, for
each run, training_probabilities_final.npy, test_probabilities_final.npy and contaminating_test_rows.npy in
results/rq1_transformer_2026_reproduction/authors_code/retraining_runs/<run>/ (runs without probabilities are
skipped). Writes results/rq1_transformer_2026_reproduction/hard_negative_check_retrained_models.json: per run a label
and the flag rates under the group names below (NaN where a group is empty), and prints the rates in per cent.
"""
import json

import numpy as np

from experiments.common.data_loading import load_test_set, load_training_set
from experiments.common.paths import TRANSFORMER_2026_RESULTS_DIR
from experiments.rq1_transformer_2026_reproduction.authors_code.file_locations import (HARD_NEGATIVE_MASKS_FILE,
                                                                                    RETRAINING_RUNS_DIR)

OUTPUT_FILE = TRANSFORMER_2026_RESULTS_DIR / "hard_negative_check_retrained_models.json"
# (run folder, label); the epochs are those of the run's last full evaluation
RUNS = [("clean_seed0", "Retraining, seed 0 (8,000 epochs)"), ("clean_seed1", "Retraining, seed 1 (1,000 epochs)"),
        ("clean_seed2", "Retraining, seed 2 (1,000 epochs)"), ("clean_seed3", "Retraining, seed 3 (1,000 epochs)"),
        ("clean_seed4", "Retraining, seed 4 (1,000 epochs)"),
        ("half_test_rows_in_training_seed0", "Half of the test rows in training, seed 0 (8,000 epochs)"),
        ("half_test_rows_in_training_seed1", "Half of the test rows in training, seed 1 (1,000 epochs)"),
        ("half_test_rows_in_training_seed2", "Half of the test rows in training, seed 2 (1,000 epochs)"),
        ("all_test_rows_in_training_seed0", "All test rows in training, seed 0 (1,000 epochs)")]


def main():
    _, y_train = load_training_set()
    _, y_test = load_test_set()
    masks = np.load(HARD_NEGATIVE_MASKS_FILE, allow_pickle=True).item()
    # hard_te is defined over the test negatives, hard_tr and removed over the training negatives
    hard_test, hard_train, removed = masks["hard_te"], masks["hard_tr"], masks["removed"]
    training_negatives = np.where(y_train == 0)[0]
    test_negatives = np.where(y_test == 0)[0]
    summary = {}
    for run, label in RUNS:
        run_dir = RETRAINING_RUNS_DIR / run
        if not (run_dir / "test_probabilities_final.npy").exists():
            continue
        training_probabilities = np.load(run_dir / "training_probabilities_final.npy")
        test_probabilities = np.load(run_dir / "test_probabilities_final.npy")
        contaminating_rows = (np.load(run_dir / "contaminating_test_rows.npy")
                              if (run_dir / "contaminating_test_rows.npy").exists() else np.array([], dtype=int))
        test_row_seen = np.zeros(len(y_test), bool)
        test_row_seen[contaminating_rows] = True
        training_negative_flagged = training_probabilities[training_negatives] >= 0.5
        test_negative_flagged = test_probabilities[test_negatives] >= 0.5
        seen = test_row_seen[test_negatives]
        # the group names are the keys of the reported file
        rates = {"train negatives kept by RENN (trained on)": training_negative_flagged[~removed].mean(),
                 "train negatives removed by RENN (never seen)": training_negative_flagged[removed].mean(),
                 "  of which hard by the one-shot rule": training_negative_flagged[removed & hard_train].mean()}
        if seen.any():
            rates.update({"test negatives, hard, seen": test_negative_flagged[hard_test & seen].mean(),
                          "test negatives, hard, unseen": (test_negative_flagged[hard_test & ~seen].mean()
                                                           if (~seen).any() else np.nan),
                          "test negatives, other, seen": test_negative_flagged[~hard_test & seen].mean(),
                          "test negatives, other, unseen": (test_negative_flagged[~hard_test & ~seen].mean()
                                                            if (~seen).any() else np.nan)})
        else:
            rates.update({"test negatives, hard (never seen)": test_negative_flagged[hard_test].mean(),
                          "test negatives, other (never seen)": test_negative_flagged[~hard_test].mean()})
        rates["n hard test negatives seen / unseen"] = (f"{int((hard_test & seen).sum())} / "
                                                        f"{int((hard_test & ~seen).sum())}")
        summary[run] = {"label": label, **{group: (float(rate) if not isinstance(rate, str) else rate)
                                           for group, rate in rates.items()}}
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as handle:
        json.dump(summary, handle, indent=1)
    for run in summary:
        print(run, {group: (round(100 * rate, 2) if isinstance(rate, float) else rate)
                    for group, rate in summary[run].items() if group != "label"})


if __name__ == "__main__":
    main()
