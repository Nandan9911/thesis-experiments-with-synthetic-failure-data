"""The hard-negative comparison applied to saved probability vectors instead of loaded models (RQ1b, a diagnostic).

hard_negative_check_shipped_models.py loads the released Keras models and scores groups of negatives. For a model that
is no longer at hand, for example one of the retraining runs, the same comparison can be made from its saved
probabilities for the 60,000 training rows and the 16,000 test rows, in the row order of the authors' input tables
(train_authors_model_with_resume.py saves them as training_probabilities_final.npy and test_probabilities_final.npy).
The script prints, at the cut-off (default 0.5, a row is flagged when its probability exceeds it), the training and
test cost and the fraction flagged in each group: training negatives kept or removed by RENN, the removed ones that
are hard by the one-shot rule, the hard ones that RENN kept, hard and other test negatives, and the failures of both
files. It does not split the test rows of a test-inclusion control into seen and unseen ones;
hard_negative_check_retrained_models.py in the parent folder does that for all retraining runs at once.

    run=results/rq1_transformer_2026_reproduction/authors_code/retraining_runs/clean_seed0
    python -m experiments.rq1_transformer_2026_reproduction.authors_code.hard_negative_check_from_probabilities \\
        $run/training_probabilities_final.npy $run/test_probabilities_final.npy

Reads the label files from the authors' repository and the masks from
results/rq1_transformer_2026_reproduction/authors_code/shipped_models_exact_inputs/hard_negative_masks.npy. Prints
only.
"""
import argparse

import numpy as np
import pandas as pd

from experiments.common.data_loading import COST_OF_FALSE_ALARM, COST_OF_MISSED_FAILURE
from experiments.rq1_transformer_2026_reproduction.authors_code.file_locations import (
    AUTHORS_REPOSITORY, HARD_NEGATIVE_MASKS_FILE, TEST_LABELS, TRAINING_LABELS)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("training_probabilities", help=".npy file with one probability per training row")
    parser.add_argument("test_probabilities", help=".npy file with one probability per test row")
    parser.add_argument("cutoff", nargs="?", type=float, default=0.5, help="a row is flagged above this probability")
    return parser.parse_args()


def main():
    args = parse_arguments()
    training_probabilities = np.load(args.training_probabilities).ravel()
    test_probabilities = np.load(args.test_probabilities).ravel()
    cutoff = args.cutoff
    y_train = pd.read_csv(AUTHORS_REPOSITORY / TRAINING_LABELS)["class"].to_numpy()
    y_test = pd.read_csv(AUTHORS_REPOSITORY / TEST_LABELS)["class"].to_numpy()
    masks = np.load(HARD_NEGATIVE_MASKS_FILE, allow_pickle=True).item()
    hard_train, removed, hard_test = masks["hard_tr"], masks["removed"], masks["hard_te"]
    training_negative = y_train == 0
    training_negative_probabilities = training_probabilities[training_negative]
    test_negative_probabilities = test_probabilities[y_test == 0]
    groups = [("train negatives kept by RENN (seen)", training_negative_probabilities[~removed]),
              ("train negatives removed by RENN (never seen)", training_negative_probabilities[removed]),
              ("  of which hard by the one-shot rule", training_negative_probabilities[hard_train & removed]),
              ("train negatives hard by the one-shot rule, kept (seen)",
               training_negative_probabilities[hard_train & ~removed]),
              ("test negatives hard by the one-shot rule (never seen)", test_negative_probabilities[hard_test]),
              ("other test negatives", test_negative_probabilities[~hard_test]),
              ("train positives", training_probabilities[~training_negative]),
              ("test positives", test_probabilities[y_test == 1])]
    training_cost = (COST_OF_MISSED_FAILURE * int((training_probabilities[~training_negative] <= cutoff).sum())
                     + COST_OF_FALSE_ALARM * int((training_negative_probabilities > cutoff).sum()))
    test_cost = (COST_OF_MISSED_FAILURE * int((test_probabilities[y_test == 1] <= cutoff).sum())
                 + COST_OF_FALSE_ALARM * int((test_negative_probabilities > cutoff).sum()))
    print(f"cut-off {cutoff}; train cost {training_cost}, test cost {test_cost}")
    for label, probabilities in groups:
        print(f"  {label:58s} n={len(probabilities):6d}  flagged {(probabilities > cutoff).mean():.4f}  "
              f"mean p {probabilities.mean():.4f}")


if __name__ == "__main__":
    main()
