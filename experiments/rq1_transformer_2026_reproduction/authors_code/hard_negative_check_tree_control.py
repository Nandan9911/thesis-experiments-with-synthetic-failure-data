"""Control for the hard-negative comparison: the thesis's own tree models on the same groups of negatives (RQ1b).

hard_negative_check_shipped_models.py finds that the models released by Beikmohammadi et al. flag the training negatives
that RENN removed more often than the test negatives that RENN's rule calls hard; neither group is in the published
training table. This control measures how often models that were not trained on either group flag them: three classifier
arms of the thesis (seed 11), trained under the common evaluation protocol on the raw training rows, never on the test
rows. Every training row is scored out of fold, that is by the fold model that did not train on it, and every test row
by the average of the five fold models, so no score in either group comes from a model that saw the row. Each arm flags
at its own out-of-fold threshold (a row is flagged at or above it). If the hard test negatives are as hard as the
removed training negatives, these models flag them at similar rates; the reported run found 61 to 72 % of the hard test
negatives flagged, against 64 to 78 % of the hard training negatives that RENN removed.

As in the main comparison, the test labels are used only to define the groups (diagnostic).

    python -m experiments.rq1_transformer_2026_reproduction.authors_code.hard_negative_check_tree_control

Reads the label files from the authors' repository, the masks from
shipped_models_exact_inputs/hard_negative_masks.npy, the arm records from results/classifier_arms/arm_records.jsonl
(the out-of-fold threshold) and the arms' saved probabilities from results/classifier_arms/predicted_probabilities/.
Writes
results/rq1_transformer_2026_reproduction/authors_code/shipped_models_exact_inputs/hard_negative_check_tree_control.json:
per arm the threshold and the fraction flagged in each group.
"""
import json

import numpy as np
import pandas as pd

from experiments.common.paths import ARM_RECORDS_FILE, arm_probability_file
from experiments.rq1_transformer_2026_reproduction.authors_code.file_locations import (
    AUTHORS_REPOSITORY, HARD_NEGATIVE_MASKS_FILE, SHIPPED_MODELS_EXACT_INPUTS_DIR, TEST_LABELS, TRAINING_LABELS)

OUTPUT_FILE = SHIPPED_MODELS_EXACT_INPUTS_DIR / "hard_negative_check_tree_control.json"
CONTROL_ARMS = ("lightgbm_tuned", "xgboost_fixed_settings", "ensemble_without_missingness_indicators")
SEED = 11


def main():
    y_train = pd.read_csv(AUTHORS_REPOSITORY / TRAINING_LABELS)["class"].to_numpy()
    y_test = pd.read_csv(AUTHORS_REPOSITORY / TEST_LABELS)["class"].to_numpy()
    masks = np.load(HARD_NEGATIVE_MASKS_FILE, allow_pickle=True).item()
    hard_train, removed, hard_test = masks["hard_tr"], masks["removed"], masks["hard_te"]
    training_negative = y_train == 0
    records = {}
    with open(ARM_RECORDS_FILE) as handle:
        for line in handle:
            if line.strip():   # where an arm and seed have several records, the last one counts
                record = json.loads(line)
                records[(record["arm"], record["seed"])] = record
    summary = {}
    for arm in CONTROL_ARMS:
        out_of_fold_probabilities = np.load(arm_probability_file(arm, SEED, "out_of_fold"))
        test_probabilities = np.load(arm_probability_file(arm, SEED, "test"))
        threshold = records[(arm, SEED)]["out_of_fold"]["threshold"]
        training_negative_probabilities = out_of_fold_probabilities[training_negative]
        test_negative_probabilities = test_probabilities[y_test == 0]
        # the group names are the keys of the reported file
        summary[arm] = {
            "threshold": threshold,
            "train neg kept by RENN": float((training_negative_probabilities[~removed] >= threshold).mean()),
            "train neg removed by RENN": float((training_negative_probabilities[removed] >= threshold).mean()),
            "train neg hard, removed": float((training_negative_probabilities[hard_train & removed]
                                              >= threshold).mean()),
            "train neg hard, kept": float((training_negative_probabilities[hard_train & ~removed]
                                           >= threshold).mean()),
            "test neg hard": float((test_negative_probabilities[hard_test] >= threshold).mean()),
            "test neg rest": float((test_negative_probabilities[~hard_test] >= threshold).mean()),
            "train pos": float((out_of_fold_probabilities[~training_negative] >= threshold).mean()),
            "test pos": float((test_probabilities[y_test == 1] >= threshold).mean())}
        print(arm, {key: (round(value, 4) if isinstance(value, float) else value)
                    for key, value in summary[arm].items()})
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as handle:
        json.dump(summary, handle, indent=1)


if __name__ == "__main__":
    main()
