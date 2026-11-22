"""Follow-up to the hard-negative comparison of the models released by Beikmohammadi et al.: training negatives split by
rule and RENN.

Part of RQ1b (diagnostic). hard_negative_check_shipped_models.py compares the training negatives that RENN removed
with the test negatives that the one-shot rule calls hard (both are defined there). The two groups are not defined
in quite the same way: RENN removes rows over several repetitions while the table changes, and the one-shot rule is
applied once against the final table. This follow-up therefore scores the training negatives that the one-shot rule
itself calls hard, split by whether RENN removed them (never seen in training) or kept them (seen), together with
the removed rows that the rule does not call hard and the hard test negatives, with each of the five released
models. As in the main comparison, the test labels are used only to define the groups.

    python -m experiments.rq1_transformer_2026_reproduction.authors_code.hard_negative_check_shipped_models_followup

Needs the exact inputs in the authors' repository (prepare_authors_inputs.py), the masks in
results/rq1_transformer_2026_reproduction/authors_code/shipped_models_exact_inputs/hard_negative_masks.npy, and loads
the five released models. Writes hard_negative_check_followup.json in the same folder (per group: n, and per run
flag_rate_0.5, mean_p, median_p) and prints the mean flag rate of each group.
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

from experiments.rq1_transformer_2026_reproduction.authors_code.file_locations import (
    AUTHORS_REPOSITORY, HARD_NEGATIVE_MASKS_FILE, SHIPPED_MODELS_EXACT_INPUTS_DIR, TEST_LABELS, TEST_TABLE_SCALED,
    TRAINING_LABELS, TRAINING_TABLE_SCALED, shipped_model_folders)

OUTPUT_FILE = SHIPPED_MODELS_EXACT_INPUTS_DIR / "hard_negative_check_followup.json"


def main():
    import tf_keras as keras
    X_train = pd.read_csv(AUTHORS_REPOSITORY / TRAINING_TABLE_SCALED).to_numpy(np.float32)
    y_train = pd.read_csv(AUTHORS_REPOSITORY / TRAINING_LABELS)["class"].to_numpy()
    X_test = pd.read_csv(AUTHORS_REPOSITORY / TEST_TABLE_SCALED).to_numpy(np.float32)
    y_test = pd.read_csv(AUTHORS_REPOSITORY / TEST_LABELS)["class"].to_numpy()
    masks = np.load(HARD_NEGATIVE_MASKS_FILE, allow_pickle=True).item()
    hard_train, removed, hard_test = masks["hard_tr"], masks["removed"], masks["hard_te"]
    training_negatives = X_train[y_train == 0]
    # the group names are the keys of the reported file
    groups = {"train neg, hard by the one-shot rule, removed by RENN (never seen)":
                  training_negatives[hard_train & removed],
              "train neg, hard by the one-shot rule, kept by RENN (seen)": training_negatives[hard_train & ~removed],
              "train neg, removed but not hard by the one-shot rule (never seen)":
                  training_negatives[~hard_train & removed],
              "test neg, hard by the one-shot rule (never seen)": X_test[y_test == 0][hard_test]}
    summary = {group: {"n": int(len(rows))} for group, rows in groups.items()}
    for model_folder in shipped_model_folders():
        run = model_folder.name[-4:]
        model = keras.models.load_model(str(model_folder), compile=False)
        for group, rows in groups.items():
            probabilities = model.predict(rows.reshape(-1, rows.shape[1], 1), batch_size=256, verbose=0).ravel()
            summary[group][run] = {"flag_rate_0.5": float((probabilities > 0.5).mean()),
                                   "mean_p": float(probabilities.mean()), "median_p": float(np.median(probabilities))}
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as handle:
        json.dump(summary, handle, indent=1)
    for group in groups:
        runs = [key for key in summary[group] if key.startswith("run")]
        flag_rates = [summary[group][run]["flag_rate_0.5"] for run in runs]
        mean_probability = np.mean([summary[group][run]["mean_p"] for run in runs])
        print(f"{group:70s} n={summary[group]['n']:5d} flagged at 0.5: {np.mean(flag_rates):.4f} "
              f"(runs {', '.join(f'{rate:.3f}' for rate in flag_rates)})  mean probability {mean_probability:.4f}",
              flush=True)


if __name__ == "__main__":
    main()
