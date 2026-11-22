"""Tree models on the preprocessing pipeline of Beikmohammadi et al. under the common evaluation protocol, with
ablations of the resampling (RQ1b).

The study of Beikmohammadi et al. reports few false alarms at the 0.5 cut-off for its cost-sensitive Transformer.
Its preprocessing also changes the training table a great deal: missing values filled by a BayesianRidge iterative
imputer, SVM-SMOTE adding synthetic failure rows up to a ratio of 1:2, and RENN removing healthy rows near the class
boundary (preprocessing_2026.py explains each step). This script trains the thesis's tree learners on that
preprocessing, so that the effect of the preprocessing can be separated from the effect of the network. Four
variants:

  full_pipeline    imputation, SVM-SMOTE, then RENN (as published)
  without_renn     imputation and SVM-SMOTE only
  renn_only        imputation and RENN only (no synthetic rows)
  imputation_only  imputation only

The protocol is the common evaluation protocol: stratified k-fold cross-validation on the 60,000 training rows, with the
column selection, the imputer and the resampling fitted inside each training fold only; the decision threshold minimises
the cost on the out-of-fold predictions (each training row predicted by the fold model that did not see it); and with
--final-test the official test set, imputed by each fold's own imputer, is scored once with the average of the fold
models' probabilities at that frozen threshold. Tree learners need no scaling, so step 5 of the published pipeline is
left out. The cost at the 0.5 cut-off is recorded as well, because that is where the resampling changes the probability
scale.

The reported runs (seed 11, three folds, 10 imputer rounds, compared with the three-fold baseline arm
lightgbm_tuned_3_folds), from the repository root:

    python -m experiments.rq1_transformer_2026_reproduction.run_tree_models_on_2026_preprocessing \\
        --variant full_pipeline --seeds 11 --folds 3 --imputer-rounds 10 --final-test
    ... and the same with --variant without_renn, renn_only and imputation_only

Reads the two data files. Appends one record per seed to results/classifier_arms/arm_records.jsonl (or, as
run_experiment_arm.py does, to the file named by the environment variable APS_ARM_RECORDS_FILE), under the arm name
<classifier>_on_2026_preprocessing_<variant> (e.g. lightgbm_on_2026_preprocessing_full_pipeline), and saves the
out-of-fold and test probabilities as
results/classifier_arms/predicted_probabilities/<arm>/seed<k>_{out_of_fold,test}.npy. The imputed folds are cached
in cache/.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from experiments.common.classifiers import CLASSIFIER_FACTORIES, load_tuned_hyperparameters
from experiments.common.cost_and_evaluation import (bootstrap_cost_confidence_interval, cost_curve,
                                                    cost_minimising_threshold)
from experiments.common.data_loading import (CONSTANT_ATTRIBUTE, TEST_SET_ROWS, load_test_set, load_training_set,
                                             misclassification_cost)
from experiments.common.paths import ARM_RECORDS_FILE, arm_probability_file
from experiments.rq1_transformer_2026_reproduction.preprocessing_2026 import (impute_fold_as_in_2026_study,
                                                                               resample_as_in_2026_study)

# which resampling steps each variant applies after the imputation
VARIANTS = {"full_pipeline": dict(use_smote=True, use_renn=True),
            "without_renn": dict(use_smote=True, use_renn=False),
            "renn_only": dict(use_smote=False, use_renn=True),
            "imputation_only": dict(use_smote=False, use_renn=False)}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="full_pipeline", choices=list(VARIANTS))
    parser.add_argument("--model", default="lightgbm", choices=sorted(CLASSIFIER_FACTORIES),
                        help="classifier (a key of CLASSIFIER_FACTORIES), with its tuned hyperparameters if any")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--imputer-rounds", type=int, default=25, help="maximum rounds of the iterative imputer")
    parser.add_argument("--final-test", action="store_true", help="also score the official test set once")
    return parser.parse_args()


def main():
    args = parse_arguments()
    X, y = load_training_set()
    X = X.drop(columns=[CONSTANT_ATTRIBUTE])
    X_test, y_test = load_test_set()
    X_test = X_test.drop(columns=[CONSTANT_ATTRIBUTE])
    classifier_parameters = load_tuned_hyperparameters(args.model)
    factory = CLASSIFIER_FACTORIES[args.model]
    arm = f"{args.model}_on_2026_preprocessing_{args.variant}"
    records_file = Path(os.environ.get("APS_ARM_RECORDS_FILE") or ARM_RECORDS_FILE)
    for seed in args.seeds:
        started = time.time()
        folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        out_of_fold_probabilities = np.zeros(len(y))
        fold_of_row = np.zeros(len(y), dtype=int)
        test_probabilities_per_fold = []
        for fold, (train_index, held_out_index) in enumerate(folds.split(X, y)):
            X_train, X_held_out, _, _, X_test_imputed = impute_fold_as_in_2026_study(
                X.iloc[train_index], X.iloc[held_out_index], cache_tag=f"seed{seed}_fold{fold}",
                imputer_rounds=args.imputer_rounds, seed=seed, X_additional=X_test)
            X_train, y_train = resample_as_in_2026_study(X_train, y[train_index], seed=seed * 100 + fold,
                                                         **VARIANTS[args.variant])
            model = factory(seed * 100 + fold, **classifier_parameters)
            model.fit(X_train, y_train)
            out_of_fold_probabilities[held_out_index] = model.predict_proba(X_held_out)[:, 1]
            fold_of_row[held_out_index] = fold
            if X_test_imputed is not None:
                test_probabilities_per_fold.append(model.predict_proba(X_test_imputed)[:, 1])
            fold_threshold, fold_cost = cost_minimising_threshold(out_of_fold_probabilities[held_out_index],
                                                                  y[held_out_index])
            fold_cost_at_half = misclassification_cost(y[held_out_index],
                                                       (out_of_fold_probabilities[held_out_index] >= 0.5).astype(int))
            print(f"  [{arm}] fold {fold}: cost {fold_cost} at the fold's own best threshold {fold_threshold:.4f}; "
                  f"cost at 0.5 {fold_cost_at_half['cost']}", flush=True)
        threshold, _ = cost_minimising_threshold(out_of_fold_probabilities, y)
        at_threshold = misclassification_cost(y, (out_of_fold_probabilities >= threshold).astype(int))
        at_half = misclassification_cost(y, (out_of_fold_probabilities >= 0.5).astype(int))
        fold_costs = []
        for fold in range(args.folds):
            in_fold = fold_of_row == fold
            fold_decisions = (out_of_fold_probabilities[in_fold] >= threshold).astype(int)
            fold_costs.append(misclassification_cost(y[in_fold], fold_decisions)["cost"])
        # the record schema of results/classifier_arms/arm_records.jsonl
        record = {"arm": arm, "classifier": args.model, "preprocessing": f"2026_preprocessing_{args.variant}",
                  "augmentation": args.variant, "classifier_parameters": classifier_parameters, "seed": seed,
                  "folds": args.folds, "imputer_rounds": args.imputer_rounds,
                  "out_of_fold": {**at_threshold, "threshold": threshold, "fold_costs": fold_costs,
                                  "cost_scaled_to_test_set_size": at_threshold["cost"] * TEST_SET_ROWS / len(y),
                                  "cost_at_0.5": at_half["cost"], "fn_at_0.5": at_half["fn"],
                                  "fp_at_0.5": at_half["fp"]},
                  "cross_validation_seconds": time.time() - started}
        probability_file = arm_probability_file(arm, seed, "out_of_fold")
        probability_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(probability_file, out_of_fold_probabilities)
        if args.final_test and test_probabilities_per_fold:
            test_started = time.time()
            # primary result: the average of the fold models, at the OOF threshold
            test_probabilities = np.mean(test_probabilities_per_fold, axis=0)
            test_decisions = (test_probabilities >= threshold).astype(int)
            test_result = misclassification_cost(y_test, test_decisions)
            test_result["threshold"] = threshold
            test_result["cost_95_percent_interval"] = list(bootstrap_cost_confidence_interval(y_test, test_decisions))
            test_at_half = misclassification_cost(y_test, (test_probabilities >= 0.5).astype(int))
            record["test"] = test_result
            record["test_at_0.5"] = test_at_half
            # the lowest cost any threshold reaches on the test labels (diagnostic)
            record["test_oracle_minimum_cost"] = int(cost_curve(test_probabilities, y_test)["cost"].min())
            np.save(arm_probability_file(arm, seed, "test"), test_probabilities)
            record["final_test_seconds"] = time.time() - test_started
            print(f"   TEST (average of fold models) cost {test_result['cost']} (FN {test_result['fn']}, "
                  f"FP {test_result['fp']}) at threshold {threshold:.4f}; at 0.5: {test_at_half['cost']} "
                  f"(FN {test_at_half['fn']}, FP {test_at_half['fp']})", flush=True)
        print(f"   OOF cost {at_threshold['cost']} (FN {at_threshold['fn']}, FP {at_threshold['fp']}) at threshold "
              f"{threshold:.4f}; at 0.5: {at_half['cost']}; {time.time() - started:.0f}s", flush=True)
        records_file.parent.mkdir(parents=True, exist_ok=True)
        with open(records_file, "a") as handle:
            handle.write(json.dumps(record, default=float) + "\n")


if __name__ == "__main__":
    main()
