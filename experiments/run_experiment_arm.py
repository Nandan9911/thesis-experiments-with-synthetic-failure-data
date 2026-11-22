"""Run one experiment arm (see experiment_arms.py) on one or more seeds.

This is the runner behind every classifier arm of RQ1 (the reproduction of the conditional GAN of Ranasinghe et al. and
its duplication control, which adds copies of real failures instead of generated rows) and RQ2 (the survey of other
generators and classifiers), and behind the reference arms they are compared with. It implements the evaluation protocol
described in the thesis. For every seed the protocol is:

1. Five-fold stratified cross-validation on the 60,000 training rows (stratified: every fold has the same share of
   failures). Inside each fold the preprocessing and the generator are fitted on the training part only, and the
   classifier is trained on the (augmented) training part and predicts the held-out part. Putting the five held-out
   predictions together gives one *out-of-fold* probability for every training row, from a model that never saw
   that row.
2. The decision threshold is the one that minimises the challenge cost (500 per missed failure + 10 per false alarm)
   on these out-of-fold probabilities.
3. With --final-test, the five fold models predict the official 16,000-row test set, their probabilities are
   averaged, and the frozen threshold from step 2 turns them into decisions; a 95 % bootstrap interval of the test
   cost is added (test rows resampled, decisions held fixed). This is the only place where the test labels are used.
4. With --final-test and unless --no-refit is given, one extra model is refitted on all 60,000 training rows and
   scored on the test set with the same threshold. This is a secondary check only: the refitted model's
   probabilities are not on the scale on which the threshold was chosen.

Each seed fixes the fold split and the random state of every generator and classifier in it; within a seed all arms
share the same folds, so their fold costs can be compared in pairs. The arms that use tuned hyperparameters read them
from results/hyperparameter_search/lightgbm_seed11_best_parameters.json.

One JSON record per seed is appended to results/classifier_arms/arm_records.jsonl (or to the file named by the
environment variable APS_ARM_RECORDS_FILE: a cloud session writes its records to its own file, merged later by
cloud_runs/merge_cloud_results.py), and the probability vectors are saved under
results/classifier_arms/predicted_probabilities/<arm>/seed<k>_{out_of_fold,test,test_full_refit}.npy.
When the environment variables APS_CLOUD_STORE and APS_JOB_NAME are set (by the cloud queue runner), every finished
fold is cached under <APS_CLOUD_STORE>/ckpt/<APS_JOB_NAME>/, so that a cloud runtime that is replaced half-way
resumes at the next fold (see cost_and_evaluation.run_cross_validation).

Examples (from the repository root):

    python -m experiments.run_experiment_arm --arm lightgbm_tuned --seeds 11 29 47 71 101 --final-test
    python -m experiments.run_experiment_arm --arm lightgbm_with_conditional_gan_2019_rows --seeds 11 --final-test
    python -m experiments.run_experiment_arm --arm lightgbm_tuned --seeds 11 --folds 3 --final-test --no-refit
    python -m experiments.run_experiment_arm --arm lightgbm_tuned --seeds 11 --smoke-test-rows 3000
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from experiments.common.cost_and_evaluation import (bootstrap_cost_confidence_interval, cost_curve,
                                                    cost_minimising_threshold, run_cross_validation)
from experiments.common.data_loading import (CONSTANT_ATTRIBUTE, load_test_set, load_training_set,
                                             misclassification_cost)
from experiments.common.paths import ARM_RECORDS_FILE, CLASSIFIER_ARMS_DIR, arm_probability_file
from experiments.experiment_arms import ARMS, build_arm


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS), help="name of the arm in experiment_arms.py")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11],
                        help="random seeds; the thesis uses 11 29 47 (and 71 101 for the main arms)")
    parser.add_argument("--folds", type=int, default=5,
                        help="number of cross-validation folds (default 5; another number adds _<k>_folds to the "
                             "arm name in the records)")
    parser.add_argument("--final-test", action="store_true", help="also evaluate once on the official test set")
    parser.add_argument("--no-refit", action="store_true",
                        help="skip the secondary model refitted on all training rows")
    parser.add_argument("--override", type=str, default="{}",
                        help='JSON object merged into the arm options, e.g. \'{"use_gpu_histograms": true}\'')
    parser.add_argument("--smoke-test-rows", type=int, default=None,
                        help="quick check only: use a stratified sample of this many training rows; the record goes "
                             "to arm_records_smoke_test.jsonl and the probabilities to smoke_test_<arm>/")
    return parser.parse_args()


def main():
    args = parse_arguments()

    override = args.override.strip()
    if override.startswith("{{") and override.endswith("}}"):   # a command written with str.format-style doubled braces
        override = override[1:-1]
    options = dict(ARMS[args.arm])
    options.update(json.loads(override))
    make_classifier, preprocess, augment, description = build_arm(args.arm, options)

    X_train, y_train = load_training_set()
    X_train = X_train.drop(columns=[CONSTANT_ATTRIBUTE])
    X_test, y_test = load_test_set()
    X_test = X_test.drop(columns=[CONSTANT_ATTRIBUTE])

    records_file = Path(os.environ.get("APS_ARM_RECORDS_FILE") or ARM_RECORDS_FILE)
    if args.smoke_test_rows:
        from sklearn.model_selection import train_test_split
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=args.smoke_test_rows, stratify=y_train,
                                                  random_state=0)
        X_train = X_train.reset_index(drop=True)
        records_file = CLASSIFIER_ARMS_DIR / "arm_records_smoke_test.jsonl"
    records_file.parent.mkdir(parents=True, exist_ok=True)

    # A run with another number of folds gets its own name, so that a 3-fold run never mixes with the 5-fold run of
    # the same arm.
    run_name = args.arm if args.folds == 5 else f"{args.arm}_{args.folds}_folds"
    if args.smoke_test_rows:
        run_name = f"smoke_test_{run_name}"    # never overwrite the probability files of a real run

    for seed in args.seeds:
        started = time.time()
        print(f"== {run_name} seed {seed}", flush=True)
        fold_cache_dir = None
        if os.environ.get("APS_CLOUD_STORE") and os.environ.get("APS_JOB_NAME") and not args.smoke_test_rows:
            fold_cache_dir = os.path.join(os.environ["APS_CLOUD_STORE"], "ckpt", os.environ["APS_JOB_NAME"])
        # With the fold cache, the test set is scored inside the fold loop (a fold restored from the cache has no
        # model left to score it afterwards); otherwise the fold models score it after the cross-validation.
        result = run_cross_validation(run_name, make_classifier, X_train, y_train, n_splits=args.folds, seed=seed,
                                      augment=augment, preprocess=preprocess, cache_dir=fold_cache_dir,
                                      X_test=(X_test if (fold_cache_dir and args.final_test) else None))
        threshold, _ = cost_minimising_threshold(result.out_of_fold_probabilities, result.y)
        out_of_fold_summary = result.summary(threshold)
        record = {**description, "arm": run_name, "seed": seed, "folds": args.folds,
                  "out_of_fold": {key: value for key, value in out_of_fold_summary.items() if key != "name"},
                  "cross_validation_seconds": time.time() - started}
        probability_file = arm_probability_file(run_name, seed, "out_of_fold")
        probability_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(probability_file, result.out_of_fold_probabilities)

        if args.final_test:
            test_started = time.time()
            # Primary result: the average of the five fold models, with the out-of-fold threshold.
            test_probabilities = result.fold_average_probabilities(X_test)
            test_decisions = (test_probabilities >= threshold).astype(int)
            test_result = misclassification_cost(y_test, test_decisions)
            test_result["threshold"] = threshold
            test_result["cost_95_percent_interval"] = list(bootstrap_cost_confidence_interval(y_test, test_decisions))
            record["test"] = test_result
            # the lowest cost any threshold could reach on the test set (diagnostic)
            record["test_oracle_minimum_cost"] = int(cost_curve(test_probabilities, y_test)["cost"].min())
            np.save(arm_probability_file(run_name, seed, "test"), test_probabilities)

            # Secondary result: one model refitted on all training rows, same threshold. Its seed (seed * 1000)
            # differs from those of the fold models (seed * 100 + fold).
            if not args.no_refit:
                X_refit, y_refit, X_test_refit = X_train, y_train, X_test
                if preprocess is not None:
                    transform = preprocess(X_refit, y_refit)
                    X_refit, X_test_refit = transform(X_refit), transform(X_test)
                if augment is not None:
                    X_refit, y_refit = augment(X_refit, y_refit, seed * 1000)
                refit_model = make_classifier(seed * 1000)
                refit_model.fit(X_refit, y_refit)
                refit_probabilities = refit_model.predict_proba(X_test_refit)[:, 1]
                refit_result = misclassification_cost(y_test, (refit_probabilities >= threshold).astype(int))
                refit_result["threshold"] = threshold
                record["test_full_refit"] = refit_result
                record["test_full_refit_oracle_minimum_cost"] = int(
                    cost_curve(refit_probabilities, y_test)["cost"].min())
                np.save(arm_probability_file(run_name, seed, "test_full_refit"), refit_probabilities)
            record["final_test_seconds"] = time.time() - test_started
            message = (f"   TEST (average of fold models) cost {test_result['cost']} (FN {test_result['fn']}, "
                       f"FP {test_result['fp']}) at threshold {threshold:.4f}; "
                       f"95% interval {test_result['cost_95_percent_interval']}")
            if "test_full_refit" in record:
                refit = record["test_full_refit"]
                message += f"; full refit {refit['cost']} (FN {refit['fn']}, FP {refit['fp']})"
            print(message, flush=True)

        print(f"   OUT-OF-FOLD cost {out_of_fold_summary['cost']} (FN {out_of_fold_summary['fn']}, "
              f"FP {out_of_fold_summary['fp']}) at threshold {threshold:.4f}; "
              f"{out_of_fold_summary['cost_scaled_to_test_set_size']:.0f} scaled to the 16,000 rows of the test set; "
              f"{time.time() - started:.0f}s", flush=True)
        with open(records_file, "a") as handle:
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
