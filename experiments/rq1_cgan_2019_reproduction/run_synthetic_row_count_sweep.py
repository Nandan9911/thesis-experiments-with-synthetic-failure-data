"""The cost as a function of the number of synthetic failure rows, with the count chosen out of fold and, as a
diagnostic, on the test set (RQ1a).

The study of Ranasinghe et al. appended 2,000 generated failure rows to the training data; the notebook attributed to it
explores counts from 250 to 5,000. The main RQ1 arms (experiment_arms.py) use 2,000 rows. This script asks how the
cost depends on the count and how much the way the count is chosen matters. It runs, for each seed:

(a) Out-of-fold selection, as in the common evaluation protocol. Five-fold stratified cross-validation
    on the 60,000 training rows (stratified: every fold keeps the training file's share of failures). In each fold
    one generator is fitted on the fold's training failures and sampled once, for the largest count; the classifier
    is then trained once per count on the fold's training rows plus the first <count> generated rows, and predicts
    the held-out fold. The held-out predictions of the five folds together are the out-of-fold (OOF) predictions:
    one prediction per training row from a model that did not see it. For every count the cost-minimising threshold
    and the cost are computed on the OOF predictions, and the count with the lowest OOF cost is recorded as
    "out_of_fold_selected_count" (count 0, no synthetic rows, competes too). Because every count draws from the same
    pool, the generator's own randomness is held fixed across counts.
    The same loop runs a duplication control: instead of generated rows, <count> copies of randomly chosen real
    failure rows of the fold. If generated rows do no better than copies, their effect is only a heavier weight on
    the failure class.
(b) Test-set selection, simulated deliberately to measure its effect (a diagnostic). One generator is fitted on all
    1,000 training failures, one classifier is trained on all training rows plus the first <count> generated rows,
    and every count's classifier is scored on the official 16,000-row test set at its test-optimal ("oracle")
    threshold. The minimum over the counts ("test_selected_minimum_oracle_cost") is the cost obtained when both the
    count and the threshold are read off the test set. Comparing the minimum with the out-of-fold numbers of (a)
    measures how much choosing the count on the test set changes the cost for this re-implementation.
    The test cost at each count's OOF threshold is recorded as well (unless --skip-out-of-fold).

The generators are the cGAN of Ranasinghe et al. (conditional_gan_2019.py) and Forest-Flow (RQ2,
forest_flow_generator.py). The classifier is any key of experiments.common.classifiers.CLASSIFIER_FACTORIES, with its
tuned hyperparameters where a search was run (only LightGBM was tuned). Synthetic rows get their missing values put back
(reapply_missingness).

Composition of the pool: the cGAN splits its generated rows evenly between its two k-means clusters and by default
returns them cluster by cluster. The sweep asks it for alternating clusters instead (alternate_clusters=True), so
that every count contains the same number of rows from each cluster, as do the RQ1 arms with 2,000 rows (1,000 per
cluster) and the notebook, which generates blocks of 250 rows with 125 from each cluster. Forest-Flow draws a
cluster label at random for every row, so its pool is mixed without reordering.

The reported run (cGAN with the notebook's 30,000 training steps, LightGBM, seed 11), from the repository root:

    python -m experiments.rq1_cgan_2019_reproduction.run_synthetic_row_count_sweep --generator conditional_gan_2019 \\
        --model lightgbm --seeds 11 --gan-training-steps 30000

Reads the two data files. Appends one JSON record per seed to
results/rq1_cgan_2019_reproduction/synthetic_row_count_sweep.jsonl and saves the probability vectors as
results/rq1_cgan_2019_reproduction/synthetic_row_count_sweep_probabilities/
<generator>_<classifier>_seed<k>_<count>_rows_{out_of_fold,test}.npy.

Fields of a record: generator, classifier, seed, counts, classifier_parameters, pool_cluster_order ("alternating"
for the cGAN, "random" for Forest-Flow), out_of_fold (per count: cost, fn,
fp, threshold, cost/fn/fp at the 0.5 cut-off, and for counts above 0 the duplication control as
duplication_control_cost, duplication_control_fn, duplication_control_fp, duplication_control_cost_at_0.5),
out_of_fold_selected_count, test (per count: cost_at_0.5, oracle_minimum_cost with oracle_fn and oracle_fp, and
cost_at_out_of_fold_threshold, fn, fp, out_of_fold_threshold), test_selected_minimum_oracle_cost and seconds. The
duplication control is computed out of fold only.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from experiments.common.classifiers import CLASSIFIER_FACTORIES, load_tuned_hyperparameters
from experiments.common.cost_and_evaluation import cost_curve, cost_minimising_threshold
from experiments.common.data_loading import (CONSTANT_ATTRIBUTE, load_test_set, load_training_set,
                                             misclassification_cost)
from experiments.common.paths import CGAN_2019_RESULTS_DIR
from experiments.rq1_cgan_2019_reproduction.conditional_gan_2019 import ConditionalGan2019
from experiments.rq2_survey_of_methods.forest_flow_generator import ForestFlowFailureGenerator

SWEEP_RECORDS_FILE = CGAN_2019_RESULTS_DIR / "synthetic_row_count_sweep.jsonl"
SWEEP_PROBABILITIES_DIR = CGAN_2019_RESULTS_DIR / "synthetic_row_count_sweep_probabilities"
# the counts of the notebook attributed to the study of Ranasinghe et al.
SYNTHETIC_ROW_COUNTS = [250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000]
GENERATORS = ("conditional_gan_2019", "forest_flow")


def make_generator(generator_name, seed, gan_training_steps):
    if generator_name == "conditional_gan_2019":
        return ConditionalGan2019(steps=gan_training_steps, seed=seed)
    if generator_name == "forest_flow":
        return ForestFlowFailureGenerator(seed=seed)
    raise ValueError(generator_name)


def sample_pool(generator, n_rows, reapply_missingness, seed):
    """The pool of n_rows generated failure rows from which every count takes its first rows (see the docstring)."""
    if isinstance(generator, ConditionalGan2019):
        return generator.sample(n_rows, reapply_missingness=reapply_missingness, seed=seed, alternate_clusters=True)
    return generator.sample(n_rows, reapply_missingness=reapply_missingness, seed=seed)


def probability_file(generator_name, classifier_name, seed, count, kind):
    return SWEEP_PROBABILITIES_DIR / f"{generator_name}_{classifier_name}_seed{seed}_{count}_rows_{kind}.npy"


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generator", default="conditional_gan_2019", choices=GENERATORS)
    parser.add_argument("--model", default="lightgbm", choices=sorted(CLASSIFIER_FACTORIES),
                        help="classifier (a key of CLASSIFIER_FACTORIES)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11])
    parser.add_argument("--gan-training-steps", type=int, default=10000,
                        help="training steps of the cGAN (the notebook uses 30,000; ignored for forest_flow)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--skip-out-of-fold", action="store_true", help="only run the test-set selection (b)")
    return parser.parse_args()


def main():
    args = parse_arguments()
    counts = SYNTHETIC_ROW_COUNTS
    X, y = load_training_set()
    X = X.drop(columns=[CONSTANT_ATTRIBUTE])
    X_test, y_test = load_test_set()
    X_test = X_test.drop(columns=[CONSTANT_ATTRIBUTE])
    classifier_parameters = load_tuned_hyperparameters(args.model)
    factory = CLASSIFIER_FACTORIES[args.model]
    if args.model in ("random_forest", "sklearn_gradient_boosting"):
        # These two cannot handle missing values. The medians are taken over the whole training file, not per fold,
        # and the generator is then fitted on complete rows, so there are no missing values to put back.
        medians = X.median()
        X = X.fillna(medians)
        X_test = X_test.fillna(medians)
    reapply_missingness = True
    SWEEP_PROBABILITIES_DIR.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        record = {"generator": args.generator, "classifier": args.model, "seed": seed, "counts": counts,
                  "classifier_parameters": classifier_parameters,
                  "pool_cluster_order": "alternating" if args.generator == "conditional_gan_2019" else "random"}
        started = time.time()
        # ------------------------------------------- (a) out-of-fold selection: a generator refitted inside each fold
        if not args.skip_out_of_fold:
            out_of_fold_probabilities = {count: np.zeros(len(y)) for count in counts + [0]}
            duplicate_out_of_fold_probabilities = {count: np.zeros(len(y)) for count in counts}
            folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
            for fold, (train_index, held_out_index) in enumerate(folds.split(X, y)):
                X_train, y_train, X_held_out = X.iloc[train_index], y[train_index], X.iloc[held_out_index]
                fold_seed = seed * 100 + fold
                generator = make_generator(args.generator, fold_seed, args.gan_training_steps)
                generator = generator.fit(X_train[y_train == 1])
                pool = sample_pool(generator, max(counts), reapply_missingness, seed=fold_seed + 1)
                real_failures = X_train[y_train == 1]
                rng = np.random.default_rng(fold_seed)
                for count in [0] + counts:
                    X_augmented = pd.concat([X_train, pool.iloc[:count]], ignore_index=True) if count else X_train
                    y_augmented = np.concatenate([y_train, np.ones(count, dtype=y_train.dtype)]) if count else y_train
                    model = factory(fold_seed, **classifier_parameters)
                    model.fit(X_augmented, y_augmented)
                    out_of_fold_probabilities[count][held_out_index] = model.predict_proba(X_held_out)[:, 1]
                    if count:
                        copies = real_failures.iloc[rng.integers(0, len(real_failures), count)]
                        X_duplicated = pd.concat([X_train, copies], ignore_index=True)
                        y_duplicated = np.concatenate([y_train, np.ones(count, dtype=y_train.dtype)])
                        duplicate_model = factory(fold_seed, **classifier_parameters)
                        duplicate_model.fit(X_duplicated, y_duplicated)
                        duplicate_out_of_fold_probabilities[count][held_out_index] = (
                            duplicate_model.predict_proba(X_held_out)[:, 1])
                print(f"  seed {seed} fold {fold} done ({time.time() - started:.0f}s)", flush=True)
            record["out_of_fold"] = {}
            for count in [0] + counts:
                threshold, cost = cost_minimising_threshold(out_of_fold_probabilities[count], y)
                at_threshold = misclassification_cost(y, (out_of_fold_probabilities[count] >= threshold).astype(int))
                at_half = misclassification_cost(y, (out_of_fold_probabilities[count] >= 0.5).astype(int))
                record["out_of_fold"][count] = {"cost": cost, "fn": at_threshold["fn"], "fp": at_threshold["fp"],
                                                "threshold": threshold, "cost_at_0.5": at_half["cost"],
                                                "fn_at_0.5": at_half["fn"], "fp_at_0.5": at_half["fp"]}
                if count:
                    duplicate_threshold, duplicate_cost = cost_minimising_threshold(
                        duplicate_out_of_fold_probabilities[count], y)
                    duplicate_at_threshold = misclassification_cost(
                        y, (duplicate_out_of_fold_probabilities[count] >= duplicate_threshold).astype(int))
                    duplicate_at_half = misclassification_cost(
                        y, (duplicate_out_of_fold_probabilities[count] >= 0.5).astype(int))
                    record["out_of_fold"][count]["duplication_control_cost"] = duplicate_cost
                    record["out_of_fold"][count]["duplication_control_fn"] = duplicate_at_threshold["fn"]
                    record["out_of_fold"][count]["duplication_control_fp"] = duplicate_at_threshold["fp"]
                    record["out_of_fold"][count]["duplication_control_cost_at_0.5"] = duplicate_at_half["cost"]
                np.save(probability_file(args.generator, args.model, seed, count, "out_of_fold"),
                        out_of_fold_probabilities[count])
            # min() keeps the first of equal costs in this order: ties go to the smaller count, and 0 (no synthetic
            # rows) is selected only when it is strictly the cheapest
            record["out_of_fold_selected_count"] = min(counts + [0],
                                                       key=lambda count: record["out_of_fold"][count]["cost"])
            print("  out-of-fold cost per count: "
                  + ", ".join(f"{count}:{record['out_of_fold'][count]['cost']}" for count in [0] + counts), flush=True)
        # ------------------------------------------------------------------------ (b) test-set selection (diagnostic)
        # one generator on all training failures, every count scored on the test set
        generator = make_generator(args.generator, seed * 1000, args.gan_training_steps).fit(X[y == 1])
        pool = sample_pool(generator, max(counts), reapply_missingness, seed=seed * 1000 + 1)
        record["test"] = {}
        for count in [0] + counts:
            X_augmented = pd.concat([X, pool.iloc[:count]], ignore_index=True) if count else X
            y_augmented = np.concatenate([y, np.ones(count, dtype=y.dtype)]) if count else y
            model = factory(seed * 1000, **classifier_parameters)
            model.fit(X_augmented, y_augmented)
            test_probabilities = model.predict_proba(X_test)[:, 1]
            curve = cost_curve(test_probabilities, y_test)
            # the "oracle" threshold is the one that minimises the cost on the test labels themselves
            entry = {"cost_at_0.5": misclassification_cost(y_test, (test_probabilities >= 0.5).astype(int))["cost"],
                     "oracle_minimum_cost": int(curve["cost"].min()),
                     "oracle_fn": int(curve.loc[curve.cost.idxmin(), "fn"]),
                     "oracle_fp": int(curve.loc[curve.cost.idxmin(), "fp"])}
            if not args.skip_out_of_fold:
                out_of_fold_threshold = record["out_of_fold"][count]["threshold"]
                at_out_of_fold_threshold = misclassification_cost(
                    y_test, (test_probabilities >= out_of_fold_threshold).astype(int))
                entry.update({"cost_at_out_of_fold_threshold": at_out_of_fold_threshold["cost"],
                              "fn": at_out_of_fold_threshold["fn"], "fp": at_out_of_fold_threshold["fp"],
                              "out_of_fold_threshold": out_of_fold_threshold})
            record["test"][count] = entry
            np.save(probability_file(args.generator, args.model, seed, count, "test"), test_probabilities)
        record["test_selected_minimum_oracle_cost"] = min(record["test"][count]["oracle_minimum_cost"]
                                                          for count in counts)
        record["seconds"] = time.time() - started
        print("  test oracle minimum cost per count: "
              + ", ".join(f"{count}:{record['test'][count]['oracle_minimum_cost']}" for count in [0] + counts),
              flush=True)
        SWEEP_RECORDS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SWEEP_RECORDS_FILE, "a") as handle:
            handle.write(json.dumps(record, default=float) + "\n")


if __name__ == "__main__":
    main()
