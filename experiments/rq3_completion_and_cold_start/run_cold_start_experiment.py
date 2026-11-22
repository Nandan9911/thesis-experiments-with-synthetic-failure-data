"""Cold start (RQ3): what synthetic failure rows are worth when only few real failures are labelled.

A new vehicle model or a new fleet starts with very few recorded failures. This experiment imitates that situation
on APS and measures the cost as a function of the number of real failures available for training, with and
without synthetic failure rows. For every seed and every failure count n in --failure-counts:

  * The 60,000 training rows are split into five stratified folds, as for every arm.
  * Inside each training fold only the first n failures of a fixed random order are kept (the order is drawn once
    per seed and fold, so the subsets are nested: the 25 failures are part of the 50, the 50 of the 100, and so on;
    a training fold holds 800 failures, so 1000 means all of them). All healthy rows are kept.
  * The generator is fitted on those n failures and adds --synthetic-rows synthetic failures (2,000), and the tuned
    LightGBM of the thesis is trained on the result.
  * The validation fold and the official test set are left as they are: failures still occur as often as before,
    only the labelled failures available for training are scarce.
  * The decision threshold is the one that minimises the cost on the out-of-fold predictions (searched over 2,001
    quantiles of the probabilities), and the test set is scored once, with the average of the five fold models and
    that frozen threshold.

Generators (--generator)
  no_synthetic_rows           nothing is added: the reference curve
  duplicated_real_failures    copies of the n real failures, drawn with replacement: the duplication control for
                              every generator (more weight on the failure class, no new information)
  smote                       SMOTE: new rows interpolated between a failure and one of its nearest failure
                              neighbours (k = 5, fewer when n is tiny), on median-imputed attributes; 200 healthy rows
                              are added only because SMOTE needs two classes. The missing-value patterns of randomly
                              drawn real failures are put back into the new rows.
  conditional_gan_2019        the conditional GAN of Ranasinghe et al. with min-max inputs and a single sigmoid, the
                              default variant (rq1_cgan_2019_reproduction/conditional_gan_2019.py)
  tabebm                      TabEBM (rq2_survey_of_methods/tabebm_generator.py)
  forest_flow                 Forest-Flow, reduced profile (rq2_survey_of_methods/forest_flow_generator.py); its
                              k-means conditioning needs a few dozen rows and is used only from n = 40 on
  forest_flow_inpainting, unmasking_trees_inpainting
                              a completion engine of completion_engines.py used as a generator: --synthetic-rows real
                              failures are drawn with replacement from the n, each known cell of these copies is hidden
                              with probability hidden_fraction (0.5), and the engine, fitted on the same n failures,
                              completes the copies with one draw each. Cells that were missing in the real failure
                              stay missing.

--override takes a JSON object of generator options: gan_training_steps (training steps of the GAN, 30,000),
langevin_steps (Langevin sampling steps of TabEBM, 200), noise_levels, duplicates_per_row, trees_per_regressor and
tree_depth (Forest-Flow: 10, 10, 20 and 5; see forest_flow_generator.py), hidden_fraction (0.5),
unmasking_trees_depth and unmasking_trees_duplicates_per_row (Unmasking Trees discretises each attribute into
2^depth bins and masks every training row in this many random orders: 4 and 5), gpu_hist (train the Forest-Flow
trees on a GPU) and xgboost_kwargs (passed to XGBoost by Unmasking Trees, e.g. {"device": "cuda"}). gpu_hist and
xgboost_kwargs keep the keyword names of the classes they are passed to, under which they are also stored in the
records.

    python -m experiments.rq3_completion_and_cold_start.run_cold_start_experiment --generator no_synthetic_rows \\
        --seeds 11 29 47 --failure-counts 25 50 100 250 1000
    python -m experiments.rq3_completion_and_cold_start.run_cold_start_experiment --generator forest_flow \\
        --seeds 11 --failure-counts 25 50 100 250 1000 --override '{"gpu_hist": true}'

Reads the training and test files and the tuned hyperparameters (results/hyperparameter_search/). Appends one JSON
line per (generator, seed, n) to results/rq3_completion_and_cold_start/cold_start_runs.jsonl, or to the file named by
the environment variable APS_COLD_START_RECORDS_FILE (a cloud session writes its own file), and saves the averaged
test probabilities to <generator>_seed<k>_<n>_failures.npy in results/rq3_completion_and_cold_start/
cold_start_test_probabilities/. With --smoke-test-rows the run uses a stratified sample of the training rows and
writes cold_start_runs_smoke_test.jsonl and *_smoke_test.npy files instead.

Fields of a record: generator; seed; labelled_failures (n, the failures kept per training fold); synthetic_rows (rows
added per fold, 0 for no_synthetic_rows); folds; options (the --override options); seconds; out_of_fold, the
out-of-fold result at the chosen threshold: cost, tp, fn, fp, tn, recall, precision, threshold, fold_costs (the cost of
each fold at that threshold), cost_scaled_to_test_set_size (the cost scaled from 60,000 to 16,000 rows) and
fp_at_fn_le_15 (the fewest false alarms with at most 15 missed failures); and test, the result on the test set: the
same counts, threshold, cost_95_percent_interval (a bootstrap 95 % interval of the cost), and two diagnostics on the
test labels: fp_at_fn_le_6 (the fewest false alarms with at most 6 missed failures) and oracle_minimum_cost (the
lowest cost any threshold would reach).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from experiments.common.classifiers import CLASSIFIER_FACTORIES, load_tuned_hyperparameters
from experiments.common.cost_and_evaluation import (bootstrap_cost_confidence_interval, cost_curve,
                                                    cost_minimising_threshold)
from experiments.common.data_loading import (CONSTANT_ATTRIBUTE, TEST_SET_ROWS, load_test_set, load_training_set,
                                             misclassification_cost)
from experiments.common.paths import COMPLETION_AND_COLD_START_RESULTS_DIR

RECORDS_FILE = (Path(os.environ["APS_COLD_START_RECORDS_FILE"]) if os.environ.get("APS_COLD_START_RECORDS_FILE")
                else COMPLETION_AND_COLD_START_RESULTS_DIR / "cold_start_runs.jsonl")
SMOKE_TEST_RECORDS_FILE = COMPLETION_AND_COLD_START_RESULTS_DIR / "cold_start_runs_smoke_test.jsonl"
TEST_PROBABILITIES_DIR = COMPLETION_AND_COLD_START_RESULTS_DIR / "cold_start_test_probabilities"
COLD_START_GENERATORS = ["no_synthetic_rows", "duplicated_real_failures", "smote", "conditional_gan_2019", "tabebm",
                         "forest_flow", "forest_flow_inpainting", "unmasking_trees_inpainting"]


def make_generator(kind: str, synthetic_rows: int, options: dict):
    """Returns generate(X_failures, seed) -> DataFrame of synthetic_rows synthetic failure rows (missing values
    allowed), or None for no_synthetic_rows. A generator with the attribute needs_healthy_rows also takes the
    training fold's healthy rows as X_healthy."""
    if kind == "no_synthetic_rows":
        return None
    if kind == "duplicated_real_failures":
        def generate(X_failures, seed):
            rng = np.random.default_rng(seed)
            return X_failures.iloc[rng.integers(0, len(X_failures), synthetic_rows)].reset_index(drop=True)
        return generate
    if kind == "smote":
        from imblearn.over_sampling import SMOTE

        def generate(X_failures, seed, X_healthy=None):
            neighbours = min(5, len(X_failures) - 1)
            medians = X_failures.median()
            failures_imputed = X_failures.fillna(medians).fillna(0.0)
            # SMOTE interpolates between failure rows; a few healthy rows are needed only to form a two-class problem
            healthy_imputed = X_healthy.iloc[:200].fillna(medians).fillna(0.0)
            X_two_classes = pd.concat([failures_imputed, healthy_imputed], ignore_index=True)
            y_two_classes = np.r_[np.ones(len(failures_imputed), int), np.zeros(len(healthy_imputed), int)]
            smote = SMOTE(sampling_strategy={1: len(failures_imputed) + synthetic_rows}, k_neighbors=neighbours,
                          random_state=seed)
            X_resampled, _ = smote.fit_resample(X_two_classes.to_numpy(np.float32), y_two_classes)
            synthetic = pd.DataFrame(X_resampled[len(X_two_classes):], columns=X_failures.columns)
            # put the real failure missingness back
            donors = np.random.default_rng(seed).integers(0, len(X_failures), len(synthetic))
            patterns = X_failures.isna().to_numpy()[donors]
            return synthetic.mask(patterns)
        generate.needs_healthy_rows = True
        return generate
    if kind == "conditional_gan_2019":
        from experiments.rq1_cgan_2019_reproduction.conditional_gan_2019 import ConditionalGan2019

        def generate(X_failures, seed):
            generator = ConditionalGan2019(seed=seed, steps=options.get("gan_training_steps", 30000))
            return generator.fit(X_failures).sample(synthetic_rows, reapply_missingness=True, seed=seed + 1)
        return generate
    if kind == "tabebm":
        from experiments.rq2_survey_of_methods.tabebm_generator import TabEBMFailureGenerator

        def generate(X_failures, seed):
            generator = TabEBMFailureGenerator(seed=seed, sgld_steps=options.get("langevin_steps", 200))
            return generator.fit(X_failures).sample(synthetic_rows, reapply_missingness=True, seed=seed + 1)
        return generate
    if kind == "forest_flow":
        from experiments.rq2_survey_of_methods.forest_flow_generator import ForestFlowFailureGenerator

        def generate(X_failures, seed):
            condition_on_clusters = len(X_failures) >= 40    # k-means conditioning needs a few dozen rows
            generator = ForestFlowFailureGenerator(seed=seed, condition_on_clusters=condition_on_clusters,
                                                   gpu_hist=options.get("gpu_hist", False),
                                                   n_t=options.get("noise_levels", 10),
                                                   duplicate_K=options.get("duplicates_per_row", 10),
                                                   n_estimators=options.get("trees_per_regressor", 20),
                                                   max_depth=options.get("tree_depth", 5))
            return generator.fit(X_failures).sample(synthetic_rows, reapply_missingness=True, seed=seed + 1)
        return generate
    if kind in ("forest_flow_inpainting", "unmasking_trees_inpainting"):
        from experiments.rq3_completion_and_cold_start.completion_engines import (ForestFlowCompletion,
                                                                                  UnmaskingTreesCompletion,
                                                                                  hide_cells_completely_at_random)
        hidden_fraction = options.get("hidden_fraction", 0.5)

        def generate(X_failures, seed):
            rng = np.random.default_rng(seed)
            if kind == "forest_flow_inpainting":
                engine = ForestFlowCompletion(seed=seed, verbose=False, gpu_hist=options.get("gpu_hist", False),
                                              n_t=options.get("noise_levels", 10),
                                              duplicate_K=options.get("duplicates_per_row", 10),
                                              n_estimators=options.get("trees_per_regressor", 20),
                                              max_depth=options.get("tree_depth", 5))
            else:
                engine = UnmaskingTreesCompletion(seed=seed, verbose=False,
                                                  depth=options.get("unmasking_trees_depth", 4),
                                                  duplicate_K=options.get("unmasking_trees_duplicates_per_row", 5),
                                                  xgboost_kwargs=options.get("xgboost_kwargs"))
            engine.fit(X_failures)
            anchors = X_failures.iloc[rng.integers(0, len(X_failures), synthetic_rows)].reset_index(drop=True)
            X_masked, _ = hide_cells_completely_at_random(anchors, hidden_fraction, seed=seed + 7)
            point, _, _ = engine.complete(X_masked, draws=1)   # one stochastic completion per anchor
            synthetic = point.copy()
            synthetic[anchors.isna().to_numpy()] = np.nan    # the anchors' own missing cells stay missing
            return synthetic
        return generate
    raise ValueError(kind)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generator", required=True, choices=COLD_START_GENERATORS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11])
    parser.add_argument("--failure-counts", type=int, nargs="+", default=[25, 50, 100, 250, 1000],
                        help="numbers of real failures kept in each training fold (1000 keeps all of them)")
    parser.add_argument("--synthetic-rows", type=int, default=2000,
                        help="synthetic failure rows added to each training fold")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--override", default="{}", help="JSON object of generator options (see above)")
    parser.add_argument("--smoke-test-rows", type=int, default=None,
                        help="quick check only: use a stratified sample of this many training rows; the outputs get "
                             "smoke_test in their names")
    return parser.parse_args()


def main():
    args = parse_arguments()
    options = json.loads(args.override)
    X, y = load_training_set()
    X = X.drop(columns=[CONSTANT_ATTRIBUTE])
    X_test, y_test = load_test_set()
    X_test = X_test.drop(columns=[CONSTANT_ATTRIBUTE])
    records_file = RECORDS_FILE
    if args.smoke_test_rows:
        from sklearn.model_selection import train_test_split
        X, _, y, _ = train_test_split(X, y, train_size=args.smoke_test_rows, stratify=y, random_state=0)
        X = X.reset_index(drop=True)
        records_file = SMOKE_TEST_RECORDS_FILE
    records_file.parent.mkdir(parents=True, exist_ok=True)
    TEST_PROBABILITIES_DIR.mkdir(parents=True, exist_ok=True)
    classifier_parameters = load_tuned_hyperparameters("lightgbm")
    generate = make_generator(args.generator, args.synthetic_rows, options)
    for seed in args.seeds:
        folds = list(StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed).split(X, y))
        # one fixed random order of the failures per (seed, fold): nested subsets across the failure counts
        failure_orders = [np.random.default_rng(seed * 100 + fold).permutation(np.where(y[train_rows] == 1)[0])
                          for fold, (train_rows, _) in enumerate(folds)]
        for failure_count in args.failure_counts:
            started = time.time()
            out_of_fold_probabilities = np.zeros(len(y))
            fold_of_row = np.zeros(len(y), dtype=int)
            models = []
            for fold, (train_rows, validation_rows) in enumerate(folds):
                kept_failures = train_rows[failure_orders[fold][:failure_count]]
                healthy = train_rows[y[train_rows] == 0]
                kept = np.concatenate([healthy, kept_failures])
                X_train, y_train = X.iloc[kept], y[kept]
                if generate is not None:
                    X_failures = X.iloc[kept_failures].reset_index(drop=True)
                    if getattr(generate, "needs_healthy_rows", False):
                        synthetic = generate(X_failures, seed * 100 + fold, X_healthy=X.iloc[healthy])
                    else:
                        synthetic = generate(X_failures, seed * 100 + fold)
                    X_train = pd.concat([X_train, synthetic[X_train.columns]], ignore_index=True)
                    y_train = np.concatenate([y_train, np.ones(len(synthetic), dtype=y.dtype)])
                model = CLASSIFIER_FACTORIES["lightgbm"](seed * 100 + fold, **classifier_parameters)
                model.fit(X_train, y_train)
                out_of_fold_probabilities[validation_rows] = model.predict_proba(X.iloc[validation_rows])[:, 1]
                fold_of_row[validation_rows] = fold
                models.append(model)
            threshold, _ = cost_minimising_threshold(out_of_fold_probabilities, y)
            out_of_fold_result = misclassification_cost(y, (out_of_fold_probabilities >= threshold).astype(int))
            out_of_fold_result["threshold"] = threshold
            fold_costs = []   # the cost of each fold at the common threshold
            for fold in range(args.folds):
                in_fold = fold_of_row == fold
                fold_decisions = (out_of_fold_probabilities[in_fold] >= threshold).astype(int)
                fold_costs.append(misclassification_cost(y[in_fold], fold_decisions)["cost"])
            out_of_fold_result["fold_costs"] = fold_costs
            out_of_fold_result["cost_scaled_to_test_set_size"] = out_of_fold_result["cost"] * TEST_SET_ROWS / len(y)
            curve = cost_curve(out_of_fold_probabilities, y)
            out_of_fold_result["fp_at_fn_le_15"] = (int(curve.loc[curve["fn"] <= 15, "fp"].min())
                                                        if (curve["fn"] <= 15).any() else None)
            test_probabilities = np.mean([model.predict_proba(X_test)[:, 1] for model in models], axis=0)
            test_result = misclassification_cost(y_test, (test_probabilities >= threshold).astype(int))
            test_result["threshold"] = threshold
            test_result["cost_95_percent_interval"] = list(bootstrap_cost_confidence_interval(
                y_test, (test_probabilities >= threshold).astype(int)))
            test_curve = cost_curve(test_probabilities, y_test)
            test_result["fp_at_fn_le_6"] = (int(test_curve.loc[test_curve["fn"] <= 6, "fp"].min())
                                            if (test_curve["fn"] <= 6).any() else None)
            test_result["oracle_minimum_cost"] = int(test_curve["cost"].min())
            record = {"generator": args.generator, "seed": seed, "labelled_failures": failure_count,
                      "synthetic_rows": 0 if generate is None else args.synthetic_rows, "folds": args.folds,
                      "options": options, "out_of_fold": out_of_fold_result, "test": test_result,
                      "seconds": time.time() - started}
            name = f"{args.generator}_seed{seed}_{failure_count}_failures"
            if args.smoke_test_rows:
                name += "_smoke_test"      # never overwrite the probabilities of a real run
            np.save(TEST_PROBABILITIES_DIR / f"{name}.npy", test_probabilities)
            with open(records_file, "a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(f"== {args.generator} seed {seed}, {failure_count} labelled failures: "
                  f"out-of-fold cost {out_of_fold_result['cost']} (FN {out_of_fold_result['fn']}, "
                  f"FP {out_of_fold_result['fp']}; {out_of_fold_result['cost_scaled_to_test_set_size']:.0f} scaled to "
                  f"the 16,000 rows of the test set) | test cost {test_result['cost']} (FN {test_result['fn']}, "
                  f"FP {test_result['fp']}) | {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
