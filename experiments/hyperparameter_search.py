"""Hyperparameter search for LightGBM with Optuna, minimising the out-of-fold cost.

Part of the preparation for RQ1 and RQ2. The search was run once, for LightGBM on seed 11, before any arm, and its
best configuration (results/hyperparameter_search/lightgbm_seed11_best_parameters.json) is used unchanged by every
arm that trains a LightGBM (see classifiers.load_tuned_hyperparameters). LightGBM is the classifier of the baseline
and of nearly every augmented arm, so the tuning budget went to it; tuning it once and reusing the configuration
everywhere means that differences between arms cannot come from differently tuned learners. The search spaces for
XGBoost, CatBoost, the random forest and histogram gradient boosting below are defined, but those searches were not
run for the thesis, and the corresponding arms use the fixed settings of classifiers.py.

Objective. Each trial proposes a configuration, runs the five-fold stratified cross-validation of the protocol on the
60,000 training rows (cost_and_evaluation.run_cross_validation, with the folds of --seed) and returns the cost at the
cost-minimising threshold of its out-of-fold predictions, i.e. the quantity that the thesis reports for the arms. The
official test set is never read. Because the best of many trials is selected on the same folds, the cost of the best
trial is optimistic for those folds (selection optimism); the arms are also run on seeds that the search never saw.

Search algorithm. Optuna's tree-structured Parzen estimator (TPE) is a sequential, model-based search. After a number
of random start-up trials (8 here), it divides the trials so far into a good and a bad group by their cost, fits a
kernel density estimate (a Parzen estimator) to the parameter values of each group, and proposes next the
configuration for which the density of the good group is largest relative to that of the bad group. The sampler is
seeded with --seed, so a search repeated with the same seed proposes the same configurations.

Search space for LightGBM: 200 to 1,500 trees in steps of 50, learning rate 0.01 to 0.2, 7 to 127 leaves, a minimum
of 5 to 100 rows per leaf, row subsampling 0.5 to 1 and column subsampling 0.2 to 1 per tree, L2 penalty 0.001 to 10,
L1 penalty 0.001 to 5, and a weight of 1 to 60 on the failure class (scale_pos_weight multiplies the weight of every
failure row in the training loss). Ranges that span orders of magnitude are searched on a logarithmic scale.

Usage (from the repository root):

    python -m experiments.hyperparameter_search --model lightgbm --trials 40 --timeout 3300 --seed 11

The search stops after --trials trials or once --timeout seconds have passed, whichever comes first (a trial that has
started is finished). It writes, in results/hyperparameter_search/,
    <model>_seed<seed>_trials.csv            one row per finished trial, rewritten after every trial so that an
                                             interrupted search loses nothing: trial number, out-of-fold cost
                                             (cost_oof), FN and FP at the cost-minimising threshold, the threshold,
                                             the seconds the trial took, and the configuration
    <model>_seed<seed>_best_parameters.json  the best configuration (best_params) and its out-of-fold cost
"""
from __future__ import annotations

import argparse
import json
import time

import optuna
import pandas as pd

from experiments.common.classifiers import (CPU_THREADS_PER_MODEL, make_catboost, make_histogram_gradient_boosting,
                                            make_lightgbm, make_random_forest, make_xgboost)
from experiments.common.cost_and_evaluation import cost_minimising_threshold, run_cross_validation
from experiments.common.data_loading import CONSTANT_ATTRIBUTE, load_training_set
from experiments.common.paths import HYPERPARAMETER_SEARCH_DIR

SEARCHABLE_FACTORIES = {"lightgbm": make_lightgbm, "xgboost": make_xgboost, "catboost": make_catboost,
                        "random_forest": make_random_forest,
                        "histogram_gradient_boosting": make_histogram_gradient_boosting}


def suggest_configuration(trial: optuna.Trial, model: str) -> dict:
    """Ask the trial for one configuration of `model`. The order of the suggest calls is part of what the seeded
    sampler reproduces, so it must not change if earlier searches are to be repeated exactly."""
    if model == "lightgbm":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1500, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 7, 127, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5, log=True),
            scale_pos_weight=trial.suggest_float("scale_pos_weight", 1, 60, log=True),
        )
    if model == "xgboost":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1500, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_child_weight=trial.suggest_float("min_child_weight", 0.5, 20, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            gamma=trial.suggest_float("gamma", 1e-3, 5, log=True),
            scale_pos_weight=trial.suggest_float("scale_pos_weight", 1, 60, log=True),
        )
    if model == "catboost":
        return dict(
            iterations=trial.suggest_int("iterations", 300, 1500, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            depth=trial.suggest_int("depth", 4, 8),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 20, log=True),
            scale_pos_weight=trial.suggest_float("scale_pos_weight", 1, 60, log=True),
            rsm=trial.suggest_float("rsm", 0.2, 1.0),
        )
    if model == "random_forest":
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"])
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
            max_features=trial.suggest_float("max_features", 0.05, 0.6, log=True),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10, log=True),
            max_depth=trial.suggest_categorical("max_depth", [None, 12, 20, 30]),
            class_weight=class_weight,
        )
    if model == "histogram_gradient_boosting":
        return dict(
            max_iter=trial.suggest_int("max_iter", 200, 1200, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 7, 127, log=True),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 100, log=True),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10, log=True),
            class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
        )
    raise ValueError(model)


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="lightgbm", choices=sorted(SEARCHABLE_FACTORIES),
                        help="classifier to tune (default lightgbm, the only one tuned for the thesis)")
    parser.add_argument("--trials", type=int, default=40, help="maximum number of trials (default 40)")
    parser.add_argument("--timeout", type=int, default=3300,
                        help="seconds after which no new trial is started (default 3300, i.e. 55 minutes)")
    parser.add_argument("--seed", type=int, default=11, help="seed of the folds and of the TPE sampler (default 11)")
    parser.add_argument("--folds", type=int, default=5, help="number of cross-validation folds (default 5)")
    parser.add_argument("--impute", default="none",
                        help="none|median (median needed for models without NaN support; always used for "
                             "random_forest)")
    args = parser.parse_args()

    X, y = load_training_set()
    X = X.drop(columns=[CONSTANT_ATTRIBUTE])
    if args.impute == "median" or args.model == "random_forest":
        # The random forest was treated as unable to take missing values, so here they are filled once, with the
        # medians of all 60,000 training rows, instead of inside each fold as in the arms. For tree splits this is
        # harmless: the medians are computed without the labels and are one constant per attribute, so they cannot
        # carry label information from a held-out fold into training, and they hardly differ from the medians of
        # any four-fifths training part. (The thesis searched only LightGBM, which needs no imputation.)
        training_medians = X.median()
        X = X.fillna(training_medians)

    trial_log = []

    def objective(trial):
        configuration = suggest_configuration(trial, args.model)
        started = time.time()

        def make_model(seed, configuration=configuration):
            return SEARCHABLE_FACTORIES[args.model](seed, **configuration)

        result = run_cross_validation(f"{args.model}#{trial.number}", make_model, X, y,
                                      n_splits=args.folds, seed=args.seed, verbose=False)
        threshold, cost = cost_minimising_threshold(result.out_of_fold_probabilities, result.y)
        summary = result.summary(threshold)
        trial_record = {"trial": trial.number, "cost_oof": int(cost), "fn": summary["fn"], "fp": summary["fp"],
                        "threshold": threshold, "seconds": time.time() - started, **configuration}
        trial_log.append(trial_record)
        print(json.dumps(trial_record), flush=True)
        HYPERPARAMETER_SEARCH_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(trial_log).to_csv(HYPERPARAMETER_SEARCH_DIR / f"{args.model}_seed{args.seed}_trials.csv",
                                       index=False)
        return cost

    # the first 8 trials are drawn at random; TPE proposes the rest
    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=8)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)
    best = {"model": args.model, "best_cost_oof": study.best_value, "best_params": study.best_params,
            "n_trials": len(study.trials), "seed": args.seed, "folds": args.folds, "n_jobs": CPU_THREADS_PER_MODEL}
    HYPERPARAMETER_SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    with open(HYPERPARAMETER_SEARCH_DIR / f"{args.model}_seed{args.seed}_best_parameters.json", "w") as handle:
        json.dump(best, handle, indent=1)
    print("BEST", json.dumps(best))


if __name__ == "__main__":
    main()
