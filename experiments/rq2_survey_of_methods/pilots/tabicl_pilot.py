"""TabICLv2 pilot on single folds: a reduced version of the arm tabicl_bagged_contexts that fits on a CPU (RQ2).

The full arm (three contexts per fold, each with all failures of the training fold and 10,000 healthy rows; five
folds; the fold models averaged on the test set) needs about 5.5 GB per 10,800-row context and hours of CPU time for
the 76,000 rows it has to classify. The pilot reported in the thesis ran on a machine with two CPU cores and 8 GB of
memory, with a single context per fold (all failures of the training fold plus --healthy-rows-per-context random
healthy rows), TabICL's internal ensemble reduced to one view, and only the folds asked for. TabICLv2 is an
in-context learner: it classifies the validation rows of the fold by reading the labelled context rows, without any
training (see rq2_survey_of_methods/tabicl_classifier.py).

Its predictions are compared with the out-of-fold predictions of the tuned LightGBM baseline (arm lightgbm_tuned)
on exactly the same validation rows; the folds are the same because both use five stratified folds with the same
seed. Each learner is scored at the threshold that minimises its cost on the fold itself, which is equally
optimistic for both, and by threshold-free measures. No test-set evaluation.

    python -m experiments.rq2_survey_of_methods.pilots.tabicl_pilot --folds 0 1 --healthy-rows-per-context 2500 \\
        --seed 11

Reads the training file and results/classifier_arms/predicted_probabilities/lightgbm_tuned/seed<k>_out_of_fold.npy
(if present). Writes results/rq2_survey_of_methods/tabicl_pilot.json (rewritten after every fold) and the TabICL
probabilities of each fold to results/rq2_survey_of_methods/tabicl_pilot_probabilities/seed<k>_fold<f>.npy.

Fields of the JSON file: seed, healthy_rows_per_context, n_estimators, and per fold n_val (validation
rows), seconds, and the metrics of "tabicl" and "lightgbm_tuned_same_rows" (the baseline on the same rows):
fold_opt_cost, fn and fp at the fold-optimal threshold, that threshold, auc (ROC AUC), ap (average precision) and
min_fp_at_fn<k>, the smallest number of false alarms at which at most k failures of the fold are missed.
"""
import argparse
import json
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from experiments.common.cost_and_evaluation import cost_curve, cost_minimising_threshold
from experiments.common.data_loading import CONSTANT_ATTRIBUTE, load_training_set, misclassification_cost
from experiments.common.paths import SURVEY_RESULTS_DIR, arm_probability_file
from experiments.rq2_survey_of_methods.tabicl_classifier import BaggedTabICLClassifier

PILOT_RESULTS_FILE = SURVEY_RESULTS_DIR / "tabicl_pilot.json"
PILOT_PROBABILITIES_DIR = SURVEY_RESULTS_DIR / "tabicl_pilot_probabilities"


def fold_metrics(probabilities, y_fold):
    """Cost at the fold-optimal threshold and threshold-free measures for one fold."""
    threshold, cost = cost_minimising_threshold(probabilities, y_fold)
    at_threshold = misclassification_cost(y_fold, (probabilities >= threshold).astype(int))
    curve = cost_curve(probabilities, y_fold)
    metrics = {"fold_opt_cost": int(cost), "fn": int(at_threshold["fn"]), "fp": int(at_threshold["fp"]),
               "threshold": float(threshold), "auc": float(roc_auc_score(y_fold, probabilities)),
               "ap": float(average_precision_score(y_fold, probabilities))}
    for k in (2, 5, 10):
        at_most_k_missed = curve[curve.fn <= k]
        metrics[f"min_fp_at_fn{k}"] = int(at_most_k_missed.fp.min()) if len(at_most_k_missed) else None
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", type=int, nargs="+", default=[0], help="folds to run (0-4)")
    parser.add_argument("--healthy-rows-per-context", type=int, default=2500,
                        help="random healthy rows added to the failures of the training fold in the context")
    parser.add_argument("--n-estimators", type=int, default=1, help="TabICL's internal ensemble size")
    parser.add_argument("--seed", type=int, default=11, help="seed of the folds (and of the baseline compared with)")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads for PyTorch")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    X, y = load_training_set()
    X = X.drop(columns=[CONSTANT_ATTRIBUTE])
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    baseline_file = arm_probability_file("lightgbm_tuned", args.seed, "out_of_fold")
    baseline_probabilities = np.load(baseline_file) if baseline_file.exists() else None
    pilot = {"seed": args.seed, "healthy_rows_per_context": args.healthy_rows_per_context,
             "n_estimators": args.n_estimators, "folds": {}}
    PILOT_PROBABILITIES_DIR.mkdir(parents=True, exist_ok=True)
    for fold, (train_rows, validation_rows) in enumerate(folds.split(X, y)):
        if fold not in args.folds:
            continue
        started = time.time()
        model = BaggedTabICLClassifier(n_bags=1, n_neg=args.healthy_rows_per_context, n_estimators=args.n_estimators,
                                       seed=args.seed * 100 + fold, threads=args.threads)
        model.fit(X.iloc[train_rows], y[train_rows])
        probabilities = model.predict_proba(X.iloc[validation_rows])[:, 1]
        np.save(PILOT_PROBABILITIES_DIR / f"seed{args.seed}_fold{fold}.npy", probabilities)
        fold_record = {"n_val": int(len(validation_rows)), "seconds": time.time() - started,
                       "tabicl": fold_metrics(probabilities, y[validation_rows])}
        if baseline_probabilities is not None:
            fold_record["lightgbm_tuned_same_rows"] = fold_metrics(baseline_probabilities[validation_rows],
                                                               y[validation_rows])
        pilot["folds"][fold] = fold_record
        print(f"fold {fold}: TabICL {fold_record['tabicl']} | LightGBM {fold_record.get('lightgbm_tuned_same_rows')} | "
              f"{fold_record['seconds']:.0f}s", flush=True)
        with open(PILOT_RESULTS_FILE, "w") as handle:
            json.dump(pilot, handle, indent=1)


if __name__ == "__main__":
    main()
