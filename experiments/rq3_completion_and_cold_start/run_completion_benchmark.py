"""Completion benchmark (RQ3): how well does each engine complete a partially observed failure record, and does the
completed record still look like a failure to the classifier?

The 1,000 training failures are split into five folds (scikit-learn KFold, shuffled with --seed). In each fold the
completion engines of completion_engines.py are fitted on the 800 failures of the other folds and applied to the
200 held-out failures, which they have never seen. The held-out records are first masked, that is, some of their
known cells are hidden, in one of four ways (--masks):

  random_20_percent, random_50_percent, random_80_percent
      every known cell is hidden independently with probability 0.2, 0.5 or 0.8. The hidden cells are missing
      completely at random (MCAR): whether a cell is hidden does not depend on any value.
  real_patterns_30_percent
      every record receives the missingness pattern of a real record, drawn at random from those of the first
      20,000 healthy training rows that miss at least 30 % of their attributes. The hidden cells then follow the
      dataset's own patterns of unavailable counters rather than a random design.

and every engine completes the masked records. The metrics compare the completed cells with the true values that
were hidden, in log(1 + x) space, where the counters span a comparable range:

  nrmse        normalised root mean squared error over the hidden cells: each error is divided by the standard
               deviation of its attribute among the 800 fitting failures, so that an engine that always predicts the
               median scores about 1. Attributes with a standard deviation below 0.05 there are left out of this
               metric (n_hidden_informative counts the cells that remain).
  mae_log      mean absolute error over all hidden cells
  cover80      interval coverage: the share of hidden cells whose true value lies inside the engine's 80 % interval
               (about 0.8 for an interval that is neither too narrow nor too wide)
  width80      the mean width of that interval (log units)
  clf_*        the classifier-based check. A tuned LightGBM (the baseline's hyperparameters) is trained in each fold
               on all 59,000 healthy rows plus the 800 fitting failures, and scores the held-out failures three
               times: as they really are (true), with the hidden cells missing (masked), and as completed
               (completed). clf_prob_* is the mean predicted failure probability, clf_flag_* the share flagged as
               failures at clf_threshold, the decision threshold of the baseline arm lightgbm_tuned (the median over
               its seeds of the threshold chosen on its out-of-fold predictions). A completion that restores the
               failure signature keeps clf_flag_completed close to clf_flag_true.

The other fields of a record are fold, mask, engine, n_records, n_hidden_cells, seconds (the time the engine took
to complete the records) and options (the engine options given with --override).

    python -m experiments.rq3_completion_and_cold_start.run_completion_benchmark --folds 0 1 2 3 4 \\
        --engines median k_nearest_neighbours chained_equations forest_flow unmasking_trees tabpfn \\
        --override '{"forest_flow": {"gpu_hist": true}, "unmasking_trees": {"xgboost_kwargs": {"device": "cuda"}}}'
    python -m experiments.rq3_completion_and_cold_start.run_completion_benchmark --folds 0 \\
        --engines median k_nearest_neighbours chained_equations --smoke-test-rows 100 --draws 5

Reads the training file, the tuned hyperparameters (results/hyperparameter_search/) and the baseline's records
(results/classifier_arms/arm_records.jsonl). Appends one JSON line per (fold, mask, engine) to
results/rq3_completion_and_cold_start/completion_benchmark.jsonl, or to the file named by the environment variable
APS_COMPLETION_RECORDS_FILE (a cloud session writes its own file); with --smoke-test-rows to
completion_benchmark_smoke_test.jsonl in the same folder.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from experiments.common.classifiers import CLASSIFIER_FACTORIES, load_tuned_hyperparameters
from experiments.common.data_loading import CONSTANT_ATTRIBUTE, load_training_set
from experiments.common.paths import ARM_RECORDS_FILE, COMPLETION_AND_COLD_START_RESULTS_DIR
from experiments.rq3_completion_and_cold_start.completion_engines import (COMPLETION_ENGINES,
                                                                          hide_cells_completely_at_random,
                                                                          hide_cells_with_real_missingness_patterns)

RECORDS_FILE = (Path(os.environ["APS_COMPLETION_RECORDS_FILE"]) if os.environ.get("APS_COMPLETION_RECORDS_FILE")
                else COMPLETION_AND_COLD_START_RESULTS_DIR / "completion_benchmark.jsonl")
SMOKE_TEST_RECORDS_FILE = COMPLETION_AND_COLD_START_RESULTS_DIR / "completion_benchmark_smoke_test.jsonl"

# mask name -> (how the cells are hidden, percentage): the share of known cells hidden for the random masks, the
# minimum share of missing attributes of the donor records for the real patterns
MASKS = {"random_20_percent": ("completely_at_random", 20), "random_50_percent": ("completely_at_random", 50),
         "random_80_percent": ("completely_at_random", 80), "real_patterns_30_percent": ("real_patterns", 30)}
# attributes whose log values vary less than this among the fitting failures are left out of the NRMSE
MIN_STANDARD_DEVIATION_FOR_NRMSE = 0.05


def baseline_threshold(default=0.5):
    """The decision threshold of the baseline arm lightgbm_tuned: the median over its seeds of the threshold chosen
    on the out-of-fold predictions, from results/classifier_arms/arm_records.jsonl (`default` if there is none)."""
    thresholds = []
    if ARM_RECORDS_FILE.exists():
        with open(ARM_RECORDS_FILE) as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if record.get("arm") == "lightgbm_tuned" and "out_of_fold" in record:
                    thresholds.append(record["out_of_fold"]["threshold"])
    return float(np.median(thresholds)) if thresholds else default


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="folds to run (0-4)")
    parser.add_argument("--engines", nargs="+", choices=sorted(COMPLETION_ENGINES),
                        default=["median", "k_nearest_neighbours", "chained_equations", "forest_flow",
                                 "unmasking_trees", "tabpfn"])
    parser.add_argument("--masks", nargs="+", choices=list(MASKS), default=list(MASKS))
    parser.add_argument("--draws", type=int, default=20, help="completions drawn per record by the stochastic engines")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--override", default="{}",
                        help='JSON object of engine options by engine name, e.g. {"forest_flow": {"gpu_hist": true}}')
    parser.add_argument("--smoke-test-rows", type=int, default=None,
                        help="quick check only: use the first this many training failures; the records go to "
                             "completion_benchmark_smoke_test.jsonl")
    return parser.parse_args()


def main():
    args = parse_arguments()
    engine_options = json.loads(args.override)
    X, y = load_training_set()
    X = X.drop(columns=[CONSTANT_ATTRIBUTE])
    failure_rows = np.where(y == 1)[0]
    if args.smoke_test_rows:
        failure_rows = failure_rows[:args.smoke_test_rows]
    healthy_rows = np.where(y == 0)[0]
    folds = list(KFold(5, shuffle=True, random_state=args.seed).split(failure_rows))
    classifier_parameters = load_tuned_hyperparameters("lightgbm")
    threshold = baseline_threshold()
    records_file = RECORDS_FILE if not args.smoke_test_rows else SMOKE_TEST_RECORDS_FILE
    records_file.parent.mkdir(parents=True, exist_ok=True)
    pattern_pool = X.iloc[healthy_rows[:20000]]          # missingness patterns of real (healthy) records
    for fold in args.folds:
        fitting, held_out = folds[fold]
        fitting_failures = X.iloc[failure_rows[fitting]].reset_index(drop=True)
        heldout_failures = X.iloc[failure_rows[held_out]].reset_index(drop=True)
        # the classifier of this fold: every healthy row and the fitting failures (the held-out failures are unseen)
        started = time.time()
        X_classifier = pd.concat([X.iloc[healthy_rows], fitting_failures], ignore_index=True)
        y_classifier = np.r_[np.zeros(len(healthy_rows), int), np.ones(len(fitting_failures), int)]
        classifier = CLASSIFIER_FACTORIES["lightgbm"](args.seed * 100 + fold, **classifier_parameters).fit(X_classifier,
                                                                                                         y_classifier)
        probability_true = classifier.predict_proba(heldout_failures)[:, 1]
        print(f"fold {fold}: classifier fitted in {time.time() - started:.0f}s; true held-out failures flagged "
              f"{float((probability_true >= threshold).mean()):.3f} at t={threshold:.4f}", flush=True)
        log_standard_deviation = np.nanstd(np.log1p(np.maximum(fitting_failures.to_numpy(float), 0)), axis=0)
        informative = ~np.isnan(log_standard_deviation) & (log_standard_deviation >= MIN_STANDARD_DEVIATION_FOR_NRMSE)
        log_standard_deviation = np.where(informative, log_standard_deviation, 1.0)
        Z_true = np.log1p(np.maximum(heldout_failures.to_numpy(float), 0))
        engines = {}
        for name in args.engines:
            fit_started = time.time()
            engines[name] = COMPLETION_ENGINES[name](seed=args.seed + fold, verbose=False,
                                                     **engine_options.get(name, {})).fit(fitting_failures)
            print(f"  {name}: fitted in {time.time() - fit_started:.0f}s", flush=True)
        for mask in args.masks:
            how, percent = MASKS[mask]
            if how == "completely_at_random":
                X_masked, hidden = hide_cells_completely_at_random(heldout_failures, percent / 100,
                                                                   seed=args.seed * 10 + fold)
            else:
                X_masked, hidden = hide_cells_with_real_missingness_patterns(heldout_failures, pattern_pool,
                                                                             percent / 100, seed=args.seed * 10 + fold)
            probability_masked = classifier.predict_proba(X_masked)[:, 1]
            for name, engine in engines.items():
                completion_started = time.time()
                point, low, high = engine.complete(X_masked, draws=args.draws)
                Z_point = np.log1p(np.maximum(point.to_numpy(float), 0))
                Z_low = np.log1p(np.maximum(low.to_numpy(float), 0))
                Z_high = np.log1p(np.maximum(high.to_numpy(float), 0))
                hidden_informative = hidden & np.tile(informative, (len(Z_true), 1))
                errors = (Z_point - Z_true)[hidden_informative]
                error_scales = np.tile(log_standard_deviation, (len(Z_true), 1))[hidden_informative]
                probability_completed = classifier.predict_proba(point)[:, 1]
                record = {"fold": fold, "mask": mask, "engine": name, "n_records": int(len(heldout_failures)),
                          "n_hidden_cells": int(hidden.sum()), "n_hidden_informative": int(hidden_informative.sum()),
                          "nrmse": float(np.sqrt(np.nanmean((errors / error_scales) ** 2))),
                          "mae_log": float(np.nanmean(np.abs((Z_point - Z_true)[hidden]))),
                          "cover80": float(((Z_low <= Z_true) & (Z_true <= Z_high))[hidden].mean()),
                          "width80": float((Z_high - Z_low)[hidden].mean()),
                          "clf_threshold": threshold, "clf_prob_true": float(probability_true.mean()),
                          "clf_prob_masked": float(probability_masked.mean()),
                          "clf_prob_completed": float(probability_completed.mean()),
                          "clf_flag_true": float((probability_true >= threshold).mean()),
                          "clf_flag_masked": float((probability_masked >= threshold).mean()),
                          "clf_flag_completed": float((probability_completed >= threshold).mean()),
                          "seconds": time.time() - completion_started, "options": engine_options.get(name, {})}
                with open(records_file, "a") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(f"  [{mask}] {name:20s} nrmse {record['nrmse']:.3f}  mae {record['mae_log']:.3f}  "
                      f"cover80 {record['cover80']:.2f}  flagged true/masked/completed {record['clf_flag_true']:.2f}/"
                      f"{record['clf_flag_masked']:.2f}/{record['clf_flag_completed']:.2f}  {record['seconds']:.0f}s",
                      flush=True)


if __name__ == "__main__":
    main()
