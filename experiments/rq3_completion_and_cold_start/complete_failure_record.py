"""The RQ3 prototype as a command-line tool: complete a partially observed APS record under the failure hypothesis.

The input is a record in which only some of the 170 attributes are known: a CSV file with one row (empty or NaN for
the unknown cells), or, for a demonstration, one of the 1,000 training failures with a share of its known cells
hidden at random. The tool
  1. fits the chosen completion engine (see completion_engines.py) on failure rows only, which is what "under the
     failure hypothesis" means, and with --both-hypotheses a second engine on healthy rows;
  2. prints the completed record under each hypothesis, with an 80 % interval for every filled cell under the
     failure hypothesis;
  3. scores the true record (when known), the partial record and the completed records with the tuned LightGBM
     failure classifier of the thesis, so that one can see whether the completion restores the failure signature.
     A record is flagged ("ALARM") when its probability reaches the baseline's decision threshold: the threshold of
     arm lightgbm_tuned, seed 11, chosen on its out-of-fold predictions and read from
     results/classifier_arms/arm_records.jsonl. Without that file the probabilities are printed without decisions.
Everything is fitted on the training file; a demonstration record is left out of every fit.

    python -m experiments.rq3_completion_and_cold_start.complete_failure_record --engine unmasking_trees \\
        --demo-index 7 --hidden-fraction 0.5                     # demonstration on a training failure
    python -m experiments.rq3_completion_and_cold_start.complete_failure_record --engine forest_flow \\
        --record my_partial_record.csv                           # a real partial record
    python -m experiments.rq3_completion_and_cold_start.complete_failure_record --engine k_nearest_neighbours \\
        --demo-index 7 --hidden-fraction 0.5 --both-hypotheses --attributes-to-show 12 --output-csv completed.csv

The demonstration reported in the thesis was run on an L4 GPU with

    python -m experiments.rq3_completion_and_cold_start.complete_failure_record --engine unmasking_trees \\
        --demo-index 7 --hidden-fraction 0.5 --both-hypotheses --healthy-rows 1000 --attributes-to-show 12 \\
        --override '{"xgboost_kwargs": {"device": "cuda"}}'

and its console output is kept in results/rq3_completion_and_cold_start/prototype_demonstration_output.txt.
A demonstration run (--demo-index) writes the engine name and the printed probabilities to
results/rq3_completion_and_cold_start/prototype_demonstration.json, replacing those of the previous demonstration;
a run on your own record writes them only where --output-json says. With --output-csv the completion under the
failure hypothesis is written as well (rows "point", "lo80" and "hi80").
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common.classifiers import CLASSIFIER_FACTORIES, load_tuned_hyperparameters
from experiments.common.data_loading import load_training_set
from experiments.common.paths import ARM_RECORDS_FILE, COMPLETION_AND_COLD_START_RESULTS_DIR
from experiments.rq3_completion_and_cold_start.completion_engines import (COMPLETION_ENGINES,
                                                                          hide_cells_completely_at_random)

DEMONSTRATION_FILE = COMPLETION_AND_COLD_START_RESULTS_DIR / "prototype_demonstration.json"


def baseline_threshold():
    """The out-of-fold decision threshold of the baseline arm lightgbm_tuned, seed 11, or None if not recorded."""
    if not ARM_RECORDS_FILE.exists():
        return None
    with open(ARM_RECORDS_FILE) as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if record.get("arm") == "lightgbm_tuned" and record.get("seed") == 11 and "out_of_fold" in record:
                return float(record["out_of_fold"]["threshold"])
    return None


def fit_classifier(X: pd.DataFrame, y: np.ndarray, seed: int):
    """The tuned LightGBM of the thesis, fitted on the whole training file (minus the demonstration record), and the
    baseline's decision threshold."""
    classifier_parameters = load_tuned_hyperparameters("lightgbm", 11)
    model = CLASSIFIER_FACTORIES["lightgbm"](seed, **classifier_parameters)
    model.fit(X, y)
    return model, baseline_threshold()


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", default="unmasking_trees", choices=sorted(COMPLETION_ENGINES))
    parser.add_argument("--record", help="CSV with one row of the 170 APS attributes (empty or NaN = unknown)")
    parser.add_argument("--demo-index", type=int, default=None,
                        help="use this training failure (0-999) as the record, hiding --hidden-fraction of its "
                             "known cells")
    parser.add_argument("--hidden-fraction", type=float, default=0.5,
                        help="probability with which each known cell of the demonstration record is hidden")
    parser.add_argument("--draws", type=int, default=20, help="completions drawn per record for the interval")
    parser.add_argument("--both-hypotheses", action="store_true",
                        help="also complete under the healthy hypothesis (an engine fitted on healthy rows)")
    parser.add_argument("--healthy-rows", type=int, default=5000,
                        help="healthy rows sampled to fit the healthy-hypothesis engine")
    parser.add_argument("--attributes-to-show", type=int, default=15,
                        help="attributes to print (the hidden ones first)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-csv", help="write the completion under the failure hypothesis (point and interval) "
                                             "to this CSV file")
    parser.add_argument("--output-json", default=None,
                        help="write the engine and the printed probabilities to this JSON file (a --demo-index run "
                             "writes results/rq3_completion_and_cold_start/prototype_demonstration.json by default)")
    parser.add_argument("--override", default="{}",
                        help='JSON object of engine options, e.g. {"xgboost_kwargs": {"device": "cuda"}}')
    return parser.parse_args()


def main():
    args = parse_arguments()

    X, y = load_training_set()        # all 170 attributes, the record format (cd_000 is constant and plays no role)
    failures = X[y == 1].reset_index(drop=True)
    healthy = X[y == 0].reset_index(drop=True)
    true_record = None
    if args.record:
        record = pd.read_csv(args.record)
        record = record.reindex(columns=X.columns)
        if len(record) != 1:
            sys.exit("the record file must hold exactly one row")
        fitting_failures = failures
        X_classifier, y_classifier = X, y
    elif args.demo_index is not None:
        true_record = failures.iloc[[args.demo_index]].reset_index(drop=True)
        record, _ = hide_cells_completely_at_random(true_record, args.hidden_fraction, args.seed)
        fitting_failures = failures.drop(index=args.demo_index).reset_index(drop=True)
        keep = np.ones(len(X), dtype=bool)
        keep[np.where(y == 1)[0][args.demo_index]] = False
        X_classifier, y_classifier = X[keep], y[keep]
    else:
        sys.exit("give --record or --demo-index")

    print(f"record: {int(record.notna().sum(axis=1).iloc[0])} of {X.shape[1]} attributes observed", flush=True)
    started = time.time()
    failure_engine = COMPLETION_ENGINES[args.engine](seed=args.seed, verbose=False,
                                                     **json.loads(args.override)).fit(fitting_failures)
    failure_point, failure_low, failure_high = failure_engine.complete(record, draws=args.draws)
    print(f"failure-hypothesis completion ({args.engine}, fitted on {len(fitting_failures)} failures) in "
          f"{time.time() - started:.0f}s", flush=True)
    healthy_point = None
    if args.both_hypotheses:
        started = time.time()
        healthy_sample = healthy.sample(n=min(args.healthy_rows, len(healthy)), random_state=args.seed)
        healthy_engine = COMPLETION_ENGINES[args.engine](seed=args.seed, verbose=False,
                                                         **json.loads(args.override)).fit(healthy_sample)
        healthy_point, _, _ = healthy_engine.complete(record, draws=args.draws)
        print(f"healthy-hypothesis completion (fitted on {len(healthy_sample)} healthy rows) in "
              f"{time.time() - started:.0f}s", flush=True)

    started = time.time()
    classifier, threshold = fit_classifier(X_classifier, y_classifier, args.seed)
    print(f"tuned LightGBM failure model fitted in {time.time() - started:.0f}s"
          + (f"; out-of-fold threshold {threshold:.2e}" if threshold else ""), flush=True)

    def failure_probability(frame):
        return float(classifier.predict_proba(frame[X.columns])[:, 1][0])

    # hidden cells: unknown in the record but known in the true record (when there is one)
    hidden_columns = [column for column in X.columns
                      if record[column].isna().iloc[0] and (true_record is None or true_record[column].notna().iloc[0])]
    other_columns = [column for column in X.columns if column not in hidden_columns]
    shown = (hidden_columns + other_columns)[:args.attributes_to_show]
    table = pd.DataFrame({"attribute": shown,
                          "observed": [record[column].iloc[0] for column in shown],
                          "true": [true_record[column].iloc[0] if true_record is not None else np.nan
                                   for column in shown],
                          "failure: point": [failure_point[column].iloc[0] for column in shown],
                          "failure: 80% low": [failure_low[column].iloc[0] for column in shown],
                          "failure: 80% high": [failure_high[column].iloc[0] for column in shown]})
    if healthy_point is not None:
        table["healthy: point"] = [healthy_point[column].iloc[0] for column in shown]
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda value: f"{value:,.0f}")
    print("\n" + table.to_string(index=False))

    probabilities = {"partial record (NaNs as given)": failure_probability(record),
                     "completed under the failure hypothesis": failure_probability(failure_point)}
    if true_record is not None:
        probabilities = {"true record": failure_probability(true_record), **probabilities}
    if healthy_point is not None:
        probabilities["completed under the healthy hypothesis"] = failure_probability(healthy_point)
    print("\nfailure probability from the tuned LightGBM"
          + (f" (alarm if above {threshold:.2e})" if threshold else "") + ":")
    for label, probability in probabilities.items():
        decision = "" if threshold is None else ("  -> ALARM" if probability >= threshold else "  -> no alarm")
        print(f"  {label:44s} {probability:9.5f}{decision}")
    if true_record is not None:
        hidden = np.array([column in hidden_columns for column in X.columns])
        true_values = true_record.to_numpy(float)[0][hidden]
        log_true = np.log1p(np.maximum(true_values, 0))
        log_completed = np.log1p(np.maximum(failure_point.to_numpy(float)[0][hidden], 0))
        inside = ((failure_low.to_numpy(float)[0][hidden] <= true_values)
                  & (true_values <= failure_high.to_numpy(float)[0][hidden])).mean()
        print(f"\nhidden cells: {hidden.sum()}; "
              f"mean absolute error in log space {np.abs(log_true - log_completed).mean():.3f}; "
              f"true value inside the 80% interval for {inside:.0%} of them")
    if args.output_csv:
        completion = pd.concat([failure_point.assign(kind="point"), failure_low.assign(kind="lo80"),
                                failure_high.assign(kind="hi80")], ignore_index=True)
        completion.to_csv(args.output_csv, index=False)
        print("written", args.output_csv)
    # The stored demonstration is replaced only by another demonstration run (--demo-index), never by a run on a
    # record read with --record.
    output_json = (Path(args.output_json) if args.output_json
                   else (DEMONSTRATION_FILE if args.demo_index is not None else None))
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as handle:
            json.dump({"engine": args.engine, "probabilities": probabilities}, handle, indent=1)


if __name__ == "__main__":
    main()
