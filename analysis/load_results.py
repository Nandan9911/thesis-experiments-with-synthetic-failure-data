"""Readers for the result files, and the per-run diagnostics that the classifier-arm tables are built from.

Every experiment of the thesis writes its numbers to results/ (layout in experiments/common/paths.py). This module
reads those files for the summarise_* modules, so that each file is parsed in exactly one place. Its main function,
build_arm_table, turns the classifier-arm records (results/classifier_arms/arm_records.jsonl, one JSON record per
arm and seed, written by experiments/run_experiment_arm.py) into one table row per (arm, seed), and adds post-hoc
diagnostics computed from the saved predicted probabilities.

The cost, the out-of-fold (OOF) predictions and the OOF threshold are described in
experiments/common/cost_and_evaluation.py. The diagnostics added here use the test labels only to describe the
predictions:

  plateau threshold
      a second rule chosen on the OOF predictions alone. The OOF cost is flat near its minimum, so the minimising
      threshold is one of many nearly equivalent ones. The plateau rule takes every threshold whose OOF cost is
      within one missed failure (500) of the minimum and uses the geometric midpoint of that range (the range spans
      orders of magnitude, hence the geometric mean). Its test cost shows how sensitive the result is to the
      particular threshold.
  cost at the cut-offs 0.5 and 10/510
      fixed thresholds that need no tuning: 0.5 is the default of most libraries, and 10/510 = 0.0196 is the
      cost-optimal threshold for calibrated probabilities (flag a row when p x 500 > (1 - p) x 10), here called
      the Bayes cut-off.
  test oracle
      the lowest test cost over all thresholds. It uses the test labels and is not a legitimate result; the gap
      between an arm's test cost and its oracle is the part of the cost that is due to the threshold.
  minimum false alarms with at most 6 (or 9) missed failures
      the fewest test false alarms among the thresholds that miss at most 6 (or 9) of the 375 test failures, a
      view of how well an arm ranks the rows that does not depend on the chosen threshold. The published results
      that the thesis compares with miss between 4 and 9 test failures (PUBLISHED_OPERATING_POINTS in
      summarise_classifier_arms.py).

The oracle and the false-alarm counts search the same threshold grid as the experiments (2,001 quantiles of the
probabilities, experiments.common.cost_and_evaluation.cost_curve).
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import paths
from experiments.common.cost_and_evaluation import cost_curve
from experiments.common.data_loading import (COST_OF_FALSE_ALARM, COST_OF_MISSED_FAILURE, load_test_set,
                                             load_training_set, misclassification_cost)

TRAINING_ROWS = 60_000
DEFAULT_CUTOFF = 0.5
BAYES_CUTOFF = COST_OF_FALSE_ALARM / (COST_OF_MISSED_FAILURE + COST_OF_FALSE_ALARM)     # 10 / 510 = 0.0196

# ---------------------------------------------------------------------------------------------- result files
QUICK_BASELINE_FILE = paths.PRELIMINARY_CHECKS_DIR / "quick_baseline.jsonl"
THRESHOLD_RULE_DEMONSTRATION_FILE = paths.PRELIMINARY_CHECKS_DIR / "threshold_rule_demonstration.json"
HYPERPARAMETER_SEARCH_TRIALS_FILE = paths.HYPERPARAMETER_SEARCH_DIR / "lightgbm_seed11_trials.csv"
SYNTHETIC_ROW_COUNT_SWEEP_FILE = paths.CGAN_2019_RESULTS_DIR / "synthetic_row_count_sweep.jsonl"

HARD_NEGATIVE_CHECK_RETRAINED_MODELS_FILE = (paths.TRANSFORMER_2026_RESULTS_DIR
                                             / "hard_negative_check_retrained_models.json")
PUBLISHED_TRAINING_LOGS_DIR = paths.AUTHORS_CODE_RESULTS_DIR / "published_training_logs"
AUTHORS_CODE_LOGS_DIR = paths.AUTHORS_CODE_RESULTS_DIR / "logs"
SHIPPED_MODELS_EXACT_INPUTS_DIR = paths.AUTHORS_CODE_RESULTS_DIR / "shipped_models_exact_inputs"
RETRAINING_RUNS_DIR = paths.AUTHORS_CODE_RESULTS_DIR / "retraining_runs"

SYNTHETIC_DATA_DIAGNOSTICS_FILE = paths.SURVEY_RESULTS_DIR / "synthetic_data_diagnostics.json"
TABICL_PILOT_FILE = paths.SURVEY_RESULTS_DIR / "tabicl_pilot.json"

COMPLETION_BENCHMARK_FILE = paths.COMPLETION_AND_COLD_START_RESULTS_DIR / "completion_benchmark.jsonl"
COMPLETION_MEMORISATION_AUDIT_FILE = paths.COMPLETION_AND_COLD_START_RESULTS_DIR / "completion_memorisation_audit.json"
COLD_START_RUNS_FILE = paths.COMPLETION_AND_COLD_START_RESULTS_DIR / "cold_start_runs.jsonl"


# ---------------------------------------------------------------------------------------------- generic readers
def read_json(path: Path):
    with open(path) as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict]:
    """One dictionary per non-empty line."""
    with open(path) as file:
        return [json.loads(line) for line in file if line.strip()]


@lru_cache(maxsize=1)
def training_labels() -> np.ndarray:
    """Class labels of the 60,000 training rows (1 = failure), in file order."""
    _, labels = load_training_set()
    return labels


@lru_cache(maxsize=1)
def test_labels() -> np.ndarray:
    """Class labels of the 16,000 official test rows (1 = failure), in file order."""
    _, labels = load_test_set()
    return labels


# ---------------------------------------------------------------------------------------------- diagnostics
def plateau_threshold(probabilities: np.ndarray, labels: np.ndarray, slack: float = COST_OF_MISSED_FAILURE) -> float:
    """Geometric midpoint of the thresholds whose cost is within `slack` of the minimum cost.

    The lower end is floored at 1e-9 so that a plateau reaching the threshold 0 still has a finite geometric mean."""
    curve = cost_curve(probabilities, labels)
    near_minimum = curve[curve.cost <= curve.cost.min() + slack]
    lowest = max(near_minimum.threshold.min(), 1e-9)
    highest = max(near_minimum.threshold.max(), 1e-9)
    return float(np.sqrt(lowest * highest))


def minimum_false_alarms(curve: pd.DataFrame, max_missed_failures: int):
    """Fewest false alarms on a cost curve among the thresholds with at most `max_missed_failures` missed
    failures; None when no threshold of the grid gets there."""
    admissible = curve[curve.fn <= max_missed_failures]
    if len(admissible) == 0:
        return None
    return int(admissible.fp.min())


def cost_at_threshold(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """misclassification_cost of the rule 'flag a row when its probability is at least `threshold`'."""
    return misclassification_cost(labels, (probabilities >= threshold).astype(int))


# ---------------------------------------------------------------------------------------------- classifier arms
def load_arm_records() -> list[dict]:
    """The records of arm_records.jsonl, one per (arm, seed). Should an arm and seed have been run more than once,
    the record written last is the one that counts."""
    latest = {}
    for record in read_jsonl(paths.ARM_RECORDS_FILE):
        latest[(record["arm"], record["seed"])] = record
    return list(latest.values())


def load_arm_probabilities(arm: str, seed: int, kind: str) -> np.ndarray:
    """Saved probabilities of one arm and seed; kind is out_of_fold, test or test_full_refit."""
    return np.load(paths.arm_probability_file(arm, seed, kind))


def _arm_row(record: dict) -> dict:
    out_of_fold, test = record["out_of_fold"], record["test"]
    interval_low, interval_high = test["cost_95_percent_interval"]
    row = {
        "arm": record["arm"], "seed": record["seed"], "folds": record.get("folds", 5),
        "oof_cost": out_of_fold["cost"], "oof_fn": out_of_fold["fn"], "oof_fp": out_of_fold["fp"],
        "oof_threshold": out_of_fold["threshold"],
        "oof_cost_scaled_to_test_set_size": out_of_fold["cost_scaled_to_test_set_size"],
        "fold_costs": out_of_fold["fold_costs"],
        "cross_validation_minutes": record["cross_validation_seconds"] / 60,
        "test_cost": test["cost"], "test_fn": test["fn"], "test_fp": test["fp"],
        "test_cost_interval_low": interval_low, "test_cost_interval_high": interval_high,
        "test_oracle_minimum_cost": record.get("test_oracle_minimum_cost"),
        "full_refit_test_cost": np.nan, "full_refit_test_fn": np.nan, "full_refit_test_fp": np.nan,
    }
    if "test_full_refit" in record:
        refit = record["test_full_refit"]
        row.update({"full_refit_test_cost": refit["cost"], "full_refit_test_fn": refit["fn"],
                    "full_refit_test_fp": refit["fp"]})
    row.update(_probability_diagnostics(record["arm"], record["seed"]))
    return row


def _probability_diagnostics(arm: str, seed: int) -> dict:
    out_of_fold_probabilities = load_arm_probabilities(arm, seed, "out_of_fold")
    test_probabilities = load_arm_probabilities(arm, seed, "test")
    labels = test_labels()
    threshold = plateau_threshold(out_of_fold_probabilities, training_labels())
    at_plateau = cost_at_threshold(test_probabilities, labels, threshold)
    at_default = cost_at_threshold(test_probabilities, labels, DEFAULT_CUTOFF)
    at_bayes = cost_at_threshold(test_probabilities, labels, BAYES_CUTOFF)
    test_curve = cost_curve(test_probabilities, labels)
    return {
        "plateau_threshold": threshold,
        "test_cost_at_plateau_threshold": at_plateau["cost"], "test_fn_at_plateau_threshold": at_plateau["fn"],
        "test_fp_at_plateau_threshold": at_plateau["fp"],
        "test_cost_at_0_5": at_default["cost"], "test_fn_at_0_5": at_default["fn"], "test_fp_at_0_5": at_default["fp"],
        "test_cost_at_bayes_cutoff": at_bayes["cost"], "test_fn_at_bayes_cutoff": at_bayes["fn"],
        "test_fp_at_bayes_cutoff": at_bayes["fp"],
        "test_min_fp_at_most_6_fn": minimum_false_alarms(test_curve, 6),
        "test_min_fp_at_most_9_fn": minimum_false_alarms(test_curve, 9),
    }


def build_arm_table(records: list[dict] | None = None) -> pd.DataFrame:
    """One row per (arm, seed) with the protocol's results and the post-hoc diagnostics.

    Columns taken from the record: arm, seed, folds; oof_cost, oof_fn, oof_fp (OOF predictions of the 60,000
    training rows at the OOF threshold), oof_threshold, oof_cost_scaled_to_test_set_size (OOF cost x 16,000 /
    60,000), fold_costs (cost of each validation fold at the OOF threshold, the input of the paired tests),
    cross_validation_minutes; test_cost, test_fn, test_fp (fold-ensemble test probabilities at the OOF threshold),
    test_cost_interval_low/high (percentile bootstrap 95 % interval of that test cost, 2,000 resamples of the test
    rows with the predictions held fixed), test_oracle_minimum_cost; full_refit_test_cost/fn/fp (one model refitted
    on all training rows, at the same threshold; only for the arms and seeds where it was run).

    Columns computed here from the saved probabilities (see the module docstring): plateau_threshold and
    test_cost/fn/fp_at_plateau_threshold; test_cost/fn/fp_at_0_5; test_cost/fn/fp_at_bayes_cutoff;
    test_min_fp_at_most_6_fn and test_min_fp_at_most_9_fn.
    """
    if records is None:
        records = load_arm_records()
    return pd.DataFrame([_arm_row(record) for record in records])


def load_quick_baseline() -> dict[str, dict]:
    """The first-look baselines (fixed untuned settings, 5-fold OOF, seed 11), each keyed by its configuration name."""
    return {record["name"]: record for record in read_jsonl(QUICK_BASELINE_FILE)}


# ------------------------------------------------------------------------ RQ1b, the study of Beikmohammadi et al.
def read_test_trace(path: Path) -> list[dict]:
    """The test trace of one retraining run of the authors' code: one record per epoch, sorted by epoch.

    While training, the job scored the official test set every 50 epochs at the published 0.5 cut-off and appended
    a line to test_trace.jsonl (a diagnostic). If a run is resumed while an earlier session is still writing, both
    append the same epochs; the line written last for an epoch is kept."""
    by_epoch = {}
    for record in read_jsonl(path):
        by_epoch[record["epoch"]] = record
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def load_retraining_run(run: str) -> dict:
    """The files of one retraining run in results/rq1_transformer_2026_reproduction/authors_code/retraining_runs/.

    state                the epoch the run reached (state.json);
    final_evaluation     final_evaluation.json, written when the job ended: train and test cost at the 0.5 cut-off
                         and the best-validation checkpoint's test cost. A run that was later resumed and stopped
                         by the time budget keeps the final evaluation of its earlier stage, so its
                         'epochs_run' can be lower than the state's epoch;
    trace                read_test_trace(test_trace.jsonl);
    training_log         training_log.csv, the authors' per-epoch log (loss and validation metric).
    """
    folder = RETRAINING_RUNS_DIR / run
    return {
        "state": read_json(folder / "state.json"),
        "final_evaluation": read_json(folder / "final_evaluation.json"),
        "trace": read_test_trace(folder / "test_trace.jsonl"),
        "training_log": pd.read_csv(folder / "training_log.csv"),
    }


def load_pytorch_port_runs() -> list[dict]:
    """Result files of the PyTorch port, results/rq1_transformer_2026_reproduction/pytorch_port/<run>_seed<k>.json.

    Returns one dictionary per file: run_name, seed, record (the file's content) and evaluated_from_checkpoint, which
    is True for a run that was stopped before its last epoch and scored afterwards from its saved checkpoint by
    evaluate_port_checkpoint.py (those files carry 'epoch_saved' instead of 'epochs_run')."""
    file_name = re.compile(r"^(?P<run_name>.+)_seed(?P<seed>\d+)\.json$")
    runs = []
    for path in sorted(paths.PYTORCH_PORT_RESULTS_DIR.glob("*.json")):
        match = file_name.match(path.name)
        if match is None or "smoke_test" in path.name:
            continue
        record = read_json(path)
        runs.append({"run_name": match["run_name"], "seed": int(match["seed"]), "record": record,
                     "evaluated_from_checkpoint": "epoch_saved" in record})
    return runs


def load_shipped_model_evaluation(path: Path) -> pd.DataFrame:
    """A re-scoring of the five models released by Beikmohammadi et al. (CSV written by
    authors_code/evaluate_shipped_models.py), indexed by run (run1 ... run5). Columns: test_cost, test_FN, test_FP at
    the 0.5 cut-off; train_cost, train_FN, train_FP (run 1 only); test_AUC; best_thr_on_test and best_thr_cost (the
    lowest test cost over the eight cut-offs 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, chosen with the test labels: a
    diagnostic); cost_thr0p0196 (test cost at 10/510); secs."""
    return pd.read_csv(path).set_index("run")


def load_shipped_model_test_probabilities() -> list[np.ndarray]:
    """Test probabilities of the five released models on the inputs regenerated with the published preprocessing."""
    return [np.load(SHIPPED_MODELS_EXACT_INPUTS_DIR / f"run{run}_test_probabilities.npy") for run in range(1, 6)]


def load_tree_control_flag_rates() -> dict[str, dict]:
    """hard_negative_check_tree_control.json, keyed by arm name."""
    return read_json(SHIPPED_MODELS_EXACT_INPUTS_DIR / "hard_negative_check_tree_control.json")


def load_retrained_hard_negative_flag_rates() -> dict[str, dict]:
    """hard_negative_check_retrained_models.json, keyed by run name."""
    return read_json(HARD_NEGATIVE_CHECK_RETRAINED_MODELS_FILE)
