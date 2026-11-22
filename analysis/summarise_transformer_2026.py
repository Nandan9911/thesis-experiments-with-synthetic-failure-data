"""RQ1b: summary tables and figures of the reproduction of the cost-sensitive Transformer of Beikmohammadi et al..

The study of Beikmohammadi et al. (Cluster Computing 2026) reports a test cost of 3,440 on the APS data with a
Transformer classifier trained with focal loss (a cross-entropy that down-weights rows the model already classifies
confidently) after this preprocessing: drop the attributes with at least 10 % missing values, impute the rest with
an iterative imputer (each attribute regressed on the others with Bayesian ridge regression, 100 rounds), oversample
the failures with SVM-SMOTE (synthetic failures interpolated between failures near the class boundary found by a
support vector machine) to half the number of healthy rows, clean the result with RENN (Repeated Edited Nearest
Neighbours: repeatedly remove the healthy rows whose three nearest neighbours are not all healthy), and scale to
[0, 1] with a scaler fitted on the resampled table. The reproduction has four parts, each with a table here:

  released models re-scored   the five trained models published with the study, applied to test inputs regenerated
                              with the published preprocessing (this reproduces 3,440 exactly), and to three
                              variants of the inputs that show which preprocessing steps matter;
  retraining runs             the released training code run from scratch on the published inputs with five seeds
                              (the retraining runs), and test-inclusion controls: the same training with half (8,000) or
                              all of the 16,000 test rows deliberately placed in the training table. A run's test
                              cost is followed during training at the published 0.5 cut-off (the test trace, a
                              diagnostic);
  PyTorch port                a re-implementation of the network, trained under the published protocol (with
                              ablations of the resampling steps) and under a protocol with a validation split of
                              real rows, in which 10 % of the real training rows are held out for validation before
                              imputation, resampling and scaling are fitted on the rest;
  hard-negative comparison    where the models raise false alarms. RENN removed 1,864 of the 59,000 healthy training
                              rows (negatives) from the published training table, those close to failures, real or
                              synthetic. A negative is called hard by the one-shot rule if its three nearest
                              neighbours in the resampled table are not all negatives (one pass of the RENN rule),
                              which can be applied to the test rows as well. The comparison sets the share of hard
                              test negatives that a model flags against the share of the hard training negatives
                              that RENN removed; neither group is in the published training table.

The tables written here report measurements only.

Run from the repository root (also part of `python -m analysis.run_all_summaries`):

    python -m analysis.summarise_transformer_2026

Reads results/rq1_transformer_2026_reproduction/ (pytorch_port/*.json, hard_negative_check_retrained_models.json,
authors_code/shipped_models_exact_inputs/*.csv|json|npy, authors_code/logs/*.csv, authors_code/retraining_runs/<run>/,
authors_code/published_training_logs/run1..5.csv). Writes to results/summary_tables/:

  rq1_transformer_2026_shipped_models_rescored            the five released models on four versions of the inputs
  rq1_transformer_2026_clean_retraining_runs              the released code retrained from scratch, one row per seed
  rq1_transformer_2026_retraining_and_contamination_runs  retraining runs, test-inclusion controls and verification
                                                          reruns
  rq1_transformer_2026_pytorch_port_runs                  the PyTorch port under both protocols
  rq1_transformer_2026_epochs_reached                     epochs reached by every retraining run and PyTorch-port run
  rq1_transformer_2026_hard_negatives_shipped_models      flag rates of the released models and of tree learners
  rq1_transformer_2026_hard_negatives_retrained_models    the same flag rates for the retrained models
  rq1_transformer_2026_summary_numbers                    means, ranges and totals of the RQ1b runs quoted in the text

and to results/figures/: rq1_transformer_2026_published_training_logs, rq1_transformer_2026_clean_retraining_test_trace
and rq1_transformer_2026_retraining_and_contamination_trace.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.load_results import (AUTHORS_CODE_LOGS_DIR, PUBLISHED_TRAINING_LOGS_DIR, SHIPPED_MODELS_EXACT_INPUTS_DIR,
                                   load_pytorch_port_runs, load_retrained_hard_negative_flag_rates,
                                   load_retraining_run, load_shipped_model_evaluation, load_tree_control_flag_rates,
                                   read_json)
from analysis.write_outputs import (MISSING, cost_with_errors, fixed_decimals, is_missing, save_figure, thousands,
                                    write_summary_table)

# The published result (mean over the authors' five runs); the released models re-scored here give the same numbers.
PUBLISHED_MEAN_TEST_COST = 3440
PUBLISHED_MEAN_MISSED_FAILURES = 6.8

# ---------------------------------------------------------------------------------------------- retraining runs
CLEAN_RUNS = ["clean_seed0", "clean_seed1", "clean_seed2", "clean_seed3", "clean_seed4"]
HALF_CONTAMINATION_RUNS = ["half_test_rows_in_training_seed0", "half_test_rows_in_training_seed1",
                           "half_test_rows_in_training_seed2"]
FULL_CONTAMINATION_RUNS = ["all_test_rows_in_training_seed0"]
# Second runs of seed 0 in another session (300 epochs), made because TensorFlow training on a GPU is not
# bit-reproducible; they show the run-to-run spread and are not part of the thesis table.
VERIFICATION_RUNS = ["clean_seed0_verification", "half_test_rows_in_training_seed0_verification"]
RUN_KINDS = {**{run: "no test rows in training" for run in CLEAN_RUNS},
             **{run: "half of the test rows in training" for run in HALF_CONTAMINATION_RUNS},
             **{run: "all test rows in training" for run in FULL_CONTAMINATION_RUNS},
             **{run: "verification rerun" for run in VERIFICATION_RUNS}}
VALIDATION_METRIC = "val_cost_sensitive_metric_dl"      # the authors' name for their validation cost metric
# Readable names of the designs of the retraining runs (the part of a run name before "_seed").
RETRAINING_DESIGN_LABELS = {"clean": "Retraining",
                            "half_test_rows_in_training": "Half of the test rows in training",
                            "all_test_rows_in_training": "All test rows in training"}

# ---------------------------------------------------------------------------------------------- PyTorch port
PORT_RUN_LABELS = {
    "corrected_protocol": "Protocol with a validation split of real rows",
    "corrected_protocol_extended": "Protocol with a validation split of real rows, extended (stopped)",
    "published_protocol": "Published protocol: imputation, SVM-SMOTE, RENN",
    "published_protocol_without_renn": "Published protocol without RENN",
    "published_protocol_without_svm_smote": "Published protocol without SVM-SMOTE",
    "published_protocol_without_resampling": "Published protocol without resampling",
}

# ---------------------------------------------------------------------------------------------- re-scoring
# (label, file): the published preprocessing first, then the three variants that isolate the two shortcuts of the
# first attempt (an imputer with 5 instead of 100 rounds; the scaler fitted on the imputed instead of the resampled
# training table).
RESCORING_VARIANTS = [
    ("Published preprocessing", SHIPPED_MODELS_EXACT_INPUTS_DIR / "evaluation_scaler_on_resampled.csv"),
    ("5-round imputer, scaler on the imputed table (first attempt)",
     AUTHORS_CODE_LOGS_DIR / "evaluate_shipped_models_five_round_tables.csv"),
    ("100-round imputer, scaler on the imputed table",
     SHIPPED_MODELS_EXACT_INPUTS_DIR / "evaluation_scaler_on_imputed.csv"),
    ("5-round imputer, scaler on the resampled table",
     SHIPPED_MODELS_EXACT_INPUTS_DIR / "evaluation_five_round_tables_scaler_on_resampled.csv"),
]

# ---------------------------------------------------------------------------------------------- hard negatives
# (row label, file, group key in that file, group key in the tree-control file). The first file holds the main
# groups, the follow-up file the training negatives that the one-shot rule calls hard.
SHIPPED_MODEL_GROUPS = [
    ("Training negatives kept by RENN (in the published training table)", "hard_negative_check.json",
     "train negatives kept by RENN (seen)", "train neg kept by RENN"),
    ("Training negatives removed by RENN (not in it)", "hard_negative_check.json",
     "train negatives removed by RENN (never seen)", "train neg removed by RENN"),
    ("  of which hard by the one-shot rule", "hard_negative_check_followup.json",
     "train neg, hard by the one-shot rule, removed by RENN (never seen)", "train neg hard, removed"),
    ("Training negatives hard by the one-shot rule but kept by RENN", "hard_negative_check_followup.json",
     "train neg, hard by the one-shot rule, kept by RENN (seen)", "train neg hard, kept"),
    ("Test negatives hard by the one-shot rule", "hard_negative_check.json",
     "test negatives, hard (never seen)", "test neg hard"),
    ("Other test negatives", "hard_negative_check.json", "test negatives, rest", "test neg rest"),
    ("Training positives", "hard_negative_check.json", "train positives", "train pos"),
    ("Test positives", "hard_negative_check.json", "test positives", "test pos"),
]
TREE_CONTROL_ARMS = {"lightgbm_tuned": "LightGBM", "xgboost_fixed_settings": "XGBoost",
                     "ensemble_without_missingness_indicators": "Ensemble"}
# (group key in hard_negative_check_retrained_models.json, row label worded as in the released-model table)
RETRAINED_MODEL_GROUPS = [
    ("train negatives kept by RENN (trained on)", "Training negatives kept by RENN (trained on)"),
    ("train negatives removed by RENN (never seen)", "Training negatives removed by RENN (never seen)"),
    ("  of which hard by the one-shot rule", "of which hard by the one-shot rule"),
    ("test negatives, hard (never seen)", "Test negatives hard by the one-shot rule (never seen)"),
    ("test negatives, other (never seen)", "Other test negatives (never seen)"),
    ("test negatives, hard, seen", "Test negatives hard by the one-shot rule, seen"),
    ("test negatives, hard, unseen", "Test negatives hard by the one-shot rule, unseen"),
    ("test negatives, other, seen", "Other test negatives, seen"),
    ("test negatives, other, unseen", "Other test negatives, unseen"),
]
RETRAINED_RUN_HEADERS = {
    "clean_seed0": "Retraining, seed 0", "clean_seed1": "Retraining, seed 1", "clean_seed2": "Retraining, seed 2",
    "clean_seed3": "Retraining, seed 3", "clean_seed4": "Retraining, seed 4",
    "half_test_rows_in_training_seed0": "Half, seed 0", "half_test_rows_in_training_seed1": "Half, seed 1",
    "half_test_rows_in_training_seed2": "Half, seed 2", "all_test_rows_in_training_seed0": "All, seed 0",
}


# ---------------------------------------------------------------------------------------------- released models
def write_shipped_model_rescoring_table() -> pd.DataFrame:
    """The five released models on the published inputs and on three variants of them.

    One CSV row per variant and run plus a row 'mean' per variant. Costs are at the published cut-off (a row is
    flagged when its probability exceeds 0.5). The lowest cost over eight cut-offs is chosen with the test labels (a
    diagnostic)."""
    rows = []
    for variant, path in RESCORING_VARIANTS:
        evaluation = load_shipped_model_evaluation(path)
        for run, result in evaluation.iterrows():
            rows.append({"variant": variant, "run": run, "test_cost": result.test_cost, "test_fn": result.test_FN,
                         "test_fp": result.test_FP, "test_auc": result.test_AUC,
                         "lowest_test_cost_over_eight_cutoffs": result.best_thr_cost,
                         "cutoff_of_lowest_test_cost": result.best_thr_on_test,
                         "test_cost_at_bayes_cutoff": result.cost_thr0p0196,
                         "train_cost": result.train_cost, "train_fn": result.train_FN, "train_fp": result.train_FP})
        rows.append({"variant": variant, "run": "mean", "test_cost": evaluation.test_cost.mean(),
                     "test_fn": evaluation.test_FN.mean(), "test_fp": evaluation.test_FP.mean(),
                     "test_auc": evaluation.test_AUC.mean(),
                     "lowest_test_cost_over_eight_cutoffs": evaluation.best_thr_cost.mean(),
                     "test_cost_at_bayes_cutoff": evaluation.cost_thr0p0196.mean()})
    table = pd.DataFrame(rows)
    write_summary_table(
        "rq1_transformer_2026_shipped_models_rescored", table, _readable_rescoring_table(table),
        title="RQ1b: the five models released by Beikmohammadi et al. re-scored",
        caption="""
        Test cost (missed failures / false alarms) of the five models published by Beikmohammadi et al., each applied to
        the official test set after the published preprocessing was regenerated (100-round iterative imputer, SVM-SMOTE
        and RENN, min-max scaler fitted on the resampled table), and after three variants of it. Sources:
        authors_code/shipped_models_exact_inputs/evaluation_scaler_on_resampled.csv, evaluation_scaler_on_imputed.csv
        and evaluation_five_round_tables_scaler_on_resampled.csv, and
        authors_code/logs/evaluate_shipped_models_five_round_tables.csv (written by
        authors_code/evaluate_shipped_models.py). The mean of the published preprocessing equals the published 3,440
        (6.8 / 4.0).""",
        column_notes={
            "Run 1 ... Run 5": "test cost at the 0.5 cut-off with missed failures and false alarms (CSV: one row "
                               "per variant and run; test_cost, test_fn, test_fp).",
            "Mean": "mean over the five runs (CSV: the row with run = mean).",
            "test AUC": "area under the ROC curve on the test set (CSV: test_auc).",
            "lowest cost over eight cut-offs": "the lowest test cost over the cut-offs 0.02, 0.05, 0.1, 0.2, 0.3, "
                                               "0.5, 0.7 and 0.9, chosen with the test labels: a diagnostic (CSV: "
                                               "lowest_test_cost_over_eight_cutoffs, cutoff_of_lowest_test_cost).",
            "CSV only": "test_cost_at_bayes_cutoff (cut-off 10/510); train_cost, train_fn, train_fp (the 60,000 "
                        "training rows, scored for run 1 only).",
        })
    return table


def _readable_rescoring_table(table: pd.DataFrame) -> pd.DataFrame:
    runs = [run for run in table.run.unique() if run != "mean"]
    header = ["Inputs"] + [run.replace("run", "Run ") for run in runs] + ["Mean"]
    lines = []
    for variant, _ in RESCORING_VARIANTS:
        rows = table[table.variant == variant].set_index("run")
        mean = rows.loc["mean"]
        mean_cell = f"{thousands(mean.test_cost)} ({mean.test_fn:.1f} / {mean.test_fp:.1f})"
        lines.append([variant] + [cost_with_errors(rows.loc[run].test_cost, rows.loc[run].test_fn,
                                                   rows.loc[run].test_fp) for run in runs] + [mean_cell])
        if variant == RESCORING_VARIANTS[0][0]:
            lines.append(["  test AUC"] + [f"{rows.loc[run].test_auc:.4f}" for run in runs] + [f"{mean.test_auc:.4f}"])
            lines.append(["  lowest cost over eight cut-offs (test labels, diagnostic)"]
                         + [thousands(rows.loc[run].lowest_test_cost_over_eight_cutoffs) for run in runs]
                         + [thousands(mean.lowest_test_cost_over_eight_cutoffs)])
    return pd.DataFrame(lines, columns=header)


# ---------------------------------------------------------------------------------------------- retraining runs
def _trace_record_at_epoch(trace: list[dict], epoch: int) -> dict:
    for record in trace:
        if record["epoch"] == epoch:
            return record
    return {}


def _estimated_gpu_hours(trace: list[dict], epochs: int):
    """Seconds per epoch from the time stamps of the first and last trace record, times the epochs the run reached.

    Any pause between an interruption and the resumption of a run falls between those time stamps and is counted,
    so this is an estimate of the occupied GPU time rather than of pure training time."""
    if len(trace) < 2 or not epochs:
        return np.nan, np.nan
    seconds_per_epoch = (trace[-1]["time"] - trace[0]["time"]) / max(trace[-1]["epoch"] - trace[0]["epoch"], 1)
    return seconds_per_epoch * epochs / 3600, seconds_per_epoch


def write_clean_retraining_table() -> pd.DataFrame:
    """The released training code retrained from scratch on the published inputs (the retraining runs), one row per
    seed."""
    rows = []
    for run in CLEAN_RUNS:
        files = load_retraining_run(run)
        epochs = files["state"]["epoch"]
        final = files["final_evaluation"]
        trace = files["trace"]
        lowest = min(trace, key=lambda record: record["test_cost_at_0.5"])
        gpu_hours, seconds_per_epoch = _estimated_gpu_hours(trace, epochs)
        validation_metric = files["training_log"][VALIDATION_METRIC]
        running_median = validation_metric.rolling(50, min_periods=1).median()      # over the last 50 epochs
        best_checkpoint = final.get("test_best_ckpt", {})
        rows.append({
            "run": run, "seed": int(run.replace("clean_seed", "")), "epochs": epochs,
            "final_evaluation_epochs": final.get("epochs_run"),
            "estimated_gpu_hours": gpu_hours, "seconds_per_epoch": seconds_per_epoch,
            "validation_metric_minimum": float(validation_metric.min()),
            "validation_metric_minimum_epoch": int(validation_metric.idxmin()) + 1,
            "validation_metric_last": float(validation_metric.iloc[-1]),
            "validation_metric_last_running_median": float(running_median.iloc[-1]),
            "train_cost": final["train"]["cost_at_0.5"], "train_fn": final["train"]["fn"],
            "train_fp": final["train"]["fp"],
            "test_cost": final["test"]["cost_at_0.5"], "test_fn": final["test"]["fn"], "test_fp": final["test"]["fp"],
            "best_checkpoint_epoch": best_checkpoint.get("epoch"),
            "best_checkpoint_test_cost": best_checkpoint.get("cost_at_0.5"),
            "best_checkpoint_test_fn": best_checkpoint.get("fn"), "best_checkpoint_test_fp": best_checkpoint.get("fp"),
            "trace_points": len(trace), "trace_lowest_test_cost": lowest["test_cost_at_0.5"],
            "trace_lowest_test_cost_epoch": lowest["epoch"],
            "trace_last_test_cost": trace[-1]["test_cost_at_0.5"], "trace_last_epoch": trace[-1]["epoch"],
        })
    table = pd.DataFrame(rows)
    earlier_stage_notes = [f"{row.run} reached {row.epochs:,} epochs, its final evaluation is from epoch "
                           f"{row.final_evaluation_epochs:,}" for row in table.itertuples()
                           if row.final_evaluation_epochs != row.epochs]
    readable = pd.DataFrame({
        "Seed": table.seed.astype(str),
        "Epochs": table.epochs.map(thousands),
        "GPU hours": table.estimated_gpu_hours.map(lambda value: thousands(value, 1)),
        "Validation metric, min (epoch)": [f"{thousands(value, 1)} ({thousands(epoch)})" for value, epoch in
                                           zip(table.validation_metric_minimum, table.validation_metric_minimum_epoch)],
        "Validation metric, last": table.validation_metric_last_running_median.map(lambda value: thousands(value, 1)),
        "Train cost at 0.5 (FN / FP)": [cost_with_errors(cost, fn, fp) for cost, fn, fp in
                                        zip(table.train_cost, table.train_fn, table.train_fp)],
        "Test cost at 0.5, final model (FN / FP)": [cost_with_errors(cost, fn, fp) for cost, fn, fp in
                                                    zip(table.test_cost, table.test_fn, table.test_fp)],
        "Test cost at 0.5, best-validation checkpoint (FN / FP; epoch)": [
            f"{thousands(cost)} ({thousands(fn)} / {thousands(fp)}; {thousands(epoch)})" for cost, fn, fp, epoch in
            zip(table.best_checkpoint_test_cost, table.best_checkpoint_test_fn, table.best_checkpoint_test_fp,
                table.best_checkpoint_epoch)],
        "Lowest test cost along training (epoch)": [f"{thousands(cost)} ({thousands(epoch)})" for cost, epoch in
                                                    zip(table.trace_lowest_test_cost,
                                                        table.trace_lowest_test_cost_epoch)],
    })
    write_summary_table(
        "rq1_transformer_2026_clean_retraining_runs", table, readable,
        title="RQ1b: retraining with the released code, no test rows in the training data",
        caption=f"""
        The authors' training code (TrainingCode.py, run through
        experiments/rq1_transformer_2026_reproduction/authors_code/train_authors_model_with_resume.py) on the
        published inputs, one run per seed, within a GPU time budget; the published runs trained for 8,000 epochs.
        Sources: authors_code/retraining_runs/clean_seed<k>/ (state.json, final_evaluation.json, test_trace.jsonl,
        training_log.csv). final_evaluation.json is written when a job ends; a run resumed for a longer budget that
        stopped before writing a new one keeps the evaluation of its earlier stage (final_evaluation_epochs in the
        CSV; here: {'; '.join(earlier_stage_notes) or 'none'}). The test trace scores the test set every 50 epochs
        and is a diagnostic only.""",
        column_notes={
            "Epochs": "epochs the run reached (CSV: epochs, from state.json).",
            "GPU hours": "estimated occupied GPU time, epochs x seconds per epoch between the first and last trace "
                         "record (CSV: estimated_gpu_hours, seconds_per_epoch).",
            "Validation metric, min (epoch)": "lowest value of the authors' validation cost metric and its epoch, "
                                              "counted from 1 (CSV: validation_metric_minimum, _minimum_epoch).",
            "Validation metric, last": "running median over the last 50 epochs of the validation metric at the end "
                                       "(CSV: validation_metric_last_running_median; the raw last value is "
                                       "validation_metric_last).",
            "Train cost at 0.5 (FN / FP)": "final model on the 60,000 training rows (CSV: train_cost, _fn, _fp).",
            "Test cost at 0.5, final model (FN / FP)": "final model on the test set (CSV: test_cost, _fn, _fp).",
            "Test cost at 0.5, best-validation checkpoint": "the checkpoint with the best validation metric "
                                                            "(CSV: best_checkpoint_*).",
            "Lowest test cost along training (epoch)": "lowest cost in the test trace, chosen with the test labels: "
                                                       "a diagnostic (CSV: trace_lowest_test_cost, _epoch; the "
                                                       "trace's last point is trace_last_test_cost, trace_last_epoch).",
        })
    return table


def retraining_and_contamination_rows() -> pd.DataFrame:
    """One row per retraining run (without test rows, test-inclusion controls, verification reruns) with its test cost
    at 1,000 epochs and at its last epoch, its final evaluation, and the cost on the test rows that were placed in the
    training table (seen) and on the others (unseen)."""
    rows = []
    for run in CLEAN_RUNS + HALF_CONTAMINATION_RUNS + FULL_CONTAMINATION_RUNS + VERIFICATION_RUNS:
        files = load_retraining_run(run)
        trace = files["trace"]
        last = trace[-1]
        at_thousand = _trace_record_at_epoch(trace, 1000)
        final = files["final_evaluation"]
        row = {
            "run": run, "kind": RUN_KINDS[run], "epochs": files["state"]["epoch"],
            "test_cost_at_1000_epochs": at_thousand.get("test_cost_at_0.5", np.nan),
            "test_fn_at_1000_epochs": at_thousand.get("fn", np.nan),
            "test_fp_at_1000_epochs": at_thousand.get("fp", np.nan),
            "last_trace_epoch": last["epoch"], "test_cost_at_last_epoch": last["test_cost_at_0.5"],
            "test_fn_at_last_epoch": last["fn"], "test_fp_at_last_epoch": last["fp"],
            "final_evaluation_epochs": final.get("epochs_run"),
            "final_evaluation_test_cost": final["test"]["cost_at_0.5"],
            "final_evaluation_test_fn": final["test"]["fn"], "final_evaluation_test_fp": final["test"]["fp"],
            "best_checkpoint_test_cost": final.get("test_best_ckpt", {}).get("cost_at_0.5", np.nan),
            "best_checkpoint_epoch": final.get("test_best_ckpt", {}).get("epoch", np.nan),
            "final_evaluation_train_cost": final["train"]["cost_at_0.5"],
        }
        for part in ("seen", "unseen"):
            if part in last:
                row.update({f"{part}_test_rows": last[part]["n"],
                            f"{part}_test_cost_at_last_epoch": last[part]["cost_at_0.5"],
                            f"{part}_test_fn_at_last_epoch": last[part]["fn"],
                            f"{part}_test_fp_at_last_epoch": last[part]["fp"]})
        rows.append(row)
    return pd.DataFrame(rows)


def write_retraining_and_contamination_table(table: pd.DataFrame) -> None:
    thesis_rows = table[table.kind != "verification rerun"]
    verification_rows = table[table.kind == "verification rerun"]
    sections = [("Runs in the thesis table", _readable_retraining_rows(thesis_rows)),
                ("Verification reruns of seed 0 (300 epochs, another session)",
                 _readable_retraining_rows(verification_rows))]
    write_summary_table(
        "rq1_transformer_2026_retraining_and_contamination_runs", table, sections,
        title="RQ1b: retraining runs and test-inclusion controls",
        caption="""
        The released training code run from scratch on the published inputs (clean_seed<k>) and with half (8,000
        rows, drawn at random) or all of the 16,000 test rows deliberately added to the training table (the
        test-inclusion controls). All costs are on the official test set at the published 0.5 cut-off, from the test
        trace (scored every 50 epochs, a diagnostic) and from final_evaluation.json. For the test-inclusion controls
        the test set splits into the rows that were placed in the training table (seen) and the others (unseen).
        Sources: authors_code/retraining_runs/<run>/ (test_trace.jsonl, final_evaluation.json, state.json). The
        verification reruns repeat seed 0 for 300 epochs because GPU training is not bit-reproducible.""",
        column_notes={
            "Run": "design and seed of the run (CSV: run, the run name, for example clean_seed0).",
            "Epochs": "epochs the run reached (CSV: epochs; kind says no, half or all of the test rows in "
                      "training, or verification rerun).",
            "At 1,000 epochs (FN / FP)": "test cost in the trace at epoch 1,000 (CSV: test_*_at_1000_epochs).",
            "At the last epoch (FN / FP)": "last point of the trace (CSV: test_*_at_last_epoch, last_trace_epoch).",
            "Best-validation checkpoint (epoch)": "test cost of the checkpoint with the best validation metric, from "
                                                  "final_evaluation.json; shown only when that evaluation covers all "
                                                  "the epochs the run reached (CSV: best_checkpoint_test_cost, "
                                                  "best_checkpoint_epoch, final_evaluation_epochs).",
            "Seen rows / Unseen rows, last epoch": "cost on the test rows inside / outside the training table at the "
                                                   "last trace point; 'none' when every test row was in the training "
                                                   "table (CSV: seen_* and unseen_*, with the row counts).",
            "CSV only": "final_evaluation_test_cost/_fn/_fp and final_evaluation_train_cost.",
        })


def _retraining_run_label(run: str) -> str:
    """Readable name of a retraining run: its design and seed, worded as in rq1_transformer_2026_epochs_reached."""
    design, seed_and_suffix = run.split("_seed")
    label = f"{RETRAINING_DESIGN_LABELS.get(design, design)}, seed {seed_and_suffix.split('_')[0]}"
    if seed_and_suffix.endswith("_verification"):
        label += ", verification rerun"
    return label


def _readable_retraining_rows(rows: pd.DataFrame) -> pd.DataFrame:
    best_checkpoint_cells = []
    for row in rows.itertuples(index=False):
        if row.final_evaluation_epochs == row.epochs and not is_missing(row.best_checkpoint_test_cost):
            best_checkpoint_cells.append(f"{thousands(row.best_checkpoint_test_cost)} "
                                         f"({thousands(row.best_checkpoint_epoch)})")
        else:
            best_checkpoint_cells.append(MISSING)
    seen_cells, unseen_cells = [], []
    for row in rows.itertuples(index=False):
        seen_cells.append(_part_cell(row, "seen"))
        if row.kind == "all test rows in training":
            unseen_cells.append("none")
        else:
            unseen_cells.append(_part_cell(row, "unseen"))
    return pd.DataFrame({
        "Run": rows.run.map(_retraining_run_label),
        "Epochs": rows.epochs.map(thousands),
        "At 1,000 epochs (FN / FP)": [cost_with_errors(cost, fn, fp) for cost, fn, fp in
                                      zip(rows.test_cost_at_1000_epochs, rows.test_fn_at_1000_epochs,
                                          rows.test_fp_at_1000_epochs)],
        "At the last epoch (FN / FP)": [cost_with_errors(cost, fn, fp) for cost, fn, fp in
                                        zip(rows.test_cost_at_last_epoch, rows.test_fn_at_last_epoch,
                                            rows.test_fp_at_last_epoch)],
        "Best-validation checkpoint (epoch)": best_checkpoint_cells,
        "Seen rows, last epoch": seen_cells,
        "Unseen rows, last epoch": unseen_cells,
    })


def _part_cell(row, part: str) -> str:
    cost = getattr(row, f"{part}_test_cost_at_last_epoch", np.nan)
    if is_missing(cost):
        return MISSING
    return cost_with_errors(cost, getattr(row, f"{part}_test_fn_at_last_epoch"),
                            getattr(row, f"{part}_test_fp_at_last_epoch"))


# ---------------------------------------------------------------------------------------------- PyTorch port
def write_pytorch_port_table() -> pd.DataFrame:
    """The PyTorch port: best-validation checkpoint at 0.5 and at the validation-optimal threshold, last epoch."""
    rows = []
    for run in load_pytorch_port_runs():
        record = run["record"]
        best_at_default = record["test_best_ckpt"]["at_0.5"]
        best_at_validation_threshold = record["test_best_ckpt"]["at_val_threshold"]
        last_epoch = record.get("test_final_epoch", record.get("test_last_saved_epoch", {})).get("at_0.5", {})
        interval = best_at_default.get("ci95") or [np.nan, np.nan]
        if run["evaluated_from_checkpoint"]:
            epochs = record["epoch_saved"] + 1          # epochs are counted from 0 in the checkpoint
        else:
            epochs = record["epochs_run"]
        rows.append({
            "run_name": run["run_name"], "label": PORT_RUN_LABELS.get(run["run_name"], run["run_name"]),
            "seed": run["seed"], "epochs": epochs, "best_epoch": record.get("best_epoch"),
            "test_cost_best_checkpoint": best_at_default["cost"], "test_fn_best_checkpoint": best_at_default["fn"],
            "test_fp_best_checkpoint": best_at_default["fp"],
            "test_cost_interval_low": interval[0], "test_cost_interval_high": interval[1],
            "test_cost_at_validation_threshold": best_at_validation_threshold["cost"],
            "test_fn_at_validation_threshold": best_at_validation_threshold["fn"],
            "test_fp_at_validation_threshold": best_at_validation_threshold["fp"],
            "test_cost_last_epoch": last_epoch.get("cost"),
            "evaluated_from_checkpoint": run["evaluated_from_checkpoint"],
        })
    table = pd.DataFrame(rows)
    order = {name: position for position, name in enumerate(PORT_RUN_LABELS)}
    table["order"] = [order.get(name, len(order)) for name in table.run_name]
    table = table.sort_values(["order", "seed"]).drop(columns="order").reset_index(drop=True)
    readable = pd.DataFrame({
        "Variant": table.label,
        "Seed": table.seed.astype(str),
        "Epochs": [thousands(epochs) + (" (a)" if from_checkpoint else "") for epochs, from_checkpoint in
                   zip(table.epochs, table.evaluated_from_checkpoint)],
        "Best epoch": table.best_epoch.map(thousands),
        "Test, best checkpoint at 0.5 (FN / FP)": [cost_with_errors(cost, fn, fp) for cost, fn, fp in
                                                   zip(table.test_cost_best_checkpoint, table.test_fn_best_checkpoint,
                                                       table.test_fp_best_checkpoint)],
        "At validation threshold": [cost_with_errors(cost, fn, fp) for cost, fn, fp in
                                    zip(table.test_cost_at_validation_threshold, table.test_fn_at_validation_threshold,
                                        table.test_fp_at_validation_threshold)],
        "Last epoch at 0.5": table.test_cost_last_epoch.map(thousands),
    })
    write_summary_table(
        "rq1_transformer_2026_pytorch_port_runs", table, readable,
        title="RQ1b: the PyTorch port of the cost-sensitive Transformer of Beikmohammadi et al.",
        caption="""
        The network re-implemented in PyTorch (transformer_2026_pytorch_port.py) and trained by
        run_transformer_2026_port.py for 500 epochs per seed (10 imputer rounds): under the protocol with a validation
        split of real rows (run names corrected_protocol*: 10 % of the real training rows held out for validation first,
        imputation, resampling and scaling fitted on the other 90 %) and under the published protocol (the whole
        training table imputed, resampled and scaled, then 10 % of the resampled table held out for validation), the
        latter also without RENN, without SVM-SMOTE and without any resampling. The best checkpoint is the epoch with
        the lowest validation cost at 0.5. (a) The extended run under the protocol with a validation split of real rows
        continued seed 0 towards 4,000 epochs, was stopped by the compute budget and was scored afterwards from its last
        saved checkpoint by evaluate_port_checkpoint.py. Sources:
        rq1_transformer_2026_reproduction/pytorch_port/<run>_seed<k>.json.""",
        column_notes={
            "Variant": "protocol and ablation (CSV: run_name, the file name stem, and label).",
            "Epochs, Best epoch": "epochs trained and the epoch of the best validation cost (CSV: epochs, best_epoch, "
                                  "evaluated_from_checkpoint).",
            "Test, best checkpoint at 0.5 (FN / FP)": "test cost of the best-validation checkpoint at the 0.5 cut-off "
                                                      "(CSV: test_*_best_checkpoint; its bootstrap 95 % interval is "
                                                      "test_cost_interval_low/high).",
            "At validation threshold": "the same checkpoint at the threshold that minimises the validation cost "
                                       "(CSV: test_*_at_validation_threshold).",
            "Last epoch at 0.5": "the weights of the last epoch (or last saved checkpoint) at 0.5 "
                                 "(CSV: test_cost_last_epoch).",
        })
    return table


# ---------------------------------------------------------------------------------------------- epochs reached
def write_epochs_reached_table(retraining: pd.DataFrame, port: pd.DataFrame) -> None:
    """How far every training run of RQ1b got: the retraining runs of the authors' code (the rows of
    retraining_and_contamination_rows) and the runs of the PyTorch port (the rows of write_pytorch_port_table)."""
    rows = []
    for row in retraining.itertuples(index=False):
        design, seed_and_suffix = row.run.split("_seed")
        label = RETRAINING_DESIGN_LABELS.get(design, design)
        if seed_and_suffix.endswith("_verification"):
            label += ", verification rerun"
        rows.append({"implementation": "authors_code", "run": row.run, "label": label,
                     "seed": int(seed_and_suffix.split("_")[0]), "epochs_reached": row.epochs,
                     "source": "state.json: epoch"})
    for row in port.itertuples(index=False):
        rows.append({"implementation": "pytorch_port", "run": row.run_name, "label": row.label, "seed": row.seed,
                     "epochs_reached": row.epochs,
                     "source": "epoch_saved + 1" if row.evaluated_from_checkpoint else "epochs_run"})
    table = pd.DataFrame(rows)
    sections = []
    for implementation, heading in (("authors_code", "Retraining with the released code"),
                                    ("pytorch_port", "The PyTorch port")):
        part = table[table.implementation == implementation]
        sections.append((heading, pd.DataFrame({"Run": part.label, "Seed": part.seed.astype(str),
                                                "Epochs reached": part.epochs_reached.map(thousands)})))
    write_summary_table(
        "rq1_transformer_2026_epochs_reached", table, sections,
        title="RQ1b: epochs reached by the training runs",
        caption="""
        The number of epochs that each training run of RQ1b reached. For the runs of the released training code
        from scratch (the retraining runs, the test-inclusion controls and the two verification reruns) it is the epoch
        in the run's state.json; for the PyTorch port it is epochs_run in the run's result file, except for the
        extended run under the protocol with a validation split of real rows, which was stopped and scored afterwards
        from its last saved checkpoint: there it is the saved epoch plus one, since the checkpoint counts epochs from 0.
        The number of epochs each run was planned to reach is not stored in the result files and is not shown. Sources,
        in results/rq1_transformer_2026_reproduction/: authors_code/retraining_runs/<run>/state.json and
        pytorch_port/<run>_seed<k>.json.""",
        column_notes={
            "Run": "design of the run (CSV: label; run is the run name, implementation is authors_code or "
                   "pytorch_port).",
            "Epochs reached": "CSV: epochs_reached; source names the field it was read from.",
        })


# ---------------------------------------------------------------------------------------------- hard negatives
def _flag_rate_cell(share) -> str:
    """Flag rates as the thesis prints them: two decimals below 1 %, one decimal above."""
    if is_missing(share):
        return MISSING
    percent = 100 * share
    if percent < 1:
        return f"{percent:.2f} %"
    return f"{percent:.1f} %"


def write_shipped_model_hard_negative_table() -> None:
    """Share of rows flagged as failures, per group of rows: the released Transformers and three tree learners.

    The Transformer rate is the mean over the five released models of the share with probability above 0.5. The tree
    learners are the thesis's own arms (seed 11): out-of-fold probabilities for training rows, so no training row is
    scored by a model that saw it, and fold-ensemble probabilities for test rows, each at the arm's OOF threshold."""
    files = {name: read_json(SHIPPED_MODELS_EXACT_INPUTS_DIR / name)
             for name in ("hard_negative_check.json", "hard_negative_check_followup.json")}
    tree_control = load_tree_control_flag_rates()
    rows = []
    for label, file_name, group_key, tree_key in SHIPPED_MODEL_GROUPS:
        group = files[file_name][group_key]
        run_rates = [group[run]["flag_rate_0.5"] for run in group if run.startswith("run")]
        row = {"group": label.strip(), "source_file": file_name, "source_key": group_key, "rows": group["n"],
               "shipped_transformers": float(np.mean(run_rates))}
        for arm in TREE_CONTROL_ARMS:
            row[arm] = tree_control[arm][tree_key]
        rows.append(row)
    table = pd.DataFrame(rows)
    readable = pd.DataFrame({"Rows": table.group,
                             "n": table.rows.map(thousands),
                             "Released Transformers (0.5)": table.shipped_transformers.map(_flag_rate_cell)})
    for arm, header in TREE_CONTROL_ARMS.items():
        readable[header] = table[arm].map(_flag_rate_cell)
    write_summary_table(
        "rq1_transformer_2026_hard_negatives_shipped_models", table, readable,
        title="RQ1b: where the released models flag rows (hard-negative comparison)",
        caption="""
        Share of each group of rows flagged as failures. Training negatives are split by whether RENN kept them in the
        published training table (the models trained on them) or removed them (never seen); test negatives by the
        one-shot RENN rule (hard if their three nearest neighbours in the resampled table are not all negatives).
        Neither the hard test negatives nor the training negatives that RENN removed are in the published training
        table. Sources: authors_code/shipped_models_exact_inputs/hard_negative_check.json,
        hard_negative_check_followup.json and hard_negative_check_tree_control.json. The tree learners are the tuned
        LightGBM, XGBoost (fixed settings) and the ensemble without missingness indicators, all with seed 11.""",
        column_notes={
            "Rows": "the group of rows (CSV: group; source_file and source_key locate it in the result files).",
            "n": "number of rows in the group (CSV: rows).",
            "Released Transformers (0.5)": "mean over the five released models of the share with probability above 0.5 "
                                          "(CSV: shipped_transformers, a share between 0 and 1).",
            "LightGBM, XGBoost, Ensemble": "share flagged by the tree arm at its own OOF threshold; out-of-fold "
                                           "probabilities for training rows (CSV: lightgbm_tuned, "
                                           "xgboost_fixed_settings, ensemble_without_missingness_indicators).",
        })


def write_retrained_model_hard_negative_table() -> None:
    """The same flag rates for the retrained models' final epoch, per run (groups as rows, runs as columns)."""
    flag_rates = load_retrained_hard_negative_flag_rates()
    runs = [run for run in RETRAINED_RUN_HEADERS if run in flag_rates]
    rows = []
    for group_key, label in RETRAINED_MODEL_GROUPS:
        values = [flag_rates[run].get(group_key, np.nan) for run in runs]
        if all(is_missing(value) for value in values):
            continue           # a group that exists only for other kinds of runs
        rows.append({"group": label, "source_key": group_key, **dict(zip(runs, values))})
    table = pd.DataFrame(rows)
    readable = pd.DataFrame({"Group of negatives": table.group})
    for run in runs:
        readable[RETRAINED_RUN_HEADERS[run]] = [fixed_decimals(100 * value, 1) if not is_missing(value) else MISSING
                                                for value in table[run]]
    write_summary_table(
        "rq1_transformer_2026_hard_negatives_retrained_models", table, readable,
        title="RQ1b: hard-negative comparison for the retrained models (% flagged at 0.5)",
        caption="""
        The comparison of rq1_transformer_2026_hard_negatives_shipped_models repeated with the models retrained from
        scratch (final epoch; epochs per run in rq1_transformer_2026_epochs_reached): the percentage of each group of
        negatives with probability of at least 0.5, on the same masks (RENN kept or removed; hard by the one-shot rule).
        For the test-inclusion controls the test negatives split into those in the training table (seen) and the rest
        (unseen). Source: hard_negative_check_retrained_models.json (written by
        hard_negative_check_retrained_models.py).""",
        column_notes={
            "Group of negatives": "CSV: group; source_key is the group's key in the result file.",
            "Retraining, seed 0 ... 4": "clean_seed0 ... clean_seed4 (CSV column names are the run names).",
            "Half, seed 0 ... 2": "half_test_rows_in_training_seed0 ... seed2 (8,000 test rows in the training "
                                  "table).",
            "All, seed 0": "all_test_rows_in_training_seed0 (every test row in the training table).",
            "Values": "percent of the group flagged; the CSV holds shares between 0 and 1.",
        })


# ---------------------------------------------------------------------------------------------- numbers in the text
def write_summary_numbers(rescoring: pd.DataFrame, clean_runs: pd.DataFrame,
                          retraining: pd.DataFrame, port: pd.DataFrame) -> None:
    """Aggregates over the tables above that the thesis quotes in its text (means, ranges, totals)."""
    numbers = []

    def add(quantity: str, value, decimals: int = 0) -> None:
        numbers.append({"quantity": quantity, "value": value, "decimals": decimals})

    published = rescoring[rescoring.variant == RESCORING_VARIANTS[0][0]]
    published_runs = published[published.run != "mean"]
    add("Released models, published preprocessing: mean test cost", published_runs.test_cost.mean())
    add("Released models, published preprocessing: lowest test cost of a run", published_runs.test_cost.min())
    add("Released models, published preprocessing: highest test cost of a run", published_runs.test_cost.max())
    add("Released models, published preprocessing: mean missed failures", published_runs.test_fn.mean(), 1)
    add("Released models, published preprocessing: mean false alarms", published_runs.test_fp.mean(), 1)
    run_one = published_runs[published_runs.run == "run1"].iloc[0]
    add("Released model run 1 on the 60,000 training rows: cost at 0.5", run_one.train_cost)
    add("Released model run 1 on the 60,000 training rows: missed failures", run_one.train_fn)
    add("Released model run 1 on the 60,000 training rows: false alarms", run_one.train_fp)

    add("Retraining runs without test rows: number of seeds", len(clean_runs))
    add("Retraining runs without test rows: median test cost of the final evaluation", clean_runs.test_cost.median())
    add("Retraining runs without test rows: lowest test cost of the final evaluation", clean_runs.test_cost.min())
    add("Retraining runs without test rows: highest test cost of the final evaluation", clean_runs.test_cost.max())
    add("Retraining runs without test rows: fewest epochs", clean_runs.epochs.min())
    add("Retraining runs without test rows: most epochs", clean_runs.epochs.max())
    add("Retraining runs without test rows: estimated GPU hours, total", clean_runs.estimated_gpu_hours.sum(), 1)
    clean = retraining[retraining.kind == "no test rows in training"]
    add("Retraining runs without test rows: mean test cost at 1,000 epochs", clean.test_cost_at_1000_epochs.mean())
    add("Retraining runs without test rows: lowest test cost at 1,000 epochs", clean.test_cost_at_1000_epochs.min())
    add("Retraining runs without test rows: highest test cost at 1,000 epochs", clean.test_cost_at_1000_epochs.max())
    half = retraining[(retraining.kind == "half of the test rows in training")]
    add("Test-inclusion controls with half of the test rows: number of seeds", int(half.seen_test_rows.notna().sum()))

    for run_name, label in PORT_RUN_LABELS.items():
        runs = port[(port.run_name == run_name) & ~port.evaluated_from_checkpoint]
        if len(runs) == 0:
            continue
        label = label[0].lower() + label[1:]     # it follows "PyTorch port, " in the row name
        add(f"PyTorch port, {label}: mean test cost, best checkpoint at 0.5", runs.test_cost_best_checkpoint.mean())
        add(f"PyTorch port, {label}: lowest test cost, best checkpoint at 0.5", runs.test_cost_best_checkpoint.min())
        add(f"PyTorch port, {label}: highest test cost, best checkpoint at 0.5", runs.test_cost_best_checkpoint.max())
        add(f"PyTorch port, {label}: lowest test cost at the validation threshold",
            runs.test_cost_at_validation_threshold.min())
        add(f"PyTorch port, {label}: highest test cost at the validation threshold",
            runs.test_cost_at_validation_threshold.max())
    table = pd.DataFrame(numbers)
    readable = pd.DataFrame({"Quantity": table.quantity,
                             "Value": [thousands(value, decimals) for value, decimals in
                                       zip(table.value, table.decimals)]})
    write_summary_table(
        "rq1_transformer_2026_summary_numbers", table[["quantity", "value"]], readable,
        title="RQ1b: means, ranges and totals quoted in the text",
        caption="""
        Aggregates of the RQ1b tables that the thesis quotes in its running text, computed from the same files:
        rq1_transformer_2026_shipped_models_rescored, _clean_retraining_runs, _retraining_and_contamination_runs and
        _pytorch_port_runs. Ranges over PyTorch port runs leave out the extended run scored from a checkpoint.""")


# ---------------------------------------------------------------------------------------------- figures
def plot_published_training_logs() -> None:
    """The authors' own training logs: the validation metric along the 8,000 epochs of each published run."""
    figure, axes = plt.subplots(figsize=(9, 4.5))
    for run in range(1, 6):
        log = pd.read_csv(PUBLISHED_TRAINING_LOGS_DIR / f"run{run}.csv")
        metric = log[VALIDATION_METRIC]
        axes.plot(log.epoch, metric.rolling(50, min_periods=1).median(), lw=1.0, label=f"published run {run}")
        axes.scatter([metric.idxmin()], [metric.min()], s=14, zorder=5)
    axes.set_xlabel("epoch")
    axes.set_ylabel("validation cost metric (50-epoch running median)")
    axes.set_title("Training logs published by Beikmohammadi et al. (dots: lowest value of each run)")
    axes.legend(fontsize=8)
    save_figure(figure, "rq1_transformer_2026_published_training_logs")


def plot_clean_retraining_test_trace() -> None:
    """Test cost and missed failures at 0.5 along training, retraining runs without test rows (diagnostic trace)."""
    figure, (cost_axes, missed_axes) = plt.subplots(1, 2, figsize=(11, 4.5))
    for run in CLEAN_RUNS:
        trace = load_retraining_run(run)["trace"]
        epochs = [record["epoch"] for record in trace]
        label = run.replace("clean_seed", "retraining, seed ")
        costs = [record["test_cost_at_0.5"] for record in trace]
        cost_axes.plot(epochs, costs, marker="o", ms=2.5, lw=0.9, label=label)
        missed_axes.plot(epochs, [record["fn"] for record in trace], marker="o", ms=2.5, lw=0.9, label=label)
    cost_axes.axhline(PUBLISHED_MEAN_TEST_COST, color="k", lw=0.7, ls=":", label="published mean 3,440")
    cost_axes.set_yscale("log")
    cost_axes.set_xlabel("epoch")
    cost_axes.set_ylabel("test cost at the 0.5 cut-off (log scale)")
    cost_axes.legend(fontsize=8)
    missed_axes.axhline(PUBLISHED_MEAN_MISSED_FAILURES, color="k", lw=0.7, ls=":", label="published mean 6.8")
    missed_axes.set_xlabel("epoch")
    missed_axes.set_ylabel("missed failures at 0.5 (of 375)")
    missed_axes.legend(fontsize=8)
    figure.suptitle("Retraining with the released code, no test rows in the training data: test trace (diagnostic)")
    save_figure(figure, "rq1_transformer_2026_clean_retraining_test_trace")


def plot_retraining_and_contamination_trace() -> None:
    """Test cost along training for the retraining runs and the test-inclusion controls. For the control runs with half
    of the test rows in the training data the costs on the 8,000 seen and the 8,000 unseen test rows are doubled, so
    that they are on the scale of the full 16,000-row test set."""
    figure, axes = plt.subplots(figsize=(10, 5.5))
    for run in CLEAN_RUNS:
        trace = load_retraining_run(run)["trace"]
        axes.plot([record["epoch"] for record in trace], [record["test_cost_at_0.5"] for record in trace], "-",
                  lw=1.3, label=run.replace("clean_seed", "retraining, seed "))
    for run in HALF_CONTAMINATION_RUNS:
        trace = load_retraining_run(run)["trace"]
        epochs = [record["epoch"] for record in trace]
        axes.plot(epochs, [2 * record["seen"]["cost_at_0.5"] for record in trace], "--", lw=1.3, color="#c9553d",
                  label=f"half of the test rows in training, seed {run[-1]}: seen test rows (x2)")
        axes.plot(epochs, [2 * record["unseen"]["cost_at_0.5"] for record in trace], "--", lw=1.3, color="#8c6bb1",
                  label=f"half of the test rows in training, seed {run[-1]}: unseen test rows (x2)")
    for run in FULL_CONTAMINATION_RUNS:
        trace = load_retraining_run(run)["trace"]
        axes.plot([record["epoch"] for record in trace], [record["seen"]["cost_at_0.5"] for record in trace], ":",
                  lw=1.3, color="#c9553d", label=f"all test rows in training, seed {run[-1]}: seen test rows")
    axes.axhline(PUBLISHED_MEAN_TEST_COST, color="k", lw=0.8, ls="-.", label="published mean (3,440)")
    axes.set_xlabel("epoch")
    axes.set_ylabel("cost on the official test set at the 0.5 cut-off (log scale)")
    axes.set_yscale("log")
    axes.set_title("Retraining runs and test-inclusion controls: test trace (diagnostic)")
    axes.legend(fontsize=7.5, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save_figure(figure, "rq1_transformer_2026_retraining_and_contamination_trace")


def main() -> None:
    print("Reproduction of the cost-sensitive Transformer of Beikmohammadi et al. (RQ1b)")
    rescoring = write_shipped_model_rescoring_table()
    clean_runs = write_clean_retraining_table()
    retraining = retraining_and_contamination_rows()
    write_retraining_and_contamination_table(retraining)
    port = write_pytorch_port_table()
    write_epochs_reached_table(retraining, port)
    write_shipped_model_hard_negative_table()
    write_retrained_model_hard_negative_table()
    write_summary_numbers(rescoring, clean_runs, retraining, port)
    plot_published_training_logs()
    plot_clean_retraining_test_trace()
    plot_retraining_and_contamination_trace()


if __name__ == "__main__":
    main()
