"""Supporting results: the data audit, the first-look baselines, the hyperparameter search, the threshold-rule
demonstration, the TabICL pilot, the synthetic-data diagnostics and the synthetic-row count sweep.

These are the smaller studies that the method and results chapters lean on:

  data audit                   integrity checks of the two UCI files (experiments/common/data_loading.py): sizes,
                               class counts, missing values, checksums, and the cost of two trivial classifiers;
  first-look baselines         three gradient-boosting libraries with fixed, untuned settings, run before any tuning
                               and scored by their 5-fold out-of-fold cost with seed 11, without the test set
                               (experiments/quick_baseline_check.py);
  hyperparameter search        the Optuna search of the LightGBM hyperparameters (experiments/hyperparameter_search.py,
                               26 trials, each scored by its 5-fold out-of-fold cost with seed 11);
  threshold-rule demonstration how much the rule for the decision threshold matters, with the learners of the earlier
                               phase of the project (experiments/run_threshold_rule_demonstration.py): each learner is
                               trained once on 80 % of the training rows, the threshold is chosen on the other 20 %
                               under four rules, and the test set is scored under each;
  TabICL pilot                 TabICLv2, a tabular foundation model that classifies by in-context learning (the
                               training rows are given to a pretrained transformer as context instead of fitting
                               weights), on single folds with one context each, against the tuned LightGBM on the
                               same validation rows (rq2_survey_of_methods/pilots/tabicl_pilot.py);
  synthetic-data diagnostics   how far the synthetic failure rows of each generator are from real failure rows
                               (rq2_survey_of_methods/run_synthetic_data_diagnostics.py): detectability AUC (how well
                               a classifier tells synthetic from real rows; 0.5 = indistinguishable, 1.0 = trivially
                               separable), distance to the closest real record, and differences of the marginals;
  synthetic-row count sweep    the cGAN of Ranasinghe et al. with 250 to 5,000 added rows, the count chosen on the OOF
                               cost (the common evaluation protocol) and, to measure the effect of test-set selection,
                               on the test set (rq1_cgan_2019_reproduction/run_synthetic_row_count_sweep.py).

Run from the repository root (also part of `python -m analysis.run_all_summaries`):

    python -m analysis.summarise_supporting_results

Reads results/data_audit.json, results/hyperparameter_search/lightgbm_seed11_trials.csv,
results/preliminary_checks/quick_baseline.jsonl and threshold_rule_demonstration.json,
results/rq2_survey_of_methods/tabicl_pilot.json and synthetic_data_diagnostics.json,
results/rq1_cgan_2019_reproduction/synthetic_row_count_sweep.jsonl, and the training data file (for the data
overview figure). Writes to results/summary_tables/: data_audit, quick_baseline_check, hyperparameter_search_trials,
threshold_rule_demonstration, tabicl_pilot, synthetic_data_diagnostics and synthetic_row_count_sweep; and to
results/figures/: data_overview, hyperparameter_search_history and synthetic_row_count_sweep_<generator> (one per
generator in the sweep file).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.load_results import (HYPERPARAMETER_SEARCH_TRIALS_FILE, SYNTHETIC_DATA_DIAGNOSTICS_FILE,
                                   SYNTHETIC_ROW_COUNT_SWEEP_FILE, TABICL_PILOT_FILE,
                                   THRESHOLD_RULE_DEMONSTRATION_FILE, TRAINING_ROWS, load_quick_baseline, read_json,
                                   read_jsonl)
from analysis.write_outputs import (MISSING, cost_with_errors, fixed_decimals, save_figure, scientific, thousands,
                                    write_summary_table)
from experiments.common.data_loading import TEST_SET_ROWS, load_training_set
from experiments.common.paths import DATA_AUDIT_FILE

# ---------------------------------------------------------------------------------------------- data audit
DATA_AUDIT_QUANTITIES = {
    "train_rows": "Training rows",
    "train_pos": "Failures among the training rows",
    "test_rows": "Test rows",
    "test_pos": "Failures among the test rows",
    "n_features": "Attributes",
    "features_identical": "Same attributes, in the same order, in both files",
    "train_missing_cells_frac": "Share of missing cells, training file",
    "test_missing_cells_frac": "Share of missing cells, test file",
    "pos_row_missing_frac": "Share of missing cells in failure rows (training)",
    "neg_row_missing_frac": "Share of missing cells in healthy rows (training)",
    "features_over_50pct_missing": "Attributes with more than 50 % missing (training)",
    "features_over_70pct_missing": "Attributes with more than 70 % missing (training)",
    "constant_features": "Constant attributes (training)",
    "duplicate_rows_train_test": "Test rows identical to a training row (values rounded to 3 decimals)",
    "cost_check_all_neg": "Test cost when nothing is flagged",
    "cost_check_all_pos": "Test cost when everything is flagged",
    "train_sha256": "SHA-256 of the training file",
    "test_sha256": "SHA-256 of the test file",
}

# ---------------------------------------------------------------------------------------------- first-look baselines
# Configuration names of quick_baseline.jsonl with the labels of the thesis table, in its order. "Fixed settings" are
# those of the factories in experiments/common/classifiers.py; "failure weight 30" is scale_pos_weight = 30.
QUICK_BASELINE_LABELS = {
    "lightgbm_fixed_settings": "LightGBM, fixed settings (600 trees)",
    "lightgbm_scale_pos_weight_30": "LightGBM, fixed settings, failure weight 30",
    "xgboost_fixed_settings": "XGBoost, fixed settings (600 trees)",
    "histogram_gradient_boosting_fixed_settings": "HistGradientBoosting, fixed settings (at most 600 iterations)",
}
# Missed-failure counts at which quick_baseline_check.py records the fewest false alarms ("fp_at_fn").
QUICK_BASELINE_MISSED_FAILURE_COUNTS = (10, 20, 30, 40, 60)

# ---------------------------------------------------------------------------------------------- threshold rules
THRESHOLD_RULES = {
    "linear_0.01": "Linear grid from 0.01",
    "all_values": "All validation scores",
    "fixed_0.5": "Fixed 0.5",
    "bayes_0.0196": "Bayes 0.0196",
}
# Learner names of the result file, written out where the file abbreviates them.
THRESHOLD_RULE_LEARNER_LABELS = {
    "RF (300 trees, depth 20, balanced_subsample)": "Random forest (300 trees, depth 20, balanced_subsample)",
}

# ---------------------------------------------------------------------------------------------- TabICL pilot
TABICL_PILOT_LEARNERS = {"tabicl": "TabICLv2 (pilot)", "lightgbm_tuned_same_rows": "LightGBM (tuned)"}

# ---------------------------------------------------------------------------------------------- diagnostics
# Rows of the synthetic-data diagnostics table, in the thesis order; a generator in the file but not listed here is
# appended with its key as label.
SYNTHETIC_DATA_LABELS = {
    "reference_real_heldout": "Real held-out failures (reference)",
    "duplicated_real_failures": "Duplication control: copied real failures (reference)",
    "conditional_gan_2019": "cGAN (min-max inputs, single sigmoid), with missingness masks",
    "conditional_gan_2019_without_missingness": "cGAN (min-max inputs, single sigmoid), no masks",
    "conditional_gan_2019_literal_notebook": "cGAN with the notebook's settings",
    "forest_flow": "Forest-Flow, with missingness masks",
    "smote": "SMOTE",
    "borderline_smote": "Borderline-SMOTE",
    "adasyn": "ADASYN",
    "svm_smote": "SVM-SMOTE",
    "gaussian_copula": "Gaussian copula",
    "tvae": "TVAE",
    "ctgan": "CTGAN",
    "ctab_gan_plus": "CTAB-GAN+",
    "tabddpm": "TabDDPM",
    "great": "GReaT (Qwen2.5-0.5B)",
    "tabebm": "TabEBM",
}
DIAGNOSTIC_FIELDS = ["detect_auc", "detect_auc_nomask", "dcr", "frac_closer", "colwise_w1", "mia_auc",
                     "near_copy_frac", "dcr_fit_vs_heldout"]

# ---------------------------------------------------------------------------------------------- count sweep
# Readable names of the generator and classifier keys of synthetic_row_count_sweep.jsonl. The sweep uses the tuned
# hyperparameters where a search was run, which is the case for LightGBM only.
COUNT_SWEEP_GENERATOR_LABELS = {"conditional_gan_2019": "cGAN of Ranasinghe et al.", "forest_flow": "Forest-Flow"}
COUNT_SWEEP_CLASSIFIER_LABELS = {"lightgbm": "tuned LightGBM"}


# ---------------------------------------------------------------------------------------------- data audit
def write_data_audit_table() -> None:
    audit = read_json(DATA_AUDIT_FILE)
    rows = []
    for field, quantity in DATA_AUDIT_QUANTITIES.items():
        value = audit[field]
        if isinstance(value, list):
            value = ", ".join(value) if value else "none"
        rows.append({"quantity": quantity, "field": field, "value": value})
    table = pd.DataFrame(rows)
    write_summary_table(
        "data_audit", table, pd.DataFrame({"Quantity": table.quantity, "Value": table.value.map(_audit_cell)}),
        title="Data audit of the two UCI files",
        caption="""
        Integrity checks of aps_failure_training_set.csv and aps_failure_test_set.csv ('na' read as missing),
        written by `python -m experiments.common.data_loading` to results/data_audit.json. The two costs at the
        end are the cost measure (500 per missed failure, 10 per false alarm) of the trivial rules.""",
        column_notes={"Value": "the value in data_audit.json (CSV: value; field is its key in the file)."})


def _audit_cell(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (int, np.integer)):
        return thousands(value)
    return str(value)


def plot_data_overview() -> None:
    """Class counts, and the share of missing values by class for the 40 attributes most often missing among
    failures (failure rows have more missing values than healthy rows, which is why missingness indicators and the
    re-application of missingness to synthetic rows matter)."""
    features, labels = load_training_set()
    missing_share = pd.DataFrame({"failure": features[labels == 1].isna().mean(),
                                  "healthy": features[labels == 0].isna().mean()})
    missing_share = missing_share.sort_values("failure", ascending=False).head(40)
    class_counts = [int((labels == 0).sum()), int(labels.sum())]
    figure, (count_axes, missing_axes) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1, 2.2]})
    count_axes.bar(["healthy", "failure"], class_counts, color=["#1b4965", "#c9553d"])
    count_axes.set_yscale("log")
    count_axes.set_ylabel("rows (log scale)")
    count_axes.set_title("Class counts, training file")
    for position, count in enumerate(class_counts):
        count_axes.text(position, count * 1.15, f"{count:,}", ha="center", fontsize=9)
    positions = range(len(missing_share))
    missing_axes.plot(positions, missing_share.failure.values, "o-", ms=3, lw=0.8, label="failure rows")
    missing_axes.plot(positions, missing_share.healthy.values, "s-", ms=3, lw=0.8, label="healthy rows")
    missing_axes.set_xlabel("attribute rank (the 40 attributes most often missing among failures)")
    missing_axes.set_ylabel("share missing")
    missing_axes.set_title("Missing values by class")
    missing_axes.legend(fontsize=8)
    save_figure(figure, "data_overview")


# ---------------------------------------------------------------------------------------------- first-look baselines
def write_quick_baseline_table() -> None:
    """The first-look baselines: each configuration's out-of-fold cost at its own cost-minimising threshold."""
    baseline = load_quick_baseline()
    names = [name for name in QUICK_BASELINE_LABELS if name in baseline]
    names += [name for name in baseline if name not in QUICK_BASELINE_LABELS]
    rows = []
    for name in names:
        record = baseline[name]
        rows.append({
            "configuration": name, "label": QUICK_BASELINE_LABELS.get(name, name),
            "oof_cost": record["cost"], "oof_fn": record["fn"], "oof_fp": record["fp"],
            "oof_cost_scaled_to_test_set_size": record["cost_scaled_to_test_set_size"],
            "oof_threshold": record["threshold"], "fold_costs": record["fold_costs"],
            **{f"fp_at_exactly_{count}_fn": record["fp_at_fn"].get(str(count))
               for count in QUICK_BASELINE_MISSED_FAILURE_COUNTS},
            "fit_seconds": record["fit_seconds_total"],
        })
    table = pd.DataFrame(rows)
    false_alarm_columns = [f"fp_at_exactly_{count}_fn" for count in QUICK_BASELINE_MISSED_FAILURE_COUNTS]
    table[false_alarm_columns] = table[false_alarm_columns].astype("Int64")      # whole numbers, empty where none
    readable = pd.DataFrame({
        "Configuration": table.label,
        "OOF cost (FN / FP)": [cost_with_errors(cost, fn, fp) for cost, fn, fp in
                               zip(table.oof_cost, table.oof_fn, table.oof_fp)],
        "Cost scaled to the 16,000 rows of the test set": table.oof_cost_scaled_to_test_set_size.map(thousands),
        "Threshold": table.oof_threshold.map(lambda value: scientific(value, 1)),
    })
    write_summary_table(
        "quick_baseline_check", table, readable,
        title="First-look baselines with fixed settings",
        caption="""
        The first models of the project, run before any tuning: three gradient-boosting libraries with the fixed
        settings of the factories in experiments/common/classifiers.py (for LightGBM and XGBoost 600 trees at
        learning rate 0.03), which are not the libraries' own defaults. Each configuration is evaluated by
        stratified 5-fold cross-validation on the 60,000 training rows with seed 11, at the threshold that
        minimises its out-of-fold cost; the test set is not used. Source:
        results/preliminary_checks/quick_baseline.jsonl (written by experiments/quick_baseline_check.py).""",
        column_notes={
            "Configuration": "learner and settings; failure weight 30 is LightGBM's scale_pos_weight (CSV: label; "
                             "configuration is the name in the result file).",
            "OOF cost (FN / FP)": "cost of the 60,000 out-of-fold predictions at the cost-minimising threshold, with "
                                  "its missed failures and false alarms (CSV: oof_cost, oof_fn, oof_fp).",
            "Cost scaled to the 16,000 rows of the test set": "the OOF cost x 16,000 / 60,000 (CSV: "
                                                              "oof_cost_scaled_to_test_set_size).",
            "Threshold": "the cost-minimising out-of-fold threshold (CSV: oof_threshold).",
            "CSV only": "fold_costs (cost of each validation fold at that threshold), fp_at_exactly_<k>_fn (fewest "
                        "false alarms with exactly k = 10, 20, 30, 40 or 60 missed failures; empty when no candidate "
                        "threshold gives exactly k) and fit_seconds.",
        })


# ---------------------------------------------------------------------------------------------- hyperparameter search
def write_hyperparameter_search_table() -> None:
    """All trials of the search, sorted by OOF cost (the file's own columns, unrounded, in the CSV)."""
    trials = pd.read_csv(HYPERPARAMETER_SEARCH_TRIALS_FILE).sort_values("cost_oof", kind="stable")
    readable = pd.DataFrame({
        "Trial": trials.trial.astype(int).astype(str),
        "Cost": trials.cost_oof.map(thousands),
        "FN": trials.fn.astype(int).map(thousands),
        "FP": trials.fp.astype(int).map(thousands),
        "Threshold": trials.threshold.map(lambda value: f"{value:.1e}"),
        "Trees": trials.n_estimators.astype(int).map(thousands),
        "Learning rate": trials.learning_rate.map(lambda value: f"{value:.3f}"),
        "Leaves": trials.num_leaves.astype(int).map(thousands),
        "Min child": trials.min_child_samples.astype(int).map(thousands),
        "Subsample": trials.subsample.map(fixed_decimals),
        "Colsample": trials.colsample_bytree.map(fixed_decimals),
        "L2": trials.reg_lambda.map(fixed_decimals),
        "Positive weight": trials.scale_pos_weight.map(lambda value: f"{value:.1f}"),
    })
    write_summary_table(
        "hyperparameter_search_trials", trials, readable,
        title="Hyperparameter search of the tuned LightGBM",
        caption="""
        Every Optuna trial of the LightGBM search, sorted by its out-of-fold cost (5 folds, seed 11, 60,000
        training rows, at the trial's own cost-minimising threshold). The best trial gives the hyperparameters of
        every 'tuned' LightGBM arm. Source: results/hyperparameter_search/lightgbm_seed11_trials.csv.""",
        column_notes={
            "Cost, FN, FP, Threshold": "OOF cost with its missed failures and false alarms at the trial's OOF "
                                       "threshold (CSV: cost_oof, fn, fp, threshold).",
            "Trees ... Positive weight": "LightGBM's n_estimators, learning_rate, num_leaves, min_child_samples, "
                                         "subsample, colsample_bytree, reg_lambda and scale_pos_weight (the weight of "
                                         "the failure class).",
            "CSV only": "reg_alpha (L1 penalty) and seconds (time of the trial).",
        })


def plot_hyperparameter_search_history() -> None:
    """OOF cost of every trial and the best so far, with the first-look baseline's untuned LightGBM for scale."""
    trials = pd.read_csv(HYPERPARAMETER_SEARCH_TRIALS_FILE)
    scale = TEST_SET_ROWS / TRAINING_ROWS
    untuned_cost = load_quick_baseline()["lightgbm_fixed_settings"]["cost"]
    figure, axes = plt.subplots(figsize=(8, 4.2))
    axes.plot(trials.trial, trials.cost_oof * scale, "o", ms=4, alpha=0.8, label="trial")
    axes.plot(trials.trial, trials.cost_oof.cummin() * scale, "-", lw=1, label="best so far")
    axes.axhline(untuned_cost * scale, ls=":", c="k", lw=0.9,
                 label="LightGBM, fixed untuned settings (first-look baseline)")
    axes.set_xlabel("Optuna trial")
    axes.set_ylabel("OOF cost, scaled to the 16,000-row test set")
    axes.set_title("Hyperparameter search of LightGBM (seed 11)")
    axes.legend(fontsize=8)
    save_figure(figure, "hyperparameter_search_history")


# ---------------------------------------------------------------------------------------------- threshold rules
def write_threshold_rule_table() -> None:
    """Test cost of one model per learner under four threshold rules, the test oracle and the false-alarm count.

    Rules: linear_0.01 searches 0.01, 0.02, ..., 0.99 on the validation rows (the grid of the evaluation code of the
    earlier phase of the project); all_values searches every distinct validation score; fixed_0.5 is the default
    cut-off; bayes_0.0196 is 10/510, cost-optimal for calibrated probabilities."""
    demonstration = read_json(THRESHOLD_RULE_DEMONSTRATION_FILE)
    rows = []
    for learner, results in demonstration.items():
        row = {"learner": learner, "fit_seconds": results.get("fit_seconds")}
        for rule in THRESHOLD_RULES:
            row.update({f"{rule}_threshold": results[rule]["threshold"], f"{rule}_test_cost": results[rule]["cost"],
                        f"{rule}_test_fn": results[rule]["fn"], f"{rule}_test_fp": results[rule]["fp"],
                        f"{rule}_validation_cost": results[rule].get("val_cost")})
        row.update({"oracle_test_cost": results["oracle"]["cost"], "oracle_threshold": results["oracle"]["threshold"],
                    "min_fp_at_most_6_fn": results["min_fp_at_fn6"]})
        rows.append(row)
    table = pd.DataFrame(rows)
    readable = pd.DataFrame({"Learner": [THRESHOLD_RULE_LEARNER_LABELS.get(learner, learner)
                                         for learner in table.learner]})
    for rule, header in THRESHOLD_RULES.items():
        readable[header] = [cost_with_errors(cost, fn, fp) for cost, fn, fp in
                            zip(table[f"{rule}_test_cost"], table[f"{rule}_test_fn"], table[f"{rule}_test_fp"])]
    readable["Oracle"] = table.oracle_test_cost.map(thousands)
    readable["FP at FN<=6"] = table.min_fp_at_most_6_fn.map(thousands)
    write_summary_table(
        "threshold_rule_demonstration", table, readable,
        title="Decision-rule demonstration with the learners of the earlier phase of the project",
        caption="""
        Each learner trained once on 80 % of the training rows (missing values set to 0, as the notebooks of the
        earlier phase of the project did); the threshold is chosen on the remaining 20 % under four rules and the
        official test set is scored under each: test cost (missed failures / false alarms). The oracle and the
        false-alarm count use the test labels and are diagnostics. Source:
        results/preliminary_checks/threshold_rule_demonstration.json.""",
        column_notes={
            "Learner": "the learner and its settings (CSV: learner, the name in the result file).",
            "Linear grid from 0.01": "best of 0.01, 0.02, ..., 0.99 on the validation rows (CSV: linear_0.01_*; "
                                     "_validation_cost is the cost on the 20 % validation rows).",
            "All validation scores": "best of every distinct validation score (CSV: all_values_*).",
            "Fixed 0.5": "the default cut-off (CSV: fixed_0.5_*).",
            "Bayes 0.0196": "the cut-off 10/510 (CSV: bayes_0.0196_*).",
            "Oracle": "lowest test cost over all thresholds (CSV: oracle_test_cost, oracle_threshold).",
            "FP at FN<=6": "fewest test false alarms with at most 6 missed failures (CSV: min_fp_at_most_6_fn).",
        })


# ---------------------------------------------------------------------------------------------- TabICL pilot
def write_tabicl_pilot_table() -> None:
    """TabICLv2 against the tuned LightGBM on the same validation rows of single folds.

    Both learners are scored at their own fold-optimal threshold (equally optimistic, since the threshold is chosen
    on the rows it is scored on) and by measures that need no threshold: AUC, average precision (AP, the area under
    the precision-recall curve) and the fewest false alarms with at most 2, 5 or 10 missed failures. The LightGBM
    numbers come from its out-of-fold probabilities, restricted to the fold's validation rows."""
    pilot = read_json(TABICL_PILOT_FILE)
    rows = []
    for fold, results in sorted(pilot["folds"].items(), key=lambda item: int(item[0])):
        for key, label in TABICL_PILOT_LEARNERS.items():
            if key not in results:
                continue
            rows.append({"fold": int(fold), "learner": key, "label": label, "validation_rows": results.get("n_val"),
                         **results[key]})
    table = pd.DataFrame(rows)
    readable = pd.DataFrame({
        "Fold": table.fold.astype(str), "Learner": table.label,
        "Fold-optimal cost": table.fold_opt_cost.map(thousands),
        "FN / FP": [f"{fn} / {fp}" for fn, fp in zip(table.fn, table.fp)],
        "AUC": table.auc.map(lambda value: f"{value:.3f}"), "AP": table.ap.map(lambda value: f"{value:.3f}"),
        "FP at FN<=2": table.min_fp_at_fn2.map(thousands), "FP at FN<=5": table.min_fp_at_fn5.map(thousands),
        "FP at FN<=10": table.min_fp_at_fn10.map(thousands),
    })
    write_summary_table(
        "tabicl_pilot", table, readable, left_aligned_columns=2,
        title="TabICLv2 pilot on single folds",
        caption=f"""
        One TabICLv2 context per fold (all failures of the training fold and {pilot.get('healthy_rows_per_context'):,}
        random healthy rows, {pilot.get('n_estimators')} estimator), seed {pilot.get('seed')}, scored on the fold's
        validation rows; the tuned LightGBM on the same rows for comparison. No test-set evaluation. Source:
        results/rq2_survey_of_methods/tabicl_pilot.json (the LightGBM entries are stored under
        'lightgbm_tuned_same_rows').""",
        column_notes={
            "Fold-optimal cost, FN / FP": "cost at the threshold that minimises it on the same validation rows "
                                          "(CSV: fold_opt_cost, fn, fp, threshold).",
            "AUC, AP": "area under the ROC curve and average precision (CSV: auc, ap).",
            "FP at FN<=k": "fewest false alarms with at most k missed failures (CSV: min_fp_at_fn2, _fn5, _fn10).",
            "CSV only": "validation_rows, learner (key in the result file).",
        })


# ---------------------------------------------------------------------------------------------- diagnostics
def write_synthetic_data_diagnostics_table() -> None:
    """Fidelity and memorisation diagnostics of each generator's synthetic failure rows.

    The columns are statistics of experiments/common/synthetic_data_diagnostics.py, where they are defined, as
    computed by rq2_survey_of_methods/run_synthetic_data_diagnostics.py: each generator's rows are compared with the
    real failures it was fitted on. The CSV also keeps the memorisation statistics (mia_auc, near_copy_frac,
    dcr_fit_vs_heldout) where they were computed."""
    diagnostics = read_json(SYNTHETIC_DATA_DIAGNOSTICS_FILE)
    generators = [key for key, value in diagnostics.items() if isinstance(value, dict) and "detect_auc" in value]
    ordered = [key for key in SYNTHETIC_DATA_LABELS if key in generators]
    ordered += [key for key in generators if key not in SYNTHETIC_DATA_LABELS]
    rows = []
    for key in ordered:
        rows.append({"generator": key, "label": SYNTHETIC_DATA_LABELS.get(key, key.replace("_", " ")),
                     **{field: diagnostics[key].get(field, np.nan) for field in DIAGNOSTIC_FIELDS}})
    table = pd.DataFrame(rows)
    # the held-out reference row has no value with missing cells filled; the thesis repeats its other value there
    shown_without_masks = table.detect_auc_nomask.where(table.detect_auc_nomask.notna(), table.detect_auc)
    readable = pd.DataFrame({"Rows": table.label,
                             "Detectability AUC": table.detect_auc.map(fixed_decimals),
                             "same, missing values filled": shown_without_masks.map(fixed_decimals),
                             "Relative NN distance": table.dcr.map(fixed_decimals),
                             "Share closer": table.frac_closer.map(fixed_decimals),
                             "Marginal W1": table.colwise_w1.map(lambda value: fixed_decimals(value, 3))})
    write_summary_table(
        "synthetic_data_diagnostics", table, readable,
        title="Synthetic-data diagnostics per generator",
        caption=f"""
        Each generator fitted on the {diagnostics.get('n_real_fit')} real failures of the training part of one fold
        (seed 11, fold 0), and its synthetic rows compared with these fitting failures. The two reference rows are
        compared with the fitting failures in the same way: the {diagnostics.get('n_real_heldout')} real failures
        held out from that fold (what an ideal generator would reach; they are also the non-members of the
        membership-inference statistic mia_auc in the CSV) and the duplication control, copies of fitting failures
        (pure copying). From every generator 2,000 rows were sampled; the duplication control comes from a run with
        300 copies, as does the memorisation reference of the file (reference_duplicates_memorisation). The statistics
        are defined in experiments/common/synthetic_data_diagnostics.py; no test rows are involved. Source:
        results/rq2_survey_of_methods/synthetic_data_diagnostics.json.""",
        column_notes={
            "Detectability AUC": "out-of-fold AUC of a LightGBM trained to tell the synthetic rows from the fitting "
                                 "failures, missing values as they are (CSV: detect_auc; 0.5 = indistinguishable).",
            "same, missing values filled": "the same after filling the missing cells of both sets with the medians of "
                                           "the fitting failures (CSV: detect_auc_nomask; the held-out reference "
                                           "row has no such value, and its value with missing values as they are "
                                           "is repeated, as in the thesis).",
            "Relative NN distance": "median distance of a synthetic row to its nearest fitting failure, divided by the "
                                    "fitting failures' median leave-one-out nearest-neighbour distance (CSV: dcr).",
            "Share closer": "share of synthetic rows closer to a fitting failure than the 5th percentile of the "
                            "fitting failures' leave-one-out nearest-neighbour distances (CSV: frac_closer).",
            "Marginal W1": "mean over the attributes of the 1-Wasserstein distance between the fitting failures' and "
                           "the synthetic values, min-max scaled (CSV: colwise_w1).",
            "CSV only": "mia_auc, near_copy_frac and dcr_fit_vs_heldout (memorisation statistics, where computed).",
        })


# ---------------------------------------------------------------------------------------------- count sweep
def count_sweep_rows() -> pd.DataFrame:
    """One row per (generator, classifier, seed, rows added) of synthetic_row_count_sweep.jsonl.

    OOF columns: the generator is refitted inside each training fold, the count's OOF cost at its own OOF threshold;
    duplication_control_* is the duplication control, the same count of copied real failures instead of generated
    rows. Test columns: one generator fitted on all training failures, one classifier per count on all training rows,
    scored on the test set at its test-optimal threshold (the oracle, a diagnostic of how much selecting the count on
    the test set could buy) and at the count's OOF threshold."""
    rows = []
    for record in read_jsonl(SYNTHETIC_ROW_COUNT_SWEEP_FILE):
        counts = [0] + record["counts"]
        test_oracles = {count: record["test"][str(count)]["oracle_minimum_cost"] for count in counts}
        lowest_test_oracle = min(test_oracles[count] for count in record["counts"])
        for count in counts:
            out_of_fold = record.get("out_of_fold", {}).get(str(count), {})
            test = record["test"][str(count)]
            rows.append({
                "generator": record["generator"], "classifier": record.get("classifier"), "seed": record["seed"],
                "rows_added": count,
                "oof_cost": out_of_fold.get("cost"), "oof_fn": out_of_fold.get("fn"), "oof_fp": out_of_fold.get("fp"),
                "oof_threshold": out_of_fold.get("threshold"), "oof_cost_at_0_5": out_of_fold.get("cost_at_0.5"),
                "duplication_control_oof_cost": out_of_fold.get("duplication_control_cost"),
                "duplication_control_oof_fn": out_of_fold.get("duplication_control_fn"),
                "duplication_control_oof_fp": out_of_fold.get("duplication_control_fp"),
                "test_oracle_minimum_cost": test["oracle_minimum_cost"], "test_oracle_fn": test.get("oracle_fn"),
                "test_oracle_fp": test.get("oracle_fp"),
                "test_cost_at_oof_threshold": test.get("cost_at_out_of_fold_threshold"),
                "test_cost_at_0_5": test.get("cost_at_0.5"),
                "selected_by_oof_cost": count == record.get("out_of_fold_selected_count"),
                "lowest_test_oracle_among_counts": count > 0 and test_oracles[count] == lowest_test_oracle,
            })
    return pd.DataFrame(rows)


def write_count_sweep_table(sweep: pd.DataFrame) -> None:
    sections = []
    for (generator, classifier, seed), rows in sweep.groupby(["generator", "classifier", "seed"], sort=False):
        readable = pd.DataFrame({
            "Rows added": rows.rows_added.map(thousands),
            "OOF cost": rows.oof_cost.map(thousands),
            "Duplication control": [thousands(value) if count else MISSING for value, count in
                                    zip(rows.duplication_control_oof_cost, rows.rows_added)],
            "Test oracle (full-data generator)": rows.test_oracle_minimum_cost.map(thousands),
        })
        selected = rows[rows.selected_by_oof_cost].rows_added.tolist()
        lowest = rows[rows.lowest_test_oracle_among_counts]
        chosen_count = thousands(selected[0]) if selected else MISSING
        heading = (f"{COUNT_SWEEP_GENERATOR_LABELS.get(generator, generator)}, "
                   f"{COUNT_SWEEP_CLASSIFIER_LABELS.get(classifier, classifier)}, seed {seed}: the OOF cost chooses "
                   f"{chosen_count} added rows; the lowest test oracle over the counts is "
                   f"{thousands(lowest.test_oracle_minimum_cost.iloc[0])} at "
                   f"{thousands(lowest.rows_added.iloc[0])} rows")
        sections.append((heading, readable))
    write_summary_table(
        "synthetic_row_count_sweep", sweep, sections,
        title="RQ1: synthetic-row count sweep of the cGAN of Ranasinghe et al.",
        caption="""
        The number of added cGAN rows varied from 250 to 5,000 as in the notebook attributed to the study of
        Ranasinghe et al.; every count takes the first rows of one generated pool in which the two k-means clusters
        alternate. OOF cost: 60,000 training rows, generator refitted inside each training fold, at the count's own OOF
        threshold (the common evaluation protocol's way to choose a count); duplication control: the same number of
        copied real failures. Test oracle: a generator fitted on all training failures, the lowest test cost over all
        thresholds. Choosing the count by this number is test-set selection, simulated deliberately to measure its
        effect: the minimum over the counts is the best of several test evaluations rather than an independent estimate
        of the cost on new data. Source: results/rq1_cgan_2019_reproduction/synthetic_row_count_sweep.jsonl.""",
        column_notes={
            "OOF cost": "CSV: oof_cost, with oof_fn, oof_fp, oof_threshold and oof_cost_at_0_5.",
            "Duplication control": "OOF cost with copied real failures (CSV: duplication_control_oof_cost, _fn, "
                                   "_fp).",
            "Test oracle (full-data generator)": "CSV: test_oracle_minimum_cost, with test_oracle_fn, test_oracle_fp; "
                                                 "also test_cost_at_oof_threshold and test_cost_at_0_5.",
            "CSV only": "selected_by_oof_cost and lowest_test_oracle_among_counts mark the two selections.",
        })


def plot_count_sweep(sweep: pd.DataFrame) -> None:
    """Left: OOF cost (scaled to the 16,000 rows of the test set) against the number of added rows, generated and copied
    (the duplication control). Right: the test oracle against the count, with the no-augmentation value and the minimum
    and median over the counts."""
    scale = TEST_SET_ROWS / TRAINING_ROWS
    for generator, generator_rows in sweep.groupby("generator", sort=False):
        figure, (oof_axes, test_axes) = plt.subplots(1, 2, figsize=(11, 4.5))
        for seed, rows in generator_rows.groupby("seed"):
            oof_axes.plot(rows.rows_added, rows.oof_cost * scale, "-o", ms=3, lw=1,
                          label=f"generated rows (seed {seed})")
            duplicated = rows[rows.rows_added > 0]
            oof_axes.plot(duplicated.rows_added, duplicated.duplication_control_oof_cost * scale, ":^", ms=3, lw=1,
                          label=f"duplication control (seed {seed})")
            with_rows = rows[rows.rows_added > 0]
            without_rows = rows[rows.rows_added == 0].test_oracle_minimum_cost.iloc[0]
            test_axes.plot(with_rows.rows_added, with_rows.test_oracle_minimum_cost, "-s", ms=3, lw=1, c="#1b4965")
            test_axes.axhline(without_rows, ls="--", c="grey", lw=0.8,
                              label=f"no augmentation: {thousands(without_rows)}")
            test_axes.axhline(with_rows.test_oracle_minimum_cost.min(), ls=":", c="k", lw=0.8,
                              label=f"minimum over counts: {thousands(with_rows.test_oracle_minimum_cost.min())}")
            test_axes.axhline(float(np.median(with_rows.test_oracle_minimum_cost)), ls=":", c="k", lw=0.5,
                              label=f"median over counts: {thousands(np.median(with_rows.test_oracle_minimum_cost))}")
        oof_axes.set_xlabel("added failure rows")
        oof_axes.set_ylabel("OOF cost, scaled to the 16,000-row test set")
        oof_axes.set_title("Out-of-fold estimate")
        oof_axes.legend(fontsize=8, loc="lower right")
        test_axes.set_xlabel("added failure rows")
        test_axes.set_ylabel("test cost at the test-optimal threshold")
        test_axes.set_title("Selection on the test set (diagnostic)")
        # below the plotted values, so that the legend covers none of them
        low, high = test_axes.get_ylim()
        test_axes.set_ylim(low - 0.45 * (high - low), high)
        test_axes.legend(fontsize=8, loc="lower right", ncol=1)
        save_figure(figure, f"synthetic_row_count_sweep_{generator}")


def main() -> None:
    print("Supporting results")
    write_data_audit_table()
    plot_data_overview()
    write_quick_baseline_table()
    write_hyperparameter_search_table()
    plot_hyperparameter_search_history()
    write_threshold_rule_table()
    write_tabicl_pilot_table()
    write_synthetic_data_diagnostics_table()
    sweep = count_sweep_rows()
    write_count_sweep_table(sweep)
    plot_count_sweep(sweep)


if __name__ == "__main__":
    main()
