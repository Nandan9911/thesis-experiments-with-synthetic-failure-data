"""RQ1 and RQ2: summary tables and figures of the classifier arms.

An experiment arm is one way of training a failure classifier on the 60,000 training rows: a learner (LightGBM with
the tuned hyperparameters, XGBoost, a random forest, ...), a preprocessing step and, for most arms, 2,000 synthetic
failure rows from some generator, added inside each training fold only. Every arm is cross-validated on stratified
folds (5, or 3 for the four arms on the preprocessing pipeline of Beikmohammadi et al. and their LightGBM baseline),
mostly with the seeds 11, 29 and 47 (the main arms also with 71 and 101), and the official 16,000-row test set is scored
once with the averaged fold models at the threshold that minimises the out-of-fold (OOF) cost. The protocol and the cost
are described in experiments/common/cost_and_evaluation.py (a cost scaled to the 16,000 rows of the test set is the OOF
cost x 16,000 / 60,000), the post-hoc diagnostics in load_results.py.

RQ1 reproduces two studies. For the study of Ranasinghe et al., the question is whether 2,000 rows from its conditional
GAN (cGAN) still lower the cost when every step is fitted inside the training folds. Each cGAN arm is compared fold by
fold with its matched baseline, the same learner without synthetic rows; a duplication control (2,000 copies of real
failures: the same extra weight on the failure class, without any new information) separates the effect of the new rows
from that of re-weighting. For the study of Beikmohammadi et al., the tree learner is also trained on the authors'
preprocessing (imputation, SVM-SMOTE oversampling, RENN cleaning, see summarise_transformer_2026.py), with three folds,
and compared with a 3-fold LightGBM. RQ2 surveys other generators and other uses of the data; every survey arm is
compared with the tuned LightGBM without synthetic rows, with Holm's adjustment over the survey and the two-tier
criterion of paired_statistics.evidence_tier.

Run from the repository root (also part of `python -m analysis.run_all_summaries`):

    python -m analysis.summarise_classifier_arms

Reads results/classifier_arms/arm_records.jsonl, the saved probabilities in
results/classifier_arms/predicted_probabilities/, the class labels of the two data files, and the test
probabilities of the models released by Beikmohammadi et al. (for the frontier figure). Writes to
results/summary_tables/:

  classifier_arms_per_seed            every arm and seed: OOF and test results and all diagnostics (appendix table)
  classifier_arms_main_results        medians over seeds of the main arms (the main results table)
  classifier_arms_decision_rules      the seed-11 test predictions of each main arm under five threshold rules
  classifier_arms_paired_comparisons  paired fold-level comparison of each main arm with its matched baseline
  rq2_survey_against_tuned_lightgbm   the RQ2 survey: paired differences, Holm-adjusted p-values, evidence tiers
  classifier_arms_runtimes            cross-validation minutes per seed and what dominates them

and to results/figures/: classifier_arms_oof_cost_against_threshold, classifier_arms_test_false_alarm_frontier,
classifier_arms_paired_differences and rq2_survey_paired_differences.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.load_results import (BAYES_CUTOFF, DEFAULT_CUTOFF, TRAINING_ROWS, build_arm_table,
                                   load_arm_probabilities, load_shipped_model_test_probabilities, test_labels,
                                   training_labels)
from analysis.paired_statistics import (compare_with_baseline, evidence_tier, holm_adjusted_p_values,
                                        paired_fold_differences)
from analysis.write_outputs import (MISSING, bracketed_interval, cost_with_errors, fixed_decimals, is_missing,
                                    median_with_range, save_figure, scientific, signed_thousands, thousands,
                                    write_summary_table)
from experiments.common.cost_and_evaluation import cost_curve
from experiments.common.data_loading import TEST_SET_ROWS

# The main arms of the results chapter, in the order of its tables, with the labels the thesis uses. The Forest-Flow
# arm with the paper's settings did not complete (no records); it is listed so that it would appear if it were run.
# lightgbm_tuned_3_folds and the lightgbm_on_2026_preprocessing_* arms are not in experiments/experiment_arms.py: the
# first is lightgbm_tuned run by run_experiment_arm.py with --folds 3, the others are written by
# rq1_transformer_2026_reproduction/run_tree_models_on_2026_preprocessing.py.
ARM_LABELS = {
    "lightgbm_tuned": "LightGBM (tuned), no augmentation",
    "xgboost_fixed_settings": "XGBoost (fixed settings), no augmentation",
    "random_forest_fixed_settings": "Random forest, no augmentation",
    "sklearn_gradient_boosting": "sklearn GBM, no augmentation",
    "lightgbm_tuned_3_folds": "LightGBM (tuned), no augmentation, 3 folds",
    "lightgbm_with_conditional_gan_2019_rows": "cGAN +2,000 rows, LightGBM",
    "random_forest_fixed_settings_with_conditional_gan_2019_rows": "cGAN +2,000 rows, random forest",
    "sklearn_gradient_boosting_with_conditional_gan_2019_rows": "cGAN +2,000 rows, sklearn GBM",
    "random_forest_with_conditional_gan_2019_literal_notebook_rows": "cGAN with the notebook's settings, random forest",
    "lightgbm_with_duplicated_real_failures": "2,000 duplicated real failures, LightGBM",
    "lightgbm_with_forest_flow_rows_reduced_profile": "Forest-Flow +2,000 rows, LightGBM",
    "lightgbm_with_forest_flow_rows_paper_profile": "Forest-Flow (paper settings) +2,000 rows, LightGBM",
    "lightgbm_with_tabebm_rows": "TabEBM +2,000 rows, LightGBM (GPU)",
    "lightgbm_tuned_with_missingness_indicators": "LightGBM (tuned) + missingness indicators",
    "ensemble_with_missingness_indicators": "Ensemble (LightGBM + XGBoost + CatBoost) + missingness indicators",
    "ensemble_without_missingness_indicators": "Ensemble (LightGBM + XGBoost + CatBoost), no indicators",
    "tabicl_bagged_contexts": "TabICLv2, bagged contexts, no tuning",
    "lightgbm_on_2026_preprocessing_full_pipeline":
        "Preprocessing of Beikmohammadi et al. (SVM-SMOTE and RENN), LightGBM",
    "lightgbm_on_2026_preprocessing_renn_only": "Preprocessing of Beikmohammadi et al. (RENN only), LightGBM",
    "lightgbm_on_2026_preprocessing_without_renn": "Preprocessing of Beikmohammadi et al. (SVM-SMOTE only), LightGBM",
    "lightgbm_on_2026_preprocessing_imputation_only":
        "Preprocessing of Beikmohammadi et al. (imputation only), LightGBM",
}

# The matched baseline of an arm: the same learner, seeds and folds without the arm's change. Arms not listed here
# are LightGBM arms and are matched with the tuned LightGBM.
DEFAULT_BASELINE = "lightgbm_tuned"
MATCHED_BASELINE = {
    "random_forest_fixed_settings_with_conditional_gan_2019_rows": "random_forest_fixed_settings",
    "random_forest_with_conditional_gan_2019_literal_notebook_rows": "random_forest_fixed_settings",
    "sklearn_gradient_boosting_with_conditional_gan_2019_rows": "sklearn_gradient_boosting",
    "lightgbm_on_2026_preprocessing_full_pipeline": "lightgbm_tuned_3_folds",
    "lightgbm_on_2026_preprocessing_renn_only": "lightgbm_tuned_3_folds",
    "lightgbm_on_2026_preprocessing_without_renn": "lightgbm_tuned_3_folds",
    "lightgbm_on_2026_preprocessing_imputation_only": "lightgbm_tuned_3_folds",
}
BASELINE_ARMS = {"lightgbm_tuned", "random_forest_fixed_settings", "sklearn_gradient_boosting",
                 "lightgbm_tuned_3_folds"}

# The RQ2 survey: (arm, label, family). Three arms have no records and are skipped: GReaT failed (the language
# model's rows were cut off), and of the three Forest-Flow profiles only the reduced one completed.
SURVEY_BASELINE = "lightgbm_tuned"
DUPLICATION_CONTROL = "lightgbm_with_duplicated_real_failures"
SURVEY_ARMS = [
    (DUPLICATION_CONTROL, "Duplication control: 2,000 copied real failures", "control"),
    ("lightgbm_with_conditional_gan_2019_rows", "cGAN of Ranasinghe et al., +2,000 rows", "adversarial"),
    ("lightgbm_with_smote_rows", "SMOTE, +2,000 rows", "classical"),
    ("lightgbm_with_borderline_smote_rows", "Borderline-SMOTE, +2,000 rows", "classical"),
    ("lightgbm_with_adasyn_rows", "ADASYN, +2,000 rows", "classical"),
    ("lightgbm_with_svm_smote_rows", "SVM-SMOTE, +2,000 rows", "classical"),
    ("lightgbm_with_gaussian_copula_rows", "Gaussian copula, +2,000 rows", "statistical"),
    ("lightgbm_with_tvae_rows", "TVAE, +2,000 rows", "deep"),
    ("lightgbm_with_ctgan_rows", "CTGAN, +2,000 rows", "deep"),
    ("lightgbm_with_ctab_gan_plus_rows", "CTAB-GAN+, +2,000 rows", "deep"),
    ("lightgbm_with_tabddpm_rows", "TabDDPM, +2,000 rows", "deep"),
    ("lightgbm_with_great_language_model_rows", "GReaT (Qwen2.5-0.5B), +2,000 rows", "language model"),
    ("lightgbm_with_forest_flow_rows_reduced_profile", "Forest-Flow (reduced profile), +2,000 rows", "flow"),
    ("lightgbm_with_forest_flow_rows_middle_profile", "Forest-Flow (middle profile), +2,000 rows", "flow"),
    ("lightgbm_with_forest_flow_rows_paper_profile", "Forest-Flow (paper settings), +2,000 rows", "flow"),
    ("lightgbm_with_tabebm_rows", "TabEBM, +2,000 rows", "energy"),
    ("tabicl_bagged_contexts", "TabICLv2, bagged contexts (no synthetic rows)", "foundation model"),
    ("lightgbm_tuned_with_missingness_indicators", "LightGBM + missingness indicators", "constructive"),
    ("ensemble_without_missingness_indicators", "Ensemble LightGBM+XGBoost+CatBoost", "constructive"),
    ("ensemble_with_missingness_indicators", "Ensemble + missingness indicators", "constructive"),
    ("ensemble_with_missingness_indicators_and_tabicl", "Ensemble + indicators + TabICLv2 member", "constructive"),
    ("xgboost_fixed_settings", "XGBoost alone", "learner"),
]
SURVEY_LABELS = {arm: label for arm, label, _ in SURVEY_ARMS}

# Sections of the per-seed table (the appendix table groups the arms by research question).
PER_SEED_GROUPS = [
    ("Baselines and other learners", ["lightgbm_tuned", "lightgbm_tuned_3_folds", "xgboost_fixed_settings",
                                      "random_forest_fixed_settings", "sklearn_gradient_boosting"]),
    ("RQ1: the cGAN of Ranasinghe et al. and its controls", ["lightgbm_with_conditional_gan_2019_rows",
                                             "lightgbm_with_duplicated_real_failures",
                                             "random_forest_fixed_settings_with_conditional_gan_2019_rows",
                                             "random_forest_with_conditional_gan_2019_literal_notebook_rows",
                                             "sklearn_gradient_boosting_with_conditional_gan_2019_rows"]),
    ("RQ1: trees on the preprocessing pipeline of Beikmohammadi et al.",
                                             ["lightgbm_on_2026_preprocessing_full_pipeline",
                                              "lightgbm_on_2026_preprocessing_without_renn",
                                              "lightgbm_on_2026_preprocessing_renn_only",
                                              "lightgbm_on_2026_preprocessing_imputation_only"]),
    ("RQ2: classical oversampling (+2,000 rows, LightGBM)", ["lightgbm_with_smote_rows",
                                                             "lightgbm_with_borderline_smote_rows",
                                                             "lightgbm_with_adasyn_rows",
                                                             "lightgbm_with_svm_smote_rows"]),
    ("RQ2: statistical and deep generators (+2,000 rows, LightGBM)", ["lightgbm_with_gaussian_copula_rows",
                                                                      "lightgbm_with_tvae_rows",
                                                                      "lightgbm_with_ctgan_rows",
                                                                      "lightgbm_with_ctab_gan_plus_rows",
                                                                      "lightgbm_with_tabddpm_rows",
                                                                      "lightgbm_with_forest_flow_rows_reduced_profile",
                                                                      "lightgbm_with_forest_flow_rows_paper_profile",
                                                                      "lightgbm_with_tabebm_rows"]),
    ("RQ2: other learners and ensembles (no synthetic rows)", ["tabicl_bagged_contexts",
                                                               "lightgbm_tuned_with_missingness_indicators",
                                                               "ensemble_without_missingness_indicators",
                                                               "ensemble_with_missingness_indicators",
                                                               "ensemble_with_missingness_indicators_and_tabicl"]),
]

# What dominates the cross-validation time of each main arm (written down when the runs were made).
DOMINANT_RUNTIME_COST = {
    "lightgbm_tuned": "LightGBM fits",
    "xgboost_fixed_settings": "XGBoost fits",
    "random_forest_fixed_settings": "random forest fits",
    "sklearn_gradient_boosting": "sklearn GBM fits",
    "lightgbm_tuned_3_folds": "LightGBM fits",
    "lightgbm_with_conditional_gan_2019_rows": "cGAN training (30k steps per fold)",
    "random_forest_fixed_settings_with_conditional_gan_2019_rows": "cGAN + random forest",
    "sklearn_gradient_boosting_with_conditional_gan_2019_rows": "sklearn GBM fits",
    "random_forest_with_conditional_gan_2019_literal_notebook_rows": "cGAN + random forest",
    "lightgbm_with_duplicated_real_failures": "LightGBM fits",
    "lightgbm_with_forest_flow_rows_reduced_profile": "Forest-Flow fits (1,690 XGBoost models per fold)",
    "lightgbm_with_forest_flow_rows_paper_profile": "Forest-Flow fits (paper settings, 8,450 models per fold)",
    "lightgbm_tuned_with_missingness_indicators": "LightGBM fits",
    "ensemble_with_missingness_indicators": "CatBoost, XGBoost and LightGBM fits",
    "ensemble_without_missingness_indicators": "CatBoost, XGBoost and LightGBM fits",
    "tabicl_bagged_contexts": "TabICL inference",
    "lightgbm_on_2026_preprocessing_full_pipeline": "imputer, SVM-SMOTE, RENN",
    "lightgbm_on_2026_preprocessing_renn_only": "imputer, RENN",
    "lightgbm_on_2026_preprocessing_without_renn": "imputer, SVM-SMOTE",
    "lightgbm_on_2026_preprocessing_imputation_only": "imputer",
}

# Published operating points on the official test set, for the frontier figure, as label: (missed failures, false
# alarms): the cGAN of Ranasinghe et al. with gradient boosting and with a random forest, the winner of the IDA 2016
# challenge, and the cost-sensitive Transformer of Beikmohammadi et al. (means over its five runs).
PUBLISHED_OPERATING_POINTS = {
    "Ranasinghe et al.,\ncGAN + gradient boosting (6, 255)": (6, 255),
    "Ranasinghe et al.,\ncGAN + random forest (4, 405)": (4, 405),
    "IDA 2016 challenge winner (9, 542)": (9, 542),
    "Beikmohammadi et al.,\ncost-sensitive Transformer (6.8, 4)": (6.8, 4)}
# Where each label is written, in data coordinates (missed failures, false alarms): in the empty band between the
# released models' curves and the tree curves, with leader lines that do not cross.
PUBLISHED_LABEL_POSITIONS = {
    "Ranasinghe et al.,\ncGAN + gradient boosting (6, 255)": (10, 140),
    "Ranasinghe et al.,\ncGAN + random forest (4, 405)": (10, 46),
    "IDA 2016 challenge winner (9, 542)": (10, 1800),
    "Beikmohammadi et al.,\ncost-sensitive Transformer (6.8, 4)": (10, 14.5)}
FIGURE_SEED = 11
COST_CURVE_FIGURE_ARMS = ["lightgbm_tuned", "lightgbm_with_conditional_gan_2019_rows",
                          "lightgbm_with_duplicated_real_failures", "lightgbm_with_forest_flow_rows_reduced_profile",
                          "ensemble_with_missingness_indicators", "random_forest_fixed_settings",
                          "lightgbm_on_2026_preprocessing_full_pipeline",
                          "lightgbm_on_2026_preprocessing_imputation_only"]
FRONTIER_FIGURE_ARMS = COST_CURVE_FIGURE_ARMS[:6] + ["random_forest_fixed_settings_with_conditional_gan_2019_rows",
                                                     "lightgbm_on_2026_preprocessing_full_pipeline",
                                                     "lightgbm_on_2026_preprocessing_imputation_only"]
TIER_COLOURS = {"demonstrated gain": "#1b7837", "positive sign": "#5fa777", "no evidence": "#1b4965",
                "reference": "#7f7f7f"}


def arm_label(arm: str) -> str:
    """The thesis label of an arm: the main-table label if it has one, otherwise the survey label."""
    return ARM_LABELS.get(arm) or SURVEY_LABELS.get(arm) or arm


def arms_present(arm_table: pd.DataFrame, arms) -> list[str]:
    """The given arms that have at least one record, in the given order."""
    recorded = set(arm_table.arm)
    return [arm for arm in arms if arm in recorded]


def _summary_counts_note(arm_table: pd.DataFrame) -> str:
    return f"{len(arm_table)} records of {arm_table.arm.nunique()} arms"


# ---------------------------------------------------------------------------------------------- per seed
def write_per_seed_table(arm_table: pd.DataFrame) -> None:
    """Every record with all diagnostics (CSV), and the appendix table of per-seed results (Markdown).

    The CSV has every column of load_results.build_arm_table plus label and group (the section of the appendix
    table); the columns are explained there and in the Markdown file."""
    group_of = {arm: heading for heading, arms in PER_SEED_GROUPS for arm in arms}
    order_of = {arm: position for position, arm in enumerate(arm for _, arms in PER_SEED_GROUPS for arm in arms)}
    ungrouped = sorted(set(arm_table.arm) - set(group_of))
    if ungrouped:
        print("  per-seed table: arms without a section are listed last:", ungrouped)
    table = arm_table.copy()
    table.insert(1, "label", [arm_label(arm) for arm in table.arm])
    table.insert(2, "group", [group_of.get(arm, "Other arms") for arm in table.arm])
    table["arm_position"] = [order_of.get(arm, len(order_of)) for arm in table.arm]
    table = table.sort_values(["arm_position", "arm", "seed"]).drop(columns="arm_position")
    sections = []
    for heading in [heading for heading, _ in PER_SEED_GROUPS] + ["Other arms"]:
        rows = table[table.group == heading]
        if len(rows):
            sections.append((heading, _readable_per_seed_rows(rows)))
    write_summary_table(
        "classifier_arms_per_seed", table, sections,
        title="Per-seed results of every classifier arm",
        caption=f"""
        One row per arm and seed ({_summary_counts_note(arm_table)}) from
        results/classifier_arms/arm_records.jsonl; the plateau rule, the fixed cut-offs and the false-alarm counts
        are computed from the saved probabilities in results/classifier_arms/predicted_probabilities/. The OOF cost
        is over the 60,000 training rows and the threshold is chosen on it; the test costs are on the 16,000-row
        official test set. Oracle and FP at FN<=6 use the test labels and are diagnostics only. The CSV has further
        columns: the OOF cost scaled to the 16,000 rows of the test set, the fold costs, the costs at the 0.5 and
        10/510 cut-offs, the plateau threshold, the fewest false alarms at <= 9 missed failures and the
        cross-validation minutes.""",
        column_notes={
            "Arm": "the arm's label (CSV: arm is the name in arm_records.jsonl, label and group the display names).",
            "Seed": "random seed of the folds, the models and the generators (CSV: seed).",
            "OOF cost": "cost of the out-of-fold predictions of the 60,000 training rows at the OOF threshold "
                        "(CSV: oof_cost, with oof_fn and oof_fp).",
            "Threshold": "the threshold that minimises the OOF cost (CSV: oof_threshold).",
            "Test cost, FN, FP": "cost, missed failures and false alarms on the test set with the fold models' "
                                 "average probability at the OOF threshold (CSV: test_cost, test_fn, test_fp).",
            "Test cost 95 % interval": "percentile bootstrap interval of the test cost with the predictions held "
                                       "fixed (CSV: test_cost_interval_low, test_cost_interval_high).",
            "Full refit": "test cost of one model refitted on all training rows, at the same threshold; only for the "
                          "arms and seeds where it was run (CSV: full_refit_test_cost, _fn, _fp).",
            "Plateau rule": "test cost at the plateau threshold (CSV: test_cost_at_plateau_threshold, "
                            "plateau_threshold).",
            "Oracle": "lowest test cost over all thresholds (CSV: test_oracle_minimum_cost).",
            "FP at FN<=6": "fewest test false alarms with at most 6 missed failures (CSV: test_min_fp_at_most_6_fn).",
        })


def _readable_per_seed_rows(rows: pd.DataFrame) -> pd.DataFrame:
    readable = pd.DataFrame({
        "Arm": rows.label,
        "Seed": rows.seed.astype(int).astype(str),
        "OOF cost": rows.oof_cost.map(thousands),
        "Threshold": rows.oof_threshold.map(scientific),
        "Test cost": rows.test_cost.map(thousands),
        "FN": rows.test_fn.map(thousands),
        "FP": rows.test_fp.map(thousands),
        "Test cost 95 % interval": [bracketed_interval(low, high) for low, high in
                                    zip(rows.test_cost_interval_low, rows.test_cost_interval_high)],
        "Full refit": rows.full_refit_test_cost.map(thousands),
        "Plateau rule": rows.test_cost_at_plateau_threshold.map(thousands),
        "Oracle": rows.test_oracle_minimum_cost.map(thousands),
        "FP at FN<=6": rows.test_min_fp_at_most_6_fn.map(thousands),
    })
    return readable


# ---------------------------------------------------------------------------------------------- medians over seeds
def write_main_results_table(arm_table: pd.DataFrame) -> pd.DataFrame:
    """Medians over seeds of the main arms (the main results table)."""
    rows = []
    for arm in arms_present(arm_table, ARM_LABELS):
        records = arm_table[arm_table.arm == arm]
        rows.append({
            "arm": arm, "label": ARM_LABELS[arm], "seeds": records.seed.nunique(), "folds": int(records.folds.iloc[0]),
            "oof_cost_scaled_to_test_set_size_median": records.oof_cost_scaled_to_test_set_size.median(),
            "test_cost_median": records.test_cost.median(), "test_cost_min": records.test_cost.min(),
            "test_cost_max": records.test_cost.max(),
            "test_fn_median": records.test_fn.median(), "test_fp_median": records.test_fp.median(),
            "test_oracle_minimum_cost_median": records.test_oracle_minimum_cost.median(),
            "test_min_fp_at_most_6_fn_median": records.test_min_fp_at_most_6_fn.median(),
        })
    table = pd.DataFrame(rows)
    readable = pd.DataFrame({
        "Arm": table.label,
        "Seeds": table.seeds.astype(str),
        "Folds": table.folds.astype(str),
        "OOF cost": table.oof_cost_scaled_to_test_set_size_median.map(thousands),
        "Test cost [range]": [median_with_range(median, low, high) for median, low, high in
                              zip(table.test_cost_median, table.test_cost_min, table.test_cost_max)],
        "Test FN / FP": [f"{thousands(fn)} / {thousands(fp)}" for fn, fp in
                         zip(table.test_fn_median, table.test_fp_median)],
        "Oracle": table.test_oracle_minimum_cost_median.map(thousands),
        "FP at FN<=6": table.test_min_fp_at_most_6_fn_median.map(thousands),
    })
    write_summary_table(
        "classifier_arms_main_results", table, readable,
        title="Main results: medians over seeds",
        caption="""
        Medians over the seeds of each main arm, from classifier_arms_per_seed (arm_records.jsonl and the saved
        probabilities). The OOF cost is scaled to the 16,000 rows of the test set, so that it can be read next to the
        test cost. Oracle and FP at FN<=6 use the test labels and are diagnostics only.""",
        column_notes={
            "Seeds, Folds": "number of seeds and of cross-validation folds (CSV: seeds, folds).",
            "OOF cost": "median OOF cost x 16,000 / 60,000 (CSV: oof_cost_scaled_to_test_set_size_median).",
            "Test cost [range]": "median, smallest and largest test cost at the OOF threshold "
                                 "(CSV: test_cost_median, test_cost_min, test_cost_max).",
            "Test FN / FP": "median missed failures and median false alarms on the test set; the two medians may "
                            "come from different seeds (CSV: test_fn_median, test_fp_median).",
            "Oracle": "median of the lowest test cost over all thresholds (CSV: test_oracle_minimum_cost_median).",
            "FP at FN<=6": "median of the fewest test false alarms with at most 6 missed failures "
                           "(CSV: test_min_fp_at_most_6_fn_median).",
        })
    return table


# ---------------------------------------------------------------------------------------------- decision rules
def write_decision_rule_table(arm_table: pd.DataFrame) -> None:
    """The same test predictions (lowest seed of each arm, which is 11 for every main arm) under five rules."""
    rows = []
    for arm in arms_present(arm_table, ARM_LABELS):
        record = arm_table[arm_table.arm == arm].sort_values("seed").iloc[0]
        rows.append({
            "arm": arm, "label": ARM_LABELS[arm], "seed": int(record.seed),
            "test_cost_at_0_5": record.test_cost_at_0_5, "test_fn_at_0_5": record.test_fn_at_0_5,
            "test_fp_at_0_5": record.test_fp_at_0_5,
            "test_cost_at_bayes_cutoff": record.test_cost_at_bayes_cutoff,
            "test_fn_at_bayes_cutoff": record.test_fn_at_bayes_cutoff,
            "test_fp_at_bayes_cutoff": record.test_fp_at_bayes_cutoff,
            "test_cost_at_oof_threshold": record.test_cost, "test_fn_at_oof_threshold": record.test_fn,
            "test_fp_at_oof_threshold": record.test_fp,
            "test_cost_at_plateau_threshold": record.test_cost_at_plateau_threshold,
            "test_fn_at_plateau_threshold": record.test_fn_at_plateau_threshold,
            "test_fp_at_plateau_threshold": record.test_fp_at_plateau_threshold,
            "test_oracle_minimum_cost": record.test_oracle_minimum_cost,
        })
    table = pd.DataFrame(rows)

    def cells(rule: str) -> list[str]:
        return [cost_with_errors(cost, fn, fp) for cost, fn, fp in
                zip(table[f"test_cost_at_{rule}"], table[f"test_fn_at_{rule}"], table[f"test_fp_at_{rule}"])]

    readable = pd.DataFrame({
        "Arm": table.label, "Seed": table.seed.astype(str),
        "0.5": cells("0_5"), "0.0196": cells("bayes_cutoff"),
        "OOF threshold": cells("oof_threshold"), "OOF plateau": cells("plateau_threshold"),
        "Oracle": table.test_oracle_minimum_cost.map(thousands),
    })
    write_summary_table(
        "classifier_arms_decision_rules", table, readable,
        title="The same test predictions under five decision rules",
        caption=f"""
        Test cost (missed failures / false alarms) of one set of test predictions per arm (seed 11) under five
        threshold rules. Only the OOF threshold and the plateau rule are chosen on training data; 0.5 and
        {BAYES_CUTOFF:.4f} are fixed in advance, and the oracle is chosen with the test labels (a diagnostic).
        Source: classifier_arms_per_seed.""",
        column_notes={
            "0.5": "the default cut-off (CSV: test_cost_at_0_5, test_fn_at_0_5, test_fp_at_0_5).",
            "0.0196": "the Bayes cut-off 10/510, cost-optimal for calibrated probabilities "
                      "(CSV: test_*_at_bayes_cutoff).",
            "OOF threshold": "the protocol's rule, the threshold minimising the OOF cost "
                             "(CSV: test_*_at_oof_threshold).",
            "OOF plateau": "geometric midpoint of the thresholds within 500 of the minimum OOF cost "
                           "(CSV: test_*_at_plateau_threshold).",
            "Oracle": "lowest test cost over all thresholds (CSV: test_oracle_minimum_cost).",
        })


# ---------------------------------------------------------------------------------------------- RQ1 paired tests
def paired_comparisons(arm_table: pd.DataFrame) -> pd.DataFrame:
    """compare_with_baseline for every main arm that is not itself a baseline."""
    rows = []
    for arm in arms_present(arm_table, ARM_LABELS):
        if arm in BASELINE_ARMS:
            continue
        baseline = MATCHED_BASELINE.get(arm, DEFAULT_BASELINE)
        comparison = compare_with_baseline(arm_table, arm, baseline)
        if comparison is None:
            continue
        rows.append({
            "arm": arm, "label": ARM_LABELS[arm], "baseline": baseline, "baseline_label": ARM_LABELS[baseline],
            "seeds": len(comparison["seeds"]), "folds": comparison["folds"], "pairs": comparison["pairs"],
            **{key: comparison[key] for key in (
                "mean_difference_fold", "median_difference_fold", "standard_deviation_fold", "mean_difference",
                "interval_low", "interval_high", "t_corrected", "p_corrected_t", "p_wilcoxon", "wins", "losses",
                "ties")},
        })
    return pd.DataFrame(rows)


def write_paired_comparison_table(comparisons: pd.DataFrame) -> None:
    readable = pd.DataFrame({
        "Arm": comparisons.label,
        "Baseline": comparisons.baseline_label,
        "Pairs": comparisons.pairs.astype(str),
        "Mean difference": comparisons.mean_difference.map(signed_thousands),
        "95 % interval": [bracketed_interval(low, high) for low, high in
                          zip(comparisons.interval_low, comparisons.interval_high)],
        "Corrected t": [MISSING if is_missing(value) else f"{value:+.2f}" for value in comparisons.t_corrected],
        "p (corrected t)": comparisons.p_corrected_t.map(fixed_decimals),
        "p (Wilcoxon)": comparisons.p_wilcoxon.map(fixed_decimals),
        "Wins / losses": [f"{wins} / {losses}" for wins, losses in zip(comparisons.wins, comparisons.losses)],
    })
    write_summary_table(
        "classifier_arms_paired_comparisons", comparisons, readable, left_aligned_columns=2,
        title="Paired fold-level comparisons with the matched baseline",
        caption="""
        Each main arm against the same learner without its change, on the same seeds and folds (the arms
        on the preprocessing pipeline of Beikmohammadi et al. against the 3-fold LightGBM). The differences are arm
        minus baseline, per validation fold at each arm's OOF threshold, pooled over the seeds and rescaled to the
        16,000-row test set; negative values favour the arm. The interval and the p-value come from the Nadeau-Bengio
        corrected resampled t-test; analysis/paired_statistics.py explains the method and its assumptions. Source: the
        fold costs in arm_records.jsonl.""",
        column_notes={
            "Baseline": "the matched baseline (CSV: baseline, baseline_label).",
            "Pairs": "number J of paired fold differences, seeds x folds (CSV: pairs, seeds, folds).",
            "Mean difference": "mean paired difference scaled to the 16,000 rows of the test set (CSV: "
                               "mean_difference; in fold-cost units mean_difference_fold, with "
                               "median_difference_fold and standard_deviation_fold).",
            "95 % interval": "interval of the corrected t-test scaled to the 16,000 rows of the test set (CSV: "
                             "interval_low, interval_high).",
            "Corrected t": "mean / corrected standard error (CSV: t_corrected).",
            "p (corrected t)": "two-sided p-value, J - 1 degrees of freedom (CSV: p_corrected_t).",
            "p (Wilcoxon)": "two-sided Wilcoxon signed-rank p-value, not corrected for the overlap of the training "
                            "sets (CSV: p_wilcoxon).",
            "Wins / losses": "folds in which the arm was cheaper / dearer than the baseline (CSV: wins, losses, ties).",
        })


# ---------------------------------------------------------------------------------------------- RQ2 survey
def survey_table(arm_table: pd.DataFrame) -> pd.DataFrame:
    """The RQ2 survey rows: paired comparison with the tuned LightGBM, Holm-adjusted p-values and the tier.

    All columns of a row are computed over the seeds that the arm shares with the baseline."""
    baseline_rows = arm_table[arm_table.arm == SURVEY_BASELINE]
    rows = []
    for arm, label, family in SURVEY_ARMS:
        comparison = compare_with_baseline(arm_table, arm, SURVEY_BASELINE)
        if comparison is None:
            continue
        seeds = comparison["seeds"]
        arm_rows = arm_table[(arm_table.arm == arm) & arm_table.seed.isin(seeds)]
        matched_baseline_rows = baseline_rows[baseline_rows.seed.isin(seeds)]
        rows.append({
            "arm": arm, "label": label, "family": family, "seeds": len(seeds), "pairs": comparison["pairs"],
            "oof_cost_scaled_to_test_set_size_median": arm_rows.oof_cost_scaled_to_test_set_size.median(),
            "test_cost_median": arm_rows.test_cost.median(), "test_cost_min": arm_rows.test_cost.min(),
            "test_cost_max": arm_rows.test_cost.max(),
            "baseline_test_cost_median": matched_baseline_rows.test_cost.median(),
            "test_min_fp_at_most_6_fn_median": arm_rows.test_min_fp_at_most_6_fn.median(),
            "baseline_test_min_fp_at_most_6_fn_median": matched_baseline_rows.test_min_fp_at_most_6_fn.median(),
            "test_oracle_minimum_cost_median": arm_rows.test_oracle_minimum_cost.median(),
            **{key: comparison[key] for key in ("mean_difference", "interval_low", "interval_high", "p_corrected_t",
                                                "p_wilcoxon", "wins", "losses", "ties", "all_seeds_favour_arm")},
        })
    table = pd.DataFrame(rows)
    is_candidate = table.arm != DUPLICATION_CONTROL
    table["p_holm"] = np.nan
    table.loc[is_candidate, "p_holm"] = holm_adjusted_p_values(table.loc[is_candidate, "p_corrected_t"].values)
    control = table[table.arm == DUPLICATION_CONTROL]
    control_mean_difference = float(control.mean_difference.iloc[0]) if len(control) else 0.0
    table["tier"] = [evidence_tier(row, control_mean_difference) if row.arm != DUPLICATION_CONTROL else "reference"
                     for _, row in table.iterrows()]
    return table


def _survey_baseline_row(arm_table: pd.DataFrame) -> dict:
    baseline_rows = arm_table[arm_table.arm == SURVEY_BASELINE]
    return {"arm": SURVEY_BASELINE, "label": "Baseline: tuned LightGBM", "family": "baseline",
            "seeds": baseline_rows.seed.nunique(),
            "oof_cost_scaled_to_test_set_size_median": baseline_rows.oof_cost_scaled_to_test_set_size.median(),
            "test_cost_median": baseline_rows.test_cost.median(), "test_cost_min": baseline_rows.test_cost.min(),
            "test_cost_max": baseline_rows.test_cost.max(),
            "test_min_fp_at_most_6_fn_median": baseline_rows.test_min_fp_at_most_6_fn.median(),
            "test_oracle_minimum_cost_median": baseline_rows.test_oracle_minimum_cost.median(), "tier": "baseline"}


def write_survey_table(arm_table: pd.DataFrame, survey: pd.DataFrame) -> None:
    full_table = pd.concat([survey, pd.DataFrame([_survey_baseline_row(arm_table)])], ignore_index=True)
    readable = pd.DataFrame({
        "Arm": full_table.label,
        "Seeds": full_table.seeds.astype(int).astype(str),
        "OOF": full_table.oof_cost_scaled_to_test_set_size_median.map(thousands),
        "Test [range]": [median_with_range(median, low, high) for median, low, high in
                         zip(full_table.test_cost_median, full_table.test_cost_min, full_table.test_cost_max)],
        "Paired difference": full_table.mean_difference.map(signed_thousands),
        "95 % interval": [bracketed_interval(low, high) for low, high in
                          zip(full_table.interval_low, full_table.interval_high)],
        "p (Holm)": full_table.p_holm.map(_holm_cell),
        "FP at FN<=6": full_table.test_min_fp_at_most_6_fn_median.map(thousands),
        "Tier": full_table.tier,
    })
    candidates = int((survey.arm != DUPLICATION_CONTROL).sum())
    write_summary_table(
        "rq2_survey_against_tuned_lightgbm", full_table, readable, left_aligned_columns=["Arm", "Tier"],
        title="RQ2: survey of methods against the tuned LightGBM",
        caption=f"""
        Every survey arm against the tuned LightGBM without synthetic rows (last row), paired fold by fold on the
        seeds both share; {candidates} arms are tested, the duplication control is the reference and is not part of
        the Holm family. Paired differences are scaled to the 16,000 rows of the test set, negative values favour
        the arm. The tier follows the two-tier criterion (demonstrated gain / positive sign / no evidence) defined in
        analysis/paired_statistics.py, evidence_tier. GReaT and two of the three Forest-Flow profiles have no
        records (GReaT failed; the other profiles did not complete). Source: arm_records.jsonl and the saved
        probabilities.""",
        column_notes={
            "Seeds": "seeds shared with the baseline (CSV: seeds; pairs = seeds x folds).",
            "OOF": "median OOF cost scaled to the 16,000 rows of the test set (CSV: "
                   "oof_cost_scaled_to_test_set_size_median).",
            "Test [range]": "median, smallest and largest test cost (CSV: test_cost_median, test_cost_min, "
                            "test_cost_max; the baseline's median on the same seeds is baseline_test_cost_median).",
            "Paired difference": "mean fold-level difference arm - baseline, scaled to the 16,000 rows of the test "
                                 "set (CSV: mean_difference; all_seeds_favour_arm says whether every seed's mean "
                                 "difference is negative).",
            "95 % interval": "interval of the corrected resampled t-test (CSV: interval_low, interval_high; the "
                             "unadjusted p-values are p_corrected_t and p_wilcoxon).",
            "p (Holm)": "corrected t-test p-value after Holm's adjustment over the tested arms (CSV: p_holm).",
            "FP at FN<=6": "median of the fewest test false alarms with at most 6 missed failures (CSV: "
                           "test_min_fp_at_most_6_fn_median; the baseline's on the same seeds is "
                           "baseline_test_min_fp_at_most_6_fn_median).",
            "Tier": "demonstrated gain, positive sign or no evidence; 'reference' for the duplication control "
                    "(CSV: tier; family groups the methods).",
            "CSV only": "test_oracle_minimum_cost_median, wins, losses, ties.",
        })


def _holm_cell(p_value) -> str:
    if is_missing(p_value):
        return MISSING
    if p_value < 0.01:
        return "<0.01"
    return f"{p_value:.2f}"


# ---------------------------------------------------------------------------------------------- run times
def write_runtime_table(arm_table: pd.DataFrame) -> None:
    rows = []
    for arm in arms_present(arm_table, ARM_LABELS):
        records = arm_table[arm_table.arm == arm]
        rows.append({"arm": arm, "label": ARM_LABELS[arm], "seeds": records.seed.nunique(),
                     "cross_validation_minutes_median": records.cross_validation_minutes.median(),
                     "dominant_cost": DOMINANT_RUNTIME_COST.get(arm, "")})
    table = pd.DataFrame(rows)
    readable = pd.DataFrame({"Arm": table.label,
                             "Minutes per seed": table.cross_validation_minutes_median.map(thousands),
                             "Dominant cost": table.dominant_cost})
    write_summary_table(
        "classifier_arms_runtimes", table, readable, left_aligned_columns=["Arm", "Dominant cost"],
        title="Cross-validation time per seed",
        caption="""
        Median over seeds of the wall-clock minutes of the cross-validation (all folds, generators included; the
        final test evaluation is not counted), from cross_validation_seconds in arm_records.jsonl. The seeds ran on
        different machines (records with a 'source' field ran in a cloud session), so the minutes indicate orders of
        magnitude rather than exact costs.""",
        column_notes={"Minutes per seed": "CSV: cross_validation_minutes_median.",
                      "Dominant cost": "what takes most of the time, noted when the runs were made (CSV: "
                                       "dominant_cost)."})


# ---------------------------------------------------------------------------------------------- figures
def plot_oof_cost_against_threshold(arm_table: pd.DataFrame) -> None:
    """OOF cost (scaled to the 16,000 rows of the test set) as a function of the threshold, seed 11: how flat the curve
    is near its minimum, and how far the default 0.5 and the Bayes cut-off are from it."""
    labels = training_labels()
    figure, axes = plt.subplots(figsize=(9, 5))
    for arm in arms_present(arm_table, COST_CURVE_FIGURE_ARMS):
        curve = cost_curve(load_arm_probabilities(arm, FIGURE_SEED, "out_of_fold"), labels)
        curve = curve[(curve.threshold >= 1e-6) & (curve.threshold <= 0.95)]
        axes.plot(curve.threshold, curve.cost * TEST_SET_ROWS / TRAINING_ROWS, lw=1.2, label=ARM_LABELS[arm])
    axes.set_xscale("log")
    axes.set_ylim(5000, 40000)
    axes.axvline(BAYES_CUTOFF, ls=":", c="k", lw=0.8)
    axes.text(BAYES_CUTOFF * 0.9, 38500, "Bayes cut-off 10/510 = 0.0196", fontsize=8, ha="right")
    axes.axvline(DEFAULT_CUTOFF, ls=":", c="grey", lw=0.8)
    axes.text(DEFAULT_CUTOFF * 1.1, 38500, "default 0.5", fontsize=8)
    axes.set_xlabel("decision threshold on the out-of-fold probability (log scale)")
    axes.set_ylabel("out-of-fold cost, scaled to the 16,000-row test set")
    axes.set_title(f"Out-of-fold cost against the threshold (seed {FIGURE_SEED})")
    axes.legend(fontsize=8, loc="upper left")
    save_figure(figure, "classifier_arms_oof_cost_against_threshold")


def _false_alarm_frontier(probabilities: np.ndarray, labels: np.ndarray, max_missed_failures: int = 25) -> pd.Series:
    """Fewest false alarms for each number of missed failures up to `max_missed_failures`, on the threshold grid."""
    curve = cost_curve(probabilities, labels)
    curve = curve[curve.fn <= max_missed_failures].sort_values("fn")
    return curve.groupby("fn").fp.min()


def plot_test_false_alarm_frontier(arm_table: pd.DataFrame) -> None:
    """For each number of missed test failures, the fewest false alarms any threshold achieves (seed 11), with the
    five models released by Beikmohammadi et al. and the published operating points. Uses test labels: a diagnostic of ranking."""
    labels = test_labels()
    figure, axes = plt.subplots(figsize=(9, 6.5))
    for arm in arms_present(arm_table, FRONTIER_FIGURE_ARMS):
        frontier = _false_alarm_frontier(load_arm_probabilities(arm, FIGURE_SEED, "test"), labels)
        axes.step(frontier.index, frontier.values, where="post", lw=1.2, label=ARM_LABELS[arm])
    for run, probabilities in enumerate(load_shipped_model_test_probabilities()):
        frontier = _false_alarm_frontier(probabilities, labels)
        legend_label = "Models released by Beikmohammadi et al., regenerated inputs (five runs)" if run == 0 else None
        axes.step(frontier.index, frontier.values, where="post", lw=0.9, ls="--", color="k", alpha=0.7,
                  label=legend_label)
    for name, (missed_failures, false_alarms) in PUBLISHED_OPERATING_POINTS.items():
        axes.scatter([missed_failures], [false_alarms], marker="*", s=90, zorder=5, c="k")
        # leader lines of the two-line labels start at the middle of the label's left edge
        anchor = (0.5, 0.5) if "\n" not in name else (0, 0.5)
        axes.annotate(name, (missed_failures, false_alarms), xytext=PUBLISHED_LABEL_POSITIONS[name],
                      textcoords="data", fontsize=8, ha="left", va="center",
                      arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "0.35", "shrinkA": 1.5, "shrinkB": 4,
                                  "relpos": anchor})
    axes.set_yscale("log")
    axes.set_xlabel("missed failures (of 375 test failures)")
    axes.set_ylabel("fewest false alarms over all thresholds (log scale)")
    axes.set_title(f"Test set: false alarms against missed failures (seed {FIGURE_SEED}; uses test labels)")
    axes.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    save_figure(figure, "classifier_arms_test_false_alarm_frontier")


def plot_paired_differences(arm_table: pd.DataFrame, comparisons: pd.DataFrame) -> None:
    """Every fold difference (dots) with the corrected 95 % interval (bar) and the mean (tick), per main arm."""
    figure, axes = plt.subplots(figsize=(11, 0.6 * len(comparisons) + 1.6))
    for position, row in enumerate(comparisons.itertuples(index=False)):
        folds, differences_by_seed = paired_fold_differences(arm_table, row.arm, row.baseline)
        scale_to_test_set = TEST_SET_ROWS / (TRAINING_ROWS / folds)
        differences = np.concatenate(list(differences_by_seed.values())) * scale_to_test_set
        axes.scatter(differences, np.full(len(differences), position), s=14, alpha=0.6)
        axes.plot([row.interval_low, row.interval_high], [position, position], c="k", lw=1.6)
        axes.plot([row.mean_difference], [position], "k|", ms=10)
    axes.axvline(0, ls=":", c="k", lw=0.8)
    axes.set_yticks(range(len(comparisons)))
    tick_labels = [f"{row.label}\nagainst: {row.baseline_label}" for row in comparisons.itertuples()]
    axes.set_yticklabels(tick_labels, fontsize=8)
    axes.invert_yaxis()
    axes.set_xlabel("paired fold cost difference, arm - matched baseline\n"
                    "(scaled to the 16,000 rows of the test set; negative favours the arm)")
    axes.set_title("Paired fold-level differences with corrected 95 % intervals")
    save_figure(figure, "classifier_arms_paired_differences")


def plot_survey_paired_differences(survey: pd.DataFrame) -> None:
    """The survey arms' mean paired difference and corrected interval, coloured by tier, with the duplication
    control's interval as a grey band."""
    rows_bottom_up = survey.iloc[::-1]
    figure, axes = plt.subplots(figsize=(10, 0.35 * len(survey) + 1.4))
    control = survey[survey.arm == DUPLICATION_CONTROL]
    if len(control):
        axes.axvspan(control.interval_low.iloc[0], control.interval_high.iloc[0], color="#cccccc", alpha=0.5,
                     label="duplication control (95 % interval)")
    axes.axvline(0, color="k", lw=0.8)
    for position, row in enumerate(rows_bottom_up.itertuples(index=False)):
        colour = TIER_COLOURS[row.tier]
        axes.plot([row.interval_low, row.interval_high], [position, position], color=colour, lw=1.8)
        axes.plot(row.mean_difference, position, "o", color=colour, ms=4)
    axes.set_yticks(np.arange(len(rows_bottom_up)))
    axes.set_yticklabels([f"{row.label}  [{row.tier}]" for row in rows_bottom_up.itertuples()], fontsize=8)
    axes.set_xlabel("paired fold cost difference against the tuned LightGBM\n"
                    "(scaled to the 16,000 rows of the test set; negative favours the arm)")
    axes.set_title("RQ2 survey: paired differences with corrected 95 % intervals")
    axes.legend(loc="lower right", fontsize=8, frameon=False)
    save_figure(figure, "rq2_survey_paired_differences")


def main(arm_table: pd.DataFrame | None = None) -> None:
    print("Classifier arms (RQ1, RQ2)")
    if arm_table is None:
        arm_table = build_arm_table()
    write_per_seed_table(arm_table)
    write_main_results_table(arm_table)
    write_decision_rule_table(arm_table)
    comparisons = paired_comparisons(arm_table)
    write_paired_comparison_table(comparisons)
    survey = survey_table(arm_table)
    write_survey_table(arm_table, survey)
    write_runtime_table(arm_table)
    plot_oof_cost_against_threshold(arm_table)
    plot_test_false_alarm_frontier(arm_table)
    plot_paired_differences(arm_table, comparisons)
    plot_survey_paired_differences(survey)


if __name__ == "__main__":
    main()
