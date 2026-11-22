"""Paired fold-level comparisons of two classifier arms: corrected resampled t-test, Wilcoxon test, Holm's
adjustment and the two-tier criterion of the survey (RQ1 and RQ2).

Every classifier arm is cross-validated with the same stratified folds for a given seed: the fold assignment
depends only on the seed, the number of folds and the class labels, so arm A and arm B with seed 11 are validated
on exactly the same 12,000 rows in fold 0, the same rows in fold 1, and so on. The record of each arm and seed
stores the cost of every validation fold at the arm's out-of-fold threshold (record["out_of_fold"]["fold_costs"]).
A comparison of an arm with its baseline therefore rests on paired differences

    d = fold cost of the arm - fold cost of the baseline     (negative favours the arm),

one per fold and seed: J = seeds x folds pairs (25 for five seeds and five folds).

Why the t-test is corrected. The J differences are not independent. Any two training sets of a 5-fold
cross-validation share three fifths of their rows, and the repetitions with other seeds reuse the same 60,000 rows,
so the differences are positively correlated. The ordinary paired t-test divides the sample variance by J as if
they were independent; it underestimates the variance of the mean difference and declares differences significant
far too often. Nadeau and Bengio (2003, "Inference for the generalization error", Machine Learning 52) proposed the
corrected resampled t-test, which replaces the factor 1/J by

    1/J + n_test / n_train,

where n_test / n_train is the ratio of validation to training rows; the extra term stands for the correlation
between the differences. In k-fold cross-validation n_test / n_train = (N/k) / (N(k-1)/k) = 1/(k-1), so the factor
is 1/J + 1/4 for 5 folds and 1/J + 1/2 for 3 folds. The statistic t = mean(d) / sqrt((1/J + 1/(k-1)) s^2) is
compared with a t distribution with J - 1 degrees of freedom (the "corrected repeated k-fold cv test" of Bouckaert
and Frank, 2004). The correction is a heuristic: it assumes approximately normal differences and a correlation of
the size implied by the overlap of the training sets. It makes the test much more cautious than the naive one,
which is the intended direction; with only three to five seeds the intervals are wide, and a non-significant result
means that the experiment cannot tell the arms apart, not that they are equal.

Units. A fold cost is the cost of one validation fold (12,000 rows with 5 folds, 20,000 with 3). The differences
and intervals reported in the tables are multiplied by 16,000 / fold rows, which puts them on the scale of the
16,000-row test set and makes 5-fold and 3-fold comparisons readable side by side. The scaling is
linear and does not change t or p.

The Wilcoxon signed-rank test is reported next to the t-test as a check that does not assume normal differences
(two-sided; zero differences are split evenly between the positive and the negative ranks, zero_method="zsplit").
It assumes independent pairs and has no correction for the overlap of the training sets, so its p-values are too
small for the same reason as those of the naive t-test; it is never used on its own.

Holm's step-down adjustment controls the family-wise error rate when several arms are tested against the same
baseline: the m p-values are sorted, the i-th smallest is multiplied by (m - i + 1), and a running maximum makes the
adjusted values non-decreasing in that order (they are capped at 1). It is applied to the corrected t-test p-values
of the survey arms (RQ2), with the duplication control left out of the family because it is the reference, not a
candidate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from analysis.load_results import TRAINING_ROWS
from experiments.common.data_loading import TEST_SET_ROWS

SIGNIFICANCE_LEVEL = 0.05


def paired_fold_differences(arm_table: pd.DataFrame, arm: str, baseline: str):
    """Fold cost differences (arm - baseline) for every seed both arms were run with.

    Returns (folds, {seed: array of differences in fold-cost units}) or None when the arms share no seed. A seed is
    skipped if the two records have a different number of folds (they could not be paired fold by fold)."""
    arm_rows = arm_table[arm_table.arm == arm]
    baseline_rows = arm_table[arm_table.arm == baseline]
    common_seeds = sorted(set(arm_rows.seed) & set(baseline_rows.seed))
    differences = {}
    for seed in common_seeds:
        arm_fold_costs = np.array(arm_rows[arm_rows.seed == seed].iloc[0].fold_costs, dtype=float)
        baseline_fold_costs = np.array(baseline_rows[baseline_rows.seed == seed].iloc[0].fold_costs, dtype=float)
        if len(arm_fold_costs) != len(baseline_fold_costs):
            continue
        differences[seed] = arm_fold_costs - baseline_fold_costs
    if not differences:
        return None
    folds = len(next(iter(differences.values())))
    return folds, differences


def corrected_resampled_t_test(differences: np.ndarray, folds: int, confidence: float = 0.95) -> dict:
    """Nadeau-Bengio corrected resampled t-test on J pooled fold differences of a k-fold cross-validation.

    Returns mean, standard_deviation (sample, ddof=1), standard_error (corrected), t, p (two-sided),
    interval_low and interval_high (the `confidence` interval of the mean difference, with the same corrected
    standard error and J - 1 degrees of freedom). t, p and the interval are NaN when all differences are equal."""
    differences = np.asarray(differences, dtype=float)
    pairs = len(differences)
    mean = differences.mean()
    standard_deviation = differences.std(ddof=1)
    if standard_deviation > 0:
        standard_error = np.sqrt((1 / pairs + 1 / (folds - 1)) * standard_deviation ** 2)
        t_statistic = mean / standard_error
        p_value = 2 * stats.t.sf(abs(t_statistic), df=pairs - 1)
    else:
        standard_error = t_statistic = p_value = np.nan
    quantile = stats.t.ppf(0.5 + confidence / 2, pairs - 1)
    return {"mean": mean, "standard_deviation": standard_deviation, "standard_error": standard_error,
            "t": t_statistic, "p": p_value,
            "interval_low": mean - quantile * standard_error, "interval_high": mean + quantile * standard_error}


def wilcoxon_signed_rank_p_value(differences: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value of the differences (zeros split between the signs); NaN when every
    difference is zero, where the test is undefined."""
    differences = np.asarray(differences, dtype=float)
    if not np.any(differences != 0):
        return np.nan
    try:
        return float(stats.wilcoxon(differences, zero_method="zsplit").pvalue)
    except ValueError:
        return np.nan


def holm_adjusted_p_values(p_values) -> np.ndarray:
    """Holm's step-down adjusted p-values, in the order given. NaN entries are left out of the family and stay NaN."""
    p_values = np.array(p_values, dtype=float)
    adjusted = np.full_like(p_values, np.nan)
    finite = np.where(np.isfinite(p_values))[0]
    family_size = len(finite)
    running_maximum = 0.0
    for rank, position in enumerate(finite[np.argsort(p_values[finite])]):
        running_maximum = max(running_maximum, (family_size - rank) * p_values[position])
        adjusted[position] = min(1.0, running_maximum)
    return adjusted


def compare_with_baseline(arm_table: pd.DataFrame, arm: str, baseline: str):
    """Everything the comparison tables report for one arm against one baseline, or None if they share no seed.

    Keys: arm, baseline, seeds (list of the paired seeds), folds, pairs (J); in fold-cost units mean_difference_fold,
    median_difference_fold, standard_deviation_fold; scaled to the 16,000 rows of the test set:
    mean_difference, interval_low, interval_high (95 % interval of the corrected t-test); t_corrected, p_corrected_t,
    p_wilcoxon; wins, losses, ties (folds in which the arm was cheaper, dearer, equal); per_seed_mean_difference
    ({seed: mean difference scaled to the 16,000 rows of the test set}) and all_seeds_favour_arm (every seed's mean
    difference is negative)."""
    paired = paired_fold_differences(arm_table, arm, baseline)
    if paired is None:
        return None
    folds, differences_by_seed = paired
    scale_to_test_set = TEST_SET_ROWS / (TRAINING_ROWS / folds)
    differences = np.concatenate(list(differences_by_seed.values()))
    test = corrected_resampled_t_test(differences, folds)
    per_seed_mean = {seed: float(values.mean() * scale_to_test_set) for seed, values in differences_by_seed.items()}
    return {
        "arm": arm, "baseline": baseline, "seeds": sorted(differences_by_seed), "folds": folds,
        "pairs": len(differences),
        "mean_difference_fold": test["mean"], "median_difference_fold": float(np.median(differences)),
        "standard_deviation_fold": test["standard_deviation"],
        "mean_difference": test["mean"] * scale_to_test_set,
        "interval_low": test["interval_low"] * scale_to_test_set,
        "interval_high": test["interval_high"] * scale_to_test_set,
        "t_corrected": test["t"], "p_corrected_t": test["p"], "p_wilcoxon": wilcoxon_signed_rank_p_value(differences),
        "wins": int((differences < 0).sum()), "losses": int((differences > 0).sum()),
        "ties": int((differences == 0).sum()),
        "per_seed_mean_difference": per_seed_mean,
        "all_seeds_favour_arm": all(value < 0 for value in per_seed_mean.values()),
    }


def evidence_tier(row, control_mean_difference: float) -> str:
    """The two-tier criterion of the survey (RQ2) for one arm compared with the tuned LightGBM.

    demonstrated gain   the 95 % interval of the corrected paired difference lies below zero, the Holm-adjusted
                        p-value is below 0.05, the mean difference is lower (better) than that of the duplication
                        control (2,000 copies of real failures: the same amount of extra failure weight without any
                        new information), and the median test cost is not higher than the baseline's.
    positive sign       not a demonstrated gain, but the mean difference is negative, every seed's mean difference
                        is negative, the median test cost is not higher than the baseline's, and the median of the
                        fewest false alarms at <= 6 missed failures is not higher than the baseline's (when known).
    no evidence         everything else.

    `row` needs mean_difference, interval_high, p_holm, test_cost_median, baseline_test_cost_median,
    all_seeds_favour_arm, test_min_fp_at_most_6_fn_median and baseline_test_min_fp_at_most_6_fn_median."""
    test_cost_not_worse = not (np.isfinite(row["test_cost_median"])
                               and row["test_cost_median"] > row["baseline_test_cost_median"])
    if (row["interval_high"] < 0 and row["p_holm"] < SIGNIFICANCE_LEVEL
            and row["mean_difference"] < control_mean_difference and test_cost_not_worse):
        return "demonstrated gain"
    false_alarms = row["test_min_fp_at_most_6_fn_median"]
    false_alarms_not_worse = (not np.isfinite(false_alarms)
                              or false_alarms <= row["baseline_test_min_fp_at_most_6_fn_median"])
    if row["mean_difference"] < 0 and row["all_seeds_favour_arm"] and test_cost_not_worse and false_alarms_not_worse:
        return "positive sign"
    return "no evidence"
