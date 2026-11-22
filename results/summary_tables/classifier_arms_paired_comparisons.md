# Paired fold-level comparisons with the matched baseline

Each main arm against the same learner without its change, on the same seeds and folds (the arms
on the preprocessing pipeline of Beikmohammadi et al. against the 3-fold LightGBM). The differences are arm
minus baseline, per validation fold at each arm's OOF threshold, pooled over the seeds and rescaled to the
16,000-row test set; negative values favour the arm. The interval and the p-value come from the Nadeau-Bengio
corrected resampled t-test; analysis/paired_statistics.py explains the method and its assumptions. Source: the
fold costs in arm_records.jsonl.

| Arm | Baseline | Pairs | Mean difference | 95 % interval | Corrected t | p (corrected t) | p (Wilcoxon) | Wins / losses |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost (fixed settings), no augmentation | LightGBM (tuned), no augmentation | 15 | -520 | [-1,863, 823] | -0.83 | 0.42 | 0.09 | 11 / 4 |
| cGAN +2,000 rows, LightGBM | LightGBM (tuned), no augmentation | 25 | +35 | [-1,240, 1,311] | +0.06 | 0.96 | 0.82 | 14 / 11 |
| cGAN +2,000 rows, random forest | Random forest, no augmentation | 15 | -253 | [-1,228, 721] | -0.56 | 0.59 | 0.22 | 9 / 6 |
| cGAN +2,000 rows, sklearn GBM | sklearn GBM, no augmentation | 15 | -384 | [-1,932, 1,164] | -0.53 | 0.60 | 0.36 | 10 / 5 |
| cGAN with the notebook's settings, random forest | Random forest, no augmentation | 5 | +61 | [-733, 856] | +0.21 | 0.84 | 1.00 | 3 / 2 |
| 2,000 duplicated real failures, LightGBM | LightGBM (tuned), no augmentation | 25 | +218 | [-735, 1,170] | +0.47 | 0.64 | 0.31 | 10 / 15 |
| Forest-Flow +2,000 rows, LightGBM | LightGBM (tuned), no augmentation | 15 | +37 | [-1,560, 1,634] | +0.05 | 0.96 | 0.98 | 7 / 8 |
| TabEBM +2,000 rows, LightGBM (GPU) | LightGBM (tuned), no augmentation | 15 | -261 | [-1,584, 1,061] | -0.42 | 0.68 | 0.49 | 9 / 6 |
| LightGBM (tuned) + missingness indicators | LightGBM (tuned), no augmentation | 15 | -80 | [-1,085, 925] | -0.17 | 0.87 | 0.69 | 8 / 7 |
| Ensemble (LightGBM + XGBoost + CatBoost) + missingness indicators | LightGBM (tuned), no augmentation | 25 | -434 | [-1,732, 865] | -0.69 | 0.50 | 0.08 | 15 / 10 |
| Ensemble (LightGBM + XGBoost + CatBoost), no indicators | LightGBM (tuned), no augmentation | 25 | -426 | [-1,524, 673] | -0.80 | 0.43 | 0.09 | 16 / 9 |
| TabICLv2, bagged contexts, no tuning | LightGBM (tuned), no augmentation | 25 | -886 | [-2,428, 655] | -1.19 | 0.25 | 0.01 | 19 / 6 |
| Preprocessing of Beikmohammadi et al. (SVM-SMOTE and RENN), LightGBM | LightGBM (tuned), no augmentation, 3 folds | 3 | -21 | [-5,602, 5,559] | -0.02 | 0.99 | 1.00 | 1 / 2 |
| Preprocessing of Beikmohammadi et al. (RENN only), LightGBM | LightGBM (tuned), no augmentation, 3 folds | 3 | -205 | [-5,197, 4,786] | -0.18 | 0.88 | 1.00 | 2 / 1 |
| Preprocessing of Beikmohammadi et al. (SVM-SMOTE only), LightGBM | LightGBM (tuned), no augmentation, 3 folds | 3 | +19 | [-3,742, 3,780] | +0.02 | 0.98 | 1.00 | 2 / 1 |
| Preprocessing of Beikmohammadi et al. (imputation only), LightGBM | LightGBM (tuned), no augmentation, 3 folds | 3 | +363 | [-2,500, 3,225] | +0.55 | 0.64 | 0.75 | 1 / 2 |

Columns:

- Baseline: the matched baseline (CSV: baseline, baseline_label).
- Pairs: number J of paired fold differences, seeds x folds (CSV: pairs, seeds, folds).
- Mean difference: mean paired difference scaled to the 16,000 rows of the test set (CSV: mean_difference; in fold-cost units mean_difference_fold, with median_difference_fold and standard_deviation_fold).
- 95 % interval: interval of the corrected t-test scaled to the 16,000 rows of the test set (CSV: interval_low, interval_high).
- Corrected t: mean / corrected standard error (CSV: t_corrected).
- p (corrected t): two-sided p-value, J - 1 degrees of freedom (CSV: p_corrected_t).
- p (Wilcoxon): two-sided Wilcoxon signed-rank p-value, not corrected for the overlap of the training sets (CSV: p_wilcoxon).
- Wins / losses: folds in which the arm was cheaper / dearer than the baseline (CSV: wins, losses, ties).

Unrounded values: `classifier_arms_paired_comparisons.csv` in this folder.
