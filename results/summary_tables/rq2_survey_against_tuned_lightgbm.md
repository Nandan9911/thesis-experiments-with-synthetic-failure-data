# RQ2: survey of methods against the tuned LightGBM

Every survey arm against the tuned LightGBM without synthetic rows (last row), paired fold by fold on the
seeds both share; 18 arms are tested, the duplication control is the reference and is not part of
the Holm family. Paired differences are scaled to the 16,000 rows of the test set, negative values favour
the arm. The tier follows the two-tier criterion (demonstrated gain / positive sign / no evidence) defined in
analysis/paired_statistics.py, evidence_tier. GReaT and two of the three Forest-Flow profiles have no
records (GReaT failed; the other profiles did not complete). Source: arm_records.jsonl and the saved
probabilities.

| Arm | Seeds | OOF | Test [range] | Paired difference | 95 % interval | p (Holm) | FP at FN<=6 | Tier |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Duplication control: 2,000 copied real failures | 5 | 9,485 | 9,070 [8,820, 10,130] | +218 | [-735, 1,170] | -- | 647 | reference |
| cGAN of Ranasinghe et al., +2,000 rows | 5 | 9,509 | 9,500 [9,000, 11,150] | +35 | [-1,240, 1,311] | 1.00 | 719 | no evidence |
| SMOTE, +2,000 rows | 3 | 9,533 | 10,950 [9,860, 11,050] | +19 | [-1,530, 1,567] | 1.00 | 966 | no evidence |
| Borderline-SMOTE, +2,000 rows | 3 | 9,805 | 10,310 [9,540, 10,520] | +139 | [-1,596, 1,873] | 1.00 | 871 | no evidence |
| ADASYN, +2,000 rows | 3 | 10,077 | 10,630 [10,580, 10,960] | +464 | [-969, 1,897] | 1.00 | 1,015 | no evidence |
| SVM-SMOTE, +2,000 rows | 3 | 9,581 | 10,420 [9,620, 10,890] | -149 | [-1,532, 1,233] | 1.00 | 735 | no evidence |
| Gaussian copula, +2,000 rows | 3 | 9,493 | 9,400 [9,280, 9,430] | -211 | [-1,757, 1,336] | 1.00 | 679 | no evidence |
| TVAE, +2,000 rows | 3 | 9,605 | 9,950 [9,330, 10,110] | -240 | [-1,703, 1,223] | 1.00 | 815 | no evidence |
| CTGAN, +2,000 rows | 3 | 9,565 | 9,980 [9,860, 10,240] | -240 | [-1,983, 1,503] | 1.00 | 663 | no evidence |
| CTAB-GAN+, +2,000 rows | 3 | 9,997 | 8,900 [8,890, 9,770] | +136 | [-1,302, 1,574] | 1.00 | 551 | no evidence |
| TabDDPM, +2,000 rows | 3 | 9,485 | 9,550 [9,540, 9,680] | -213 | [-1,345, 918] | 1.00 | 703 | no evidence |
| Forest-Flow (reduced profile), +2,000 rows | 3 | 9,813 | 9,600 [9,370, 9,970] | +37 | [-1,560, 1,634] | 1.00 | 719 | no evidence |
| TabEBM, +2,000 rows | 3 | 9,381 | 9,700 [9,520, 10,100] | -261 | [-1,584, 1,061] | 1.00 | 735 | no evidence |
| TabICLv2, bagged contexts (no synthetic rows) | 5 | 8,373 | 7,980 [7,320, 9,700] | -886 | [-2,428, 655] | 1.00 | 471 | no evidence |
| LightGBM + missingness indicators | 3 | 9,925 | 9,180 [8,960, 10,040] | -80 | [-1,085, 925] | 1.00 | 615 | no evidence |
| Ensemble LightGBM+XGBoost+CatBoost | 5 | 8,821 | 8,220 [7,520, 9,200] | -426 | [-1,524, 673] | 1.00 | 463 | positive sign |
| Ensemble + missingness indicators | 5 | 8,653 | 8,730 [7,790, 9,920] | -434 | [-1,732, 865] | 1.00 | 479 | no evidence |
| Ensemble + indicators + TabICLv2 member | 5 | 8,429 | 8,060 [7,640, 12,180] | -672 | [-1,794, 450] | 1.00 | 495 | positive sign |
| XGBoost alone | 3 | 9,005 | 8,380 [8,250, 8,480] | -520 | [-1,863, 823] | 1.00 | 455 | no evidence |
| Baseline: tuned LightGBM | 5 | 8,877 | 9,350 [8,970, 10,360] | -- | -- | -- | 591 | baseline |

Columns:

- Seeds: seeds shared with the baseline (CSV: seeds; pairs = seeds x folds).
- OOF: median OOF cost scaled to the 16,000 rows of the test set (CSV: oof_cost_scaled_to_test_set_size_median).
- Test [range]: median, smallest and largest test cost (CSV: test_cost_median, test_cost_min, test_cost_max; the baseline's median on the same seeds is baseline_test_cost_median).
- Paired difference: mean fold-level difference arm - baseline, scaled to the 16,000 rows of the test set (CSV: mean_difference; all_seeds_favour_arm says whether every seed's mean difference is negative).
- 95 % interval: interval of the corrected resampled t-test (CSV: interval_low, interval_high; the unadjusted p-values are p_corrected_t and p_wilcoxon).
- p (Holm): corrected t-test p-value after Holm's adjustment over the tested arms (CSV: p_holm).
- FP at FN<=6: median of the fewest test false alarms with at most 6 missed failures (CSV: test_min_fp_at_most_6_fn_median; the baseline's on the same seeds is baseline_test_min_fp_at_most_6_fn_median).
- Tier: demonstrated gain, positive sign or no evidence; 'reference' for the duplication control (CSV: tier; family groups the methods).
- CSV only: test_oracle_minimum_cost_median, wins, losses, ties.

Unrounded values: `rq2_survey_against_tuned_lightgbm.csv` in this folder.
