# Main results: medians over seeds

Medians over the seeds of each main arm, from classifier_arms_per_seed (arm_records.jsonl and the saved
probabilities). The OOF cost is scaled to the 16,000 rows of the test set, so that it can be read next to the
test cost. Oracle and FP at FN<=6 use the test labels and are diagnostics only.

| Arm | Seeds | Folds | OOF cost | Test cost [range] | Test FN / FP | Oracle | FP at FN<=6 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM (tuned), no augmentation | 5 | 5 | 8,877 | 9,350 [8,970, 10,360] | 11 / 386 | 8,510 | 591 |
| XGBoost (fixed settings), no augmentation | 3 | 5 | 9,005 | 8,380 [8,250, 8,480] | 8 / 448 | 7,250 | 455 |
| Random forest, no augmentation | 3 | 5 | 10,405 | 9,760 [9,270, 9,770] | 9 / 526 | 9,240 | 648 |
| sklearn GBM, no augmentation | 3 | 5 | 11,981 | 10,650 [9,580, 11,820] | 8 / 558 | 10,090 | 831 |
| LightGBM (tuned), no augmentation, 3 folds | 3 | 3 | 10,021 | 9,200 [9,030, 9,480] | 9 / 470 | 8,380 | 591 |
| cGAN +2,000 rows, LightGBM | 5 | 5 | 9,509 | 9,500 [9,000, 11,150] | 10 / 400 | 8,870 | 719 |
| cGAN +2,000 rows, random forest | 3 | 5 | 10,051 | 9,790 [9,720, 10,100] | 10 / 510 | 9,240 | 664 |
| cGAN +2,000 rows, sklearn GBM | 3 | 5 | 11,477 | 10,600 [9,730, 11,530] | 8 / 660 | 9,530 | 727 |
| cGAN with the notebook's settings, random forest | 1 | 5 | 10,251 | 10,130 [10,130, 10,130] | 10 / 513 | 9,470 | 648 |
| 2,000 duplicated real failures, LightGBM | 5 | 5 | 9,485 | 9,070 [8,820, 10,130] | 10 / 413 | 8,570 | 647 |
| Forest-Flow +2,000 rows, LightGBM | 3 | 5 | 9,813 | 9,600 [9,370, 9,970] | 9 / 487 | 8,790 | 719 |
| TabEBM +2,000 rows, LightGBM (GPU) | 3 | 5 | 9,381 | 9,700 [9,520, 10,100] | 10 / 470 | 9,110 | 735 |
| LightGBM (tuned) + missingness indicators | 3 | 5 | 9,925 | 9,180 [8,960, 10,040] | 9 / 468 | 8,370 | 615 |
| Ensemble (LightGBM + XGBoost + CatBoost) + missingness indicators | 5 | 5 | 8,653 | 8,730 [7,790, 9,920] | 9 / 423 | 7,730 | 479 |
| Ensemble (LightGBM + XGBoost + CatBoost), no indicators | 5 | 5 | 8,821 | 8,220 [7,520, 9,200] | 7 / 472 | 7,460 | 463 |
| TabICLv2, bagged contexts, no tuning | 5 | 5 | 8,373 | 7,980 [7,320, 9,700] | 7 / 432 | 7,330 | 471 |
| Preprocessing of Beikmohammadi et al. (SVM-SMOTE and RENN), LightGBM | 1 | 3 | 9,661 | 8,850 [8,850, 8,850] | 6 / 585 | 8,670 | 567 |
| Preprocessing of Beikmohammadi et al. (RENN only), LightGBM | 1 | 3 | 9,477 | 9,840 [9,840, 9,840] | 9 / 534 | 8,830 | 583 |
| Preprocessing of Beikmohammadi et al. (SVM-SMOTE only), LightGBM | 1 | 3 | 9,701 | 9,550 [9,550, 9,550] | 9 / 505 | 9,560 | 983 |
| Preprocessing of Beikmohammadi et al. (imputation only), LightGBM | 1 | 3 | 10,045 | 9,060 [9,060, 9,060] | 7 / 556 | 9,020 | 679 |

Columns:

- Seeds, Folds: number of seeds and of cross-validation folds (CSV: seeds, folds).
- OOF cost: median OOF cost x 16,000 / 60,000 (CSV: oof_cost_scaled_to_test_set_size_median).
- Test cost [range]: median, smallest and largest test cost at the OOF threshold (CSV: test_cost_median, test_cost_min, test_cost_max).
- Test FN / FP: median missed failures and median false alarms on the test set; the two medians may come from different seeds (CSV: test_fn_median, test_fp_median).
- Oracle: median of the lowest test cost over all thresholds (CSV: test_oracle_minimum_cost_median).
- FP at FN<=6: median of the fewest test false alarms with at most 6 missed failures (CSV: test_min_fp_at_most_6_fn_median).

Unrounded values: `classifier_arms_main_results.csv` in this folder.
