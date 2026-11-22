# The same test predictions under five decision rules

Test cost (missed failures / false alarms) of one set of test predictions per arm (seed 11) under five
threshold rules. Only the OOF threshold and the plateau rule are chosen on training data; 0.5 and
0.0196 are fixed in advance, and the oracle is chosen with the test labels (a diagnostic).
Source: classifier_arms_per_seed.

| Arm | Seed | 0.5 | 0.0196 | OOF threshold | OOF plateau | Oracle |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM (tuned), no augmentation | 11 | 38,200 (76 / 20) | 21,420 (41 / 92) | 9,020 (10 / 402) | 9,020 (10 / 402) | 7,710 |
| XGBoost (fixed settings), no augmentation | 11 | 40,650 (81 / 15) | 13,760 (23 / 226) | 8,480 (8 / 448) | 8,440 (8 / 444) | 7,250 |
| Random forest, no augmentation | 11 | 56,180 (112 / 18) | 9,980 (6 / 698) | 9,760 (9 / 526) | 9,690 (9 / 519) | 9,100 |
| sklearn GBM, no augmentation | 11 | 53,180 (106 / 18) | 11,370 (16 / 337) | 11,820 (8 / 782) | 11,680 (8 / 768) | 10,090 |
| LightGBM (tuned), no augmentation, 3 folds | 11 | 38,680 (77 / 18) | 19,500 (37 / 100) | 9,200 (9 / 470) | 9,280 (9 / 478) | 8,380 |
| cGAN +2,000 rows, LightGBM | 11 | 35,200 (70 / 20) | 21,420 (41 / 92) | 9,500 (10 / 450) | 9,870 (10 / 487) | 9,030 |
| cGAN +2,000 rows, random forest | 11 | 56,150 (112 / 15) | 10,010 (6 / 701) | 9,720 (8 / 572) | 9,720 (8 / 572) | 9,010 |
| cGAN +2,000 rows, sklearn GBM | 11 | 49,250 (98 / 25) | 12,000 (16 / 400) | 9,730 (6 / 673) | 9,730 (6 / 673) | 8,830 |
| cGAN with the notebook's settings, random forest | 11 | 54,160 (108 / 16) | 10,080 (6 / 708) | 10,130 (10 / 513) | 10,340 (10 / 534) | 9,470 |
| 2,000 duplicated real failures, LightGBM | 11 | 34,720 (69 / 22) | 20,970 (40 / 97) | 8,820 (8 / 482) | 9,580 (10 / 458) | 8,730 |
| Forest-Flow +2,000 rows, LightGBM | 11 | 36,200 (72 / 20) | 19,490 (37 / 99) | 9,600 (10 / 460) | 9,610 (10 / 461) | 8,790 |
| TabEBM +2,000 rows, LightGBM (GPU) | 11 | 36,250 (72 / 25) | 20,020 (38 / 102) | 9,700 (10 / 470) | 9,740 (10 / 474) | 9,110 |
| LightGBM (tuned) + missingness indicators | 11 | 36,690 (73 / 19) | 21,430 (41 / 93) | 8,960 (7 / 546) | 9,270 (8 / 527) | 8,610 |
| Ensemble (LightGBM + XGBoost + CatBoost) + missingness indicators | 11 | 37,160 (74 / 16) | 15,930 (29 / 143) | 7,790 (7 / 429) | 8,300 (9 / 380) | 7,540 |
| Ensemble (LightGBM + XGBoost + CatBoost), no indicators | 11 | 38,670 (77 / 17) | 16,380 (30 / 138) | 7,520 (5 / 502) | 7,440 (6 / 444) | 7,460 |
| TabICLv2, bagged contexts, no tuning | 11 | 20,860 (40 / 86) | 9,070 (3 / 757) | 7,320 (5 / 482) | 7,420 (5 / 492) | 7,280 |
| Preprocessing of Beikmohammadi et al. (SVM-SMOTE and RENN), LightGBM | 11 | 16,310 (28 / 231) | 11,250 (15 / 375) | 8,850 (6 / 585) | 8,920 (6 / 592) | 8,670 |
| Preprocessing of Beikmohammadi et al. (RENN only), LightGBM | 11 | 21,310 (40 / 131) | 15,510 (26 / 251) | 9,840 (9 / 534) | 9,320 (7 / 582) | 8,830 |
| Preprocessing of Beikmohammadi et al. (SVM-SMOTE only), LightGBM | 11 | 30,970 (61 / 47) | 17,640 (32 / 164) | 9,550 (9 / 505) | 10,010 (10 / 501) | 9,560 |
| Preprocessing of Beikmohammadi et al. (imputation only), LightGBM | 11 | 38,200 (76 / 20) | 23,100 (44 / 110) | 9,060 (7 / 556) | 9,400 (8 / 540) | 9,020 |

Columns:

- 0.5: the default cut-off (CSV: test_cost_at_0_5, test_fn_at_0_5, test_fp_at_0_5).
- 0.0196: the Bayes cut-off 10/510, cost-optimal for calibrated probabilities (CSV: test_*_at_bayes_cutoff).
- OOF threshold: the protocol's rule, the threshold minimising the OOF cost (CSV: test_*_at_oof_threshold).
- OOF plateau: geometric midpoint of the thresholds within 500 of the minimum OOF cost (CSV: test_*_at_plateau_threshold).
- Oracle: lowest test cost over all thresholds (CSV: test_oracle_minimum_cost).

Unrounded values: `classifier_arms_decision_rules.csv` in this folder.
