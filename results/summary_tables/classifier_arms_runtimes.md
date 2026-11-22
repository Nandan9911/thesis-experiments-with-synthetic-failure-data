# Cross-validation time per seed

Median over seeds of the wall-clock minutes of the cross-validation (all folds, generators included; the
final test evaluation is not counted), from cross_validation_seconds in arm_records.jsonl. The seeds ran on
different machines (records with a 'source' field ran in a cloud session), so the minutes indicate orders of
magnitude rather than exact costs.

| Arm | Minutes per seed | Dominant cost |
| :--- | ---: | :--- |
| LightGBM (tuned), no augmentation | 2 | LightGBM fits |
| XGBoost (fixed settings), no augmentation | 14 | XGBoost fits |
| Random forest, no augmentation | 15 | random forest fits |
| sklearn GBM, no augmentation | 53 | sklearn GBM fits |
| LightGBM (tuned), no augmentation, 3 folds | 15 | LightGBM fits |
| cGAN +2,000 rows, LightGBM | 19 | cGAN training (30k steps per fold) |
| cGAN +2,000 rows, random forest | 26 | cGAN + random forest |
| cGAN +2,000 rows, sklearn GBM | 70 | sklearn GBM fits |
| cGAN with the notebook's settings, random forest | 28 | cGAN + random forest |
| 2,000 duplicated real failures, LightGBM | 3 | LightGBM fits |
| Forest-Flow +2,000 rows, LightGBM | 350 | Forest-Flow fits (1,690 XGBoost models per fold) |
| TabEBM +2,000 rows, LightGBM (GPU) | 80 |  |
| LightGBM (tuned) + missingness indicators | 39 | LightGBM fits |
| Ensemble (LightGBM + XGBoost + CatBoost) + missingness indicators | 62 | CatBoost, XGBoost and LightGBM fits |
| Ensemble (LightGBM + XGBoost + CatBoost), no indicators | 53 | CatBoost, XGBoost and LightGBM fits |
| TabICLv2, bagged contexts, no tuning | 6 | TabICL inference |
| Preprocessing of Beikmohammadi et al. (SVM-SMOTE and RENN), LightGBM | 37 | imputer, SVM-SMOTE, RENN |
| Preprocessing of Beikmohammadi et al. (RENN only), LightGBM | 13 | imputer, RENN |
| Preprocessing of Beikmohammadi et al. (SVM-SMOTE only), LightGBM | 14 | imputer, SVM-SMOTE |
| Preprocessing of Beikmohammadi et al. (imputation only), LightGBM | 58 | imputer |

Columns:

- Minutes per seed: CSV: cross_validation_minutes_median.
- Dominant cost: what takes most of the time, noted when the runs were made (CSV: dominant_cost).

Unrounded values: `classifier_arms_runtimes.csv` in this folder.
