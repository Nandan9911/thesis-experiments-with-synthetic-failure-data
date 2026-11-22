# RQ1b: where the released models flag rows (hard-negative comparison)

Share of each group of rows flagged as failures. Training negatives are split by whether RENN kept them in the
published training table (the models trained on them) or removed them (never seen); test negatives by the
one-shot RENN rule (hard if their three nearest neighbours in the resampled table are not all negatives).
Neither the hard test negatives nor the training negatives that RENN removed are in the published training
table. Sources: authors_code/shipped_models_exact_inputs/hard_negative_check.json,
hard_negative_check_followup.json and hard_negative_check_tree_control.json. The tree learners are the tuned
LightGBM, XGBoost (fixed settings) and the ensemble without missingness indicators, all with seed 11.

| Rows | n | Released Transformers (0.5) | LightGBM | XGBoost | Ensemble |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Training negatives kept by RENN (in the published training table) | 57,136 | 0.03 % | 1.0 % | 1.3 % | 1.5 % |
| Training negatives removed by RENN (not in it) | 1,864 | 21.4 % | 43.0 % | 55.5 % | 56.4 % |
| of which hard by the one-shot rule | 1,039 | 36.6 % | 63.5 % | 77.9 % | 78.0 % |
| Training negatives hard by the one-shot rule but kept by RENN | 387 | 2.2 % | 36.2 % | 46.3 % | 47.5 % |
| Test negatives hard by the one-shot rule | 381 | 0.89 % | 61.2 % | 69.8 % | 72.2 % |
| Other test negatives | 15,244 | 0.00 % | 1.1 % | 1.2 % | 1.5 % |
| Training positives | 1,000 | 98.2 % | 96.1 % | 96.8 % | 97.2 % |
| Test positives | 375 | 98.2 % | 97.3 % | 97.9 % | 98.7 % |

Columns:

- Rows: the group of rows (CSV: group; source_file and source_key locate it in the result files).
- n: number of rows in the group (CSV: rows).
- Released Transformers (0.5): mean over the five released models of the share with probability above 0.5 (CSV: shipped_transformers, a share between 0 and 1).
- LightGBM, XGBoost, Ensemble: share flagged by the tree arm at its own OOF threshold; out-of-fold probabilities for training rows (CSV: lightgbm_tuned, xgboost_fixed_settings, ensemble_without_missingness_indicators).

Unrounded values: `rq1_transformer_2026_hard_negatives_shipped_models.csv` in this folder.
