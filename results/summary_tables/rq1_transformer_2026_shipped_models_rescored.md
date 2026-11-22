# RQ1b: the five models released by Beikmohammadi et al. re-scored

Test cost (missed failures / false alarms) of the five models published by Beikmohammadi et al., each applied to
the official test set after the published preprocessing was regenerated (100-round iterative imputer, SVM-SMOTE
and RENN, min-max scaler fitted on the resampled table), and after three variants of it. Sources:
authors_code/shipped_models_exact_inputs/evaluation_scaler_on_resampled.csv, evaluation_scaler_on_imputed.csv
and evaluation_five_round_tables_scaler_on_resampled.csv, and
authors_code/logs/evaluate_shipped_models_five_round_tables.csv (written by
authors_code/evaluate_shipped_models.py). The mean of the published preprocessing equals the published 3,440
(6.8 / 4.0).

| Inputs | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Mean |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Published preprocessing | 3,540 (7 / 4) | 3,520 (7 / 2) | 3,060 (6 / 6) | 3,540 (7 / 4) | 3,540 (7 / 4) | 3,440 (6.8 / 4.0) |
|   test AUC | 0.9994 | 0.9994 | 0.9992 | 0.9992 | 0.9993 | 0.9993 |
|   lowest cost over eight cut-offs (test labels, diagnostic) | 1,550 | 1,990 | 2,590 | 1,220 | 2,180 | 1,906 |
| 5-round imputer, scaler on the imputed table (first attempt) | 183,510 (367 / 1) | 183,510 (367 / 1) | 179,010 (358 / 1) | 184,510 (369 / 1) | 183,510 (367 / 1) | 182,810 (365.6 / 1.0) |
| 100-round imputer, scaler on the imputed table | 136,020 (272 / 2) | 136,520 (273 / 2) | 140,580 (281 / 8) | 155,530 (311 / 3) | 143,030 (286 / 3) | 142,336 (284.6 / 3.6) |
| 5-round imputer, scaler on the resampled table | 182,510 (365 / 1) | 182,510 (365 / 1) | 179,010 (358 / 1) | 182,510 (365 / 1) | 183,010 (366 / 1) | 181,910 (363.8 / 1.0) |

Columns:

- Run 1 ... Run 5: test cost at the 0.5 cut-off with missed failures and false alarms (CSV: one row per variant and run; test_cost, test_fn, test_fp).
- Mean: mean over the five runs (CSV: the row with run = mean).
- test AUC: area under the ROC curve on the test set (CSV: test_auc).
- lowest cost over eight cut-offs: the lowest test cost over the cut-offs 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7 and 0.9, chosen with the test labels: a diagnostic (CSV: lowest_test_cost_over_eight_cutoffs, cutoff_of_lowest_test_cost).
- CSV only: test_cost_at_bayes_cutoff (cut-off 10/510); train_cost, train_fn, train_fp (the 60,000 training rows, scored for run 1 only).

Unrounded values: `rq1_transformer_2026_shipped_models_rescored.csv` in this folder.
