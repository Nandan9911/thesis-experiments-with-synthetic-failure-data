# TabICLv2 pilot on single folds

One TabICLv2 context per fold (all failures of the training fold and 2,500
random healthy rows, 1 estimator), seed 11, scored on the fold's
validation rows; the tuned LightGBM on the same rows for comparison. No test-set evaluation. Source:
results/rq2_survey_of_methods/tabicl_pilot.json (the LightGBM entries are stored under
'lightgbm_tuned_same_rows').

| Fold | Learner | Fold-optimal cost | FN / FP | AUC | AP | FP at FN<=2 | FP at FN<=5 | FP at FN<=10 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | TabICLv2 (pilot) | 7,000 | 6 / 400 | 0.986 | 0.854 | 3,738 | 1,659 | 302 |
| 0 | LightGBM (tuned) | 7,480 | 10 / 248 | 0.985 | 0.890 | 5,922 | 3,123 | 248 |
| 1 | TabICLv2 (pilot) | 6,580 | 6 / 358 | 0.995 | 0.844 | 936 | 549 | 272 |
| 1 | LightGBM (tuned) | 7,180 | 8 / 318 | 0.992 | 0.905 | 2,562 | 627 | 260 |

Columns:

- Fold-optimal cost, FN / FP: cost at the threshold that minimises it on the same validation rows (CSV: fold_opt_cost, fn, fp, threshold).
- AUC, AP: area under the ROC curve and average precision (CSV: auc, ap).
- FP at FN<=k: fewest false alarms with at most k missed failures (CSV: min_fp_at_fn2, _fn5, _fn10).
- CSV only: validation_rows, learner (key in the result file).

Unrounded values: `tabicl_pilot.csv` in this folder.
