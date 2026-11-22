# First-look baselines with fixed settings

The first models of the project, run before any tuning: three gradient-boosting libraries with the fixed
settings of the factories in experiments/common/classifiers.py (for LightGBM and XGBoost 600 trees at
learning rate 0.03), which are not the libraries' own defaults. Each configuration is evaluated by
stratified 5-fold cross-validation on the 60,000 training rows with seed 11, at the threshold that
minimises its out-of-fold cost; the test set is not used. Source:
results/preliminary_checks/quick_baseline.jsonl (written by experiments/quick_baseline_check.py).

| Configuration | OOF cost (FN / FP) | Cost scaled to the 16,000 rows of the test set | Threshold |
| :--- | ---: | ---: | ---: |
| LightGBM, fixed settings (600 trees) | 34,100 (30 / 1,910) | 9,093 | 1.8e-03 |
| LightGBM, fixed settings, failure weight 30 | 34,550 (35 / 1,705) | 9,213 | 6.1e-03 |
| XGBoost, fixed settings (600 trees) | 33,620 (32 / 1,762) | 8,965 | 4.6e-03 |
| HistGradientBoosting, fixed settings (at most 600 iterations) | 37,070 (37 / 1,857) | 9,885 | 7.1e-03 |

Columns:

- Configuration: learner and settings; failure weight 30 is LightGBM's scale_pos_weight (CSV: label; configuration is the name in the result file).
- OOF cost (FN / FP): cost of the 60,000 out-of-fold predictions at the cost-minimising threshold, with its missed failures and false alarms (CSV: oof_cost, oof_fn, oof_fp).
- Cost scaled to the 16,000 rows of the test set: the OOF cost x 16,000 / 60,000 (CSV: oof_cost_scaled_to_test_set_size).
- Threshold: the cost-minimising out-of-fold threshold (CSV: oof_threshold).
- CSV only: fold_costs (cost of each validation fold at that threshold), fp_at_exactly_<k>_fn (fewest false alarms with exactly k = 10, 20, 30, 40 or 60 missed failures; empty when no candidate threshold gives exactly k) and fit_seconds.

Unrounded values: `quick_baseline_check.csv` in this folder.
