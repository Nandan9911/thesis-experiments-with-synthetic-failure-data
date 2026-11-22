# Decision-rule demonstration with the learners of the earlier phase of the project

Each learner trained once on 80 % of the training rows (missing values set to 0, as the notebooks of the
earlier phase of the project did); the threshold is chosen on the remaining 20 % under four rules and the
official test set is scored under each: test cost (missed failures / false alarms). The oracle and the
false-alarm count use the test labels and are diagnostics. Source:
results/preliminary_checks/threshold_rule_demonstration.json.

| Learner | Linear grid from 0.01 | All validation scores | Fixed 0.5 | Bayes 0.0196 | Oracle | FP at FN<=6 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| sklearn GBM (default: 100 trees, depth 3) | 14,390 (20 / 439) | 10,820 (8 / 682) | 63,870 (127 / 37) | 16,560 (26 / 356) | 10,880 | 834 |
| Random forest (300 trees, depth 20, balanced_subsample) | 10,220 (8 / 622) | 9,960 (7 / 646) | 46,820 (91 / 132) | 9,420 (2 / 842) | 8,990 | 647 |
| LightGBM (500 trees, defaults) | 28,900 (57 / 40) | 13,460 (21 / 296) | 45,650 (91 / 15) | 30,800 (61 / 30) | 10,140 | 903 |

Columns:

- Learner: the learner and its settings (CSV: learner, the name in the result file).
- Linear grid from 0.01: best of 0.01, 0.02, ..., 0.99 on the validation rows (CSV: linear_0.01_*; _validation_cost is the cost on the 20 % validation rows).
- All validation scores: best of every distinct validation score (CSV: all_values_*).
- Fixed 0.5: the default cut-off (CSV: fixed_0.5_*).
- Bayes 0.0196: the cut-off 10/510 (CSV: bayes_0.0196_*).
- Oracle: lowest test cost over all thresholds (CSV: oracle_test_cost, oracle_threshold).
- FP at FN<=6: fewest test false alarms with at most 6 missed failures (CSV: min_fp_at_most_6_fn).

Unrounded values: `threshold_rule_demonstration.csv` in this folder.
