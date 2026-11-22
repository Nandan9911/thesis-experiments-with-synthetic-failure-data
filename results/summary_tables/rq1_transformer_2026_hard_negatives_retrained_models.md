# RQ1b: hard-negative comparison for the retrained models (% flagged at 0.5)

The comparison of rq1_transformer_2026_hard_negatives_shipped_models repeated with the models retrained from
scratch (final epoch; epochs per run in rq1_transformer_2026_epochs_reached): the percentage of each group of
negatives with probability of at least 0.5, on the same masks (RENN kept or removed; hard by the one-shot rule).
For the test-inclusion controls the test negatives split into those in the training table (seen) and the rest
(unseen). Source: hard_negative_check_retrained_models.json (written by
hard_negative_check_retrained_models.py).

| Group of negatives | Retraining, seed 0 | Retraining, seed 1 | Retraining, seed 2 | Retraining, seed 3 | Retraining, seed 4 | Half, seed 0 | Half, seed 1 | Half, seed 2 | All, seed 0 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Training negatives kept by RENN (trained on) | 0.0 | 0.1 | 0.0 | 0.1 | 0.1 | 0.0 | 0.1 | 0.0 | 0.0 |
| Training negatives removed by RENN (never seen) | 37.5 | 35.4 | 35.0 | 37.9 | 39.6 | 26.6 | 28.8 | 27.5 | 22.2 |
| of which hard by the one-shot rule | 62.1 | 57.8 | 58.5 | 63.0 | 65.0 | 45.4 | 48.2 | 47.4 | 37.4 |
| Test negatives hard by the one-shot rule (never seen) | 46.5 | 47.0 | 46.5 | 47.0 | 49.9 | -- | -- | -- | -- |
| Other test negatives (never seen) | 0.1 | 0.2 | 0.1 | 0.2 | 0.2 | -- | -- | -- | -- |
| Test negatives hard by the one-shot rule, seen | -- | -- | -- | -- | -- | 2.9 | 7.3 | 12.6 | 5.0 |
| Test negatives hard by the one-shot rule, unseen | -- | -- | -- | -- | -- | 34.5 | 43.1 | 39.2 | -- |
| Other test negatives, seen | -- | -- | -- | -- | -- | 0.0 | 0.0 | 0.0 | 0.0 |
| Other test negatives, unseen | -- | -- | -- | -- | -- | 0.1 | 0.1 | 0.1 | -- |

Columns:

- Group of negatives: CSV: group; source_key is the group's key in the result file.
- Retraining, seed 0 ... 4: clean_seed0 ... clean_seed4 (CSV column names are the run names).
- Half, seed 0 ... 2: half_test_rows_in_training_seed0 ... seed2 (8,000 test rows in the training table).
- All, seed 0: all_test_rows_in_training_seed0 (every test row in the training table).
- Values: percent of the group flagged; the CSV holds shares between 0 and 1.

Unrounded values: `rq1_transformer_2026_hard_negatives_retrained_models.csv` in this folder.
