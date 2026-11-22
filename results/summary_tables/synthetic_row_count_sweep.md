# RQ1: synthetic-row count sweep of the cGAN of Ranasinghe et al.

The number of added cGAN rows varied from 250 to 5,000 as in the notebook attributed to the study of
Ranasinghe et al.; every count takes the first rows of one generated pool in which the two k-means clusters
alternate. OOF cost: 60,000 training rows, generator refitted inside each training fold, at the count's own OOF
threshold (the common evaluation protocol's way to choose a count); duplication control: the same number of
copied real failures. Test oracle: a generator fitted on all training failures, the lowest test cost over all
thresholds. Choosing the count by this number is test-set selection, simulated deliberately to measure its
effect: the minimum over the counts is the best of several test evaluations rather than an independent estimate
of the cost on new data. Source: results/rq1_cgan_2019_reproduction/synthetic_row_count_sweep.jsonl.

## cGAN of Ranasinghe et al., tuned LightGBM, seed 11: the OOF cost chooses 0 added rows; the lowest test oracle over the counts is 8,220 at 1,500 rows

| Rows added | OOF cost | Duplication control | Test oracle (full-data generator) |
| :--- | ---: | ---: | ---: |
| 0 | 33,290 | -- | 10,480 |
| 250 | 35,390 | 38,180 | 9,370 |
| 500 | 36,050 | 35,840 | 10,750 |
| 750 | 34,190 | 35,510 | 9,450 |
| 1,000 | 34,250 | 37,310 | 8,720 |
| 1,500 | 35,630 | 34,800 | 8,220 |
| 2,000 | 36,560 | 35,900 | 9,020 |
| 2,500 | 36,620 | 37,760 | 8,730 |
| 3,000 | 35,900 | 36,680 | 9,290 |
| 4,000 | 34,730 | 37,760 | 9,250 |
| 5,000 | 36,170 | 36,170 | 9,260 |

Columns:

- OOF cost: CSV: oof_cost, with oof_fn, oof_fp, oof_threshold and oof_cost_at_0_5.
- Duplication control: OOF cost with copied real failures (CSV: duplication_control_oof_cost, _fn, _fp).
- Test oracle (full-data generator): CSV: test_oracle_minimum_cost, with test_oracle_fn, test_oracle_fp; also test_cost_at_oof_threshold and test_cost_at_0_5.
- CSV only: selected_by_oof_cost and lowest_test_oracle_among_counts mark the two selections.

Unrounded values: `synthetic_row_count_sweep.csv` in this folder.
