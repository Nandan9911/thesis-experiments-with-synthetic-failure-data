# RQ3 cold start: test cost with few labelled failures

Median test cost over seeds (fold-ensemble predictions at the OOF threshold) when only the first n of a
fixed random order of the training failures are kept in each training fold (nested subsets), all healthy
rows are kept, and a generator fitted on those n failures adds 2,000 synthetic failure rows. Validation
folds and the test set are untouched real data. In brackets: difference from the median without synthetic
rows at the same n (negative is cheaper). Source:
results/rq3_completion_and_cold_start/cold_start_runs.jsonl.
Unmasking Trees inpainting has one seed at 100, 250 and 1,000 failures; every other cell has three seeds.

| Generator | 25 failures | 50 failures | 100 failures | 250 failures | 1,000 failures |
| :--- | ---: | ---: | ---: | ---: | ---: |
| No synthetic rows | 20,350 | 21,140 | 16,100 | 11,170 | 8,470 |
| Duplicated real failures | 16,910 (-3,440) | 14,380 (-6,760) | 10,890 (-5,210) | 10,730 (-440) | 8,850 (+380) |
| SMOTE | 15,840 (-4,510) | 15,080 (-6,060) | 12,910 (-3,190) | 10,930 (-240) | 9,130 (+660) |
| cGAN of Ranasinghe et al. | 15,430 (-4,920) | 14,030 (-7,110) | 10,980 (-5,120) | 11,180 (+10) | 9,720 (+1,250) |
| TabEBM | 16,950 (-3,400) | 14,840 (-6,300) | 11,790 (-4,310) | 11,210 (+40) | 9,380 (+910) |
| Forest-Flow | 15,800 (-4,550) | 14,770 (-6,370) | 12,830 (-3,270) | 11,380 (+210) | 9,540 (+1,070) |
| Forest-Flow inpainting | 15,360 (-4,990) | 14,180 (-6,960) | 13,020 (-3,080) | 12,260 (+1,090) | 10,480 (+2,010) |
| Unmasking Trees inpainting | 16,100 (-4,250) | 13,270 (-7,870) | 15,290 (-810) | 11,490 (+320) | 8,640 (+170) |

Columns:

- n failures: labelled failures available in each training fold (CSV: labelled_failures).
- Cells: median test cost (CSV: test_cost) and difference from no synthetic rows (CSV: test_cost_difference_from_no_synthetic_rows).
- CSV only: oof_cost_scaled_to_test_set_size (median OOF cost scaled to the 16,000 rows of the test set), seeds.

Unrounded values: `rq3_cold_start_test_costs.csv` in this folder.
