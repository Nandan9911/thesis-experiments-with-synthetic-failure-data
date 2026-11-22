# RQ1b: retraining runs and test-inclusion controls

The released training code run from scratch on the published inputs (clean_seed<k>) and with half (8,000
rows, drawn at random) or all of the 16,000 test rows deliberately added to the training table (the
test-inclusion controls). All costs are on the official test set at the published 0.5 cut-off, from the test
trace (scored every 50 epochs, a diagnostic) and from final_evaluation.json. For the test-inclusion controls
the test set splits into the rows that were placed in the training table (seen) and the others (unseen).
Sources: authors_code/retraining_runs/<run>/ (test_trace.jsonl, final_evaluation.json, state.json). The
verification reruns repeat seed 0 for 300 epochs because GPU training is not bit-reproducible.

## Runs in the thesis table

| Run | Epochs | At 1,000 epochs (FN / FP) | At the last epoch (FN / FP) | Best-validation checkpoint (epoch) | Seen rows, last epoch | Unseen rows, last epoch |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Retraining, seed 0 | 8,000 | 18,120 (32 / 212) | 17,980 (32 / 198) | 19,920 (7,379) | -- | -- |
| Retraining, seed 1 | 5,950 | 18,570 (33 / 207) | 19,990 (36 / 199) | -- | -- | -- |
| Retraining, seed 2 | 1,000 | 18,460 (33 / 196) | 18,460 (33 / 196) | 18,120 (902) | -- | -- |
| Retraining, seed 3 | 1,000 | 17,060 (30 / 206) | 17,060 (30 / 206) | 19,800 (888) | -- | -- |
| Retraining, seed 4 | 1,000 | 18,240 (32 / 224) | 18,240 (32 / 224) | 17,210 (642) | -- | -- |
| Half of the test rows in training, seed 0 | 8,000 | 18,270 (35 / 77) | 15,220 (29 / 72) | 14,710 (6,010) | 3,070 (6 / 7) | 12,150 (23 / 65) |
| Half of the test rows in training, seed 1 | 1,000 | 13,040 (24 / 104) | 13,040 (24 / 104) | 13,060 (971) | 2,650 (5 / 15) | 10,390 (19 / 89) |
| Half of the test rows in training, seed 2 | 1,000 | 13,560 (25 / 106) | 13,560 (25 / 106) | 13,200 (899) | 1,740 (3 / 24) | 11,820 (22 / 82) |
| All test rows in training, seed 0 | 1,000 | 10,190 (20 / 19) | 10,190 (20 / 19) | 8,040 (875) | 10,190 (20 / 19) | none |

## Verification reruns of seed 0 (300 epochs, another session)

| Run | Epochs | At 1,000 epochs (FN / FP) | At the last epoch (FN / FP) | Best-validation checkpoint (epoch) | Seen rows, last epoch | Unseen rows, last epoch |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Retraining, seed 0, verification rerun | 300 | -- | 16,400 (28 / 240) | 14,980 (232) | -- | -- |
| Half of the test rows in training, seed 0, verification rerun | 300 | -- | 17,810 (33 / 131) | 14,640 (263) | 6,460 (12 / 46) | 11,350 (21 / 85) |

Columns:

- Run: design and seed of the run (CSV: run, the run name, for example clean_seed0).
- Epochs: epochs the run reached (CSV: epochs; kind says no, half or all of the test rows in training, or verification rerun).
- At 1,000 epochs (FN / FP): test cost in the trace at epoch 1,000 (CSV: test_*_at_1000_epochs).
- At the last epoch (FN / FP): last point of the trace (CSV: test_*_at_last_epoch, last_trace_epoch).
- Best-validation checkpoint (epoch): test cost of the checkpoint with the best validation metric, from final_evaluation.json; shown only when that evaluation covers all the epochs the run reached (CSV: best_checkpoint_test_cost, best_checkpoint_epoch, final_evaluation_epochs).
- Seen rows / Unseen rows, last epoch: cost on the test rows inside / outside the training table at the last trace point; 'none' when every test row was in the training table (CSV: seen_* and unseen_*, with the row counts).
- CSV only: final_evaluation_test_cost/_fn/_fp and final_evaluation_train_cost.

Unrounded values: `rq1_transformer_2026_retraining_and_contamination_runs.csv` in this folder.
