# RQ1b: retraining with the released code, no test rows in the training data

The authors' training code (TrainingCode.py, run through
experiments/rq1_transformer_2026_reproduction/authors_code/train_authors_model_with_resume.py) on the
published inputs, one run per seed, within a GPU time budget; the published runs trained for 8,000 epochs.
Sources: authors_code/retraining_runs/clean_seed<k>/ (state.json, final_evaluation.json, test_trace.jsonl,
training_log.csv). final_evaluation.json is written when a job ends; a run resumed for a longer budget that
stopped before writing a new one keeps the evaluation of its earlier stage (final_evaluation_epochs in the
CSV; here: clean_seed1 reached 5,950 epochs, its final evaluation is from epoch 1,000). The test trace scores the test set every 50 epochs
and is a diagnostic only.

| Seed | Epochs | GPU hours | Validation metric, min (epoch) | Validation metric, last | Train cost at 0.5 (FN / FP) | Test cost at 0.5, final model (FN / FP) | Test cost at 0.5, best-validation checkpoint (FN / FP; epoch) | Lowest test cost along training (epoch) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 8,000 | 49.1 | 5.0 (1,979) | 17.7 | 15,130 (16 / 713) | 17,980 (32 / 198) | 19,920 (36 / 192; 7,379) | 15,480 (150) |
| 1 | 5,950 | 48.6 | 13.4 (1,273) | 29.7 | 20,680 (27 / 718) | 18,570 (33 / 207) | 17,230 (30 / 223; 730) | 13,660 (350) |
| 2 | 1,000 | 5.1 | 38.3 (902) | 71.0 | 21,270 (29 / 677) | 18,460 (33 / 196) | 18,120 (32 / 212; 902) | 15,890 (350) |
| 3 | 1,000 | 5.7 | 13.1 (930) | 22.0 | 18,950 (23 / 745) | 17,060 (30 / 206) | 19,800 (36 / 180; 888) | 15,500 (300) |
| 4 | 1,000 | 5.5 | 42.2 (654) | 70.7 | 18,560 (21 / 806) | 18,240 (32 / 224) | 17,210 (30 / 221; 642) | 14,670 (150) |

Columns:

- Epochs: epochs the run reached (CSV: epochs, from state.json).
- GPU hours: estimated occupied GPU time, epochs x seconds per epoch between the first and last trace record (CSV: estimated_gpu_hours, seconds_per_epoch).
- Validation metric, min (epoch): lowest value of the authors' validation cost metric and its epoch, counted from 1 (CSV: validation_metric_minimum, _minimum_epoch).
- Validation metric, last: running median over the last 50 epochs of the validation metric at the end (CSV: validation_metric_last_running_median; the raw last value is validation_metric_last).
- Train cost at 0.5 (FN / FP): final model on the 60,000 training rows (CSV: train_cost, _fn, _fp).
- Test cost at 0.5, final model (FN / FP): final model on the test set (CSV: test_cost, _fn, _fp).
- Test cost at 0.5, best-validation checkpoint: the checkpoint with the best validation metric (CSV: best_checkpoint_*).
- Lowest test cost along training (epoch): lowest cost in the test trace, chosen with the test labels: a diagnostic (CSV: trace_lowest_test_cost, _epoch; the trace's last point is trace_last_test_cost, trace_last_epoch).

Unrounded values: `rq1_transformer_2026_clean_retraining_runs.csv` in this folder.
