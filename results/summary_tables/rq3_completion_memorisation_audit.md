# RQ3: memorisation audit of the completion engines

One cell of the completion benchmark repeated with the completed records kept: seed 11,
fold 0, 50 % of the observed cells of the
200 held-out failures hidden at random, every engine fitted on the 800 fitting failures. Distances are
measured in the min-max scaled space of the fitting failures; the measures are defined in
experiments/common/synthetic_data_diagnostics.py. Source:
results/rq3_completion_and_cold_start/completion_memorisation_audit.json.

| Records | Relative NN distance | Share closer | Near copies |
| :--- | ---: | ---: | ---: |
| Real held-out failures (reference) | 1.03 | 0.03 | 0.09 |
| Median of the training failures | 1.18 | 0.00 | 0.00 |
| k-nearest neighbours | 1.09 | 0.00 | 0.00 |
| Chained equations (Bayesian ridge) | 2.02 | 0.00 | 0.00 |
| Unmasking Trees | 1.21 | 0.00 | 0.00 |
| Forest-Flow completer | 4.96 | 0.00 | 0.00 |
| TabPFN | 1.13 | 0.00 | 0.00 |

Columns:

- Relative NN distance: median distance of a completed record to its nearest fitting failure, divided by the fitting failures' median leave-one-out nearest-neighbour distance (CSV: dcr).
- Share closer: share of completed records closer to a fitting failure than the 5th percentile of the fitting failures' leave-one-out nearest-neighbour distances (CSV: frac_closer).
- Near copies: share of completed records whose nearest fitting failure is closer than 5 % of the fitting failures' median leave-one-out nearest-neighbour distance (CSV: near_copy_frac).
- CSV only: records (key in the result file), mia_auc, dcr_fit_vs_heldout, median_nn_real, median_nn_syn_to_real (the two medians behind dcr) and seconds.

Unrounded values: `rq3_completion_memorisation_audit.csv` in this folder.
