# RQ3: completion benchmark

Medians over 5 folds of the benchmark records in
results/rq3_completion_and_cold_start/completion_benchmark.jsonl (one record per fold, mask and engine).
In each fold the engines are fitted on 800 real failures and complete the 200 held-out failures after the
mask has hidden part of their observed cells; all accuracy measures are on the hidden cells. The
classifier columns show whether the completed records still look like failures to the tuned LightGBM.

## 20 % of observed cells hidden at random (random_20_percent)

| Engine | NRMSE | MAE (log) | cover80 | width80 | flag: true | flag: masked | flag: completed | s / fold |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Median of the training failures | 1.06 | 1.98 | 0.88 | 6.15 | 0.94 | 0.94 | 0.97 | 0 |
| k-nearest neighbours | 0.65 | 0.96 | 0.96 | 6.20 | 0.94 | 0.94 | 0.96 | 0 |
| Chained equations (Bayesian ridge) | 0.51 | 0.84 | 0.90 | 3.06 | 0.94 | 0.94 | 0.96 | 28 |
| Unmasking Trees | 0.52 | 0.51 | 0.69 | 0.92 | 0.94 | 0.94 | 0.96 | 1,655 |
| Forest-Flow completer | 2.42 | 3.80 | 0.65 | 5.68 | 0.94 | 0.94 | 0.99 | 379 |
| TabPFN | 0.61 | 0.72 | 0.74 | 1.68 | 0.94 | 0.94 | 0.96 | 1,087 |

## 50 % hidden at random (random_50_percent)

| Engine | NRMSE | MAE (log) | cover80 | width80 | flag: true | flag: masked | flag: completed | s / fold |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Median of the training failures | 1.06 | 1.96 | 0.88 | 6.14 | 0.94 | 0.89 | 0.99 | 0 |
| k-nearest neighbours | 0.68 | 1.01 | 0.95 | 6.19 | 0.94 | 0.89 | 0.97 | 0 |
| Chained equations (Bayesian ridge) | 0.60 | 1.03 | 0.87 | 3.57 | 0.94 | 0.89 | 0.96 | 30 |
| Unmasking Trees | 0.55 | 0.61 | 0.71 | 1.24 | 0.94 | 0.89 | 0.97 | 3,106 |
| Forest-Flow completer | 2.01 | 3.35 | 0.72 | 6.37 | 0.94 | 0.89 | 1.00 | 318 |
| TabPFN | 0.85 | 1.28 | 0.62 | 2.30 | 0.94 | 0.89 | 0.97 | 1,120 |

## 80 % hidden at random (random_80_percent)

| Engine | NRMSE | MAE (log) | cover80 | width80 | flag: true | flag: masked | flag: completed | s / fold |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Median of the training failures | 1.04 | 1.94 | 0.88 | 6.15 | 0.94 | 0.41 | 1.00 | 0 |
| k-nearest neighbours | 0.69 | 1.10 | 0.95 | 6.26 | 0.94 | 0.41 | 0.98 | 0 |
| Chained equations (Bayesian ridge) | 0.81 | 1.52 | 0.81 | 4.47 | 0.94 | 0.41 | 0.98 | 31 |
| Unmasking Trees | 0.72 | 0.91 | 0.74 | 1.99 | 0.94 | 0.41 | 0.98 | 4,689 |
| Forest-Flow completer | 1.36 | 2.68 | 0.80 | 6.97 | 0.94 | 0.41 | 1.00 | 325 |
| TabPFN | 0.96 | 1.77 | 0.51 | 2.53 | 0.94 | 0.41 | 0.99 | 873 |

## real missingness patterns (>= 30 % missing) (real_patterns_30_percent)

| Engine | NRMSE | MAE (log) | cover80 | width80 | flag: true | flag: masked | flag: completed | s / fold |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Median of the training failures | 1.08 | 2.07 | 0.86 | 6.11 | 0.94 | 0.81 | 0.97 | 0 |
| k-nearest neighbours | 0.94 | 1.80 | 0.89 | 6.57 | 0.94 | 0.81 | 0.96 | 0 |
| Chained equations (Bayesian ridge) | 1.03 | 2.24 | 0.76 | 5.68 | 0.94 | 0.81 | 0.96 | 28 |
| Unmasking Trees | 0.95 | 1.44 | 0.72 | 2.82 | 0.94 | 0.81 | 0.97 | 1,938 |
| Forest-Flow completer | 2.38 | 3.51 | 0.70 | 6.69 | 0.94 | 0.81 | 0.97 | 289 |
| TabPFN | 1.04 | 2.14 | 0.52 | 2.97 | 0.94 | 0.81 | 0.96 | 831 |

Columns:

- NRMSE: root mean squared error in log1p space, per attribute divided by its standard deviation among the fitting failures (CSV: nrmse; cells used: n_hidden_informative of n_hidden_cells).
- MAE (log): mean absolute error in log1p space (CSV: mae_log).
- cover80: share of hidden cells inside the engine's 80 % interval (CSV: cover80).
- width80: mean width of the 80 % interval in log1p units (CSV: width80).
- flag: true / masked / completed: share of the 200 held-out failures flagged by the classifier as failures in their true, masked and completed form (CSV: clf_flag_true, clf_flag_masked, clf_flag_completed; mean probabilities clf_prob_*; the threshold clf_threshold).
- s / fold: seconds per fold for the completion step (CSV: seconds).
- CSV only: n_records (held-out failures per fold).

Unrounded values: `rq3_completion_benchmark.csv` in this folder.
