# Synthetic-data diagnostics per generator

Each generator fitted on the 800 real failures of the training part of one fold
(seed 11, fold 0), and its synthetic rows compared with these fitting failures. The two reference rows are
compared with the fitting failures in the same way: the 200 real failures
held out from that fold (what an ideal generator would reach; they are also the non-members of the
membership-inference statistic mia_auc in the CSV) and the duplication control, copies of fitting failures
(pure copying). From every generator 2,000 rows were sampled; the duplication control comes from a run with
300 copies, as does the memorisation reference of the file (reference_duplicates_memorisation). The statistics
are defined in experiments/common/synthetic_data_diagnostics.py; no test rows are involved. Source:
results/rq2_survey_of_methods/synthetic_data_diagnostics.json.

| Rows | Detectability AUC | same, missing values filled | Relative NN distance | Share closer | Marginal W1 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Real held-out failures (reference) | 0.49 | 0.49 | 0.95 | 0.04 | 0.010 |
| Duplication control: copied real failures (reference) | 0.36 | 0.36 | 0.00 | 1.00 | 0.007 |
| cGAN (min-max inputs, single sigmoid), with missingness masks | 1.00 | 1.00 | 1.00 | 0.00 | 0.022 |
| cGAN (min-max inputs, single sigmoid), no masks | 1.00 | 1.00 | 0.95 | 0.00 | 0.019 |
| cGAN with the notebook's settings | 1.00 | 1.00 | 1.06 | 0.00 | 0.073 |
| Forest-Flow, with missingness masks | 1.00 | 1.00 | 0.87 | 0.00 | 0.042 |
| SMOTE | 1.00 | 0.93 | 0.78 | 0.00 | 0.012 |
| Borderline-SMOTE | 1.00 | 0.98 | 0.55 | 0.00 | 0.044 |
| ADASYN | 1.00 | 0.96 | 0.64 | 0.00 | 0.035 |
| SVM-SMOTE | 1.00 | 0.97 | 0.62 | 0.00 | 0.036 |
| Gaussian copula | 1.00 | 1.00 | 1.79 | 0.00 | 0.021 |
| TVAE | 1.00 | 1.00 | 1.43 | 0.00 | 0.016 |
| CTGAN | 1.00 | 1.00 | 6.03 | 0.00 | 0.057 |
| CTAB-GAN+ | 1.00 | 1.00 | 1.89 | 0.00 | 0.025 |
| TabDDPM | 1.00 | 1.00 | 1.36 | 0.00 | 0.015 |
| TabEBM | 1.00 | 1.00 | 1.20 | 0.00 | 0.009 |

Columns:

- Detectability AUC: out-of-fold AUC of a LightGBM trained to tell the synthetic rows from the fitting failures, missing values as they are (CSV: detect_auc; 0.5 = indistinguishable).
- same, missing values filled: the same after filling the missing cells of both sets with the medians of the fitting failures (CSV: detect_auc_nomask; the held-out reference row has no such value, and its value with missing values as they are is repeated, as in the thesis).
- Relative NN distance: median distance of a synthetic row to its nearest fitting failure, divided by the fitting failures' median leave-one-out nearest-neighbour distance (CSV: dcr).
- Share closer: share of synthetic rows closer to a fitting failure than the 5th percentile of the fitting failures' leave-one-out nearest-neighbour distances (CSV: frac_closer).
- Marginal W1: mean over the attributes of the 1-Wasserstein distance between the fitting failures' and the synthetic values, min-max scaled (CSV: colwise_w1).
- CSV only: mia_auc, near_copy_frac and dcr_fit_vs_heldout (memorisation statistics, where computed).

Unrounded values: `synthetic_data_diagnostics.csv` in this folder.
