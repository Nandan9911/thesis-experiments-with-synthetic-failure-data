# Results

Everything the experiments of the thesis produced, organised like `experiments/`. The files were written by the
scripts named below; nothing here was edited by hand except where a note says so. The tables and figures in
`summary_tables/` and `figures/` are derived from the other files by `python -m analysis.run_all_summaries`.

Sizes: about 80 MB in total, most of it the saved probability vectors (`.npy` files: NumPy arrays, mostly float64,
that load with `numpy.load`; `hard_negative_masks.npy` holds a dictionary and needs
`numpy.load(path, allow_pickle=True).item()`).

```
data_audit.json                      checks of the two data files
hyperparameter_search/               the Optuna search that tuned LightGBM
preliminary_checks/                  the first look at the data and the threshold-rule demonstration
classifier_arms/                     every arm of RQ1 and RQ2: one record per arm and seed, and the probabilities
rq1_cgan_2019_reproduction/          the synthetic-row count sweep
rq1_transformer_2026_reproduction/   the study of Beikmohammadi et al.: the released models and code, the PyTorch port,
                                     hard negatives
rq2_survey_of_methods/               diagnostics of the synthetic rows, pilots
rq3_completion_and_cold_start/       completion benchmark, memorisation audit, prototype, cold start
summary_tables/                      tables computed by analysis/ (CSV with full precision, Markdown for reading)
figures/                             figures computed by analysis/
```

Costs are always 500 × missed failures + 10 × false alarms. `fn`, `fp`, `tp` and `tn` are the cells of the
confusion matrix (false negatives = missed failures, false positives = false alarms). Out-of-fold costs are on the
60,000 training rows unless a field says it is scaled to the 16,000-row test set.

## data_audit.json

Written by `python -m experiments.common.data_loading`. SHA-256 checksums of the two CSV files, row and failure
counts, the share of missing cells overall and in failure and healthy rows, attributes with more than 50 % and 70 %
missing values, constant attributes (`cd_000`), the number of rows shared by both files (0), and two sanity checks
of the cost function (the cost of calling every test row healthy, 187,500, and of calling every row a failure,
156,250).

## hyperparameter_search/

Written by `python -m experiments.hyperparameter_search --model lightgbm --seed 11`.

* `lightgbm_seed11_trials.csv`: one row per Optuna trial: trial number, out-of-fold cost at the best threshold
  (`cost_oof`), `fn`, `fp`, `threshold`, run time, and the sampled hyperparameters.
* `lightgbm_seed11_best_parameters.json`: the best trial's parameters (`best_params`), its out-of-fold cost, the
  number of trials completed within the 55-minute limit (26), the seed, the number of folds and CPU threads. Every
  arm marked `use_tuned_hyperparameters` reads LightGBM's settings from this file.

## preliminary_checks/

* `quick_baseline.jsonl` and `quick_baseline_<configuration>_out_of_fold.npy`: written by
  `python -m experiments.quick_baseline_check`. One record per configuration (LightGBM, LightGBM with
  `scale_pos_weight` 30, XGBoost and scikit-learn's histogram gradient boosting, all with the fixed settings of
  `classifiers.py`): the out-of-fold cost at the best threshold with its confusion matrix, the per-fold costs, the
  cost scaled to the test-set size, the run time, and `fp_at_fn`, the fewest false alarms needed to keep the missed
  failures at exactly 10, 20, 30, 40 or 60.
* `threshold_rule_demonstration.json`: written by `python -m experiments.run_threshold_rule_demonstration`. For each
  of the learners of the earlier phase of the project, the test cost under four threshold rules (a 0.01 grid, every
  distinct validation score, the fixed 0.5, and the cost-optimal cut-off 10/510 for calibrated probabilities) and
  the test oracle.

## classifier_arms/

Written by `python -m experiments.run_experiment_arm` (and, for the four arms on the preprocessing pipeline of
Beikmohammadi et al., by
`python -m experiments.rq1_transformer_2026_reproduction.run_tree_models_on_2026_preprocessing`).

`arm_records.jsonl` holds one JSON record per arm and seed (94 records). If an arm and seed occur more than once,
the analysis uses the last one. The fields:

| Field | Content |
|---|---|
| `arm` | the arm's name (see `experiments/experiment_arms.py` and the table at the end of this file) |
| `seed`, `folds` | the random seed and the number of cross-validation folds |
| `classifier`, `classifier_parameters` | the classifier and the parameters it was built with (the tuned LightGBM settings where they apply) |
| `preprocessing`, `augmentation` | what was done to the training folds |
| `arm_options` | the arm's full definition as it was run |
| `out_of_fold` | the result on the 60,000 out-of-fold predictions at the chosen threshold: `cost`, `fn`, `fp`, `tp`, `tn`, `recall`, `precision`, `threshold`, `fold_costs` (the cost of each fold at that threshold, used for the paired tests), `cost_scaled_to_test_set_size`, `fit_seconds_total` |
| `test` | the one evaluation on the 16,000 test rows (average of the five fold models, same threshold): the confusion matrix, `cost`, `threshold` and `cost_95_percent_interval` (bootstrap, 2,000 resamples of the test rows) |
| `test_oracle_minimum_cost` | the lowest test cost any threshold could have reached; a diagnostic, never used for a choice |
| `test_full_refit`, `test_full_refit_oracle_minimum_cost` | the secondary model refitted on all training rows, where it was run |
| `cross_validation_seconds`, `final_test_seconds` | run times |
| `source` | where the run was made (`colab:<session>` for Colab sessions; absent for the CPU machine) |
| `test_at_0.5`, `imputer_rounds` | only in the records of the 2026-preprocessing arms: the test result at threshold 0.5, and the number of imputer rounds |

`predicted_probabilities/<arm>/seed<k>_out_of_fold.npy` (60,000 values in the row order of the training file),
`seed<k>_test.npy` (16,000 values in the row order of the test file) and, where it exists,
`seed<k>_test_full_refit.npy` are the probabilities behind each record. All post-hoc diagnostics in the summary
tables (costs at other thresholds, false alarms at a given number of misses) are computed from them.

`verification_reruns.jsonl` holds reruns of some arms made in a Colab session as a check of repeatability (same
format). They are not used in any table; the thesis discusses them (for example, the cGAN arm's seed 11 costs
9,260 on the test set in the rerun against 9,500 in the main run).

Note on `xgboost_fixed_settings`: its records show `use_tuned_hyperparameters: true` in `arm_options` because the
arm was defined that way when it was run. No hyperparameter search was run for XGBoost, so no tuned settings
existed and the fixed settings of `classifiers.py` were used, as the empty `classifier_parameters` show. The
current definition, with `use_tuned_hyperparameters: false`, builds the same model.

## rq1_cgan_2019_reproduction/

`synthetic_row_count_sweep.jsonl` and `synthetic_row_count_sweep_probabilities/`: written by
`python -m experiments.rq1_cgan_2019_reproduction.run_synthetic_row_count_sweep --generator conditional_gan_2019
--model lightgbm --seeds 11 --gan-training-steps 30000`.
For each number of added cGAN rows (0 to 5,000) the out-of-fold result (with the duplication control at the same
count) and the test result; `out_of_fold_selected_count` is the count the protocol would choose,
`test_selected_minimum_oracle_cost` the lowest test cost over all counts, which is what choosing the count on the
test set would report. Every count takes the first rows of one generated pool in which the two k-means clusters
alternate (`pool_cluster_order`), so every count has the same number of rows from each cluster.

## rq1_transformer_2026_reproduction/

**`authors_code/`**: results of the scripts in `experiments/rq1_transformer_2026_reproduction/authors_code/` (see the
README there for the order in which they were run).

* `published_training_logs/run1.csv` … `run5.csv`: the authors' own per-epoch training logs, copied from their
  repository.
* `shipped_models_five_round_tables/`: test probabilities of the five released models on input tables made with an
  imputer capped at five rounds (the first, approximate re-scoring), and, for model 1 only, with a scaler fitted on
  the raw columns.
* `shipped_models_exact_inputs/`: the re-scoring on the exactly regenerated inputs: the evaluation of the five
  models with the scaler fitted on the resampled table as published (`evaluation_scaler_on_resampled.csv`, the
  3,440 of the paper) and on the imputed table (`evaluation_scaler_on_imputed.csv`), the five-round tables with the
  published scaler (`evaluation_five_round_tables_scaler_on_resampled.csv`), the test probabilities of each model,
  the masks of hard negatives (`hard_negative_masks.npy`) and the hard-negative comparison for the released models
  (`hard_negative_check.json`, `hard_negative_check_followup.json`) and for the thesis's own tree models
  (`hard_negative_check_tree_control.json`). `colab_cpu_session_check/` repeats the published-scaler evaluation in a
  Colab session, with the same result.
* `retraining_runs/<run>/`: the released training code run from scratch. Runs: `clean_seed0` … `clean_seed4`
  (published inputs, no test rows in the training table); test-inclusion controls with half of the test rows in the
  training table (`half_test_rows_in_training_seed0` … `seed2`) or all of them (`all_test_rows_in_training_seed0`);
  and two verification runs that repeat a retraining run and a control run with half of the test rows in the training
  data. Each folder has the training log (`training_log.csv`, the authors' CSV format), the official test cost traced
  every 50 epochs (`test_trace.jsonl`), the last epoch reached (`state.json`), the final evaluation
  (`final_evaluation.json`; the two runs extended to 8,000 epochs also keep the evaluation of their 1,000-epoch stage as
  `final_evaluation_after_1000_epochs.json`), the test probabilities of the final and of the best-validation model, the
  training probabilities of the final model, and the indices of the test rows placed in the training table
  (`contaminating_test_rows.npy`, empty for the runs without test rows).
* `logs/`: console output of the authors-code scripts (imputation, preparation of the inputs, evaluations, timing of
  one epoch, the scaler attribution run).

**`pytorch_port/`**: written by `python -m experiments.rq1_transformer_2026_reproduction.run_transformer_2026_port`. For
each run (`corrected_protocol_seed0` … `seed2` under the protocol with a validation split of real rows,
`published_protocol_seed0`/`seed1` and the three ablations without RENN, without SVM-SMOTE and without resampling) a
JSON file with the arguments, the run times, the training history (training loss and validation cost, false negatives
and false positives for every epoch), the best validation epoch, and the test result of the best checkpoint and of the
final epoch, each at threshold 0.5 and at the validation-chosen threshold; plus the test probabilities (not kept for
`published_protocol_without_svm_smote_seed1`). `corrected_protocol_extended_seed0*` is the offline evaluation of seed
0's checkpoints after the run was extended to 3,250 epochs (`evaluate_port_checkpoint.py`).

**`hard_negative_check_retrained_models.json`**: written by
`python -m experiments.rq1_transformer_2026_reproduction.hard_negative_check_retrained_models`: for each retraining
run, the share of each group of healthy rows (training rows that RENN kept or removed, hard and other test rows)
that the model flags as failures.

## rq2_survey_of_methods/

* `synthetic_data_diagnostics.json`: written by
  `python -m experiments.rq2_survey_of_methods.run_synthetic_data_diagnostics`. Each generator is fitted on the 800
  training failures of fold 0 (seed 11), and its rows are compared with those 800 failures; the 200 held-out failures
  of the fold serve as the reference (`reference_real_heldout`) and as the non-members of the membership test:
  `detect_auc` (how well a LightGBM tells the synthetic rows from real ones; 0.5 = not at all, 1.0 = perfectly),
  `detect_auc_nomask` (the same with the missing values filled, so the missingness pattern cannot help),
  `detect_auc_rounded` (after rounding to integers), `dcr` (median distance of a synthetic row to its closest real
  failure, relative to the real rows' own nearest-neighbour distance; values far below 1 indicate copying),
  `frac_closer` (share of synthetic rows closer to a real row than the real rows' 5th-percentile distance),
  `colwise_w1` (mean 1-Wasserstein distance between real and synthetic marginals on min-max scaled values), the
  shares of missing, zero and integer values in real and synthetic rows, and the run time. `reference_real_heldout`
  compares the held-out real failures in the same way (what indistinguishable rows look like).
  *Provenance note:* the file was filled by several runs, generator by generator. The entries
  `duplicated_real_failures` and `reference_duplicates_memorisation` and the top-level `n_syn` come from a run with
  300 copied rows; all other generators were sampled 2,000 times. With 2,000 copies the duplicated rows give a
  detection AUC of 0.16 instead of 0.36 and a Wasserstein distance of 0.003 instead of 0.007; the thesis table uses
  the 300-row values for this reference row. The conclusion (copies cannot be told apart and sit at distance zero)
  is the same either way.
* `tabicl_pilot.json` and `tabicl_pilot_probabilities/`: the two-fold CPU pilot of TabICLv2 with 2,500 healthy rows
  per context, against the tuned LightGBM on the same folds.
* `forest_flow_timing_pilot.log`, `tabebm_timing_pilot.log`: console output of the timing pilots, which measured fit
  and sampling times before the full runs.

## rq3_completion_and_cold_start/

* `completion_benchmark.jsonl`: written by `run_completion_benchmark.py`, one record per fold, mask and engine:
  `n_records` and `n_hidden_cells` (how many failure records and hidden cells were completed), `nrmse` (root mean
  squared error of the completed cells in log(1 + x) space, each error divided by the standard deviation of its
  attribute among the fitting failures, so that always predicting the median scores about 1; attributes that hardly vary
  are left out, and `n_hidden_informative` counts the cells that remain), `mae_log` (mean absolute error in log space
  over all hidden cells), `cover80` and `width80` (how often the true value lies inside the engine's 80 % interval, and
  the interval's width), and the classifier check: with the tuned LightGBM at its out-of-fold threshold
  (`clf_threshold`), the mean predicted probability (`clf_prob_*`) and the share flagged as failures (`clf_flag_*`) for
  the true record, the masked record and the completed record. Masks: `random_20_percent`, `random_50_percent`,
  `random_80_percent` (cells hidden completely at random) and `real_patterns_30_percent` (the missingness pattern of a
  real record, drawn from the first 20,000 healthy training rows that miss at least 30 % of their attributes).
* `completion_memorisation_audit.json`: written by `run_completion_memorisation_audit.py`: for each engine, whether
  its completions of half-hidden failures sit closer to the rows it was fitted on than real held-out failures do
  (`dcr`, `frac_closer`, `near_copy_frac`, and `mia_auc`, the AUC of a distance-based membership test).
* `cold_start_runs.jsonl`: written by `run_cold_start_experiment.py`, one
  record per generator, seed and number of labelled failures (`labelled_failures`, 25 to 1,000), with the number of
  synthetic rows added (`synthetic_rows`), the out-of-fold result and the test result (as in the arm records, plus
  the fewest false alarms at no more than 15 out-of-fold or 6 test misses, and the test oracle). The records of the
  two later Unmasking Trees runs on seeds 29 and 47 (25 and 50 failures), first written to separate files, are
  included. `cold_start_test_probabilities/` holds the test probabilities of those four runs only; the earlier runs
  did not save them.
* `prototype_demonstration.json` and `prototype_demonstration_output.txt`: the prototype on training failure no. 7
  with half of its observed cells hidden (Unmasking Trees, on an L4 GPU): the probabilities of the tuned LightGBM for
  the true, the partial and the two completed records, and the printed output. The header of the text file gives
  the command; it was added when the code was reorganised.

## summary_tables/ and figures/

Written by `python -m analysis.run_all_summaries`. Each Markdown table starts with a short description of what it
shows and where its numbers come from; the CSV file of the same name has the full-precision values and the extra
columns.

| Table | Content |
|---|---|
| `classifier_arms_main_results` | the 20 arms of the thesis's main results table (the references, the RQ1 arms, Forest-Flow, TabEBM, the learner arms and the preprocessing pipeline of Beikmohammadi et al.): medians over seeds of the out-of-fold and test costs; the other survey arms are in `rq2_survey_against_tuned_lightgbm` |
| `classifier_arms_per_seed` | every arm and seed, with all diagnostics |
| `classifier_arms_decision_rules` | seed 11: test cost at 0.5, at 10/510, at the out-of-fold threshold, with the plateau rule, and the oracle |
| `classifier_arms_paired_comparisons` | RQ1 arms against their matched baselines: corrected t-test and Wilcoxon test |
| `rq2_survey_against_tuned_lightgbm` | RQ2: every survey arm against the tuned LightGBM, Holm-adjusted, with the tier |
| `classifier_arms_runtimes` | run time per arm and seed |
| `rq1_transformer_2026_shipped_models_rescored` | the five released models on the four input variants |
| `rq1_transformer_2026_clean_retraining_runs` | the released code retrained on the published inputs (the retraining runs), one row per seed |
| `rq1_transformer_2026_retraining_and_contamination_runs` | all retraining runs including the test-inclusion controls |
| `rq1_transformer_2026_pytorch_port_runs` | the PyTorch port and its ablations |
| `rq1_transformer_2026_epochs_reached` | the epochs reached by every retraining run and every run of the PyTorch port |
| `rq1_transformer_2026_hard_negatives_shipped_models`, `..._retrained_models` | the hard-negative comparison |
| `rq1_transformer_2026_summary_numbers` | means, ranges and totals of the RQ1b runs quoted in the thesis text |
| `rq3_completion_benchmark`, `rq3_completion_memorisation_audit`, `rq3_cold_start_test_costs` | RQ3 |
| `quick_baseline_check` | the first out-of-fold look at three gradient-boosting libraries with fixed settings |
| `data_audit`, `hyperparameter_search_trials`, `threshold_rule_demonstration`, `tabicl_pilot`, `synthetic_data_diagnostics`, `synthetic_row_count_sweep` | the supporting results |

The figures show the out-of-fold cost against the threshold, the false-alarm frontier on the test set, the paired
differences of the RQ1 and RQ2 arms, the authors' training logs and the test traces of the retraining runs, the
completion benchmark, the cold-start curves, an overview of the data, the history of the hyperparameter search and
the count sweep.

## Old names inside stored files

The results were renamed with the code before publication; the computations are unchanged. A few records and logs still
contain names from before the renaming, because they are kept exactly as they were written; section 9 of the main
[README](../README.md) lists them with their present names.
