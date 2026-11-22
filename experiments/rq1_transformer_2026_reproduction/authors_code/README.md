# The code released by Beikmohammadi et al.: re-scoring the released models and retraining with the released code

This folder belongs to research question 1 of the thesis, the reproduction of Beikmohammadi et al., *A Cost-Sensitive
Transformer Model for Prognostics Under Highly Imbalanced Industrial Data* (Cluster Computing, 2026). The study
reports a mean cost of 3,440 on the official APS test set (500 per missed failure plus 10 per false alarm; on average
6.8 missed failures and 4.0 false alarms over five models). The scripts here work with what the authors released:
their public repository, the five trained Keras models included in it (the released models), and their training
code. This release is what made the examination below possible. The PyTorch port of the network and the tree models
on the same preprocessing are in the parent folder.

What was done, in order:

1. First re-scoring, approximate inputs. The repository names the intermediate tables of the authors'
   preprocessing (missing values filled by an iterative imputer, SVM-SMOTE and RENN resampling, min-max scaling) but
   does not contain them. They were first regenerated within the compute available at the time, with two deviations:
   the imputer was capped at five rounds instead of up to 100, and the scaler was fitted on the imputed training
   table instead of the resampled table. On these tables all five released models missed 358 to 369 of the 375 test
   failures.
2. Exact inputs. `prepare_authors_inputs.py` then regenerated the authors' inputs exactly (100 imputer rounds,
   SVM-SMOTE and RENN, the scaler fitted on the resampled table). On these tables the five released models reproduce
   the published cost of 3,440 (mean 6.8 missed failures, 4.0 false alarms).
3. Attribution. `run_scaler_attribution.sh` crosses the two deviations of step 1. With the exact tables but the
   scaler fitted on the imputed table the models miss on average 284.6 of the 375 failures; with the five-round
   tables and the published scaler source, 363.8. Each deviation on its own is enough for the models to miss most
   failures.
4. Retraining with the released code. `train_authors_model_with_resume.py` runs the authors' training code on the
   exact inputs, with checkpoints so that runs of thousands of epochs survive the end of a Colab session. The runs
   were made on Colab GPUs through `cloud_runs/`: retraining runs with the authors' seeds 0 to 4, with no test rows in
   the training table, and test-inclusion controls, in which a known share of the official test rows (half for seeds 0,
   1 and 2; all for seed 0) is deliberately included in the training table, to show how a model scores test rows that
   were part of its training data. Every 50 epochs each run scores the test set at the published 0.5 cut-off; this trace
   is a diagnostic and selects nothing. Within the compute available, the retrained runs did not reach the published
   cost: at 1,000 epochs, the horizon every main run reached, the five retraining runs cost 17,060 to 18,570; the
   retraining run of seed 0 at 8,000 epochs costs 17,980.
5. Hard-negative comparison. RENN removed from the published training table the healthy rows whose three nearest
   neighbours are not all healthy, so the models were not trained on these "hard negatives". Applied once to the
   test set, the same rule marks 381 of the 15,625 healthy test trucks as hard in the same sense; none of them is in
   the training file. The comparison counts how often the models flag the two groups at the 0.5 cut-off. The released
   models flag 37 % of the hard training negatives that RENN removed and 0.9 % of the hard test negatives; the
   models retrained without test rows 58 to 65 % and 46 to 50 %; the tree models of the thesis, as a control, 64 to 78 %
   and 61 to 72 %. The test-inclusion controls flag 2.9 to 12.6 % of the hard test negatives that were in their training
   table and 34 to 43 % of those that were not. The comparison is a diagnostic: it uses the test labels only to form the
   groups and selects nothing.

## Requirements

* TensorFlow with tf-keras (the released models are Keras 2 SavedModel folders; the scripts load them with
  `tf_keras`, and `train_authors_model_with_resume.py` sets `TF_USE_LEGACY_KERAS=1`), the `focal-loss` package for
  training, and scikit-learn, imbalanced-learn and pandas.
* The authors' repository, https://github.com/AliBeikmohammadi/Cost-Sensitive-Transformer-Model, at commit
  `4db444a3958f0a5e95536e29b56206c497bb6d3d`, in `third_party/cost-sensitive-transformer-model`. It is fetched by
  `third_party/fetch_third_party_code.sh`. The environment variable `APS_AUTHORS_2026_REPOSITORY` can name another
  folder instead (see `file_locations.py`).
* The two UCI data files in `data/`.

Every script is run from the repository root with `python -m`. Intermediate tables go to `cache/authors_code_work/` (the
exact tables) and `cache/authors_code_work/five_round_tables/` (the approximation). The cache can be deleted at any
time, except for the checkpoints of retraining runs that are still to be continued (`cache/retraining_runs/` in the
command below). The scripts print to the console, and the logs listed below are that output, redirected.

## The scripts, in the order they are run

First re-scoring on approximate inputs, and an early check of the scaler:

```
python -m experiments.rq1_transformer_2026_reproduction.authors_code.impute_five_rounds
python -m experiments.rq1_transformer_2026_reproduction.authors_code.evaluate_shipped_models --tables five_round
python -m experiments.rq1_transformer_2026_reproduction.authors_code.evaluate_shipped_models_raw_scale
python -m experiments.rq1_transformer_2026_reproduction.authors_code.time_one_training_epoch
```

`impute_five_rounds.py` drops the attributes with 10 % or more missing values (28 of 170) and fits the imputer with
five rounds (about 15 minutes). `evaluate_shipped_models.py` fits the scaler on the table named by its positional
argument (`train_imputed.csv` by default, `train_resampled.csv` as published) and scores the five models.
`evaluate_shipped_models_raw_scale.py` scores model 1 with the attribute ranges of the raw training file instead.
`time_one_training_epoch.py` measures the step time of the authors' architecture on the machine, which showed that
retraining needed GPUs.

Exact inputs and the attribution:

```
python -m experiments.rq1_transformer_2026_reproduction.authors_code.prepare_authors_inputs --imputer-rounds 100
python -m experiments.rq1_transformer_2026_reproduction.authors_code.evaluate_shipped_models --tables exact train_resampled.csv
bash experiments/rq1_transformer_2026_reproduction/authors_code/run_scaler_attribution.sh
```

`prepare_authors_inputs.py` writes the six input tables of the authors' `TrainingCode.py` into the authors' repository
and the unscaled tables into the cache; the 100 imputer rounds take about four hours on one core.
`run_scaler_attribution.sh` runs `resample_five_round_table.py` (SVM-SMOTE and RENN on the five-round table) and the
two crossed evaluations.

Hard-negative comparison for the released models, and the tree-model control:

```
python -m experiments.rq1_transformer_2026_reproduction.authors_code.hard_negative_check_shipped_models
python -m experiments.rq1_transformer_2026_reproduction.authors_code.hard_negative_check_shipped_models_followup
python -m experiments.rq1_transformer_2026_reproduction.authors_code.hard_negative_check_tree_control
```

The first defines the groups: training negatives kept or removed by RENN (found by exact row matching against the
resampled table) and test negatives that RENN's rule, applied once against the resampled table, calls hard. The
three masks are saved as `shipped_models_exact_inputs/hard_negative_masks.npy`, and the other hard-negative scripts
read them from there. The follow-up splits the training negatives by the same one-shot rule. The control scores the
same groups with three classifier arms of the thesis (seed 11), out of fold for the training rows; it needs their
records and probabilities in `results/classifier_arms/`.

Retraining with the released code (on GPUs; one checkpoint folder per run, and one working copy of the authors'
repository per concurrent run, because their code writes `CSV/` and `Model/` into the folder it runs in):

```
APS_AUTHORS_2026_REPOSITORY=<working copy> \
python -m experiments.rq1_transformer_2026_reproduction.authors_code.train_authors_model_with_resume \
    --seed_number 0 --num_epochs 8000 --stop-epoch 1000 --checkpoint-every 50 --test-every 50 \
    --checkpoint-dir cache/retraining_runs/clean_seed0
# test-inclusion controls: add --contamination-fraction 0.5 (half of the test rows) or --contamination-fraction 1.0 (all)
# to continue a run, repeat the command with a larger --stop-epoch
```

The checkpoint folder collects the model checkpoint (`last_model.keras`) and every output of the run; the analysis
reads the outputs from `results/rq1_transformer_2026_reproduction/authors_code/retraining_runs/<run>/`, so copy
everything except the checkpoint there when the run is finished. In the thesis every run was a sequence of stages,
each stage a queue job (`cloud_runs/run_authors_training_job.sh`) that resumed from the previous stage's checkpoint:

| Run | `--seed_number` | `--contamination-fraction` | Stages (`--stop-epoch`) | Reached |
|---|---|---|---|---|
| `clean_seed0` | 0 | none | 1,000, then 8,000 | 8,000 |
| `clean_seed1` | 1 | none | 1,000, then 8,000 | 5,950 (stopped when the compute ran out) |
| `clean_seed2` … `clean_seed4` | 2 to 4 | none | 1,000, then 2,000 | 1,000 |
| `half_test_rows_in_training_seed0` | 0 | 0.5 | 1,000, then 8,000 | 8,000 |
| `half_test_rows_in_training_seed1`, `..._seed2` | 1, 2 | 0.5 | 1,000, then 2,000 | 1,000 |
| `all_test_rows_in_training_seed0` | 0 | 1.0 | 1,000, then 2,000 | 1,000 |
| `clean_seed0_verification` | 0 | none | 300 | 300 |
| `half_test_rows_in_training_seed0_verification` | 0 | 0.5 | 300 | 300 |

Each stage overwrites `final_evaluation.json`. Before a run continues beyond a stage, the queue job keeps a copy of
the earlier evaluation as `final_evaluation_after_<epochs>_epochs.json`, named after the `epochs_run` field inside it;
when you run the stages by hand, make that copy yourself before starting the next stage.

The flags with underscores are the authors' own (`--seed_number`, `--num_epochs`, ...), with their defaults. The
hard-negative comparison for the retrained models is in the parent folder; `hard_negative_check_from_probabilities.py`
prints the same comparison for any one pair of saved probability vectors:

```
python -m experiments.rq1_transformer_2026_reproduction.hard_negative_check_retrained_models
python -m experiments.rq1_transformer_2026_reproduction.authors_code.hard_negative_check_from_probabilities \
    results/rq1_transformer_2026_reproduction/authors_code/retraining_runs/clean_seed0/training_probabilities_final.npy \
    results/rq1_transformer_2026_reproduction/authors_code/retraining_runs/clean_seed0/test_probabilities_final.npy
```

## Where the outputs are

All under `results/rq1_transformer_2026_reproduction/authors_code/`:

* `logs/`: the console output of the scripts. `impute_five_rounds.log` (with `impute_five_rounds_audit.txt`, a count of
  imputed values outside the observed ranges, made by a separate check that is not included in this repository),
  `evaluate_shipped_models_five_round_tables.log` and its table `evaluate_shipped_models_five_round_tables.csv`,
  `evaluate_shipped_models_raw_scale.log`, `time_one_training_epoch.log`, `prepare_authors_inputs.log`,
  `evaluate_shipped_models_exact_inputs.log`, `scaler_attribution.log` with the logs of its steps
  (`evaluate_shipped_models_exact_tables_scaler_on_imputed.log`, `resample_five_round_table.log`,
  `evaluate_shipped_models_five_round_tables_scaler_on_resampled.log`).
* `published_training_logs/run1.csv` to `run5.csv`: the authors' own per-epoch training logs, copied from the `CSV/`
  folder of their repository.
* `shipped_models_five_round_tables/`: the test probabilities of the five models on the five-round tables
  (`run<k>_test_probabilities.npy`) and of model 1 with the raw-range scaling (`run1_test_probabilities_raw_scale.npy`).
* `shipped_models_exact_inputs/`: on the exact inputs, the test probabilities (`run<k>_test_probabilities.npy`) and the
  table `evaluation_scaler_on_resampled.csv`; the attribution tables `evaluation_scaler_on_imputed.csv` and
  `evaluation_five_round_tables_scaler_on_resampled.csv`; the hard-negative results `hard_negative_check.json` (with its
  console output `hard_negative_check.log`), `hard_negative_check_followup.json` and
  `hard_negative_check_tree_control.json`; `hard_negative_masks.npy`, the three groups of negatives that the other
  hard-negative scripts read (see `file_locations.py`); and copies of the console output of
  `prepare_authors_inputs.py` and of the exact evaluation (`prepare_authors_inputs.log`, `evaluation.log`).
* `retraining_runs/<run>/`: one folder per retraining run (`clean_seed0` to `clean_seed4`,
  `half_test_rows_in_training_seed0` to `seed2`, `all_test_rows_in_training_seed0`, and the 300-epoch repeats
  `clean_seed0_verification` and `half_test_rows_in_training_seed0_verification`) with `state.json`,
  `training_log.csv`, `test_trace.jsonl`, `contaminating_test_rows.npy`, `final_evaluation.json`,
  `test_probabilities_final.npy`, `training_probabilities_final.npy` and `test_probabilities_best_checkpoint.npy`.
  The two runs that were extended to 8,000 epochs (`clean_seed0` and `half_test_rows_in_training_seed0`) also keep
  the evaluation of their 1,000-epoch stage as `final_evaluation_after_1000_epochs.json`; the cloud job wrapper saves
  that copy when it extends a run (`cloud_runs/run_authors_training_job.sh`).

The hard-negative comparison for the retrained models writes
`results/rq1_transformer_2026_reproduction/hard_negative_check_retrained_models.json`.
