# Synthetic failure data for Scania APS failure prediction

This repository contains the experiment code and the results behind the master's thesis

> **Inventing Failures to Predict Failures: A reproduction and comparison of synthetic failure data for truck
> prognostics**
> Nandan Ganguly, Department of Computer and Systems Sciences (DSV), Stockholm University, 2026.
> Supervisor: Tony Lindgren.

The thesis itself is not part of the repository. It explains the motivation, the methods and the findings in full;
this README explains how the code is organised, what each experiment does, where its results are, and how to run it
again. It is written for a reader who knows machine learning but has not seen this project before.

## Contents

1. [The experiments in brief](#1-the-experiments-in-brief)
2. [Terms used throughout the code](#2-terms-used-throughout-the-code)
3. [Repository layout](#3-repository-layout)
4. [Setting up](#4-setting-up)
5. [Running the experiments step by step](#5-running-the-experiments-step-by-step)
6. [A quick check that everything runs](#6-a-quick-check-that-everything-runs)
7. [Running the long jobs on Google Colab](#7-running-the-long-jobs-on-google-colab)
8. [Results, and where each thesis table comes from](#8-results-and-where-each-thesis-table-comes-from)
9. [Old names in stored result files](#9-old-names-in-stored-result-files)
10. [Reproducibility notes](#10-reproducibility-notes)
11. [References](#11-references)

## 1. The experiments in brief

**The data and the cost.** The public APS dataset (Air Pressure System, a component of the trucks' braking and
gear-changing systems) was released by Scania in 2016. Each row describes one truck with 170 anonymised numeric
attributes, most of them counters and histogram bins, and many values are missing. The training file has 60,000 rows
of which 1,000 are APS failures; the test file has 16,000 rows of which 375 are failures. A classifier is judged by
the cost that Scania attached to the data: every missed failure costs 500 and every unnecessary workshop check costs
10, so one missed failure weighs as much as fifty false alarms.

**The common evaluation protocol.** Every arm (section 2) is evaluated in the same way, by the common evaluation
protocol. Five-fold stratified cross-validation on the 60,000 training rows gives an out-of-fold probability for every
training row. Anything that is fitted to data (imputation, resampling, a generator of synthetic rows, the classifier) is
fitted inside the training part of each fold only. The decision threshold is the one with the lowest cost on the
out-of-fold probabilities. Only then is the official test set scored, once, with that frozen threshold. Each experiment
is repeated with several random seeds (11, 29 and 47, and also 71 and 101 for the main comparisons), and differences
between two experiments are judged on paired fold-level costs with a corrected t-test. The re-scoring of the released
models, the retraining with the released code and the PyTorch port of RQ1b follow the procedure of the study of
Beikmohammadi et al. instead (the port also under a validation split of real rows; step 4).

**RQ1, reproduction.** Research question 1 (RQ1) asks to what extent the test costs reported by two published studies
can be reproduced from their published descriptions, code and trained models, and which differences in procedure or
settings may explain any gap.

* RQ1a, *the conditional GAN of Ranasinghe et al.* (Ranasinghe and Parlikad, 2019; Ranasinghe et al., 2019). A
  conditional generative adversarial network was trained on the 1,000 real failures, 2,000 synthetic failures were added
  to the training data, and a gradient-boosting model reached a test cost of 5,550. The folder
  `experiments/rq1_cgan_2019_reproduction/` re-implements the GAN and runs it under the common evaluation protocol,
  together with two references: the same classifiers without synthetic rows, and 2,000 duplicated real failures instead
  of generated ones (the duplication control). The papers do not report every setting that a re-implementation needs
  (among them the network sizes, the training settings and the decision threshold); the network settings were taken from
  a notebook attributed to the study. The re-implementation therefore shows what this specification of the method
  achieves under the common evaluation protocol, not what the original implementation achieved.
* RQ1b, *the cost-sensitive Transformer of Beikmohammadi et al.* (2026), with a reported test cost of 3,440. The
  folder `experiments/rq1_transformer_2026_reproduction/` re-scores the five released models, regenerates their input
  tables exactly, retrains the released code from scratch on the published inputs and in test-inclusion controls
  (runs in which a known share of the official test rows is deliberately included in the training data), compares
  how the models flag hard negatives (the hard-negative comparison), and contains a PyTorch port of the network that
  is trained under the published protocol and under a protocol with a validation split of real rows. The authors'
  release of their code, hyperparameters and trained models is what made this examination possible.

**RQ2, survey.** `experiments/rq2_survey_of_methods/` evaluates other ways of producing synthetic failures or of using
the existing data better, all under the common evaluation protocol and at the same synthetic-row count: the SMOTE
family, a Gaussian copula, TVAE, CTGAN, CTAB-GAN+, TabDDPM, GReaT, Forest-Flow, TabEBM, the tabular foundation model
TabICLv2, missingness indicators, and an ensemble of three gradient-boosting libraries.

**RQ3, a different use for the generators.** Instead of adding rows to the training data, a generative model can fill
in the missing attributes of a record that has been observed only in part. `experiments/rq3_completion_and_cold_start/`
benchmarks six such *completion* engines, audits them for memorisation, provides a small command-line prototype, and
measures how much synthetic rows help when only a few labelled failures exist (the *cold start*).

The main numbers are in [section 8](#8-results-and-where-each-thesis-table-comes-from), together with a table that
links every table and figure of the thesis to the code and the result files behind it.

## 2. Terms used throughout the code

| Term | Meaning |
|---|---|
| research question (RQ) | The thesis has three: RQ1, with the parts RQ1a and RQ1b, the reproduction of two published studies; RQ2, the survey of other approaches; RQ3, the completion of partially observed failure records and the cold start (section 1). |
| failure, healthy | A row of class `pos` (an APS failure) or `neg` (any other truck). In the code they are 1 and 0. |
| cost | 500 × missed failures (false negatives, FN) + 10 × false alarms (false positives, FP). |
| scaled to the 16,000 rows of the test set | A cost computed on the 60,000 out-of-fold predictions, multiplied by 16,000 / 60,000 so that it can be compared with test-set costs. |
| arm | One complete recipe for training a classifier: the classifier and its settings, the preprocessing, and what (if anything) is added to the training data. The arms are listed in `experiments/experiment_arms.py`; two further kinds of record come from other scripts (`lightgbm_tuned_3_folds` from `run_experiment_arm.py --folds 3`, and the `lightgbm_on_2026_preprocessing_*` arms of RQ1b). |
| fixed settings | The settings of a classifier written in `experiments/common/classifiers.py` (for LightGBM and XGBoost, 600 trees at learning rate 0.03). They are not the libraries' own defaults. Only LightGBM was tuned. |
| out-of-fold (OOF) | In k-fold cross-validation each training row is predicted by the model that did not see it. Collecting these predictions gives one out-of-sample prediction per training row. |
| common evaluation protocol | Every step that learns from data (imputer, resampler, generator, classifier) is fitted on the training part of each cross-validation split only, the threshold is chosen on out-of-fold predictions, and the official test set is scored once per final model. The comments in the code call this fitting "inside the fold". |
| threshold | The probability above which a truck is sent to the workshop. It is chosen to minimise the cost on the out-of-fold predictions and then frozen. |
| test oracle | The lowest test cost any threshold could have reached. It uses the test labels and is reported only as a diagnostic. |
| synthetic rows | Failure rows produced by a generator fitted on the real failures of the training fold. Every generator arm adds 2,000 of them. The generators produce complete rows. In the arms whose classifier sees missing values, the RQ2 generators (Forest-Flow and TabEBM included) give each row the missing-value pattern of a randomly drawn real failure, and the cGAN of Ranasinghe et al. blanks each cell independently with the missing rate of its attribute among the fitting failures. Rows stay complete in the arms on median-imputed attributes (the SMOTE family, and the random forest and scikit-learn gradient boosting arms with or without cGAN rows) and in the cGAN with the notebook's settings. |
| cGAN variants | The conditional GAN of Ranasinghe et al. as implemented here (`conditional_gan_2019`, the default) scales its inputs to [0, 1] (min-max scaling) and applies a single sigmoid in the discriminator loss, and its rows get missing values put back. The cGAN with the notebook's settings (`..._literal_notebook`) keeps the standard scaling, the double sigmoid and the way of drawing batches of the notebook attributed to the study, and its rows stay complete. `conditional_gan_2019_without_missingness`, used only in the diagnostics of the generated rows, is the default variant without the missing values put back. `conditional_gan_2019.py` gives the reasons for both changes. |
| duplication control | 2,000 copies of real training failures instead of synthetic ones: the reference every generator has to beat. |
| missingness indicators | One extra 0/1 column per attribute that records whether the value was missing. Failure rows have about twice the missing rate of healthy rows, so the pattern itself is informative. |
| SMOTE family | Oversamplers that create new failure rows by interpolating between a real failure and one of its nearest failure neighbours: SMOTE, Borderline-SMOTE and SVM-SMOTE (which concentrate on failures near the class boundary) and ADASYN (which creates more rows where failures are hard to learn). |
| RENN | Repeated Edited Nearest Neighbours: removes every healthy row with a failure among its three nearest neighbours, and repeats until a pass removes nothing (at most 100 passes). Part of the preprocessing pipeline of Beikmohammadi et al., after SVM-SMOTE. |
| chained equations | Imputation that predicts every attribute with missing values from all the others in turns, round after round (scikit-learn's `IterativeImputer` with Bayesian ridge regression). The study of Beikmohammadi et al. used 100 rounds. |
| preprocessing pipeline of Beikmohammadi et al. | Their pipeline: drop the attributes with 10 % or more missing values, impute the rest by chained equations (Bayesian ridge regression), oversample the failures with SVM-SMOTE to one failure per two healthy rows, clean the result with RENN, and scale every attribute to [0, 1] (min-max scaling). |
| published protocol, protocol with a validation split of real rows | Two protocols for the PyTorch port of the cost-sensitive Transformer of Beikmohammadi et al. The published protocol follows the code released by Beikmohammadi et al.: imputation, resampling and scaling are fitted on the whole training table and the validation rows are drawn afterwards from the resampled table, an order that is common in applied work. The protocol with a validation split of real rows holds out 10 % of the real training rows for validation first and fits those steps on the other 90 %. In flags and run names it is called `corrected` (`--protocol corrected`, `corrected_protocol`). |
| hard negatives, hard-negative comparison | The healthy training rows that the RENN step of the preprocessing pipeline of Beikmohammadi et al. removes because a failure is among their three nearest neighbours; applied once to the test file, the same rule marks hard test negatives. The hard-negative comparison measures how often a model flags hard negatives that were and were not in its training table; its scripts are named `hard_negative_check_*`. |
| released models | The five trained Keras models that Beikmohammadi et al. published with their code. File names call them "shipped" (for example `evaluate_shipped_models.py`, `shipped_models_exact_inputs/`). |
| retraining runs | The code released by Beikmohammadi et al. retrained on the published inputs, with no test rows in the training data. Their run names are `clean_seed0` … `clean_seed4`. |
| test-inclusion control | A run in which a known share of the official test rows is deliberately included in the training data: the code released by Beikmohammadi et al. retrained with half (8,000) or all (16,000) of the test rows added to its training table. It shows how a model scores test rows that were part of its training data. File names, flags and record fields call these runs contamination runs (for example `--contamination-fraction`, `contaminating_test_rows.npy`, the `contamination` field of `final_evaluation.json`, `rq1_transformer_2026_retraining_and_contamination_runs`). |
| Gaussian copula | A statistical generator: every attribute keeps its own marginal distribution, and the dependence between the attributes is modelled as a multivariate normal distribution of their normal scores. |
| TVAE | A variational autoencoder for tables (Xu et al., 2019) that encodes each continuous attribute by its component in a fitted Gaussian mixture. |
| CTGAN | A conditional generative adversarial network for tables (Xu et al., 2019) with the same encoding as TVAE; here it is conditioned on the k-means cluster of the failure rows. |
| CTAB-GAN+ | CTGAN with an auxiliary classifier that keeps the generated rows consistent with a target column (here the cluster) and with special encodings for long-tailed columns (Zhao et al., 2022). |
| TabDDPM | A denoising diffusion model for tables (Kotelnikov et al., 2023): noise is added to the rows in many small steps, a network learns to remove it, and new rows are made by removing noise step by step, starting from pure noise. Its numerical part is re-implemented here. |
| GReaT | A generator that writes every row as a short sentence, fine-tunes a pretrained language model on the sentences and parses generated text back into rows (Borisov et al., 2023). It produced no usable rows on the APS data. |
| Forest-Flow profiles | Forest-Flow generates rows by flow matching with gradient-boosted trees. Its size is set by the number of noise levels, duplicates per row, trees and tree depth; the *reduced* profile (10, 10, 20, 5) was run, the *middle* and *paper* profiles were too slow for the compute budget. |
| TabEBM, TabPFN | TabEBM samples rows from an energy model built from TabPFN, a pretrained transformer that classifies or regresses a table from a few thousand example rows without being trained on it. TabPFN is also one of the completion engines of RQ3. |
| TabICLv2, bagged contexts | A pretrained tabular model that classifies new rows by reading labelled example rows (the context) alongside them. The training set is too large for one context, so each fold model averages three contexts, each with all training failures and a random sample of 10,000 healthy rows. |
| paired comparison | Two arms compared fold by fold on the same folds and seeds. The corrected resampled t-test of Nadeau and Bengio accounts for the overlap between training sets of different folds; Holm's method adjusts for testing many arms at once. |
| two-tier criterion | An arm shows a *demonstrated gain* if the 95 % interval of its corrected paired difference to the baseline lies below zero, the Holm-adjusted p-value is below 0.05, its mean difference is better than that of the duplication control, and its median test cost is not higher than the baseline's. It shows a *positive sign* if, without that, its mean difference is negative on average and on every seed, its median test cost is not higher, and its median number of false alarms at no more than 6 missed failures is not higher than the baseline's. Everything else is *no evidence*. |
| completion | Filling in the missing attributes of one partially observed record, conditioned on the attributes that were observed. |
| inpainting | Completion done by a generator (Forest-Flow or Unmasking Trees): the hidden cells are generated while the observed cells are held fixed. In the cold start, inpainting creates new failures from partially hidden real ones. |
| Unmasking Trees | A completion and generation method that fills in hidden cells one after another with gradient-boosted trees (McCarter, 2024). |
| completion masks | Cells hidden on purpose so that the completion can be checked against the true values: every known cell hidden independently with probability 0.2, 0.5 or 0.8 (`random_20_percent`, `random_50_percent`, `random_80_percent`; missing completely at random), or the missingness pattern of a real record that misses at least 30 % of its attributes (`real_patterns_30_percent`). |
| normalised error (NRMSE) | The root mean squared error of a completion over the hidden cells, in log(1 + x) space, with each error divided by the standard deviation of its attribute among the fitting failures, so that always predicting the median scores about 1. |
| cold start | A situation with very few labelled failures (25 to 1,000 here) in which synthetic rows might be expected to help most. |
| earlier phase of the project | The exploratory work done before the common evaluation protocol was fixed, in which the test set was scored many times and the notebook attributed to the study of Ranasinghe et al. was obtained. `run_threshold_rule_demonstration.py` measures what its way of choosing thresholds cost. |
| L4, A100 | The NVIDIA GPUs of the Colab sessions; the L4 has 24 GB of memory. |

Each module explains its own methods in more detail at the top of the file.

## 3. Repository layout

```
README.md                         this guide
requirements.txt                  exact package versions (everything)
requirements-core.txt             the minimal set for the baseline, the cGAN reproduction and the analysis
data/README.md                    how to download the UCI files (the data are not included)
third_party/                      the external repositories, cloned by fetch_third_party_code.sh (not included)
experiments/                      all experiment code (a Python package; run modules with python -m)
  common/                         shared by every experiment
    paths.py                      every input and output location
    data_loading.py               reads the CSV files, defines the cost, writes the data audit
    cost_and_evaluation.py        cross-validation, threshold choice, test evaluation, bootstrap interval
    classifiers.py                LightGBM, XGBoost, CatBoost, random forest, ..., and the ensemble
    imputation_and_oversampling.py  median imputation and the SMOTE family
    synthetic_data_diagnostics.py   detectability, distance and memorisation statistics for synthetic rows
  experiment_arms.py              the registry of the classifier arms of RQ1 and RQ2
  run_experiment_arm.py           runs one arm on one or more seeds (the main entry point)
  hyperparameter_search.py        the Optuna search that tuned LightGBM
  quick_baseline_check.py         the very first look at the data (fixed settings, no tuning)
  run_threshold_rule_demonstration.py  what the threshold rule of the earlier phase of the project cost
  rq1_cgan_2019_reproduction/     the conditional GAN of Ranasinghe et al. and the synthetic-row count sweep
  rq1_transformer_2026_reproduction/  the preprocessing pipeline of Beikmohammadi et al., the PyTorch port, the tree
                                  models on that preprocessing, and authors_code/ (scripts that run the authors' code)
  rq2_survey_of_methods/          the generators and the foundation model of the survey, their diagnostics, pilots
  rq3_completion_and_cold_start/  completion engines, benchmark, memorisation audit, prototype, cold start
analysis/                         statistics, summary tables and figures computed from results/
cloud_runs/                       how the GPU and long CPU jobs were run on Google Colab (queue, job list, notebooks)
scripts/                          shell scripts that run each step of section 5 in order, and a quick smoke test
results/                          every result of the thesis (see results/README.md)
```

`cache/` is created on the fly for intermediate tables such as imputed folds and the regenerated inputs of the code
released by Beikmohammadi et al. (about 1 GB in all). It can be deleted at any time, except for checkpoints of training
runs you still want to continue: the PyTorch port and the retraining with the released code keep them there when they
are run with `--checkpoint-dir cache/...`, as in the commands of step 4 and of `authors_code/README.md`.

## 4. Setting up

### Python and packages

The experiments ran with Python 3.13 on Linux (a two-core machine with 8 GB of memory for the CPU work, which was
enough) and on Google Colab for the GPU work. The shell scripts need bash (on Windows, use WSL). Create a fresh
environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate            # on Windows (WSL): the same; in PowerShell: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt      # or requirements-core.txt for the minimal set
```

PyTorch is pinned to the version used on the CPU machine. Without a GPU, the CPU build is much smaller
(`pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu`, then the rest of the file); with an
NVIDIA GPU, install the CUDA build that matches your driver from https://pytorch.org first. TensorFlow is only needed
for the scripts in `experiments/rq1_transformer_2026_reproduction/authors_code/`. To see whether the optional packages
are importable:

```bash
python -c "import tensorflow, tf_keras, focal_loss, tabicl, tabpfn, tabebm, utrees, ForestDiffusion, ctgan, copulas, be_great"
```

### Data and external code

Download the two CSV files as described in [`data/README.md`](data/README.md) and put them in `data/`. The external
repositories are only needed for RQ1b and for the CTAB-GAN+ arm:

```bash
bash third_party/fetch_third_party_code.sh
```

### Check

From the repository root:

```bash
python -m experiments.common.data_loading     # prints the data audit; the checksums must match data/README.md
bash scripts/smoke_test.sh                     # the main entry points on small samples (about ten minutes)
```

All commands in this README are run from the repository root. Modules are started with `python -m` so that the
`experiments` package is found without installing anything.

A few environment variables change where things go or how many resources they use; none is needed for a local run.

| Variable | Effect |
|---|---|
| `APS_CPU_THREADS_PER_MODEL` | CPU threads per classifier (default: all cores; the thesis machine used 2) |
| `APS_TABICL_DEVICE`, `APS_TABICL_QUERY_CHUNK_ROWS` | run TabICLv2 on the GPU (`cuda`) and in chunks of rows (the Colab sessions used 4,000) |
| `APS_AUTHORS_2026_REPOSITORY`, `APS_CTAB_GAN_PLUS_REPOSITORY` | other locations of the two external repositories |
| `APS_ARM_RECORDS_FILE`, `APS_COLD_START_RECORDS_FILE`, `APS_COMPLETION_RECORDS_FILE` | write the records of a run to another file (used by the cloud queue; the probability files still go to `results/`) |

A few further variables are used mainly by the cloud queue and the GPU jobs (for example `APS_CLOUD_STORE`,
`APS_JOB_NAME` and `APS_TENSORFLOW_THREADS`); each is described where it is read, in a module docstring or a script
header.

## 5. Running the experiments step by step

Each step below lists the commands that produced the results in `results/`. The same commands are collected, in
order and with comments, in the shell scripts of `scripts/` (`scripts/02_baseline.sh` and so on), which you can run
as they are or copy from.

**Every script writes into `results/`.** Records are appended to the files described in
[`results/README.md`](results/README.md), and probability files of the same arm and seed are overwritten, so a rerun
replaces part of the published results. To keep them, run the experiments in a second clone of the repository (or
check the changes afterwards with `git status` and undo them with `git checkout -- results/`). If an arm and seed
occur more than once in a records file, the analysis uses the last record.

### Time and hardware

The whole programme took about 410 hours of computing, most of it on Colab GPUs (about 935 of 979 purchased compute
units). The times below are totals for each part as it was run for the thesis; the per-seed times in the steps come
from the same records. A current 8–12-core laptop is several times faster than the two-core machine for the tree
learners and the generators.

| Part | Where it ran | Hours | Feasible on a CPU? |
|---|---|---|---|
| Hyperparameter search, first-look baselines, threshold demonstration | two-core machine | about 1.5 | yes |
| Classifier arms of RQ1 and RQ2 (94 records) | two-core machine and Colab | about 76 (31 on the machine, 44 in Colab) | yes, except the TabICLv2, TabEBM and deep-generator arms in practice |
| Count sweep and diagnostics of the generated rows | two-core machine and an L4 | about 2.5 | yes, except the GPU generators |
| Regenerating the inputs of Beikmohammadi et al. (100 imputer rounds) | one core | about 4 | yes |
| Retraining the code released by Beikmohammadi et al. from scratch | A100, two runs per GPU | about 114 (retraining runs) and 75 (test-inclusion and verification runs), hours of run time | no |
| PyTorch port of the cost-sensitive Transformer of Beikmohammadi et al. | A100, two runs per GPU | about 42, hours of run time | no |
| Completion benchmark and memorisation audit | L4 | about 24 | slow |
| Cold start | L4 and CPU sessions | about 71 | slow |

Each A100 session ran two jobs at a time (`cloud_runs/README.md`), so the hours of the two A100 rows are the run times
of the individual runs, with two runs sharing each GPU.

### Step 0. Check the data

```bash
python -m experiments.common.data_loading
```

Writes `results/data_audit.json`: row and failure counts, missing-value rates, constant columns, duplicates between the
files and the SHA-256 checksums.

### Step 1. First look, hyperparameter search and threshold rules

```bash
python -m experiments.quick_baseline_check                                  # about 25 minutes in all
python -m experiments.hyperparameter_search --model lightgbm --trials 26 --timeout 1000000 --seed 11
python -m experiments.run_threshold_rule_demonstration                      # about 13 minutes
```

`quick_baseline_check.py` runs the first-look baselines: four configurations (0.4 to 12 minutes each) with the fixed
settings of `classifiers.py` (LightGBM, LightGBM with the failures weighted 30 times, XGBoost and scikit-learn's
histogram gradient boosting), and reports their out-of-fold cost.

The hyperparameter search tunes LightGBM for the cost with Optuna. The stored search ran with a time limit and
completed 26 trials in about 50 minutes; because a limit makes the number of trials depend on the speed of the
machine, the command above asks for exactly 26 trials instead (the sampler is seeded, so it proposes the same
configurations). Its result, `results/hyperparameter_search/lightgbm_seed11_best_parameters.json`, is already in the
repository, and every arm with tuned LightGBM reads it from there; rerunning the search overwrites it.

To check the results: among the first-look baselines, LightGBM with the fixed settings has an out-of-fold cost of
34,100, and the best trial of the search (trial 15) reaches 33,290.

The threshold demonstration reruns the learners of the earlier phase of the project and shows how much their test
cost depended on the rule used to pick the threshold.

### Step 2. The baseline

```bash
python -m experiments.run_experiment_arm --arm lightgbm_tuned --seeds 11 29 47 --final-test
python -m experiments.run_experiment_arm --arm lightgbm_tuned --seeds 71 101 --final-test --no-refit
```

The tuned LightGBM without any synthetic rows: the reference for every comparison in RQ1 and RQ2 (about 3 minutes per
seed). Without `--no-refit` the runner also fits a secondary model on all training rows and scores it with the same
threshold; in the thesis this check was run for seeds 11, 29 and 47 of the baseline and of the cGAN arm only.

To check the result: seed 11 must print an out-of-fold cost of 33,290 and a test cost of 9,020 (10 missed failures,
402 false alarms). The stored values of every arm and seed are in `results/summary_tables/classifier_arms_per_seed.md`,
and those of the other steps in the summary tables listed in section 8.

### Step 3. RQ1a, the conditional GAN of Ranasinghe et al.

```bash
# the cGAN rows with LightGBM (about 22 minutes per seed), and the duplication control (about 3 minutes per seed)
python -m experiments.run_experiment_arm --arm lightgbm_with_conditional_gan_2019_rows --seeds 11 29 47 --final-test
python -m experiments.run_experiment_arm --arm lightgbm_with_conditional_gan_2019_rows --seeds 71 101 --final-test --no-refit
python -m experiments.run_experiment_arm --arm lightgbm_with_duplicated_real_failures --seeds 11 29 47 71 101 --final-test --no-refit

# the learners closest to those of the study of Ranasinghe et al., with and without the cGAN rows
# (15 to 70 minutes per seed)
python -m experiments.run_experiment_arm --arm random_forest_fixed_settings --seeds 11 29 47 --final-test --no-refit
python -m experiments.run_experiment_arm --arm random_forest_fixed_settings_with_conditional_gan_2019_rows \
    --seeds 11 29 47 --final-test --no-refit
python -m experiments.run_experiment_arm --arm sklearn_gradient_boosting --seeds 11 29 47 --final-test --no-refit
python -m experiments.run_experiment_arm --arm sklearn_gradient_boosting_with_conditional_gan_2019_rows \
    --seeds 11 29 47 --final-test --no-refit
python -m experiments.run_experiment_arm --arm random_forest_with_conditional_gan_2019_literal_notebook_rows \
    --seeds 11 --final-test --no-refit

# how the cost changes with the number of synthetic rows (about 80 minutes)
python -m experiments.rq1_cgan_2019_reproduction.run_synthetic_row_count_sweep --generator conditional_gan_2019 \
    --model lightgbm --seeds 11 --gan-training-steps 30000

# what the generated rows look like: detectability, closeness to the real failures, copying (about 6 minutes)
python -m experiments.rq2_survey_of_methods.run_synthetic_data_diagnostics --generators conditional_gan_2019 \
    conditional_gan_2019_literal_notebook conditional_gan_2019_without_missingness duplicated_real_failures
```

The cGAN arm's seed 11 gives a test cost of 9,500 on the CPU machine (a rerun in a Colab session gave 9,260; see
section 10). The stored diagnostics of the duplicated real failures come from a run with 300 copies
(`--synthetic-rows 300`), the other generators from runs with 2,000 rows; [`results/README.md`](results/README.md)
explains this. The cGAN with the notebook's settings (`..._literal_notebook`) and the variant without missing values put
back (`conditional_gan_2019_without_missingness`) are described under "cGAN variants" in section 2.

### Step 4. RQ1b, the cost-sensitive Transformer of Beikmohammadi et al.

This step has four parts. The first two run on a CPU; the PyTorch port and the retraining with the released code
need a GPU. `scripts/04_rq1_transformer_2026_reproduction.sh` lists every command with its seed.

*The released models and inputs* (TensorFlow and `third_party/cost-sensitive-transformer-model` needed). The scripts
in `experiments/rq1_transformer_2026_reproduction/authors_code/` regenerate the authors' input tables with their
published settings, score the five released models on the official test set, separate the effects of the imputer and
the scaler, and run the hard-negative comparison. Their order, options and outputs are described in
[`authors_code/README.md`](experiments/rq1_transformer_2026_reproduction/authors_code/README.md). The central commands:

```bash
python -m experiments.rq1_transformer_2026_reproduction.authors_code.prepare_authors_inputs --imputer-rounds 100
python -m experiments.rq1_transformer_2026_reproduction.authors_code.evaluate_shipped_models --tables exact train_resampled.csv
```

The second must report test costs of 3,540, 3,520, 3,060, 3,540 and 3,540 for the five models, 3,440 on average.

*The preprocessing pipeline of Beikmohammadi et al. in front of the tuned LightGBM* (CPU, three folds because the
imputer of Beikmohammadi et al. is slow, with the 3-fold baseline):

```bash
python -m experiments.run_experiment_arm --arm lightgbm_tuned --folds 3 --seeds 11 29 47 --final-test --no-refit
python -m experiments.rq1_transformer_2026_reproduction.run_tree_models_on_2026_preprocessing --variant full_pipeline \
    --folds 3 --imputer-rounds 10 --seeds 11 --final-test
#   ... and the same with --variant without_renn, renn_only and imputation_only
```

Seeds 29 and 47 of the full pipeline were also run in a Colab session, but their records were not copied out of the
session (see `cloud_runs/README.md`), so the results have seed 11 only. To check the result: seed 11 of the full
pipeline must report a test cost of 8,850 (6 missed failures, 585 false alarms).

*The PyTorch port* (GPU; 500 epochs each, about 0.6 to 5.5 hours per run):

```bash
python -m experiments.rq1_transformer_2026_reproduction.run_transformer_2026_port --protocol corrected --epochs 500 \
    --patience 100000 --imputer-rounds 10 --seed 0 --run-name corrected_protocol \
    --checkpoint-dir cache/port_checkpoints --checkpoint-every 50
python -m experiments.rq1_transformer_2026_reproduction.run_transformer_2026_port --protocol published --epochs 500 \
    --patience 100000 --imputer-rounds 10 --seed 0 --run-name published_protocol \
    --checkpoint-dir cache/port_checkpoints --checkpoint-every 50
#   protocol with a validation split of real rows (--protocol corrected) on seeds 0, 1, 2; published protocol and
#   its three ablations on seeds 0 and 1:
#   --no-renn (run name published_protocol_without_renn), --no-smote (published_protocol_without_svm_smote),
#   --no-smote --no-renn (published_protocol_without_resampling)
```

In the stored runs, seed 0 of the protocol with a validation split of real rows reached its best checkpoint at epoch
138, with a test cost of 12,540 (17 missed failures, 404 false alarms); GPU runs do not repeat bit for bit (section 10).

*Retraining with the released code* (GPU; about 190 hours of run time in all, with two runs sharing each A100). Each run
is a sequence of resumable stages of `authors_code/train_authors_model_with_resume.py`: retraining runs on seeds 0 to 4,
test-inclusion controls with `--contamination-fraction 0.5` (seeds 0 to 2) or `1.0` (seed 0), and two short verification
runs. The thesis runs went through the Colab queue (`cloud_runs/run_authors_training_job.sh`, section 7);
`authors_code/README.md` lists every run with its stages. At epoch 1,000 the stored trace of `clean_seed0` shows a test
cost of 18,120 (32 missed failures, 212 false alarms). Afterwards,

```bash
python -m experiments.rq1_transformer_2026_reproduction.hard_negative_check_retrained_models
```

runs the hard-negative comparison for the retrained models.

### Step 5. RQ2, the survey

```bash
# classical oversampling, 2,000 synthetic failures each, on median-imputed attributes (30–40 minutes per seed)
for arm in lightgbm_with_smote_rows lightgbm_with_borderline_smote_rows lightgbm_with_adasyn_rows \
           lightgbm_with_svm_smote_rows; do
    python -m experiments.run_experiment_arm --arm $arm --seeds 11 29 47 --final-test --no-refit
done

# the Gaussian copula and Forest-Flow (reduced profile) on a CPU (about 100 minutes and 6 hours per seed)
python -m experiments.run_experiment_arm --arm lightgbm_with_gaussian_copula_rows --seeds 11 29 47 --final-test --no-refit
python -m experiments.run_experiment_arm --arm lightgbm_with_forest_flow_rows_reduced_profile --seeds 11 29 47 \
    --final-test --no-refit

# deep generators and TabEBM (GPU; an L4 needs 8 to 80 minutes per seed, CTAB-GAN+ needs third_party/CTAB-GAN-Plus)
for arm in lightgbm_with_tvae_rows lightgbm_with_ctgan_rows lightgbm_with_ctab_gan_plus_rows \
           lightgbm_with_tabddpm_rows lightgbm_with_tabebm_rows; do
    python -m experiments.run_experiment_arm --arm $arm --seeds 11 29 47 --final-test --no-refit
done

# changing the learner instead of the data
python -m experiments.run_experiment_arm --arm xgboost_fixed_settings --seeds 11 29 47 --final-test --no-refit
python -m experiments.run_experiment_arm --arm lightgbm_tuned_with_missingness_indicators --seeds 11 29 47 \
    --final-test --no-refit
python -m experiments.run_experiment_arm --arm ensemble_without_missingness_indicators --seeds 11 29 47 71 101 \
    --final-test --no-refit
python -m experiments.run_experiment_arm --arm ensemble_with_missingness_indicators --seeds 11 29 47 71 101 \
    --final-test --no-refit
# TabICLv2 alone and in the ensemble (GPU, about 10 minutes per seed on an L4)
export APS_TABICL_DEVICE=cuda APS_TABICL_QUERY_CHUNK_ROWS=4000
python -m experiments.run_experiment_arm --arm tabicl_bagged_contexts --seeds 11 29 47 71 101 --final-test --no-refit
python -m experiments.run_experiment_arm --arm ensemble_with_missingness_indicators_and_tabicl --seeds 11 29 47 71 101 \
    --final-test --no-refit

# diagnostics of the generated rows (the second command on a GPU)
python -m experiments.rq2_survey_of_methods.run_synthetic_data_diagnostics --generators forest_flow gaussian_copula \
    smote borderline_smote adasyn svm_smote
python -m experiments.rq2_survey_of_methods.run_synthetic_data_diagnostics --generators tabebm ctgan tvae tabddpm \
    ctab_gan_plus

# the pilots that measured run time and memory before the full runs
python -m experiments.rq2_survey_of_methods.pilots.tabicl_pilot --folds 0 1 --healthy-rows-per-context 2500
python -m experiments.rq2_survey_of_methods.pilots.tabebm_timing_pilot 300,3,100
python -m experiments.rq2_survey_of_methods.pilots.forest_flow_timing_pilot 10,10,20,5
```

In the stored runs, TabICLv2 (`tabicl_bagged_contexts`) on seed 11 has a test cost of 7,320 (5 missed failures,
482 false alarms).

Three arms are defined in the registry but were not completed, and the thesis reports them as such: GReaT
(`lightgbm_with_great_language_model_rows`) produced no usable rows, and the middle and paper profiles of Forest-Flow
need about a day of GPU time per seed. `results/classifier_arms/verification_reruns.jsonl` holds three of the arms
above (baseline, cGAN and duplication control, seed 11) rerun with the same commands in a Colab CPU session, with
their records sent to that file by `APS_ARM_RECORDS_FILE=results/classifier_arms/verification_reruns.jsonl`.

### Step 6. RQ3, completion and cold start

The thesis ran this step on L4 GPUs; `scripts/06_rq3_completion_and_cold_start.sh` runs it on a CPU, or with
`USE_GPU=1` with the GPU options used then.

```bash
# the completion benchmark: 6 engines x 4 masks, one cross-validation fold of the failures per command
# (about 24 L4-hours in all)
for fold in 0 1 2 3 4; do
    python -m experiments.rq3_completion_and_cold_start.run_completion_benchmark --folds $fold \
        --engines median k_nearest_neighbours chained_equations forest_flow unmasking_trees tabpfn
done

# memorisation audit of the six engines
python -m experiments.rq3_completion_and_cold_start.run_completion_memorisation_audit \
    --engines median k_nearest_neighbours chained_equations unmasking_trees forest_flow tabpfn

# the prototype: complete training failure no. 7 after hiding half of its observed cells (about 75 minutes on an L4)
python -m experiments.rq3_completion_and_cold_start.complete_failure_record --engine unmasking_trees --demo-index 7 \
    --hidden-fraction 0.5 --both-hypotheses --healthy-rows 1000 --attributes-to-show 12

# the cold start: 25 to 1,000 labelled failures, with and without 2,000 synthetic rows (about 71 hours in all;
# conditional_gan_2019, tabebm, forest_flow and the two inpainting generators ran on L4 GPUs)
for generator in no_synthetic_rows duplicated_real_failures smote conditional_gan_2019 tabebm forest_flow \
                 forest_flow_inpainting; do
    for seed in 11 29 47; do
        python -m experiments.rq3_completion_and_cold_start.run_cold_start_experiment --generator $generator \
            --seeds $seed --failure-counts 25 50 100 250 1000
    done
done
python -m experiments.rq3_completion_and_cold_start.run_cold_start_experiment --generator unmasking_trees_inpainting \
    --seeds 11 --failure-counts 25 50 100 250 1000
python -m experiments.rq3_completion_and_cold_start.run_cold_start_experiment --generator unmasking_trees_inpainting \
    --seeds 29 47 --failure-counts 25 50
```

On a GPU, the tree-based engines are told to use it through `--override`, for example
`--override '{"forest_flow": {"gpu_hist": true}, "unmasking_trees": {"xgboost_kwargs": {"device": "cuda"}}}'` for the
benchmark and the audit; the script shows the options of every run. Keys that an engine or a generator does not
recognise are ignored without a warning, so check their spelling against the names listed in the module docstring. Without synthetic rows, the cold start's median
test cost falls from 20,350 with 25 failures to 8,470 with 1,000.

### Step 7. Statistics, tables and figures

```bash
python -m analysis.run_all_summaries
```

Reads everything in `results/` and writes the summary tables to `results/summary_tables/` (each as CSV with full
precision and as a Markdown table) and the figures to `results/figures/`. This is where the paired tests, the Holm
adjustment and the two-tier criterion are computed. It takes well under a minute and trains nothing. On an unchanged
repository it rewrites the stored tables with identical content, which `git status` confirms.

## 6. A quick check that everything runs

```bash
bash scripts/smoke_test.sh
```

runs the main entry points on small stratified samples of the training data (a few thousand rows, a few epochs) so
that a broken installation shows up in minutes rather than hours; it takes about ten minutes on a two-core machine.
It needs only the packages of `requirements-core.txt`; the import check in section 4 covers the others. The numbers
it prints mean nothing. All its outputs have `smoke_test` in their names and are deleted at the end.

## 7. Running the long jobs on Google Colab

The GPU jobs and many CPU jobs ran on Google Colab with paid compute units, from a job queue kept on Google Drive: 57 of
the 94 arm records, the retraining runs and the PyTorch port of RQ1b, and the whole completion benchmark and cold start
(section 5 gives the hours). Several Colab sessions (with A100, L4 or CPU-only runtimes) ran the same queue runner, each
taking the next job its hardware could run; every job checkpointed to Drive, so a session that was disconnected could be
replaced without losing more than one fold or a few epochs. The queue, the job list with every command of the queue, and
the notebook used to start a session are in `cloud_runs/`, with their own [README](cloud_runs/README.md). Results
written by the sessions were merged into `results/` by `cloud_runs/merge_cloud_results.py`.

## 8. Results, and where each thesis table comes from

All results are in `results/`, organised like `experiments/`; [`results/README.md`](results/README.md) describes every
file and the fields of every record. The tables and figures that summarise them are in `results/summary_tables/` and
`results/figures/` (step 7).

The main numbers, as reported in the thesis (test-set costs; lower is better):

| Experiment | Result |
|---|---|
| Baseline: tuned LightGBM, out-of-fold threshold | 9,350 (median of 5 seeds; range 8,970–10,360) |
| Rows from the cGAN of Ranasinghe et al. + LightGBM | 9,500 (9,000–11,150); paired difference to the baseline +35, 95 % interval [−1,240, 1,311]; duplication control +218; a classifier tells the generated rows from real ones with an AUC of 1.00; the published 5,550 was not reproduced with this re-implementation, which fills in settings that the papers do not report |
| Models released by Beikmohammadi et al. | re-scoring the five released models on inputs regenerated by repeating the steps and settings of the released preprocessing code reproduces the published mean of 3,440 exactly |
| Code released by Beikmohammadi et al., retrained from scratch | 17,060–18,570 at epoch 1,000 (5 seeds), 17,980 for seed 0 at 8,000 epochs: the published cost was not reached within the compute available |
| Hard-negative comparison | the released models flag 37 % of the hard training negatives that RENN removed and 0.9 % of the hard test negatives; the retrained models without test rows 58–65 % and 46–50 %; the test-inclusion controls 2.9–12.6 % of the hard test negatives that were in their training table and 34–43 % of those that were not. |
| RQ2 survey | 18 arms; none shows a demonstrated gain; the ensemble without indicators (−426) and the ensemble with indicators and TabICLv2 (−672) show a positive sign; lowest test cost of any arm under the common evaluation protocol 7,320 (TabICLv2, seed 11) |
| RQ3 completion | normalised error 0.51–0.60 for the best engines under 20 % and 50 % random masks against 1.06 for the median (0.69 against 1.04 at 80 %); under real missingness patterns 0.94–0.95 against 1.08; no engine produces near copies of its training rows |
| RQ3 cold start | synthetic rows lower the cost by 3,000 to 7,900 with 25–100 labelled failures, duplicated real failures by 3,440 to 6,760; the gain is gone at 250 failures |

Where each table and figure of the thesis comes from (summary files in `results/summary_tables/` as `.md` and `.csv`,
figures in `results/figures/`):

| Thesis | Content | Summary table or figure | Raw results (in `results/`) | Code |
|---|---|---|---|---|
| Table 4.1, Figure 5.1 | the data | `data_audit`, `data_overview.png` | `data_audit.json` | `common/data_loading.py` |
| Table 5.1 | first-look baselines | `quick_baseline_check` | `preliminary_checks/quick_baseline.jsonl` | `quick_baseline_check.py` |
| Figure 5.2, Table B.1 | hyperparameter search | `hyperparameter_search_trials`, `hyperparameter_search_history.png` | `hyperparameter_search/` | `hyperparameter_search.py` |
| Table 5.2, Figure 5.3 | thresholds | `classifier_arms_decision_rules`, `classifier_arms_oof_cost_against_threshold.png` | `classifier_arms/` | `run_experiment_arm.py` |
| Table 5.3 | threshold rules of the earlier phase | `threshold_rule_demonstration` | `preliminary_checks/threshold_rule_demonstration.json` | `run_threshold_rule_demonstration.py` |
| Table 5.4, Figure 5.4, Figure 5.5 | main results of the arms | `classifier_arms_main_results`, `classifier_arms_paired_differences.png`, `classifier_arms_test_false_alarm_frontier.png` | `classifier_arms/` | `run_experiment_arm.py` |
| Table 5.16 | arms against their matched baselines | `classifier_arms_paired_comparisons` | `classifier_arms/` | `analysis/paired_statistics.py` |
| Table 5.5 | diagnostics of the generated rows | `synthetic_data_diagnostics` | `rq2_survey_of_methods/synthetic_data_diagnostics.json` | `rq2_survey_of_methods/run_synthetic_data_diagnostics.py` |
| Figure 5.6 | count sweep | `synthetic_row_count_sweep`, `synthetic_row_count_sweep_conditional_gan_2019.png` | `rq1_cgan_2019_reproduction/` | `rq1_cgan_2019_reproduction/run_synthetic_row_count_sweep.py` |
| Table 5.6 | the models released by Beikmohammadi et al. re-scored | `rq1_transformer_2026_shipped_models_rescored` | `rq1_transformer_2026_reproduction/authors_code/shipped_models_*` | `authors_code/evaluate_shipped_models.py` |
| Table 5.7 | hard negatives, released models | `rq1_transformer_2026_hard_negatives_shipped_models` | `.../shipped_models_exact_inputs/hard_negative_check*.json` | `authors_code/hard_negative_check_*.py` |
| Figure 5.7 | the authors' training logs | `rq1_transformer_2026_published_training_logs.png` | `.../authors_code/published_training_logs/` | — |
| Table 5.8, Figure 5.8 | retraining and test-inclusion controls | `rq1_transformer_2026_retraining_and_contamination_runs`, `rq1_transformer_2026_clean_retraining_runs`, `rq1_transformer_2026_retraining_and_contamination_trace.png` | `.../authors_code/retraining_runs/` | `authors_code/train_authors_model_with_resume.py` |
| Table 5.9 | hard negatives, retrained models | `rq1_transformer_2026_hard_negatives_retrained_models` | `rq1_transformer_2026_reproduction/hard_negative_check_retrained_models.json` | `hard_negative_check_retrained_models.py` |
| Table 5.10 | PyTorch port and ablations | `rq1_transformer_2026_pytorch_port_runs` | `rq1_transformer_2026_reproduction/pytorch_port/` | `run_transformer_2026_port.py` |
| Table A.2 | epochs reached by the GPU runs | `rq1_transformer_2026_epochs_reached` | `.../retraining_runs/*/state.json`, `pytorch_port/` | — |
| Table 5.11, Figure 5.9 | RQ2 survey | `rq2_survey_against_tuned_lightgbm`, `rq2_survey_paired_differences.png` | `classifier_arms/` | `analysis/paired_statistics.py` |
| Table 5.12 | TabICLv2 pilot | `tabicl_pilot` | `rq2_survey_of_methods/tabicl_pilot.json` | `rq2_survey_of_methods/pilots/tabicl_pilot.py` |
| Table 5.13, Figure 5.10 | completion benchmark | `rq3_completion_benchmark`, `rq3_completion_benchmark.png` | `rq3_completion_and_cold_start/completion_benchmark.jsonl` | `rq3_completion_and_cold_start/run_completion_benchmark.py` |
| Table 5.14 | memorisation audit | `rq3_completion_memorisation_audit` | `.../completion_memorisation_audit.json` | `run_completion_memorisation_audit.py` |
| Table 5.15, Figure 5.11 | cold start | `rq3_cold_start_test_costs`, `rq3_cold_start_curves.png` | `.../cold_start_runs.jsonl` | `run_cold_start_experiment.py` |
| Figure 5.12 | the prototype | `results/rq3_completion_and_cold_start/prototype_demonstration_output.txt` | `prototype_demonstration.json` | `complete_failure_record.py` |
| Table A.1 | run times | `classifier_arms_runtimes` | `classifier_arms/` | — |
| Table B.2 | every arm and seed | `classifier_arms_per_seed` | `classifier_arms/` | — |

Code paths are relative to `experiments/` unless they start with `analysis/`. The means, ranges and totals of RQ1b
that the thesis quotes in its text are collected in `rq1_transformer_2026_summary_numbers`.

## 9. Old names in stored result files

The code and the experiments were given their present names before publication; the computations are unchanged, and
the thesis refers to the code by the present names. A few stored result files still contain older names, because
records and logs are kept exactly as they were written. Those names are listed here for that reason only:

| Result file | Old name in it | Present name |
|---|---|---|
| `results/rq1_transformer_2026_reproduction/pytorch_port/*.json`, field `args.checkpoint_dir` | `port_seed<k>`, `abl_published_seed<k>`, `abl_norenn_seed<k>`, `abl_nosmote_seed<k>`, `abl_noresample_seed<k>` | the runs `corrected_protocol_seed<k>`, `published_protocol_seed<k>`, `published_protocol_without_renn_seed<k>`, `published_protocol_without_svm_smote_seed<k>`, `published_protocol_without_resampling_seed<k>` |
| `results/rq1_transformer_2026_reproduction/pytorch_port/corrected_protocol_extended_seed0.json`, field `ckpt` | `transformer2026_corrected_v4ext_seed0.pt` | the checkpoint of the run `corrected_protocol_extended_seed0` |
| `results/rq2_survey_of_methods/forest_flow_timing_pilot.log` | `pilot_forestflow.py`, `forestflow_aug.py` | `experiments/rq2_survey_of_methods/pilots/forest_flow_timing_pilot.py`, `experiments/rq2_survey_of_methods/forest_flow_generator.py` |
| `results/rq3_completion_and_cold_start/prototype_demonstration_output.txt` | `complete_cli.py`, the engine `utrees` | `experiments/rq3_completion_and_cold_start/complete_failure_record.py`, the engine `unmasking_trees` |

## 10. Reproducibility notes

* Every random step takes its seed from the command line, and each fold derives its own seeds from it, so a CPU run
  repeated on the same machine with the same package versions gives the same numbers. Two exceptions: the CTGAN
  library draws from NumPy's global random generator, which it does not seed, so its rows differ from run to run;
  and LightGBM, in the LightGBM arms and as the LightGBM member of the ensembles, chooses between two internal
  histogram layouts by timing both, which on a busy machine can change probabilities in the fourth digit (for the
  LightGBM arms without their own classifier parameters, passing `force_col_wise` through
  `--override '{"classifier_parameters": {"force_col_wise": true}}'` removes that).
* Across machines and on GPUs results are not bit-for-bit identical. The cGAN arm rerun in a Colab session costs 9,260
  on the test set for seed 11 against 9,500 on the original machine, well inside that arm's seed-to-seed range. GPU
  training is not deterministic: two verification runs of the authors' training script differ from the main runs by
  up to 1,950 cost units at the traced epochs. The reruns are kept in the results as checks
  (`results/classifier_arms/verification_reruns.jsonl` and the `*_verification` retraining runs).
* The threshold is searched over 2,001 quantiles of the out-of-fold probabilities rather than over every distinct
  value. On the stored out-of-fold predictions the grid's best cost is at most 0.9 % (280 cost units on the
  60,000-row scale) above the exact optimum, and at most 0.55 % (180 units) for the baseline.
* What was not completed, and is reported as such in the thesis: GReaT produced no usable rows (its code was corrected
  afterwards but not rerun); the middle and paper profiles of Forest-Flow were dropped for lack of compute; the
  retraining runs of the code released by Beikmohammadi et al. stopped at the epochs listed in the thesis (only the
  retraining run of seed 0 and seed 0 of the control runs with half of the test rows in the training data reached the
  full 8,000 epochs), and the extended PyTorch-port run stopped at 3,250 of 4,000 epochs.
* The queue runner in `cloud_runs/` is the one that ran the thesis jobs, with its bookkeeping made safer afterwards;
  `cloud_runs/README.md` lists what changed. The experiment code it runs is unchanged.
* Versions are pinned in `requirements.txt` for the CPU machine. The Colab runs used the Colab images of November
  2026 (PyTorch 2.14.1 with CUDA 13, TensorFlow 2.21); see `cloud_runs/requirements-colab.txt`.

## 11. References

* T. Lindgren and J. Biteus, *APS Failure at Scania Trucks*, UCI Machine Learning Repository, 2017.
  https://doi.org/10.24432/C51S51
* G. D. Ranasinghe and A. K. Parlikad, "Generating real-valued failure data for prognostics under the conditions of
  limited data availability", IEEE International Conference on Prognostics and Health Management (ICPHM), 2019.
  https://doi.org/10.1109/ICPHM.2019.8819392
* G. D. Ranasinghe, T. Lindgren, M. Girolami and A. K. Parlikad, "A methodology for prognostics under the conditions
  of limited failure data availability", IEEE Access, 2019. https://doi.org/10.1109/ACCESS.2019.2960310
* A. Beikmohammadi, M. H. Hamian, N. Khoeyniha, T. Lindgren, O. Steinert and S. Magnússon, "A cost-sensitive
  Transformer model for prognostics under highly imbalanced industrial data", Cluster Computing 29(6), 2026.
  https://doi.org/10.1007/s10586-026-06080-2; code: https://github.com/AliBeikmohammadi/Cost-Sensitive-Transformer-Model
* C. Nadeau and Y. Bengio, "Inference for the generalization error", Machine Learning 52, 2003.
  https://doi.org/10.1023/A:1024068626366

The generators, foundation models and completion engines of RQ2 and RQ3 are cited in the modules that use them and in
the thesis.
