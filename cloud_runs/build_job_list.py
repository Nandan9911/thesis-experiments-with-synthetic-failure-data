#!/usr/bin/env python3
"""Write cloud_runs/job_list.jsonl, the jobs that the Colab sessions of the thesis took from the queue.

Every line of the job list is one job of cloud_runs/queue_runner.py (its module docstring describes the fields and
the placeholders). The list holds every job of the Colab queue of the thesis, with the command, lane, priority,
weight, number of attempts and dependencies it ran with; the commands are those of the experiment scripts in this
repository. Jobs that were planned but taken out of the queue before they ran, because the compute budget could not
cover them, stay in the file as comment lines (the runner skips lines that start with #) together with the reason,
since the thesis reports them. Two pure housekeeping jobs of the original list are left out: one copied the error
lines and the end of four failed logs into small text files in the store, so that they could be read without
downloading the logs, and one wrote the drain file of the session l4-1 once that session's last jobs had finished,
so that its runtime was released at once.

    python cloud_runs/build_job_list.py

The output depends on nothing but this script, so running it again reproduces the file byte for byte.

How the work was ordered. Lower priorities run first. The retraining with the code released by Beikmohammadi et al. and
its test-inclusion controls are the longest runs of RQ1b and come first, in stages that fit the measured speed of an
A100 (about 13 s per epoch for the authors' Keras code with XLA and 18-19 s for the PyTorch port, so an 8,000-epoch run
takes about 30 hours): first every run to a short horizon (1,000 epochs, 300 for the two verification runs, 500 for the
port and its ablations), then four runs in lane gpu_anchor towards the full budget (the retraining runs of seeds 0 and 1
and seed 0 of the control with half of the test rows to 8,000 epochs, the port of seed 0 under the protocol with a
validation split of real rows to 4,000), then the other retraining runs to 2,000 epochs while units remained. A later
stage of a run uses the same checkpoint folder and continues where the earlier stage stopped (needs_jobs makes it wait
for that stage). The sessions were drained when the compute units ran low, so the later stages stopped where the budget
ended: the retraining run of seed 1 at 5,950 epochs, the port extension before 4,000, and the extensions to 2,000 epochs
had not started.
"""
from __future__ import annotations

import json
from pathlib import Path

OUTPUT_FILE = Path(__file__).resolve().parent / "job_list.jsonl"

ARM_RUNNER = "python -m experiments.run_experiment_arm"
TREES_ON_2026_PREPROCESSING = "python -m experiments.rq1_transformer_2026_reproduction.run_tree_models_on_2026_preprocessing"
PYTORCH_PORT = "python -m experiments.rq1_transformer_2026_reproduction.run_transformer_2026_port"
PREPARE_AUTHORS_INPUTS = "python -m experiments.rq1_transformer_2026_reproduction.authors_code.prepare_authors_inputs"
AUTHORS_TRAINING_JOB = "bash cloud_runs/run_authors_training_job.sh"
SYNTHETIC_DATA_DIAGNOSTICS = "python -m experiments.rq2_survey_of_methods.run_synthetic_data_diagnostics"
COMPLETION_BENCHMARK = "python -m experiments.rq3_completion_and_cold_start.run_completion_benchmark"
MEMORISATION_AUDIT = "python -m experiments.rq3_completion_and_cold_start.run_completion_memorisation_audit"
COMPLETE_FAILURE_RECORD = "python -m experiments.rq3_completion_and_cold_start.complete_failure_record"
COLD_START = "python -m experiments.rq3_completion_and_cold_start.run_cold_start_experiment"

EXACT_INPUTS = "inputs/authors_exact_inputs.zip"     # under the store; the retraining jobs wait for it
# The completion engines that use XGBoost or the Forest-Flow regressors were run on the GPU through these options.
GPU_COMPLETION_OPTIONS = '{"forest_flow": {"gpu_hist": true}, "unmasking_trees": {"xgboost_kwargs": {"device": "cuda"}}}'
GPU_XGBOOST_OPTIONS = '{"xgboost_kwargs": {"device": "cuda"}}'
GPU_FOREST_FLOW_OPTIONS = '{"gpu_hist": true}'

TAKEN_OUT_FOR_BUDGET = "compute budget: about a day per run on an L4"
TAKEN_OUT_FOR_BUDGET_NO_PARTIAL_RESULT = ("compute budget: about a day on an L4, with no result before all five "
                                          "folds are done")

JOBS: list[tuple[dict, str | None]] = []    # (job, reason it was taken out of the queue, or None), in output order


def add_job(name: str, lane: str | list[str], priority: int, command: str, taken_out_because: str | None = None,
            **fields) -> None:
    """Add one job; fields that are None are left out, so that the runner's defaults apply."""
    job = {"name": name, "lane": lane, "priority": priority, "command": command}
    job.update({key: value for key, value in fields.items() if value is not None})
    JOBS.append((job, taken_out_because))


def add_arm_jobs(arm: str, seeds: list[int], lane: str | list[str], priority: int, override: str = "{}",
                 taken_out_because: str | None = None, **fields) -> None:
    """One job per seed of an experiment arm (experiments/experiment_arms.py), done once its record exists."""
    for seed in seeds:
        command = f"{ARM_RUNNER} --arm {arm} --seeds {seed} --final-test --no-refit --override '{override}'"
        add_job(f"{arm}_seed{seed}", lane, priority, command, taken_out_because, seed=seed,
                done_when_arm_recorded=[arm, seed], **fields)


# ============================================================= RQ1b: the training code released by Beikmohammadi et al.
# The six input tables of the authors' TrainingCode.py, regenerated exactly (100 imputer rounds, about four hours on
# one core), zipped into the store, from where every retraining job unpacks them. If the zip is uploaded to the store
# by hand instead, this job can be marked as done (state/done/prepare_authors_exact_inputs.json).
add_job("prepare_authors_exact_inputs", "cpu", 1,
        f"{PREPARE_AUTHORS_INPUTS} --imputer-rounds 100 && "
        f'cd "${{APS_AUTHORS_2026_REPOSITORY:-third_party/cost-sensitive-transformer-model}}" && '
        f'mkdir -p "{{STORE}}/inputs" && '
        f'zip -q "{{STORE}}/{EXACT_INPUTS}" aps_failure_*_imputed*.csv aps_failure_*label*.csv')

# Retraining runs (cloud_runs/run_authors_training_job.sh): the authors' seeds 0 to 4 on the published inputs, and
# the test-inclusion controls, in which half or all of the official test rows are added to the
# training table before the authors' 90/10 split, to show how a model trained with the authors' procedure scores test
# rows that were part of its training data.
CONTAMINATION_FRACTION = {"clean": None, "half_test_rows_in_training": 0.5, "all_test_rows_in_training": 1.0}


def add_retraining_stage(job_name: str, run: str, seed: int, stop_epoch: int | str, priority: int,
                         lane: str = "gpu_heavy", needs_jobs: list[str] | None = None) -> None:
    """One stage of a retraining run: train run <run> (its checkpoint folder) until stop_epoch."""
    command = f'{AUTHORS_TRAINING_JOB} {run} {seed} "{{STORE}}" {stop_epoch}'
    fraction = CONTAMINATION_FRACTION[run.removesuffix("_verification").rsplit("_seed", 1)[0]]
    if fraction is not None:
        command += f" --contamination-fraction {fraction}"
    add_job(job_name, lane, priority, command, seed=seed, needs_file=EXACT_INPUTS, max_attempts=8,
            needs_jobs=needs_jobs)


# stage 1: every run to 1,000 epochs; two runs repeated for 300 epochs as verification (TensorFlow on a GPU is not
# bit-for-bit reproducible, and the repeats measure the run-to-run spread)
for run, seed, priority in (("clean_seed0", 0, 10), ("half_test_rows_in_training_seed0", 0, 11), ("clean_seed1", 1, 12),
                            ("clean_seed2", 2, 13), ("clean_seed3", 3, 14), ("clean_seed4", 4, 15),
                            ("half_test_rows_in_training_seed1", 1, 16), ("half_test_rows_in_training_seed2", 2, 17),
                            ("all_test_rows_in_training_seed0", 0, 18)):
    add_retraining_stage(run, run, seed, 1000, priority)
for run in ("clean_seed0_verification", "half_test_rows_in_training_seed0_verification"):
    add_retraining_stage(run, run, 0, 300, 19)
# stage 2: three runs continued to the published budget of 8,000 epochs, in lane gpu_anchor (the port's extension to
# 4,000 epochs, the fourth run of that lane, is with the port jobs below)
for run, seed, priority in (("clean_seed0", 0, 5), ("half_test_rows_in_training_seed0", 0, 6), ("clean_seed1", 1, 7)):
    add_retraining_stage(f"{run}_extended_to_full_budget", run, seed, "{FULL_TRAINING_EPOCHS}", priority,
                         lane="gpu_anchor", needs_jobs=[run])
# stage 3: the other runs to 2,000 epochs while compute units remained
for run, seed, priority in (("clean_seed2", 2, 40), ("clean_seed3", 3, 41), ("clean_seed4", 4, 42),
                            ("half_test_rows_in_training_seed1", 1, 43), ("half_test_rows_in_training_seed2", 2, 44),
                            ("all_test_rows_in_training_seed0", 0, 45)):
    add_retraining_stage(f"{run}_extended_to_2000_epochs", run, seed, 2000, priority, needs_jobs=[run])

# ===================================== RQ1b: the PyTorch port of the cost-sensitive Transformer of Beikmohammadi et al.
# 500 epochs, 10 imputer rounds; the protocol with a validation split of real rows ("corrected") with seeds 0 to 2, the
# published protocol and three ablations of its resampling with seeds 0 and 1. Each run keeps its checkpoint in
# ckpt/<job>/.
PORT_RUNS = (("corrected", "corrected_protocol", "", {0: 20, 1: 21, 2: 21}),
             ("published", "published_protocol", "", {0: 21, 1: 22}),
             ("published", "published_protocol_without_renn", " --no-renn", {0: 22, 1: 23}),
             ("published", "published_protocol_without_svm_smote", " --no-smote", {0: 22, 1: 23}),
             ("published", "published_protocol_without_resampling", " --no-smote --no-renn", {0: 22, 1: 23}))


def port_command(protocol: str, run_name: str, seed: int, epochs: int, checkpoint_folder: str,
                 ablation_flags: str = "") -> str:
    return (f"{PYTORCH_PORT} --protocol {protocol} --epochs {epochs} --patience 100000 --imputer-rounds 10 "
            f'--seed {seed} --run-name {run_name} --checkpoint-dir "{{STORE}}/ckpt/{checkpoint_folder}" '
            f"--checkpoint-every 50{ablation_flags}")


for protocol, run_name, ablation_flags, priority_of_seed in PORT_RUNS:
    for seed, priority in priority_of_seed.items():
        job_name = f"pytorch_port_{run_name}_seed{seed}"
        add_job(job_name, "gpu_heavy", priority, port_command(protocol, run_name, seed, 500, job_name, ablation_flags),
                seed=seed, max_attempts=8)
# Seed 0 of the protocol with a validation split of real rows continued towards 4,000 epochs under its own run name: the
# checkpoint is copied to the new name first, so that the run resumes from epoch 500 and the 500-epoch result files stay
# as they are.
first_stage_folder = "pytorch_port_corrected_protocol_seed0"
add_job("pytorch_port_corrected_protocol_extended_seed0", "gpu_anchor", 8,
        f'cp -n "{{STORE}}/ckpt/{first_stage_folder}/corrected_protocol_seed0.pt" '
        f'"{{STORE}}/ckpt/{first_stage_folder}/corrected_protocol_extended_seed0.pt" 2>/dev/null; '
        + port_command("corrected", "corrected_protocol_extended", 0, 4000, first_stage_folder),
        seed=0, max_attempts=8, needs_jobs=[first_stage_folder])

# ============================================ RQ1b: the preprocessing pipeline of Beikmohammadi et al. with tree models
# Seeds 29 and 47 of the full preprocessing pipeline of Beikmohammadi et al. in front of LightGBM (three folds, 10
# imputer rounds) and of the three-fold baseline they are compared with; seed 11 of both ran on the CPU machine. The
# records of the two tree runs never reached the store: run_tree_models_on_2026_preprocessing.py then appended to the
# repository's own record file (it now honours APS_ARM_RECORDS_FILE), which the runner of the thesis runs did not copy,
# so results/ and the thesis hold seed 11 of that arm only. The runner now copies the lines that a job adds to such a
# file (see "Result files" in queue_runner.py).
for seed in (29, 47):
    add_job(f"lightgbm_tuned_3_folds_seed{seed}", "cpu", 33,
            f"{ARM_RUNNER} --arm lightgbm_tuned --seeds {seed} --folds 3 --final-test --no-refit --override '{{}}'",
            seed=seed, done_when_arm_recorded=["lightgbm_tuned_3_folds", seed])
    add_job(f"lightgbm_on_2026_preprocessing_full_pipeline_seed{seed}", "cpu", 34,
            f"{TREES_ON_2026_PREPROCESSING} --variant full_pipeline --seeds {seed} --folds 3 --imputer-rounds 10 "
            f"--final-test",
            seed=seed, done_when_arm_recorded=["lightgbm_on_2026_preprocessing_full_pipeline", seed])

# ====================================================================== RQ1a and RQ2: classifier arms on the CPU
# seeds 29 and 47 of the other learners with and without rows from the cGAN of Ranasinghe et al., of XGBoost and of the
# ensemble
for arm in ("sklearn_gradient_boosting", "sklearn_gradient_boosting_with_conditional_gan_2019_rows",
            "random_forest_fixed_settings", "random_forest_fixed_settings_with_conditional_gan_2019_rows",
            "xgboost_fixed_settings", "ensemble_without_missingness_indicators"):
    add_arm_jobs(arm, [29, 47], "cpu", 32)
# the SMOTE family (2,000 synthetic failure rows each), the Gaussian copula, Forest-Flow with the reduced profile
# (seed 47; seeds 11 and 29 ran on the CPU machine) and two more seeds of the ensemble with missingness indicators
for arm in ("lightgbm_with_smote_rows", "lightgbm_with_borderline_smote_rows", "lightgbm_with_adasyn_rows",
            "lightgbm_with_svm_smote_rows"):
    add_arm_jobs(arm, [11, 29, 47], "cpu", 30)
add_arm_jobs("lightgbm_with_gaussian_copula_rows", [11, 29, 47], "cpu", 31)
add_arm_jobs("lightgbm_with_forest_flow_rows_reduced_profile", [47], "cpu", 20, max_attempts=3)
add_arm_jobs("ensemble_with_missingness_indicators", [71, 101], "cpu", 21)
# Verification: three deterministic arms of the CPU machine run again in a Colab session, into a record file of
# their own (results/classifier_arms/verification_reruns.jsonl after the merge), to compare the numbers across
# machines.
verification_records = '"{STORE}/results/{SESSION}/classifier_arms/verification_reruns.jsonl"'
add_job("verification_reruns_of_cpu_arms", "cpu", 62,
        " && ".join(f"APS_ARM_RECORDS_FILE={verification_records} {ARM_RUNNER} --arm {arm} --seeds 11 --final-test "
                    f"--no-refit --override '{{}}'"
                    for arm in ("lightgbm_tuned", "lightgbm_with_conditional_gan_2019_rows",
                                "lightgbm_with_duplicated_real_failures")))

# ====================================================================== RQ2: arms that need a GPU (L4)
add_arm_jobs("tabicl_bagged_contexts", [29], "gpu_light", 30)
add_arm_jobs("lightgbm_with_tabebm_rows", [11, 29, 47], "gpu_light", 31)
add_arm_jobs("lightgbm_with_ctgan_rows", [11, 29, 47], "gpu_light", 32)
add_arm_jobs("lightgbm_with_tvae_rows", [11, 29, 47], "gpu_light", 32)
add_arm_jobs("lightgbm_with_tabddpm_rows", [11, 29, 47], "gpu_light", 33)
add_arm_jobs("lightgbm_with_ctab_gan_plus_rows", [11, 29, 47], "gpu_light", 34)
add_arm_jobs("ensemble_with_missingness_indicators_and_tabicl", [11, 29, 47], "gpu_light", 35)
# GReaT (a language model writes the failure rows); it fits on an A100 or an L4, and it produced no usable rows
for seed, priority, taken_out_because in ((11, 34, None), (29, 90, TAKEN_OUT_FOR_BUDGET)):
    add_job(f"lightgbm_with_great_language_model_rows_seed{seed}", ["gpu_heavy", "gpu_light"], priority,
            f"{ARM_RUNNER} --arm lightgbm_with_great_language_model_rows --seeds {seed} --final-test --no-refit "
            f"--override '{{}}'",
            taken_out_because, seed=seed)
# the middle and paper profiles of Forest-Flow (only the reduced profile completed)
add_arm_jobs("lightgbm_with_forest_flow_rows_middle_profile", [11], "gpu_light", 39,
             taken_out_because=TAKEN_OUT_FOR_BUDGET_NO_PARTIAL_RESULT, max_attempts=3)
add_arm_jobs("lightgbm_with_forest_flow_rows_middle_profile", [29, 47], "gpu_light", 39,
             taken_out_because=TAKEN_OUT_FOR_BUDGET, max_attempts=3)
add_arm_jobs("lightgbm_with_forest_flow_rows_paper_profile", [11], "gpu_light", 70,
             override='{"use_gpu_histograms": true}', taken_out_because=TAKEN_OUT_FOR_BUDGET, weight=2)
# Lane "extra", run at the end by one L4 session, one job at a time: TabICL seeds 11 and 47 again (they had failed with
# CUDA out-of-memory errors when three TabICL jobs shared one L4) and seeds 71 and 101 of TabICL and of the ensemble
# with TabICL, so that these arms rest on five seeds like the baseline
for arm, priority_of_seed in (("tabicl_bagged_contexts", {11: 1, 47: 2, 71: 5, 101: 6}),
                              ("ensemble_with_missingness_indicators_and_tabicl", {71: 3, 101: 4})):
    for seed, priority in priority_of_seed.items():
        add_arm_jobs(arm, [seed], "extra", priority)
# row-quality diagnostics (detectability, distance to closest record, memorisation) of the generators' output
add_job("synthetic_data_diagnostics_cpu_generators", "cpu", 45,
        f"{SYNTHETIC_DATA_DIAGNOSTICS} --generators conditional_gan_2019 duplicated_real_failures forest_flow "
        f"gaussian_copula smote borderline_smote adasyn svm_smote")
add_job("synthetic_data_diagnostics_gpu_generators", "gpu_light", 46,
        f"{SYNTHETIC_DATA_DIAGNOSTICS} --generators tabebm ctgan tvae tabddpm ctab_gan_plus")

# ====================================================================== RQ3: completion and cold start
for fold in range(5):
    add_job(f"completion_benchmark_fold{fold}", "gpu_light", 35,
            f"{COMPLETION_BENCHMARK} --folds {fold} --engines median k_nearest_neighbours chained_equations forest_flow "
            f"unmasking_trees tabpfn --override '{GPU_COMPLETION_OPTIONS}'",
            weight=2, max_attempts=3)     # weight 2: one fold uses most of an L4's memory
add_job("completion_memorisation_audit", "extra", 7,
        f"{MEMORISATION_AUDIT} --engines median k_nearest_neighbours chained_equations unmasking_trees forest_flow "
        f"tabpfn --override '{GPU_COMPLETION_OPTIONS}'",
        max_attempts=1)
# the demonstration of the prototype shown in the thesis; its console output is the job's log
add_job("prototype_demonstration", "extra", 8,
        f"{COMPLETE_FAILURE_RECORD} --engine unmasking_trees --demo-index 7 --hidden-fraction 0.5 --both-hypotheses "
        f"--healthy-rows 1000 --attributes-to-show 12 --override '{GPU_XGBOOST_OPTIONS}'",
        max_attempts=1)


def cold_start_command(generator: str, seed: int, override: str, failure_counts: str = "25 50 100 250 1000") -> str:
    return (f"{COLD_START} --generator {generator} --seeds {seed} --failure-counts {failure_counts} "
            f"--override '{override}'")


# cold-start curves: the cost as a function of the number of labelled failures, without synthetic rows, with
# duplicated real failures and with 2,000 rows of each generator
for generator in ("no_synthetic_rows", "duplicated_real_failures", "smote"):
    for seed in (11, 29, 47):
        add_job(f"cold_start_{generator}_seed{seed}", "cpu", 40, cold_start_command(generator, seed, "{}"),
                seed=seed, max_attempts=3)
# the generators that need a GPU: every generator gets one seed before any gets three
for generator, override in (("conditional_gan_2019", "{}"), ("tabebm", "{}"), ("forest_flow", GPU_FOREST_FLOW_OPTIONS),
                            ("forest_flow_inpainting", GPU_FOREST_FLOW_OPTIONS)):
    for seed, priority in ((11, 36), (29, 37), (47, 38)):
        add_job(f"cold_start_{generator}_seed{seed}", "gpu_light", priority,
                cold_start_command(generator, seed, override), seed=seed, max_attempts=3)
add_job("cold_start_unmasking_trees_inpainting_seed11", "gpu_light", 58,
        cold_start_command("unmasking_trees_inpainting", 11, GPU_XGBOOST_OPTIONS), seed=11, weight=2)
# Unmasking Trees inpainting had the lowest cost of all generators at 25 and 50 labelled failures on seed 11; seeds 29
# and 47 at those two counts test whether that holds. Each run writes a record file of its own.
for seed in (29, 47):
    records = (f'"{{STORE}}/results/{{SESSION}}/rq3_completion_and_cold_start/'
               f'cold_start_runs_unmasking_trees_inpainting_seed{seed}.jsonl"')
    add_job(f"cold_start_unmasking_trees_inpainting_seed{seed}_25_and_50_failures", "extra", 9,
            f"APS_COLD_START_RECORDS_FILE={records} "
            + cold_start_command("unmasking_trees_inpainting", seed, GPU_XGBOOST_OPTIONS, failure_counts="25 50"),
            seed=seed)


def write_job_list() -> None:
    names = [job["name"] for job, _ in JOBS]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate job names: {duplicates}")
    lines = ["# Job list of the Colab queue (cloud_runs/queue_runner.py), written by cloud_runs/build_job_list.py:",
             "# change that script and run it again rather than editing this file. One job per line; the runner skips",
             "# lines that start with #, which keeps the jobs that were taken out of the queue together with the reason."]
    for job, taken_out_because in JOBS:
        line = json.dumps(job)
        lines.append(f"# taken out of the queue ({taken_out_because}): {line}" if taken_out_because else line)
    OUTPUT_FILE.write_text("\n".join(lines) + "\n")
    active = [job for job, taken_out_because in JOBS if not taken_out_because]
    print(f"{len(active)} jobs and {len(JOBS) - len(active)} jobs taken out of the queue -> {OUTPUT_FILE}")
    for lane in ("gpu_anchor", "gpu_heavy", "gpu_light", "cpu", "extra"):
        count = sum(1 for job in active if lane in (job["lane"] if isinstance(job["lane"], list) else [job["lane"]]))
        print(f"  {lane:10s} {count}")


if __name__ == "__main__":
    write_job_list()
