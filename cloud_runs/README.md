# Running the long jobs on Google Colab

The experiments of the thesis ran partly on a two-core CPU machine, with the commands in the main
[README](../README.md), and partly on Google Colab, in November 2026, in several sessions at the same time: runtimes
with an A100 GPU, with an L4 GPU and with CPUs only. Colab ran the jobs that need a GPU and many CPU jobs: 57 of the 94
arm records, the retraining runs and the PyTorch port of RQ1b, and the whole completion benchmark and cold start
(section 5 of the main README gives the hours). They were paid for with Colab compute units; about 935 of the 979 units
bought were used. This folder holds what was needed for that: a job queue that the sessions share through a folder in
Google Drive, the list of every job in the queue, the notebook that starts a session, and a script that merges the
sessions' results into `results/`.

## Why a queue

A Colab runtime ends after at most 24 hours, and earlier when it is disconnected, while some jobs ran for more than
a day: an 8,000-epoch run of the training code released by Beikmohammadi et al. takes about 30 hours on an A100. Several
runtimes ran at once, and the only storage that all of them can see is Google Drive. The sessions were therefore
coordinated through files in one Drive folder, called the store below. Every session runs the same queue runner, which
takes the next job that its hardware can run, runs it, and records the outcome in the store. Long jobs keep their
checkpoints in the store, so when a runtime ends, its unfinished jobs are taken up again by the next session that can
run them and continue from their last checkpoint.

## Files

| File | What it is |
|---|---|
| `queue_runner.py` | the queue runner; its module docstring describes the store, the job fields, the claim protocol and the control files in detail |
| `job_list.jsonl` | every job of the queue, one per line, with the command, lane, priority, weight, attempts and dependencies it ran with; the jobs taken out of the queue before they ran are kept as comment lines with the reason, and the six jobs `*_extended_to_2000_epochs` were still pending when the sessions were drained and never started |
| `build_job_list.py` | writes `job_list.jsonl` (`python cloud_runs/build_job_list.py`); it explains how the work was ordered |
| `run_authors_training_job.sh` | one stage of one retraining run of the code released by Beikmohammadi et al., as a queue job |
| `setup_environment.sh` | prepares a Colab runtime: data, packages, external repositories, exact inputs |
| `requirements-colab.txt` | the packages installed on top of the Colab image |
| `colab/colab_bootstrap.py` | start-up code of the notebooks and the table of sessions |
| `colab/build_colab_notebooks.py` | writes one notebook per session |
| `colab/queue_session_l4_1.ipynb` | the notebook of session `l4-1`, as an example (the others differ only in the session name and the runtime type) |
| `merge_cloud_results.py` | merges a downloaded store into `results/` |

The jobs were given full names for publication, like the experiments: `clean_seed0` was `authors_seed0`,
`half_test_rows_in_training_seed0` was `contam50_seed0`, `pytorch_port_published_protocol_without_renn_seed0` was
`abl_norenn_seed0`, `cold_start_forest_flow_inpainting_seed11` was `coldstart_inpaint_ff_s11`, and an arm job is
named after its arm and seed. The arm, generator and engine names are those of [`results/README.md`](../results/README.md).

## The store

The store of the thesis runs was `My Drive/aps_thesis`. The runner creates what it needs:

```
<store>/
  job_list.jsonl                     optional; used instead of cloud_runs/job_list.jsonl when present, so that jobs
                                     can be added or changed while sessions run
  inputs/authors_exact_inputs.zip    the six input tables of the authors' training code (see below)
  inputs/data/                       optional copies of aps_failure_training_set.csv and aps_failure_test_set.csv
  state/done/<job>.json              finished jobs
  state/failed/<job>.json            failed attempts and the last exit code
  state/locks/<job>.<session>.json   the claim and the heartbeat of the session that runs a job
  logs/<job>.log                     console output of every attempt of a job
  logs/environment_<session>.log     output of setup_environment.sh
  results/<session>/                 the result files of that session's jobs, laid out like results/
  ckpt/<job or run>/                 checkpoints: the finished folds of an arm, the state of a Transformer run
  runs/<run>/                        training logs and best-validation models of the authors' retraining runs
  control/                           files that steer running sessions (see below)
```

The retraining jobs of the authors' code wait for `inputs/authors_exact_inputs.zip`, a ZIP file of the six tables
that `experiments/rq1_transformer_2026_reproduction/authors_code/prepare_authors_inputs.py` writes into the authors'
repository. The job `prepare_authors_exact_inputs` makes it (about four hours on one CPU core). If you make the
tables elsewhere and upload the ZIP file, also create `state/done/prepare_authors_exact_inputs.json` (containing `{}`),
so that no session repeats the work.

## Lanes and sessions

Every job has a lane, and every session runs the jobs of the lanes it was started with. The lanes group the jobs by
the hardware they need. Most GPU time went to RQ1b: the training code released by Beikmohammadi et al. was retrained
from scratch on the published inputs (the retraining runs) and in test-inclusion controls, in which half or all of the
official test rows are deliberately added to the training table, to show how a model trained with the authors' procedure
scores test rows that were part of its training data; the PyTorch port of their network was trained under the published
protocol and under the protocol with a validation split of real rows (`corrected` in run names).

| Lane | Jobs | Runtime |
|---|---|---|
| `gpu_anchor` | the four runs continued towards the full budget: the authors' code with the retraining runs of seeds 0 and 1 and the control with half of the test rows (seed 0) to 8,000 epochs, the port of seed 0 under the protocol with a validation split of real rows to 4,000 | A100 |
| `gpu_heavy` | the other retraining runs of the authors' code (1,000 epochs, 300 for the verification runs, then 2,000), the PyTorch port and its ablations (500 epochs); GReaT, which may also run in `gpu_light` | A100 |
| `gpu_light` | the arms with TabICL, TabEBM, CTGAN, TVAE, TabDDPM and CTAB-GAN+ rows and the ensemble with TabICL; the completion benchmark; the cold-start curves of the cGAN of Ranasinghe et al., TabEBM, Forest-Flow and Forest-Flow inpainting; the diagnostics of the GPU generators | L4 |
| `cpu` | the tree arms, the SMOTE family, the Gaussian copula, Forest-Flow with the reduced profile, the ensembles, the preprocessing pipeline of Beikmohammadi et al. with LightGBM and its three-fold baseline, the cold-start curves without a generator or with duplicates or SMOTE, the CPU diagnostics, the verification reruns, the exact inputs | CPU, or spare slots of an L4 session |
| `extra` | added at the end: TabICL seeds that had failed for lack of GPU memory, two more seeds of TabICL and of the ensemble with TabICL, the memorisation audit, the prototype demonstration, two scarce-data cold-start runs | one L4, one job at a time |

The heavy lane ran on A100s only. The authors' Transformer took about 13 s per epoch there with XLA (TensorFlow's
compiler), the PyTorch port 18–19 s; an L4 was about three times slower per step, which made an epoch cost about as many
compute units as on the A100 but took three times as long. The sessions of the thesis runs, with their settings in
`colab/colab_bootstrap.py`:

| Session | Runtime | Lanes | Job slots | Exits when idle for |
|---|---|---|---|---|
| `a100-1`, `a100-2` | A100 | `gpu_anchor`, `gpu_heavy` | 2 | 20 minutes |
| `a100-3` | A100 | `gpu_heavy` | 2 | 20 minutes |
| `l4-1` | L4 | `gpu_light`, `cpu` | 2 | 30 minutes |
| `l4-2` | L4 | `gpu_light` | 2 | 30 minutes |
| `cpu-1` | CPU | `cpu` | 4 | 120 minutes |

A job occupies as many slots as its weight (most have weight 1; a fold of the completion benchmark has weight 2,
because it needs most of an L4's memory). The L4 sessions had two slots because three GPU jobs at once did not fit
into an L4's memory. On GPU runtimes the notebook starts NVIDIA MPS, which lets the jobs of a session share the GPU; two or three
training processes under MPS finished 25–30 % more epochs per hour than without it. Each job gets an equal share of
the CPU cores. Towards the end, `l4-1` was switched to the lane `extra` alone through its configuration file (see
below). An A100 runtime used about six compute units an hour, an L4 runtime about 2.5 and a CPU runtime well under
one.

## The job list

`job_list.jsonl` has one JSON object per line. The fields:

| Field | Meaning |
|---|---|
| `name` | unique; names the job's state, lock and log files and, for most jobs, its checkpoint folder |
| `lane` | a lane or a list of lanes (default `cpu`) |
| `priority` | lower numbers run first (default 100) |
| `command` | the shell command, run in the repository root; `{STORE}`, `{JOB}`, `{SESSION}`, `{SEED}` and `{FULL_TRAINING_EPOCHS}` (8,000 unless `--full-training-epochs` says otherwise) are replaced first |
| `weight` | job slots it occupies (default 1) |
| `max_attempts` | how often a failing job is started (default 2; 8 for the Transformer runs) |
| `needs_jobs` | jobs that must be finished first (the later stages of a run wait for the earlier ones) |
| `needs_file` | a file in the store that must exist first |
| `done_when_arm_recorded` | `[arm, seed]`: the job is finished as soon as that arm and seed have a record, in the repository's `results/classifier_arms/arm_records.jsonl` or in a session's results |
| `seed`, `threads` | the job's seed (for `{SEED}`), and CPU threads if not an equal share of the cores |

Because of `done_when_arm_recorded`, a session started from this repository skips every arm whose record is already
in `results/`. To repeat such a run, remove the field from the job (in a copy of the job list placed in the store).

## Starting a session

1. Create the store folder in Google Drive, `My Drive/aps_thesis` (another place works if you change `STORE` in the
   notebook). Optionally add `inputs/data/` with the two data files (otherwise every session downloads them from the
   UCI repository) and `inputs/authors_exact_inputs.zip` (see above).
2. Open `cloud_runs/colab/queue_session_l4_1.ipynb` in Colab (*File → Upload notebook*, or *File → Open notebook →
   GitHub*). For another session, change `SESSION` in its first code cell, or write the notebooks of all sessions with
   `python cloud_runs/colab/build_colab_notebooks.py --output-dir <folder>`.
3. Choose the session's runtime: *Runtime → Change runtime type*, then A100 GPU, L4 GPU or CPU, with high RAM.
4. The first code cell clones this repository from GitHub (`REPOSITORY_URL`). To run a modified copy instead, set
   `REPOSITORY_URL = ""` and copy the repository into the store as `repository/`; the notebook then copies it from
   there.
5. *Runtime → Run all*, and allow the access to Google Drive when Colab asks for it. Nothing else needs attention
   while the session runs.

The first code cell mounts Drive, puts a fresh copy of the repository on the runtime's disk and prepares the session
(`colab_bootstrap.prepare_session`). The second runs `setup_environment.sh`; its full output goes to
`logs/environment_<session>.log` in the store, and its last lines say whether every package imports and whether the
GPU is visible. The third runs the queue runner, which prints a line for every job it starts and ends:

```
[hh:mm:ss] START <job>: <command>
[hh:mm:ss] DONE  <job> in <hours> h (<number> result files)
[hh:mm:ss] FAIL  <job> exit <code> (attempt <k>); see logs/<job>.log
```

When nothing is left for the session, the runner exits and the last cell hands the runtime back to Colab, so that a
session that finishes at night stops using compute units. Several sessions are started in the same way, each in its
own notebook and runtime and each with its own session name. Whether a session keeps running after the browser tab
is closed depends on the Colab plan (background execution); without it, keep the tab open.

A runtime lives at most 24 hours. To move a session to a fresh runtime, use *Runtime → Disconnect and delete
runtime* in the notebook that runs it, then *Run all* again. If the old runtime of the session is still alive (its
locks are fresh and were written by another machine), the new runner waits up to 20 minutes for them to go stale and
then gives up, so that the same job never runs twice into the same checkpoint folder.

The runner does not depend on Colab: any machine that sees the store can run a session with
`python cloud_runs/queue_runner.py --store <store> --session <name> --lanes <lanes> --slots <n>` from the repository
root, after installing the packages.

## Watching the queue

From any notebook or terminal that sees the store (for example a computer with Google Drive for desktop):

```
python cloud_runs/queue_runner.py --store <store> --status
```

prints every job with its priority, lanes and state (`done`, `running in <session>`, `pending`,
`waiting (needs <job or file>)`, `FAILED x<attempts>`), and a count of the states. While a job runs, its console output
is copied to `logs/<job>.log` every minute. A retraining run also shows its progress in `ckpt/<run>/state.json` (the
last saved epoch) and `ckpt/<run>/test_trace.jsonl` (the official test cost at the 0.5 cut-off every 50 epochs, a
diagnostic only; for a test-inclusion control separately for the test rows in and out of the training table).

## Steering running sessions

Every runner reads the folder `control/` in the store about once a minute:

| File | Effect |
|---|---|
| `drain_<session>` | the session claims no new jobs and exits when its running jobs have finished |
| `drain_all` | the same for every session (used when the compute units ran low) |
| `kill_<job>` | stops the job in whichever session runs it; the job stays pending and is claimed again later |
| `kill_<job>__<session>` | the same, in that session only |
| `config_<session>.json` | e.g. `{"slots": 2, "lanes": ["extra"], "threads_per_job": 3}`: new slots, lanes or CPU threads for the jobs the session starts from then on |
| `keep_<session>` | the notebook keeps the runtime after the queue has finished (for inspection) |

## How a job resumes

A job that exits with an error is started again until it has used its `max_attempts`; the attempts are counted in
`state/failed/<job>.json`, and deleting that file gives the job a fresh set of attempts. A job stopped through a
control file, or because its runner was stopped, stays pending and is not counted as a failure. A job whose runtime
ended leaves a lock that no runner refreshes; 25 minutes later the job can be claimed again, by any session with a
matching lane. In all these cases the job continues from what it left in the store:

* an experiment arm stores every finished fold in `ckpt/<job>/` and computes only the missing folds
  (`experiments/run_experiment_arm.py`; the runner switches this on through `APS_CLOUD_STORE` and `APS_JOB_NAME`);
* a retraining run of the authors' code saves the model every 50 epochs in `ckpt/<run>/` and continues from there,
  and a later stage of the same run (a larger stop epoch) continues from the earlier stage's last epoch;
* a run of the PyTorch port saves its training state every 50 epochs in `ckpt/<job>/<run name>_seed<k>.pt`.

The other jobs (cold-start curves, completion folds, diagnostics, the preprocessing pipeline of Beikmohammadi et al.
with LightGBM) have no checkpoints and start again from the beginning.

Two sessions can try to claim the same job within seconds, and Google Drive does not prevent both from creating
their lock files. Every session therefore writes its own lock file and decides only after listing the job's locks
twice, 30 seconds apart; the earliest claim wins. A running job checks again at every heartbeat, and a duplicate that
finds an earlier claim of another session stops itself.

## Merging the results

When the sessions are finished, download the store folder (Google Drive for desktop, or *Download* on
drive.google.com, which gives a ZIP file to unpack) and run from the repository root

```
python cloud_runs/merge_cloud_results.py --store <downloaded store>             # report what would be merged
python cloud_runs/merge_cloud_results.py --store <downloaded store> --apply     # merge it
```

The script adds what `results/` does not have yet and never overwrites anything: new arm records (marked with
`"source": "colab:<session>"`) with their probability files, new cold-start and completion records (keyed on
generator, seed and number of labelled failures, and on fold, mask and engine), the verification reruns, new entries
of the diagnostics and the memorisation audit, the PyTorch port's result files, and the files of every retraining run
of the authors' code from `ckpt/<run>/` into `results/rq1_transformer_2026_reproduction/authors_code/retraining_runs/`.
Its module docstring lists the rules. Afterwards the tables and figures are rebuilt with
`python -m analysis.run_all_summaries`.

The arm runner, the cold-start study and the completion benchmark write their records straight into the session's
folder in the store (the runner sets `APS_ARM_RECORDS_FILE`,
`APS_COLD_START_RECORDS_FILE` and `APS_COMPLETION_RECORDS_FILE`). Every other result file is written into the
repository on the runtime and copied to the store when its job ends; of a record file (`*.jsonl`) only the new lines
are appended, because a fresh runtime starts from the repository's copy and would otherwise overwrite the records of
an earlier runtime. In the thesis runs the runner still copied record files whole. The cold-start records of two
sessions were therefore overwritten by later runtimes and had to be restored from earlier copies, and the records of
the two cloud runs of the preprocessing pipeline of Beikmohammadi et al. with LightGBM (seeds 29 and 47), whose script
at the time wrote into the repository's own record file (it now honours `APS_ARM_RECORDS_FILE`), were not copied at all;
`results/` holds seed 11 of that arm only.

Apart from that copying rule, the runner here differs from the one that ran the thesis jobs only in its bookkeeping:
a job that cannot start for lack of free slots no longer ends the runner's pass over the job list; a job stopped by
the runner's own shutdown is not counted as a failed attempt; `--status` only reads the store; a fresh runtime
continues the store's job log instead of starting a new one; the TabICL GPU settings are set only in GPU sessions; and
the mechanism that delivered code updates to sessions that were already running is gone. The experiment code it starts
is the code in `experiments/`.
