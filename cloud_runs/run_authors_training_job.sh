#!/usr/bin/env bash
# One stage of one retraining run with the training code released by Beikmohammadi et al. (RQ1b), as a job of the Colab
# queue.
#
#   bash cloud_runs/run_authors_training_job.sh <run name> <seed> <store> <stop epoch> [further options]
#   e.g. bash cloud_runs/run_authors_training_job.sh half_test_rows_in_training_seed0 0 /content/drive/MyDrive/aps_thesis 1000 --contamination-fraction 0.5
#
# The training is experiments/rq1_transformer_2026_reproduction/authors_code/train_authors_model_with_resume.py, which
# runs the authors' TrainingCode.py with checkpoints, a trace of the official test cost every 50 epochs and, for the
# test-inclusion controls, a share of the official test rows added to the training table (its docstring explains
# all of this). Further options are passed on to it. This wrapper
#   * makes sure that the authors' repository (third_party/cost-sensitive-transformer-model, at the pinned commit)
#     and the six exactly regenerated input tables are present; the tables come from <store>/inputs/
#     authors_exact_inputs.zip, which the queue job prepare_authors_exact_inputs writes;
#   * gives the run a working copy of TrainingCode.py and the six tables, because the authors' code writes its CSV
#     log and its model folders into the folder it runs in, and several runs share one machine;
#   * keeps the checkpoint in <store>/ckpt/<run name>/ and copies the CSV log and the best-validation model folders
#     to <store>/runs/<run name>/ when it ends, so that a new runtime or a later stage continues where it stopped;
#   * keeps the final evaluation of an earlier stage before this stage overwrites it;
#   * reports success only when the run has reached the stop epoch (a run cut short stays pending in the queue).
#
# The run names of the thesis are clean_seed<k> (k = 0..4), half_test_rows_in_training_seed<k> (k = 0..2),
# all_test_rows_in_training_seed0, clean_seed0_verification and half_test_rows_in_training_seed0_verification
# (cloud_runs/build_job_list.py). Every run first went to 1,000 epochs (the verification runs to 300); later stages
# of a run use the same run name and a larger stop epoch, and continue from its checkpoint.
#
# Environment: APS_TENSORFLOW_THREADS (CPU threads of TensorFlow per run, default 2).
set -u
RUN_NAME=$1
SEED=$2
STORE=$3
STOP_EPOCH=$4
shift 4
REPOSITORY_ROOT=$(cd "$(dirname "$0")/.." && pwd)
AUTHORS_REPOSITORY=$REPOSITORY_ROOT/third_party/cost-sensitive-transformer-model
CHECKPOINT_DIR=$STORE/ckpt/$RUN_NAME
RUN_DIR=$STORE/runs/$RUN_NAME
WORKING_COPY=$REPOSITORY_ROOT/cache/authors_training_working_copies/$RUN_NAME
cd "$REPOSITORY_ROOT" || exit 2

if [ ! -f "$AUTHORS_REPOSITORY/TrainingCode.py" ]; then
  bash third_party/fetch_third_party_code.sh \
    || { echo "the authors' repository is missing; run bash third_party/fetch_third_party_code.sh"; exit 3; }
fi
if [ ! -f "$AUTHORS_REPOSITORY/aps_failure_train_valid_set_label_resampled.csv" ]; then
  if [ -f "$STORE/inputs/authors_exact_inputs.zip" ]; then
    unzip -o -q "$STORE/inputs/authors_exact_inputs.zip" -d "$AUTHORS_REPOSITORY/" && echo "exact inputs unpacked"
  else
    echo "the exact inputs are missing in $STORE/inputs/; run the job prepare_authors_exact_inputs first"
    exit 3
  fi
fi
mkdir -p "$WORKING_COPY"
cp -n "$AUTHORS_REPOSITORY"/TrainingCode.py "$AUTHORS_REPOSITORY"/aps_failure_*imputed*.csv \
      "$AUTHORS_REPOSITORY"/aps_failure_*label*.csv "$WORKING_COPY"/ 2>/dev/null || true
# the CSV log and the best-validation model folders of an earlier session (the checkpoint itself is in the store)
if [ -d "$RUN_DIR/CSV" ]; then cp -r "$RUN_DIR/CSV" "$WORKING_COPY"/ 2>/dev/null || true; fi
if [ -d "$RUN_DIR/Model" ]; then cp -r "$RUN_DIR/Model" "$WORKING_COPY"/ 2>/dev/null || true; fi

export TF_USE_LEGACY_KERAS=1 TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2
# several runs share one machine: cap the CPU threads of each (the model is small, the threads only add contention)
THREADS=${APS_TENSORFLOW_THREADS:-2}
export TF_NUM_INTRAOP_THREADS=$THREADS TF_NUM_INTEROP_THREADS=$THREADS OMP_NUM_THREADS=$THREADS

copy_log_and_models_to_store() {
  mkdir -p "$RUN_DIR"
  cp -r "$WORKING_COPY/CSV" "$RUN_DIR/" 2>/dev/null || true
  cp -r "$WORKING_COPY/Model" "$RUN_DIR/" 2>/dev/null || true
}
trap copy_log_and_models_to_store EXIT

# Every stage ends by writing final_evaluation.json, which a later stage overwrites. Before this stage can do so,
# keep the earlier evaluation as final_evaluation_after_<E>_epochs.json, E being the epochs it evaluated (its
# epochs_run field). Nothing is copied when that file exists already, or when this job neither continues past E nor
# finds the run already past E (a stage started again after it had finished evaluates the same epoch again).
if [ -f "$CHECKPOINT_DIR/final_evaluation.json" ] && [ -f "$CHECKPOINT_DIR/state.json" ]; then
  python - "$CHECKPOINT_DIR" "$STOP_EPOCH" <<'EOF'
import json
import shutil
import sys
from pathlib import Path

checkpoint_dir, stop_epoch = Path(sys.argv[1]), int(sys.argv[2])
try:
    evaluated_epochs = int(json.loads((checkpoint_dir / "final_evaluation.json").read_text())["epochs_run"])
    current_epoch = int(json.loads((checkpoint_dir / "state.json").read_text())["epoch"])
except (OSError, KeyError, ValueError) as error:
    print(f"earlier final evaluation not kept ({error})")
    sys.exit(0)
kept = checkpoint_dir / f"final_evaluation_after_{evaluated_epochs}_epochs.json"
if max(current_epoch, stop_epoch) > evaluated_epochs and not kept.exists():
    shutil.copyfile(checkpoint_dir / "final_evaluation.json", kept)
    print(f"kept the final evaluation after {evaluated_epochs} epochs as {kept.name}")
EOF
fi

APS_AUTHORS_2026_REPOSITORY=$WORKING_COPY \
  python -m experiments.rq1_transformer_2026_reproduction.authors_code.train_authors_model_with_resume \
    --seed_number "$SEED" --num_epochs 8000 --stop-epoch "$STOP_EPOCH" \
    --checkpoint-dir "$CHECKPOINT_DIR" --checkpoint-every 50 --test-every 50 "$@"
exit_code=$?
# the job counts as finished only when the stop epoch was reached (a time limit leaves it for the next session)
if [ $exit_code -eq 0 ] && [ -f "$CHECKPOINT_DIR/state.json" ]; then
  python - "$CHECKPOINT_DIR/state.json" "$STOP_EPOCH" <<'EOF'
import json
import sys

epoch = json.load(open(sys.argv[1]))["epoch"]
target = int(sys.argv[2])
print(f"epochs run: {epoch} / {target}")
sys.exit(0 if epoch >= target else 4)
EOF
  exit_code=$?
fi
exit $exit_code
