#!/usr/bin/env bash
# Step 2 (README, section 5): the baseline of every comparison, the tuned LightGBM without synthetic rows,
# on the five seeds of the thesis. About 3 minutes per seed on the two-core machine.
# Without --no-refit the runner also fits the secondary model on all training rows (a check only); in the thesis
# that was done for seeds 11, 29 and 47.
set -euo pipefail
cd "$(dirname "$0")/.."
run() { echo; echo "+ $*"; "$@"; }

run python -m experiments.run_experiment_arm --arm lightgbm_tuned --seeds 11 29 47 --final-test
run python -m experiments.run_experiment_arm --arm lightgbm_tuned --seeds 71 101 --final-test --no-refit
