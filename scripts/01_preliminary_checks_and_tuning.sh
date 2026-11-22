#!/usr/bin/env bash
# Step 1 (README, section 5): the first look at the data, the hyperparameter search for LightGBM and the
# threshold-rule demonstration. None of this uses the test set except the demonstration, which shows how the
# test cost depends on the rule that picks the threshold.
#
# Times on the two-core machine used for the thesis: 0.4 to 12 minutes per first-look configuration (about 25 in
# all), about 50 minutes for the search and 13 minutes for the demonstration.
#
# The search overwrites results/hyperparameter_search/lightgbm_seed11_best_parameters.json, which every arm with
# tuned LightGBM reads. Skip it if you want to keep the settings used in the thesis.
set -euo pipefail
cd "$(dirname "$0")/.."
run() { echo; echo "+ $*"; "$@"; }

run python -m experiments.quick_baseline_check
# exactly 26 trials: the number that the stored search completed within its time limit of 3,300 s (README, step 1)
run python -m experiments.hyperparameter_search --model lightgbm --trials 26 --timeout 1000000 --seed 11
run python -m experiments.run_threshold_rule_demonstration
