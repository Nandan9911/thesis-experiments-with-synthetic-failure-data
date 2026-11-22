#!/usr/bin/env bash
# A quick check that the installation works: the main entry points on small stratified samples of the training data,
# with a few epochs or GAN steps where that applies. The printed numbers are meaningless. Every output file has
# smoke_test in its name and is deleted at the end; the real results are not touched.
# Needs requirements-core.txt plus the packages of the completion engines used here (none beyond scikit-learn).
# Takes about ten minutes on a laptop.
set -euo pipefail
cd "$(dirname "$0")/.."
run() { echo; echo "+ $*"; "$@"; }
cleanup() { find results cache -name '*smoke_test*' -prune -exec rm -rf {} + 2>/dev/null || true; }
trap cleanup EXIT

# an arm without synthetic rows, one with the cGAN of Ranasinghe et al. (a few hundred training steps instead of 30,000)
# and one with SMOTE
run python -m experiments.run_experiment_arm --arm lightgbm_tuned --seeds 11 --smoke-test-rows 3000 --final-test
run python -m experiments.run_experiment_arm --arm lightgbm_with_conditional_gan_2019_rows --seeds 11 --smoke-test-rows 3000 \
    --final-test --no-refit --override '{"gan_training_steps": 300}'
run python -m experiments.run_experiment_arm --arm lightgbm_with_smote_rows --seeds 11 --smoke-test-rows 3000 --final-test --no-refit

# the PyTorch port of the cost-sensitive Transformer of Beikmohammadi et al., two epochs
run python -m experiments.rq1_transformer_2026_reproduction.run_transformer_2026_port --protocol corrected --epochs 2 \
    --imputer-rounds 2 --smoke-test-rows 3000

# completion with the three classical engines on 200 failures, and the cold start with duplicated failures
run python -m experiments.rq3_completion_and_cold_start.run_completion_benchmark --folds 0 \
    --engines median k_nearest_neighbours chained_equations --smoke-test-rows 200
run python -m experiments.rq3_completion_and_cold_start.run_cold_start_experiment --generator duplicated_real_failures \
    --seeds 11 --failure-counts 25 --smoke-test-rows 5000

# the prototype with the k-nearest-neighbours engine
run python -m experiments.rq3_completion_and_cold_start.complete_failure_record --engine k_nearest_neighbours \
    --demo-index 7 --draws 5 --output-json results/rq3_completion_and_cold_start/prototype_smoke_test.json

echo
echo "Smoke test passed."
