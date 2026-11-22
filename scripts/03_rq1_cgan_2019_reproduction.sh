#!/usr/bin/env bash
# Step 3 (README, section 5): RQ1a, the conditional GAN of Ranasinghe et al. under the common evaluation protocol, with
# the same learners without synthetic rows and a duplication control as references. Times per seed on the two-core
# machine are given above each block.
set -euo pipefail
cd "$(dirname "$0")/.."
run() { echo; echo "+ $*"; "$@"; }

# the cGAN rows with the tuned LightGBM (about 22 minutes per seed), and 2,000 duplicated real failures as the
# control (about 3 minutes per seed)
run python -m experiments.run_experiment_arm --arm lightgbm_with_conditional_gan_2019_rows --seeds 11 29 47 --final-test
run python -m experiments.run_experiment_arm --arm lightgbm_with_conditional_gan_2019_rows --seeds 71 101 --final-test --no-refit
run python -m experiments.run_experiment_arm --arm lightgbm_with_duplicated_real_failures --seeds 11 29 47 71 101 --final-test --no-refit

# the learners closest to those of the study of Ranasinghe et al., each with and without the cGAN rows
# (random forest about 15 and 26 minutes per seed, scikit-learn gradient boosting about 53 and 70 minutes)
run python -m experiments.run_experiment_arm --arm random_forest_fixed_settings --seeds 11 29 47 --final-test --no-refit
run python -m experiments.run_experiment_arm --arm random_forest_fixed_settings_with_conditional_gan_2019_rows --seeds 11 29 47 --final-test --no-refit
run python -m experiments.run_experiment_arm --arm sklearn_gradient_boosting --seeds 11 29 47 --final-test --no-refit
run python -m experiments.run_experiment_arm --arm sklearn_gradient_boosting_with_conditional_gan_2019_rows --seeds 11 29 47 --final-test --no-refit
# the cGAN with the notebook's settings (about 28 minutes)
run python -m experiments.run_experiment_arm --arm random_forest_with_conditional_gan_2019_literal_notebook_rows --seeds 11 --final-test --no-refit

# the cost as a function of the number of added rows, 0 to 5,000 (about 80 minutes)
run python -m experiments.rq1_cgan_2019_reproduction.run_synthetic_row_count_sweep --generator conditional_gan_2019 --model lightgbm --seeds 11 --gan-training-steps 30000

# what the generated rows look like: detectability, closeness to the real failures, copying (about 6 minutes)
run python -m experiments.rq2_survey_of_methods.run_synthetic_data_diagnostics --generators conditional_gan_2019 \
    conditional_gan_2019_literal_notebook conditional_gan_2019_without_missingness duplicated_real_failures
