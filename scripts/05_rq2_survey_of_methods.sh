#!/usr/bin/env bash
# Step 5 (README, section 5): RQ2, the survey. Every generator adds 2,000 synthetic failures inside each training
# fold; the learner arms change the classifier instead. Times are per seed, on the machine where the thesis run was
# made (the two-core machine or a Colab session with an L4 GPU).
#
# The arms that need a GPU in practice run only with USE_GPU=1:   USE_GPU=1 bash scripts/05_rq2_survey_of_methods.sh
set -euo pipefail
cd "$(dirname "$0")/.."
run() { echo; echo "+ $*"; "$@"; }
USE_GPU=${USE_GPU:-0}

# classical oversampling on median-imputed attributes (about 30 to 40 minutes per seed)
for arm in lightgbm_with_smote_rows lightgbm_with_borderline_smote_rows lightgbm_with_adasyn_rows lightgbm_with_svm_smote_rows; do
    run python -m experiments.run_experiment_arm --arm "$arm" --seeds 11 29 47 --final-test --no-refit
done

# the Gaussian copula runs on a CPU (about 100 minutes per seed)
run python -m experiments.run_experiment_arm --arm lightgbm_with_gaussian_copula_rows --seeds 11 29 47 --final-test --no-refit

# Forest-Flow in its reduced profile, on the CPU as in the thesis runs (about 6 hours per seed on two cores; with an
# NVIDIA GPU, --override '{"use_gpu_histograms": true}' lets XGBoost build its histograms there)
run python -m experiments.run_experiment_arm --arm lightgbm_with_forest_flow_rows_reduced_profile --seeds 11 29 47 --final-test --no-refit

# changing the learner instead of the data (XGBoost about 14 minutes per seed, LightGBM with indicators about 40,
# each ensemble about an hour)
run python -m experiments.run_experiment_arm --arm xgboost_fixed_settings --seeds 11 29 47 --final-test --no-refit
run python -m experiments.run_experiment_arm --arm lightgbm_tuned_with_missingness_indicators --seeds 11 29 47 --final-test --no-refit
run python -m experiments.run_experiment_arm --arm ensemble_without_missingness_indicators --seeds 11 29 47 71 101 --final-test --no-refit
run python -m experiments.run_experiment_arm --arm ensemble_with_missingness_indicators --seeds 11 29 47 71 101 --final-test --no-refit

# diagnostics of the rows of the CPU generators (about 50 minutes, most of it Forest-Flow)
run python -m experiments.rq2_survey_of_methods.run_synthetic_data_diagnostics --generators forest_flow gaussian_copula \
    smote borderline_smote adasyn svm_smote

if [ "$USE_GPU" != 1 ]; then
    echo
    echo "Skipping the GPU arms (TVAE, CTGAN, CTAB-GAN+, TabDDPM, TabEBM, TabICLv2). Run with USE_GPU=1 to include them."
    exit 0
fi

# deep generators on an L4 (TabDDPM about 8 minutes per seed, TVAE and CTGAN about 23, CTAB-GAN+ about 66;
# CTAB-GAN+ needs third_party/CTAB-GAN-Plus, see third_party/README.md)
for arm in lightgbm_with_tvae_rows lightgbm_with_ctgan_rows lightgbm_with_ctab_gan_plus_rows lightgbm_with_tabddpm_rows; do
    run python -m experiments.run_experiment_arm --arm "$arm" --seeds 11 29 47 --final-test --no-refit
done
# TabEBM (about 80 minutes per seed on an L4)
run python -m experiments.run_experiment_arm --arm lightgbm_with_tabebm_rows --seeds 11 29 47 --final-test --no-refit
# the tabular foundation model TabICLv2, alone and as a fourth member of the ensemble (each about 10 minutes per seed);
# the two variables put TabICLv2 on the GPU and let it classify the rows in chunks of 4,000, as in the Colab sessions
export APS_TABICL_DEVICE=cuda APS_TABICL_QUERY_CHUNK_ROWS=4000
run python -m experiments.run_experiment_arm --arm tabicl_bagged_contexts --seeds 11 29 47 71 101 --final-test --no-refit
run python -m experiments.run_experiment_arm --arm ensemble_with_missingness_indicators_and_tabicl --seeds 11 29 47 71 101 \
    --final-test --no-refit
# diagnostics of the rows of the GPU generators
run python -m experiments.rq2_survey_of_methods.run_synthetic_data_diagnostics --generators tabebm ctgan tvae tabddpm ctab_gan_plus

# Not run here, as in the thesis: lightgbm_with_great_language_model_rows (it produced no usable rows) and the
# middle and paper profiles of Forest-Flow (about a day of an L4 per seed each).
