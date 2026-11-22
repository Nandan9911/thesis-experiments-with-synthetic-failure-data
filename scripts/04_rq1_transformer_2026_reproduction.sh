#!/usr/bin/env bash
# Step 4 (README, section 5): RQ1b, the cost-sensitive Transformer of Beikmohammadi et al.
#
# Part A runs on a CPU and needs TensorFlow, tf-keras and the authors' repository (bash third_party/fetch_third_party_code.sh).
# Part B runs on a CPU. Part C needs a GPU and runs only with USE_GPU=1; the retraining with the released code from
# scratch (part D) took several days of Colab time and was run through the queue in cloud_runs/, so it is only
# described here. The order and the outputs of the authors-code scripts are explained in
# experiments/rq1_transformer_2026_reproduction/authors_code/README.md.
set -euo pipefail
cd "$(dirname "$0")/.."
run() { echo; echo "+ $*"; "$@"; }
USE_GPU=${USE_GPU:-0}
RQ1B=experiments.rq1_transformer_2026_reproduction     # the package of this step

# ---- A. the released models and inputs
# the first, approximate re-scoring: an imputer capped at five rounds (about 15 minutes), then the five released
# models on those tables, the same with a scaler fitted on the raw columns, and the time of one training epoch
run python -m $RQ1B.authors_code.impute_five_rounds
run python -m $RQ1B.authors_code.evaluate_shipped_models --tables five_round
run python -m $RQ1B.authors_code.evaluate_shipped_models_raw_scale
run python -m $RQ1B.authors_code.time_one_training_epoch
# the exact regeneration of the authors' inputs with their published settings (100 imputer rounds, about four hours
# on one core), the re-scoring that reproduces the published cost, and the attribution of the earlier high cost to
# the imputer and the scaler
run python -m $RQ1B.authors_code.prepare_authors_inputs --imputer-rounds 100
run python -m $RQ1B.authors_code.evaluate_shipped_models --tables exact train_resampled.csv
run bash experiments/rq1_transformer_2026_reproduction/authors_code/run_scaler_attribution.sh
# the hard-negative comparison for the released models, and the same groups scored by three of the thesis's own arms
# (the control needs the seed-11 records of lightgbm_tuned, xgboost_fixed_settings and
# ensemble_without_missingness_indicators in results/classifier_arms/)
run python -m $RQ1B.authors_code.hard_negative_check_shipped_models
run python -m $RQ1B.authors_code.hard_negative_check_shipped_models_followup
run python -m $RQ1B.authors_code.hard_negative_check_tree_control

# ---- B. the preprocessing pipeline of Beikmohammadi et al. in front of the tuned LightGBM, on three folds because the
# imputer of Beikmohammadi et al. is slow, with the 3-fold baseline (each about 15 to 60 minutes)
run python -m experiments.run_experiment_arm --arm lightgbm_tuned --folds 3 --seeds 11 29 47 --final-test --no-refit
for variant in full_pipeline without_renn renn_only imputation_only; do
    run python -m $RQ1B.run_tree_models_on_2026_preprocessing --variant "$variant" --seeds 11 --folds 3 --imputer-rounds 10 --final-test
done

# ---- C. the PyTorch port of the network (GPU; 500 epochs each, checkpoints every 50 epochs in cache/)
if [ "$USE_GPU" = 1 ]; then
    for seed in 0 1 2; do
        run python -m $RQ1B.run_transformer_2026_port --protocol corrected --epochs 500 --patience 100000 --imputer-rounds 10 \
            --seed "$seed" --run-name corrected_protocol --checkpoint-dir cache/port_checkpoints --checkpoint-every 50
    done
    for seed in 0 1; do
        run python -m $RQ1B.run_transformer_2026_port --protocol published --epochs 500 --patience 100000 --imputer-rounds 10 \
            --seed "$seed" --run-name published_protocol --checkpoint-dir cache/port_checkpoints --checkpoint-every 50
        run python -m $RQ1B.run_transformer_2026_port --protocol published --epochs 500 --patience 100000 --imputer-rounds 10 \
            --seed "$seed" --run-name published_protocol_without_renn --no-renn --checkpoint-dir cache/port_checkpoints --checkpoint-every 50
        run python -m $RQ1B.run_transformer_2026_port --protocol published --epochs 500 --patience 100000 --imputer-rounds 10 \
            --seed "$seed" --run-name published_protocol_without_svm_smote --no-smote --checkpoint-dir cache/port_checkpoints --checkpoint-every 50
        run python -m $RQ1B.run_transformer_2026_port --protocol published --epochs 500 --patience 100000 --imputer-rounds 10 \
            --seed "$seed" --run-name published_protocol_without_resampling --no-smote --no-renn --checkpoint-dir cache/port_checkpoints --checkpoint-every 50
    done
    # seed 0 of the protocol with a validation split of real rows continued from its checkpoint towards 4,000 epochs
    # (the thesis run was stopped at 3,250 when the compute budget ran out), then its checkpoints were evaluated offline
    cp -n cache/port_checkpoints/corrected_protocol_seed0.pt cache/port_checkpoints/corrected_protocol_extended_seed0.pt
    run python -m $RQ1B.run_transformer_2026_port --protocol corrected --epochs 4000 --patience 100000 --imputer-rounds 10 \
        --seed 0 --run-name corrected_protocol_extended --checkpoint-dir cache/port_checkpoints --checkpoint-every 50
    run python -m $RQ1B.evaluate_port_checkpoint --checkpoint cache/port_checkpoints/corrected_protocol_extended_seed0.pt \
        --protocol corrected --imputer-rounds 10 --seed 0
else
    echo
    echo "Skipping the PyTorch port (part C). Run with USE_GPU=1 to include it."
fi

# ---- D. retraining with the released code: retraining runs on seeds 0 to 4 and test-inclusion controls, each
# a sequence of resumable stages; see cloud_runs/README.md and authors_code/README.md. When the runs are in
# results/rq1_transformer_2026_reproduction/authors_code/retraining_runs/, the hard-negative comparison for the
# retrained models is
run python -m $RQ1B.hard_negative_check_retrained_models
