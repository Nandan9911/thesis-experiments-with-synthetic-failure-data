#!/usr/bin/env bash
# Attribution of the high cost of the models released by Beikmohammadi et al. in the first re-scoring to its two
# deviations (RQ1b).
#
# The first re-scoring (evaluate_shipped_models.py --tables five_round) differed from the published preprocessing in
# two ways: the imputer was capped at five rounds instead of running up to 100, and the min-max scaler was fitted on
# the imputed training table instead of the resampled table. On the exact inputs both are as published and the five
# models reproduce the published cost. This script crosses the two deviations:
#   (i)  the exact 100-round tables, with the scaler fitted on the imputed training table;
#   (ii) the five-round tables, with the scaler fitted on their resampled table (the published scaler source), after
#        resampling the five-round imputed table with SVM-SMOTE and RENN.
# It needs the exact tables (prepare_authors_inputs.py) and the five-round tables (impute_five_rounds.py) in
# cache/authors_code_work/ and loads the five released models; each evaluation took 40 to 50 minutes on one core.
#
#   bash experiments/rq1_transformer_2026_reproduction/authors_code/run_scaler_attribution.sh \
#       > results/rq1_transformer_2026_reproduction/authors_code/logs/scaler_attribution.log
#
# The console output of each step goes to results/rq1_transformer_2026_reproduction/authors_code/logs/ and the tables of
# results to shipped_models_exact_inputs/evaluation_scaler_on_imputed.csv and
# shipped_models_exact_inputs/evaluation_five_round_tables_scaler_on_resampled.csv in the same folder.
set -u
cd "$(dirname "$0")/../../.."          # the repository root, so that python -m finds the experiments package
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2
LOGS=results/rq1_transformer_2026_reproduction/authors_code/logs
SCRIPTS=experiments.rq1_transformer_2026_reproduction.authors_code
log() { echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p "$LOGS"
log "START (i) exact tables, scaler on train_imputed"
nice -n 5 python3 -u -m $SCRIPTS.evaluate_shipped_models --tables exact train_imputed.csv \
    > "$LOGS/evaluate_shipped_models_exact_tables_scaler_on_imputed.log" 2>&1
log "END (i) (exit $?)"
log "START (ii) 5-round tables: SVM-SMOTE + RENN"
nice -n 5 python3 -u -m $SCRIPTS.resample_five_round_table > "$LOGS/resample_five_round_table.log" 2>&1
log "END (ii) resampling (exit $?)"
log "START (ii) 5-round tables, scaler on resampled"
nice -n 5 python3 -u -m $SCRIPTS.evaluate_shipped_models --tables five_round train_resampled.csv \
    > "$LOGS/evaluate_shipped_models_five_round_tables_scaler_on_resampled.log" 2>&1
log "END (ii) (exit $?)"
log "ATTRIBUTION DONE"
