#!/usr/bin/env bash
# Step 6 (README, section 5): RQ3, completion of partially observed failure records and the cold start.
#
# The thesis ran these on Colab L4 GPUs. With USE_GPU=1 the tree-based engines and generators are told to use the
# GPU exactly as in those runs; without it everything runs on the CPU (much more slowly for Forest-Flow,
# Unmasking Trees and TabPFN).
#     USE_GPU=1 bash scripts/06_rq3_completion_and_cold_start.sh
set -euo pipefail
cd "$(dirname "$0")/.."
run() { echo; echo "+ $*"; "$@"; }
USE_GPU=${USE_GPU:-0}

if [ "$USE_GPU" = 1 ]; then
    engine_override='{"forest_flow": {"gpu_hist": true}, "unmasking_trees": {"xgboost_kwargs": {"device": "cuda"}}}'
    unmasking_trees_override='{"xgboost_kwargs": {"device": "cuda"}}'
    forest_flow_override='{"gpu_hist": true}'
else
    engine_override='{}'
    unmasking_trees_override='{}'
    forest_flow_override='{}'
fi

# the completion benchmark: six engines x four masks, one cross-validation fold of the failures per command
for fold in 0 1 2 3 4; do
    run python -m experiments.rq3_completion_and_cold_start.run_completion_benchmark --folds "$fold" \
        --engines median k_nearest_neighbours chained_equations forest_flow unmasking_trees tabpfn --override "$engine_override"
done

# the memorisation audit: do the completions copy the failures the engines were fitted on?
run python -m experiments.rq3_completion_and_cold_start.run_completion_memorisation_audit \
    --engines median k_nearest_neighbours chained_equations unmasking_trees forest_flow tabpfn --override "$engine_override"

# the prototype: training failure no. 7 with half of its observed cells hidden, completed under the failure and the
# healthy hypothesis (about 75 minutes on an L4)
run python -m experiments.rq3_completion_and_cold_start.complete_failure_record --engine unmasking_trees --demo-index 7 \
    --hidden-fraction 0.5 --both-hypotheses --healthy-rows 1000 --attributes-to-show 12 --override "$unmasking_trees_override"

# the cold start: 25 to 1,000 labelled failures, with and without 2,000 synthetic rows from each generator
for generator in no_synthetic_rows duplicated_real_failures smote conditional_gan_2019 tabebm; do
    for seed in 11 29 47; do
        run python -m experiments.rq3_completion_and_cold_start.run_cold_start_experiment --generator "$generator" \
            --seeds "$seed" --failure-counts 25 50 100 250 1000 --override '{}'
    done
done
for generator in forest_flow forest_flow_inpainting; do
    for seed in 11 29 47; do
        run python -m experiments.rq3_completion_and_cold_start.run_cold_start_experiment --generator "$generator" \
            --seeds "$seed" --failure-counts 25 50 100 250 1000 --override "$forest_flow_override"
    done
done
# Unmasking Trees inpainting: all counts on seed 11, then the two smallest counts on seeds 29 and 47
run python -m experiments.rq3_completion_and_cold_start.run_cold_start_experiment --generator unmasking_trees_inpainting \
    --seeds 11 --failure-counts 25 50 100 250 1000 --override "$unmasking_trees_override"
for seed in 29 47; do
    run python -m experiments.rq3_completion_and_cold_start.run_cold_start_experiment --generator unmasking_trees_inpainting \
        --seeds "$seed" --failure-counts 25 50 --override "$unmasking_trees_override"
done
