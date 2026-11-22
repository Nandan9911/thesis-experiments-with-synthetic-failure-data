"""Time and memory of Forest-Flow on a CPU, fitted on all 1,000 training failures (a pilot for the RQ2 arm).

Forest-Flow (see rq2_survey_of_methods/forest_flow_generator.py) fits one XGBoost regressor per attribute, noise
level and cluster, so its cost grows quickly with its settings. This pilot fits it once with the settings given as
<noise_levels>,<duplicates_per_row>,<trees>,<depth> (default 10,10,20,5, the reduced profile of the arms), samples
2,000 rows and prints one RESULT line with the run time, the peak memory, the share of missing cells and the median
over the attributes of mean(synthetic) / mean(real), a rough check that the rows are on the right scale:

    python -m experiments.rq2_survey_of_methods.pilots.forest_flow_timing_pilot 10,10,20,5

The sampled rows are written to results/rq2_survey_of_methods/forest_flow_timing_pilot_<settings>.csv, e.g.
forest_flow_timing_pilot_10_10_20_5.csv. The console output of the reported run is
results/rq2_survey_of_methods/forest_flow_timing_pilot.log: the fit of the reduced profile took 1,986 s, and the run
then stopped at the sampling call because of an error in an earlier version of forest_flow_generator.py, so it
gives the fitting time only.
"""
import resource
import sys
import time
import warnings

import numpy as np

from experiments.common.paths import SURVEY_RESULTS_DIR

OUTPUT_DIR = SURVEY_RESULTS_DIR


def main():
    warnings.filterwarnings("ignore")
    from experiments.common.data_loading import CONSTANT_ATTRIBUTE, load_training_set
    from experiments.rq2_survey_of_methods.forest_flow_generator import ForestFlowFailureGenerator

    X, y = load_training_set()
    X = X.drop(columns=[CONSTANT_ATTRIBUTE])
    X_failures = X[y == 1]
    settings = sys.argv[1] if len(sys.argv) > 1 else "10,10,20,5"
    n_t, duplicate_K, trees, depth = [int(value) for value in settings.split(",")]
    started = time.time()
    generator = ForestFlowFailureGenerator(n_t=n_t, duplicate_K=duplicate_K, n_estimators=trees, max_depth=depth,
                                           seed=1).fit(X_failures)
    synthetic = generator.sample(2000, seed=2)
    peak_memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    mean_ratio = np.nanmedian(synthetic.mean().to_numpy() / (X_failures.mean().to_numpy() + 1e-9))
    print(f"RESULT n_t={n_t} K={duplicate_K} trees={trees} depth={depth}: total {time.time() - started:.0f}s; "
          f"peak memory {peak_memory_mb:.0f} MB; missing share {synthetic.isna().mean().mean():.3f}; "
          f"median ratio of column means, synthetic / real {mean_ratio:.3f}; shape {synthetic.shape}", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(OUTPUT_DIR / f"forest_flow_timing_pilot_{n_t}_{duplicate_K}_{trees}_{depth}.csv", index=False)


if __name__ == "__main__":
    main()
