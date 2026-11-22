"""Time and memory of TabEBM on a CPU, on subsets of the real failure rows (a pilot for the RQ2 TabEBM arm).

TabEBM (see rq2_survey_of_methods/tabebm_generator.py) runs a TabPFN model forward and backward at every step of
its Langevin sampler, so its cost grows with the number of context rows, the number of sampler steps and the
number of rows generated. This pilot measures that cost for a few settings, each given as
<context_rows>,<sgld_steps>,<generated_rows> (default 400,3,100):

    python -m experiments.rq2_survey_of_methods.pilots.tabebm_timing_pilot 300,3,100

For each setting it draws the context rows from the 1,000 training failures, generates the rows and prints one
RESULT line with the run time, the peak memory, the share of missing cells and the median over the attributes of
mean(synthetic) / mean(real), a rough check that the rows are on the right scale. The reported run used 300 context
rows, 3 steps and 100 rows and took 392 s with a peak memory of 4,286 MB on a two-core CPU (console output in
results/rq2_survey_of_methods/tabebm_timing_pilot.log). The arm, with 800 context rows, 200 steps and 2,000 rows
per fold, was run on a GPU.

Writes nothing (redirect the output to keep it). APS_THREADS sets the number of PyTorch CPU threads (default 2).
"""
import os
import resource
import sys
import time
import warnings

import numpy as np


def main():
    # TabPFN refuses more than 1,000 rows on a CPU unless this is set; it reads the setting when it is imported
    os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(int(os.environ.get("APS_THREADS", "2")))
    from experiments.common.data_loading import CONSTANT_ATTRIBUTE, load_training_set
    from experiments.rq2_survey_of_methods.tabebm_generator import TabEBMFailureGenerator

    X, y = load_training_set()
    X = X.drop(columns=[CONSTANT_ATTRIBUTE])
    X_failures = X[y == 1]
    settings = [tuple(int(value) for value in setting.split(",")) for setting in sys.argv[1:]] or [(400, 3, 100)]
    for context_rows, sgld_steps, generated_rows in settings:
        context = X_failures.sample(n=min(context_rows, len(X_failures)), random_state=0)
        started = time.time()
        generator = TabEBMFailureGenerator(sgld_steps=sgld_steps, seed=1, verbose=True).fit(context)
        synthetic = generator.sample(generated_rows, seed=2)
        peak_memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        mean_ratio = np.nanmedian(synthetic.mean().to_numpy() / (X_failures.mean().to_numpy() + 1e-9))
        print(f"RESULT ctx={context_rows} steps={sgld_steps} n={generated_rows}: {time.time() - started:.0f}s; "
              f"peak memory {peak_memory_mb:.0f} MB; missing share {synthetic.isna().mean().mean():.3f}; "
              f"median ratio of column means, synthetic / real {mean_ratio:.3f}; shape {synthetic.shape}", flush=True)


if __name__ == "__main__":
    main()
