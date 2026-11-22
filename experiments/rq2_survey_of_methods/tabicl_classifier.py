"""TabICLv2 with bagged contexts: a classifier arm of the RQ2 survey that changes the learner, not the data.

TabICLv2 (Qu, Holzmüller, Varoquaux and Le Morvan, ICML 2026) is a pretrained tabular foundation model that
classifies by in-context learning: it receives labelled example rows (the context) together with the rows to be
classified and predicts their labels in one forward pass, without any training or tuning on APS. The memory
needed grows with the size of the context, and the 48,000 rows of a training fold do not fit. Each fold model is
therefore a bag of n_bags contexts, each holding all failure rows of the training fold plus a fresh random sample
of n_neg healthy rows. Down-sampling the majority class in this way is the remedy for class imbalance recommended
for in-context models; no synthetic rows are added. The probabilities are averaged over the bags, and the decision
threshold is then chosen on the out-of-fold cost as for every other arm. Missing values are left to the model's
own preprocessing, which fills them with the attribute mean.

Used by the arms tabicl_bagged_contexts and ensemble_with_missingness_indicators_and_tabicl
(experiments/experiment_arms.py, via the "tabicl" entry of experiments/common/classifiers.py), for example

    python -m experiments.run_experiment_arm --arm tabicl_bagged_contexts --seeds 11 29 47 --final-test

Two environment variables adapt it to the machine: APS_TABICL_DEVICE ("cpu" by default; "cuda" on a GPU) and
APS_TABICL_QUERY_CHUNK_ROWS (how many rows are classified at once, 500 by default). Reads and writes nothing
itself; TabICL loads its model weights from its own cache.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch


class BaggedTabICLClassifier:
    """A bag of TabICLv2 classifiers with sub-sampled healthy rows; fit/predict_proba like scikit-learn.

    n_bags      number of contexts per model
    n_neg       healthy (negative) rows per context, sampled without replacement; every context also holds all
                failure rows. The short name is kept because the arm options store it under this name.
    n_estimators  TabICL's own ensemble size (it averages over several shuffled and normalised views of a context)
    seed        seed of the healthy-row sampling; bag b uses random_state seed + b inside TabICL
    threads     limits PyTorch's CPU threads
    """

    def __init__(self, n_bags=3, n_neg=10000, n_estimators=2, seed=0, device="cpu", verbose=True, threads=None):
        self.n_bags, self.n_neg, self.n_estimators, self.seed = n_bags, n_neg, n_estimators, seed
        self.device, self.verbose = device, verbose
        if threads:
            torch.set_num_threads(threads)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        from tabicl import TabICLClassifier
        self.models_, rng = [], np.random.default_rng(self.seed)
        failures = np.where(y == 1)[0]
        healthy = np.where(y == 0)[0]
        X_values = X.to_numpy(dtype=np.float32)
        started = time.time()
        for bag in range(self.n_bags):
            context = np.concatenate([failures, rng.choice(healthy, min(self.n_neg, len(healthy)), replace=False)])
            model = TabICLClassifier(n_estimators=self.n_estimators, device=self.device, random_state=self.seed + bag)
            model.fit(X_values[context], y[context])
            self.models_.append(model)
        if self.verbose:
            print(f"    TabICL bags fitted ({self.n_bags} x [{len(failures)} failures + {self.n_neg} healthy rows]) "
                  f"in {time.time() - started:.0f}s", flush=True)
        return self

    def predict_proba(self, X: pd.DataFrame, chunk_rows: int = 500) -> np.ndarray:
        # The rows to classify are passed in chunks: the attention over context plus query rows grows with the
        # chunk, and a 3,300-row context with 12,000 query rows at once needs more memory than an 8 GB machine has.
        X_values = X.to_numpy(dtype=np.float32)
        started = time.time()
        chunk_rows = int(os.environ.get("APS_TABICL_QUERY_CHUNK_ROWS", chunk_rows))
        per_bag = [np.concatenate([model.predict_proba(X_values[i:i + chunk_rows])[:, 1]
                                   for i in range(0, len(X_values), chunk_rows)])
                   for model in self.models_]
        probabilities = np.mean(per_bag, axis=0)
        if self.verbose:
            print(f"    TabICL predict {len(X_values)} rows in {time.time() - started:.0f}s", flush=True)
        return np.column_stack([1 - probabilities, probabilities])


def bagged_tabicl_factory(n_bags=3, n_neg=10000, n_estimators=2, device=None):
    """Returns make(seed) -> an unfitted BaggedTabICLClassifier, the form classifiers.py expects. The device
    defaults to APS_TABICL_DEVICE, else the CPU."""
    device = device or os.environ.get("APS_TABICL_DEVICE", "cpu")

    def make(seed):
        return BaggedTabICLClassifier(n_bags=n_bags, n_neg=n_neg, n_estimators=n_estimators, seed=seed, device=device)
    return make
