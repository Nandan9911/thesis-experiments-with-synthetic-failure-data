"""Forest-Flow as a generator of synthetic failure rows (RQ2 survey; also used by the RQ1 count sweep and RQ3).

Forest-Flow (Jolicoeur-Martineau, Fatras and Kachman, AISTATS 2024; package ForestDiffusion) is a flow-matching
generator built from gradient-boosted trees instead of a neural network. Flow matching learns a velocity field
that carries a sample of Gaussian noise to a data row along a straight line; a new row is made by drawing noise and
following the learned velocity from t = 0 (noise) to t = 1 (data) in small Euler steps. Forest-Flow learns that
velocity with XGBoost: the time axis is cut into n_t noise levels, and at every level one regressor per attribute
predicts how that attribute should move. The training pairs are made by repeating every real row duplicate_K times,
each time with a fresh draw of noise, which gives the trees more (row, noise) combinations to learn from. XGBoost
can split on missing values, so the model is fitted on the failure rows as they are, without imputation.

How it is used here:
  * The generator is fitted on the failure rows of the training fold only, in log(1 + x) space (the APS attributes
    are heavy-tailed counters).
  * It is conditioned on the two k-means cluster labels of the failure rows used by the study of Ranasinghe et al.:
    ForestDiffusion fits a separate set of regressors for every cluster and gives new rows cluster labels in the real
    proportions.
  * The sampler fills every cell, so each generated row receives the missing-value pattern of a real failure row
    drawn at random, as for the other generators of RQ2.

Three profiles appear in experiments/experiment_arms.py. The default settings of this class are the reduced
profile (n_t = 10 noise levels, duplicate_K = 10 copies per row, 20 trees of depth 5 per regressor), the only one
that was run on three seeds. The paper's defaults are n_t = 50, duplicate_K = 100, 100 trees of depth 7; the middle
profile uses n_t = 50 with duplicate_K = 20.

The generator class and forest_flow_augmenter, like the Forest-Flow completion engine of RQ3, take ForestDiffusion's
keyword names. The arm options of experiment_arms.py and the --override options of
rq3_completion_and_cold_start/run_cold_start_experiment.py give the same settings descriptive names:

    noise_levels         n_t            number of noise levels
    duplicates_per_row   duplicate_K    noisy copies of every training row
    trees_per_regressor  n_estimators   trees of each XGBoost regressor
    tree_depth           max_depth      depth of those trees
    use_gpu_histograms   gpu_hist       train the trees on a GPU (the cold-start options use the name gpu_hist)

Used through the arm runner, for example

    python -m experiments.run_experiment_arm --arm lightgbm_with_forest_flow_rows_reduced_profile --seeds 11 \\
        --final-test

Reads and writes nothing itself.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class ForestFlowFailureGenerator:
    """Forest-Flow fitted on real failure rows; `fit(X_failures)`, then `sample(n, reapply_missingness, seed)`.

    n_t and duplicate_K are ForestDiffusion's names for the number of noise levels and the number of noisy copies
    of every row; n_estimators and max_depth set the size of each XGBoost regressor. diffusion_type "flow" is flow
    matching (ForestDiffusion's "vp" diffusion variant is used only by the completion engine of RQ3). n_batch >= 1
    makes ForestDiffusion stream the duplicated rows to XGBoost through a data iterator instead of building the
    K-times larger table in memory. gpu_hist trains the trees on a GPU. n_jobs defaults to APS_CPU_THREADS_PER_MODEL,
    else to all cores.
    """

    def __init__(self, n_t=10, duplicate_K=10, n_estimators=20, max_depth=5, diffusion_type="flow",
                 condition_on_clusters=True, n_clusters=2, n_jobs=None, n_batch=1, seed=0, verbose=True,
                 log_space=True, gpu_hist=False):
        self.n_t, self.duplicate_K, self.n_estimators, self.max_depth = n_t, duplicate_K, n_estimators, max_depth
        self.diffusion_type, self.condition_on_clusters = diffusion_type, condition_on_clusters
        self.n_clusters = n_clusters
        self.n_jobs = n_jobs or max(1, int(os.environ.get("APS_CPU_THREADS_PER_MODEL") or (os.cpu_count() or 1)))
        self.n_batch, self.seed, self.verbose = n_batch, seed, verbose
        self.log_space, self.gpu_hist = log_space, gpu_hist

    def fit(self, X_failures: pd.DataFrame):
        from ForestDiffusion import ForestDiffusionModel
        started = time.time()
        self.columns = list(X_failures.columns)
        values = X_failures.to_numpy(dtype=np.float64)
        self.missingness_patterns_ = np.isnan(values)
        self.shift_ = np.nanmin(np.where(np.isnan(values), np.inf, values), axis=0)
        self.shift_ = np.minimum(self.shift_, 0.0)    # 0 for non-negative counters; guards log1p otherwise
        Z = np.log1p(values - self.shift_) if self.log_space else values
        cluster_labels = None
        if self.condition_on_clusters:
            # k-means needs complete rows: missing cells are set to the attribute's median (0 if it has none)
            Z_filled = np.where(np.isnan(Z), np.nanmedian(Z, axis=0), Z)
            Z_filled = np.where(np.isnan(Z_filled), 0.0, Z_filled)
            self.kmeans_ = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.seed).fit(Z_filled)
            cluster_labels = self.kmeans_.labels_
            self.cluster_proportions_ = np.bincount(cluster_labels, minlength=self.n_clusters) / len(cluster_labels)
        self.model_ = ForestDiffusionModel(Z, label_y=cluster_labels, n_t=self.n_t, duplicate_K=self.duplicate_K,
                                           n_estimators=self.n_estimators, max_depth=self.max_depth,
                                           diffusion_type=self.diffusion_type, model="xgboost", gpu_hist=self.gpu_hist,
                                           n_jobs=self.n_jobs, n_batch=self.n_batch, seed=self.seed)
        self.fit_seconds_ = time.time() - started
        if self.verbose:
            print(f"    ForestFlow fit: n_t={self.n_t} K={self.duplicate_K} trees={self.n_estimators} "
                  f"depth={self.max_depth} cond={self.condition_on_clusters} in {self.fit_seconds_:.0f}s", flush=True)
        return self

    def sample(self, n: int, reapply_missingness=True, seed: int | None = None) -> pd.DataFrame:
        seed = self.seed if seed is None else seed
        started = time.time()
        # generate() draws its starting noise and the cluster labels of the new rows from NumPy's global random
        # generator; the model's own seed attribute is used by its training code only and is set for completeness
        np.random.seed(seed)
        self.model_.seed = seed
        generated = self.model_.generate(batch_size=n)
        Z = np.asarray(generated, dtype=np.float64)
        if self.condition_on_clusters:      # the last column is the generated cluster label
            Z = Z[:, :len(self.columns)]
        values = (np.expm1(Z) + self.shift_) if self.log_space else Z
        values = np.maximum(values, 0.0)
        if reapply_missingness:
            rng = np.random.default_rng(seed)
            patterns = self.missingness_patterns_[rng.integers(0, len(self.missingness_patterns_), len(values))]
            values = np.where(patterns, np.nan, values)
        if self.verbose:
            print(f"    ForestFlow sample: {n} rows in {time.time() - started:.0f}s", flush=True)
        return pd.DataFrame(values.astype(np.float32), columns=self.columns)


def forest_flow_augmenter(n_synthetic=2000, reapply_missingness=True, **generator_options):
    """Augmenter for run_cross_validation: fits ForestFlowFailureGenerator (generator_options are its constructor
    arguments, e.g. n_t, duplicate_K, n_estimators, max_depth, condition_on_clusters, gpu_hist) on the training
    fold's failure rows with the fold's seed, and appends n_synthetic rows sampled with seed + 1."""
    def augment(X_train: pd.DataFrame, y_train: np.ndarray, seed: int):
        generator = ForestFlowFailureGenerator(seed=seed, **generator_options).fit(X_train[y_train == 1])
        X_synthetic = generator.sample(n_synthetic, reapply_missingness=reapply_missingness, seed=seed + 1)
        X_augmented = pd.concat([X_train, X_synthetic], ignore_index=True)
        y_augmented = np.concatenate([y_train, np.ones(len(X_synthetic), dtype=y_train.dtype)])
        return X_augmented, y_augmented
    return augment
