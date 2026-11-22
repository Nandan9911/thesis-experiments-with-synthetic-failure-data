"""TabEBM as a generator of synthetic failure rows (RQ2 survey; also used by the RQ3 cold-start experiment).

TabEBM (Margeloiu et al., NeurIPS 2024) turns a pretrained TabPFN classifier into an energy-based model of one
class. TabPFN classifies by in-context learning: it reads labelled example rows together with the rows to be
classified and needs no training of its own. For the class of interest, TabEBM shows TabPFN the real rows of that
class next to a handful of artificial "surrogate negatives" placed far away, at corners of a hypercube with
coordinates of +/- distance_negative_class in the standardised space. The classifier's logits then define an energy
that is low near the real rows and high elsewhere. New rows are drawn by stochastic gradient Langevin dynamics
(SGLD): start from real rows plus a little noise (starting_point_noise_std), then repeat sgld_steps times a small
step down the energy gradient (sgld_step_size) with Gaussian noise added (sgld_noise_std). There is no training
phase, and a GPU is needed in practice because every SGLD step runs TabPFN forward and backward.

Here TabEBM is applied to the failure rows of the training fold only, after median imputation, log(1 + x) and
standardisation, and the generated rows receive the missing-value patterns of randomly drawn real failure rows.

Backend: tabpfn==2.1.3 (the TabPFN v2 classifier weights, which are not gated) and tabebm==2025.8.19.

Used through the arm runner, for example

    python -m experiments.run_experiment_arm --arm lightgbm_with_tabebm_rows --seeds 11 --final-test

Reads and writes nothing itself (TabPFN loads its model weights from its own cache).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch


class TabEBMFailureGenerator:
    """TabEBM on real failure rows; `fit(X_failures)` only stores the transformed rows, `sample` runs SGLD.

    The SGLD parameters keep the names of the tabebm package; n_threads limits PyTorch's CPU threads.
    """

    def __init__(self, sgld_steps=200, sgld_step_size=0.1, sgld_noise_std=0.01,
                 starting_point_noise_std=0.01, distance_negative_class=5.0,
                 n_threads=None, seed=0, verbose=True):
        self.sgld_steps, self.sgld_step_size = sgld_steps, sgld_step_size
        self.sgld_noise_std, self.starting_point_noise_std = sgld_noise_std, starting_point_noise_std
        self.distance_negative_class, self.seed, self.verbose = distance_negative_class, seed, verbose
        if n_threads:
            torch.set_num_threads(n_threads)

    def fit(self, X_failures: pd.DataFrame):
        """Store the transformed failure rows; TabEBM has no training phase."""
        self.columns = list(X_failures.columns)
        values = X_failures.to_numpy(dtype=np.float64)
        self.missingness_patterns_ = np.isnan(values)       # real missingness patterns, drawn again at sampling
        self.medians_ = np.nanmedian(values, axis=0)
        self.medians_ = np.where(np.isnan(self.medians_), 0.0, self.medians_)
        values = np.where(np.isnan(values), self.medians_[None, :], values)
        self.shift_ = np.minimum(values.min(axis=0), 0.0)   # APS attributes are non-negative counts; guard anyway
        log_values = np.log1p(values - self.shift_)
        self.means_, self.standard_deviations_ = log_values.mean(axis=0), log_values.std(axis=0) + 1e-6
        self.Z_ = (log_values - self.means_) / self.standard_deviations_
        # attributes that are constant among the failure rows are left out (TabPFN v2 does not cope well with them)
        # and restored in _inverse
        self.varying_attributes_ = self.standard_deviations_ > 1e-5
        return self

    def _inverse(self, Z_varying: np.ndarray) -> np.ndarray:
        Z = np.tile(self.Z_.mean(axis=0), (len(Z_varying), 1))
        Z[:, self.varying_attributes_] = Z_varying
        log_values = Z * self.standard_deviations_ + self.means_
        values = np.expm1(log_values) + self.shift_
        return np.maximum(values, 0.0)

    def sample(self, n: int, reapply_missingness=True, seed: int | None = None) -> pd.DataFrame:
        from tabebm.TabEBM import TabEBM
        seed = self.seed if seed is None else seed
        started = time.time()
        ebm = TabEBM()
        Z_varying = self.Z_[:, self.varying_attributes_].astype(np.float32)
        labels = np.zeros(len(Z_varying), dtype=int)   # a single class: one energy model (failures vs surrogates)
        generated = ebm.generate(Z_varying, labels, num_samples=n, sgld_steps=self.sgld_steps,
                                 sgld_step_size=self.sgld_step_size, sgld_noise_std=self.sgld_noise_std,
                                 starting_point_noise_std=self.starting_point_noise_std,
                                 distance_negative_class=self.distance_negative_class, seed=seed)
        Z = np.asarray(generated["class_0"], dtype=np.float64)
        if Z.ndim == 3:   # the whole sampling path (steps, n, d) was returned: keep the final step
            Z = Z[-1]
        values = self._inverse(Z)
        if reapply_missingness:
            rng = np.random.default_rng(seed)
            patterns = self.missingness_patterns_[rng.integers(0, len(self.missingness_patterns_), len(values))]
            values = np.where(patterns, np.nan, values)
        if self.verbose:
            print(f"    TabEBM: {n} rows in {time.time() - started:.0f}s (sgld_steps={self.sgld_steps})", flush=True)
        return pd.DataFrame(values.astype(np.float32), columns=self.columns)


def tabebm_augmenter(n_synthetic=2000, reapply_missingness=True, **generator_options):
    """Augmenter for run_cross_validation: fits TabEBMFailureGenerator (generator_options are its constructor
    arguments, e.g. sgld_steps) on the training fold's failure rows with the fold's seed, and appends n_synthetic
    rows sampled with seed + 1."""
    def augment(X_train: pd.DataFrame, y_train: np.ndarray, seed: int):
        generator = TabEBMFailureGenerator(seed=seed, **generator_options).fit(X_train[y_train == 1])
        X_synthetic = generator.sample(n_synthetic, reapply_missingness=reapply_missingness, seed=seed + 1)
        X_augmented = pd.concat([X_train, X_synthetic], ignore_index=True)
        y_augmented = np.concatenate([y_train, np.ones(len(X_synthetic), dtype=y_train.dtype)])
        return X_augmented, y_augmented
    return augment
