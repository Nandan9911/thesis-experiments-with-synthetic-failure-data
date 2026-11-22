"""Six completion engines for partially observed APS records, the core of the RQ3 prototype.

Part of RQ3. Completion and imputation both fill in missing values, but they answer different questions.
Imputation fills gaps so that a model can be trained or applied, and is judged by that model. Completion takes one
record in which only some of the 170 counters are known and returns the complete record that is most consistent
with the known counters under a stated hypothesis, here "this record is a failure": the engine is fitted on failure
rows only, and the same engine fitted on healthy rows gives the healthy counterpart. Every filled cell also gets an
80 % interval, so that the answer is a range of plausible values rather than a single guess.

The engines (keys of COMPLETION_ENGINES):
  median                 the median of each attribute among the fitting rows, with the interval median +/- 1.2816
                         standard deviations (the 10th and 90th percentiles of a normal distribution); a baseline
  k_nearest_neighbours   scikit-learn's KNNImputer: for each missing attribute, its mean over the 10 nearest fitting
                         rows that have it (distances over the attributes both rows have); interval as for the
                         median; a baseline
  chained_equations      chained equations, as in multiple imputation (scikit-learn's IterativeImputer with Bayesian
                         ridge regression): each attribute is regressed on all others in turn, for ten rounds. The
                         imputer draws from the regression's predictive distribution (sample_posterior), so repeated
                         calls give different completions; the median and the 10th and 90th percentiles of these
                         draws are reported. A baseline.
  unmasking_trees        Unmasking Trees (McCarter, 2024). Training rows are masked attribute by attribute in random
                         orders, and gradient-boosted tree classifiers learn to predict ("unmask") each attribute,
                         discretised into 2^depth bins, from the attributes that are still visible. Completion
                         samples the missing attributes one at a time, so every call gives a different completion.
  forest_flow            Forest-Diffusion (Jolicoeur-Martineau et al., 2024) in its imputation mode. The tree-based
                         diffusion model, trained on the fitting rows, generates the missing cells while the known
                         cells are held at their values (REPAINT); `draws` independent completions per record.
                         Imputation is defined only for the variance-preserving ("vp") diffusion variant, not for
                         flow matching, so this engine is not the flow-matching generator of RQ2.
  tabpfn                 one TabPFN v2 regression per missing attribute (Hollmann et al., 2025). TabPFN is a pretrained
                         in-context model: the fitting rows are its context, the missing attribute is the target and
                         all other attributes (missing ones as NaN) are the inputs; it predicts quantiles directly
                         and has no training phase.

All engines work on log(1 + x) of the counters, whose values span many orders of magnitude, and return values in
the original units. complete(X_partial, draws) returns (point, low, high): a point estimate and the bounds of an 80 %
interval for every cell. Known cells are passed through unchanged, and no filled value may leave the range that the
attribute has among the fitting rows. `draws` is the number of completions drawn per record by the stochastic
engines (chained_equations makes between 2 and 10); the median, k-nearest-neighbour and TabPFN engines ignore it.

The two masking functions at the end hide known cells of real records, so that completions can be scored against
the true values. Used by run_completion_benchmark.py, run_completion_memorisation_audit.py,
complete_failure_record.py and, with unmasking_trees and forest_flow as generators, run_cold_start_experiment.py,
for example

    python -m experiments.rq3_completion_and_cold_start.complete_failure_record --engine k_nearest_neighbours \\
        --demo-index 7

Reads and writes nothing itself.
"""
from __future__ import annotations

import importlib
import os
import time

import numpy as np
import pandas as pd

# 10th and 90th percentiles of a normal distribution, in standard deviations: the bounds of a central 80 % interval
NORMAL_80_PERCENT_HALF_WIDTH = 1.2816


def _to_log(values, shift):
    return np.log1p(np.maximum(values - shift, 0.0))


def _from_log(Z, shift):
    return np.maximum(np.expm1(Z) + shift, 0.0)


class _CompletionEngine:
    """Shared log transform, clipping and output format of the engines; subclasses define fit and complete."""
    name = "base"

    def __init__(self, seed=0, verbose=True, **kwargs):
        self.seed, self.verbose = seed, verbose

    def _prepare_fitting_rows(self, X: pd.DataFrame):
        """Log-transform the fitting rows (missing cells stay NaN) and store the per-attribute statistics that the
        engines and the clipping need."""
        self.columns = list(X.columns)
        values = X.to_numpy(dtype=np.float64)
        self.shift_ = np.nanmin(np.where(np.isnan(values), np.inf, values), axis=0)
        self.shift_ = np.minimum(np.where(np.isfinite(self.shift_), self.shift_, 0.0), 0.0)   # 0 for counters
        Z = _to_log(np.where(np.isnan(values), np.nan, values), self.shift_)
        Z = np.where(np.isnan(values), np.nan, Z)
        self.medians_ = np.nanmedian(Z, axis=0)
        self.medians_ = np.where(np.isnan(self.medians_), 0.0, self.medians_)
        self.standard_deviations_ = np.nanstd(Z, axis=0)
        self.standard_deviations_ = np.where(np.isnan(self.standard_deviations_) | (self.standard_deviations_ < 1e-6),
                                             1e-6, self.standard_deviations_)
        self.log_minimum_ = np.where(np.isnan(np.nanmin(Z, axis=0)), 0.0, np.nanmin(Z, axis=0))
        self.log_maximum_ = np.where(np.isnan(np.nanmax(Z, axis=0)), 0.0, np.nanmax(Z, axis=0))
        return Z

    def _prepare_records(self, X_partial: pd.DataFrame):
        """The records to complete in log space, with NaN in the cells to be filled."""
        values = X_partial[self.columns].to_numpy(dtype=np.float64)
        return np.where(np.isnan(values), np.nan, _to_log(np.where(np.isnan(values), 0.0, values), self.shift_))

    def _assemble_completion(self, Z_partial, point, low, high):
        """Clip to the fitting range, keep the known cells, and return (point, low, high) as DataFrames in counter
        units."""
        # no completion may leave the range the engine was fitted on (chained-equation imputers extrapolate badly)
        point, low, high = (np.clip(np.nan_to_num(M, nan=0.0), self.log_minimum_, self.log_maximum_)
                            for M in (point, low, high))
        known = ~np.isnan(Z_partial)
        point = np.where(known, Z_partial, point)
        low = np.where(known, Z_partial, low)
        high = np.where(known, Z_partial, high)

        def to_frame(M):
            return pd.DataFrame(_from_log(M, self.shift_).astype(np.float32), columns=self.columns)
        return to_frame(point), to_frame(low), to_frame(high)

    def fit(self, X: pd.DataFrame):
        raise NotImplementedError

    def complete(self, X_partial: pd.DataFrame, draws: int = 20):
        raise NotImplementedError


# ---------------------------------------------------------------------------------------------------------- baselines
class MedianCompletion(_CompletionEngine):
    name = "median"

    def fit(self, X):
        self._prepare_fitting_rows(X)
        return self

    def complete(self, X_partial, draws=20):
        Z_partial = self._prepare_records(X_partial)
        point = np.tile(self.medians_, (len(Z_partial), 1))
        half_width = NORMAL_80_PERCENT_HALF_WIDTH * self.standard_deviations_
        return self._assemble_completion(Z_partial, point, point - half_width, point + half_width)


class KNearestNeighboursCompletion(_CompletionEngine):
    name = "k_nearest_neighbours"

    def __init__(self, n_neighbors=10, **kwargs):
        super().__init__(**kwargs)
        self.n_neighbors = n_neighbors

    def fit(self, X):
        from sklearn.impute import KNNImputer
        Z = self._prepare_fitting_rows(X)
        self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors).fit(Z)
        return self

    def complete(self, X_partial, draws=20):
        Z_partial = self._prepare_records(X_partial)
        point = self.imputer_.transform(Z_partial)
        # KNNImputer does not expose the spread of the neighbours' values, so the attribute's standard deviation
        # gives a crude interval
        half_width = NORMAL_80_PERCENT_HALF_WIDTH * self.standard_deviations_
        return self._assemble_completion(Z_partial, point, point - half_width, point + half_width)


class ChainedEquationsCompletion(_CompletionEngine):
    """IterativeImputer with BayesianRidge and sample_posterior=True; max_iter is the number of rounds."""
    name = "chained_equations"

    def __init__(self, max_iter=10, **kwargs):
        super().__init__(**kwargs)
        self.max_iter = max_iter

    def fit(self, X):
        importlib.import_module("sklearn.experimental.enable_iterative_imputer")   # makes IterativeImputer importable
        from sklearn.impute import IterativeImputer
        from sklearn.linear_model import BayesianRidge
        Z = self._prepare_fitting_rows(X)
        self.imputer_ = IterativeImputer(estimator=BayesianRidge(), max_iter=self.max_iter, sample_posterior=True,
                                         random_state=self.seed, skip_complete=True).fit(Z)
        return self

    def complete(self, X_partial, draws=20):
        Z_partial = self._prepare_records(X_partial)
        # every transform continues the imputer's random stream, so repeated calls give different draws;
        # between 2 and 10 draws are made
        completions = np.stack([self.imputer_.transform(Z_partial) for _ in range(max(2, min(draws, 10)))])
        return self._assemble_completion(Z_partial, np.median(completions, 0), np.quantile(completions, 0.1, 0),
                                         np.quantile(completions, 0.9, 0))


# -------------------------------------------------------------------------------------------------------- Forest-Flow
class ForestFlowCompletion(_CompletionEngine):
    """Forest-Diffusion in its variance-preserving (vp) diffusion form, used for imputation with REPAINT.

    REPAINT runs the reverse diffusion on the missing cells while the known cells are re-noised to the current
    noise level from their observed values at every step, which keeps the two consistent. After every jump of
    ceil(repaint_jump_fraction * n_t) steps the sampler goes back the same number of steps and repeats them, so each
    jump is run repaint_resamplings times (ForestDiffusion's arguments j and r). n_t, duplicate_K, n_estimators,
    max_depth and gpu_hist are as for the generator in rq2_survey_of_methods/forest_flow_generator.py, whose docstring
    also gives the descriptive names that the cold-start options use for them.
    """
    name = "forest_flow"

    def __init__(self, n_t=10, duplicate_K=10, n_estimators=20, max_depth=5, repaint=True, repaint_resamplings=5,
                 repaint_jump_fraction=0.1, n_jobs=None, gpu_hist=False, **kwargs):
        super().__init__(**kwargs)
        self.n_t, self.duplicate_K, self.n_estimators, self.max_depth = n_t, duplicate_K, n_estimators, max_depth
        self.repaint, self.repaint_resamplings = repaint, repaint_resamplings
        self.repaint_jump_fraction = repaint_jump_fraction
        self.gpu_hist = gpu_hist
        self.n_jobs = n_jobs or max(1, int(os.environ.get("APS_CPU_THREADS_PER_MODEL") or (os.cpu_count() or 1)))

    def fit(self, X):
        from ForestDiffusion import ForestDiffusionModel
        started = time.time()
        Z = self._prepare_fitting_rows(X)
        self.model_ = ForestDiffusionModel(Z, n_t=self.n_t, duplicate_K=self.duplicate_K,
                                           n_estimators=self.n_estimators, max_depth=self.max_depth,
                                           diffusion_type="vp", model="xgboost", gpu_hist=self.gpu_hist,
                                           n_jobs=self.n_jobs, seed=self.seed)
        if self.verbose:
            print(f"    Forest-Flow completion engine fitted on {Z.shape} in {time.time() - started:.0f}s", flush=True)
        return self

    def complete(self, X_partial, draws=20):
        Z_partial = self._prepare_records(X_partial)
        started = time.time()
        completions = []
        for i in range(draws):
            # impute() draws its noise from NumPy's global random generator, which is seeded per draw; the model's
            # own seed attribute is used by its training code only and is set for completeness
            self.model_.seed = self.seed + 1000 + i
            np.random.seed(self.seed + 1000 + i)
            completion = self.model_.impute(k=1, X=Z_partial, repaint=self.repaint, r=self.repaint_resamplings,
                                            j=self.repaint_jump_fraction)
            completion = np.asarray(completion, dtype=np.float64)
            if completion.ndim == 3:
                completion = completion[0]
            completions.append(completion)
        completions = np.stack(completions)
        if self.verbose:
            print(f"    Forest-Flow completion engine: {len(Z_partial)} records x {draws} draws in "
                  f"{time.time() - started:.0f}s", flush=True)
        return self._assemble_completion(Z_partial, np.median(completions, 0), np.quantile(completions, 0.1, 0),
                                         np.quantile(completions, 0.9, 0))


# ---------------------------------------------------------------------------------------------------- Unmasking Trees
class UnmaskingTreesCompletion(_CompletionEngine):
    """Unmasking Trees (package utrees). depth: each attribute is discretised into 2^depth bins; duplicate_K: number
    of random masking orders per training row; xgboost_kwargs: passed to XGBoost (e.g. {"device": "cuda"})."""
    name = "unmasking_trees"

    def __init__(self, depth=4, duplicate_K=5, xgboost_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.depth, self.duplicate_K = depth, duplicate_K
        self.xgboost_kwargs = xgboost_kwargs or {}

    def fit(self, X):
        from utrees import UnmaskingTrees
        started = time.time()
        Z = self._prepare_fitting_rows(X)
        self.model_ = UnmaskingTrees(depth=self.depth, duplicate_K=self.duplicate_K, xgboost_kwargs=self.xgboost_kwargs,
                                     random_state=self.seed).fit(Z.astype(np.float32))
        if self.verbose:
            print(f"    Unmasking Trees completion engine fitted on {Z.shape} in {time.time() - started:.0f}s",
                  flush=True)
        return self

    def complete(self, X_partial, draws=20):
        Z_partial = self._prepare_records(X_partial)
        started = time.time()
        completions = np.asarray(self.model_.impute(n_impute=draws, X=Z_partial.astype(np.float32)),
                                 dtype=np.float64)   # (draws, records, attributes)
        if self.verbose:
            print(f"    Unmasking Trees completion engine: {len(Z_partial)} records x {draws} draws in "
                  f"{time.time() - started:.0f}s", flush=True)
        return self._assemble_completion(Z_partial, np.median(completions, 0), np.quantile(completions, 0.1, 0),
                                         np.quantile(completions, 0.9, 0))


# ------------------------------------------------------------------------------------------------------------- TabPFN
class TabPFNCompletion(_CompletionEngine):
    """One TabPFN regressor per missing attribute.

    The context is the fitting rows (at most max_context_rows, drawn at random; NaNs are kept, TabPFN v2 handles
    them), the target is the missing attribute and the inputs are all other attributes of the records, with their
    missing cells as NaN. An attribute that fewer than 20 context rows observe, or that is constant in the context,
    keeps the median and the median-based interval. APS_TABPFN_MAX_ATTRIBUTES limits the number of attributes
    completed, for smoke tests on a CPU.
    """
    name = "tabpfn"

    def __init__(self, device=None, max_context_rows=1000, **kwargs):
        super().__init__(**kwargs)
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_context_rows = max_context_rows

    def fit(self, X):
        Z = self._prepare_fitting_rows(X)
        if len(Z) > self.max_context_rows:
            rng = np.random.default_rng(self.seed)
            Z = Z[rng.choice(len(Z), self.max_context_rows, replace=False)]
        self.Z_context_ = Z.astype(np.float32)
        return self

    def complete(self, X_partial, draws=20):
        from tabpfn import TabPFNRegressor
        os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
        Z_partial = self._prepare_records(X_partial).astype(np.float32)
        started = time.time()
        n_attributes = Z_partial.shape[1]
        point = np.tile(self.medians_, (len(Z_partial), 1))
        low = np.tile(self.medians_ - NORMAL_80_PERCENT_HALF_WIDTH * self.standard_deviations_, (len(Z_partial), 1))
        high = np.tile(self.medians_ + NORMAL_80_PERCENT_HALF_WIDTH * self.standard_deviations_, (len(Z_partial), 1))
        attributes_to_complete = np.where(np.isnan(Z_partial).any(axis=0))[0]
        if os.environ.get("APS_TABPFN_MAX_ATTRIBUTES"):   # smoke tests on a CPU: complete only the first few attributes
            attributes_to_complete = attributes_to_complete[:int(os.environ["APS_TABPFN_MAX_ATTRIBUTES"])]
        for target in attributes_to_complete:
            records_missing_it = np.where(np.isnan(Z_partial[:, target]))[0]
            context_rows = ~np.isnan(self.Z_context_[:, target])
            if context_rows.sum() < 20 or np.nanstd(self.Z_context_[context_rows, target]) < 1e-6:
                continue                      # constant or nearly empty attribute: keep the median
            inputs = [column for column in range(n_attributes) if column != target]
            regressor = TabPFNRegressor(device=self.device, random_state=self.seed)
            regressor.fit(self.Z_context_[context_rows][:, inputs], self.Z_context_[context_rows, target])
            quantiles = regressor.predict(Z_partial[records_missing_it][:, inputs], output_type="quantiles",
                                          quantiles=[0.1, 0.5, 0.9])
            quantiles = np.asarray(quantiles)   # (3, n) in recent tabpfn versions; (n, 3) in older ones
            if quantiles.shape[0] != 3:
                quantiles = quantiles.T
            point[records_missing_it, target] = quantiles[1]
            low[records_missing_it, target] = quantiles[0]
            high[records_missing_it, target] = quantiles[2]
        if self.verbose:
            print(f"    TabPFN completion engine: {len(Z_partial)} records, {len(attributes_to_complete)} attributes "
                  f"completed in {time.time() - started:.0f}s [{self.device}]", flush=True)
        return self._assemble_completion(Z_partial, point, low, high)


COMPLETION_ENGINES = {"median": MedianCompletion, "k_nearest_neighbours": KNearestNeighboursCompletion,
                      "chained_equations": ChainedEquationsCompletion, "forest_flow": ForestFlowCompletion,
                      "unmasking_trees": UnmaskingTreesCompletion, "tabpfn": TabPFNCompletion}


# -------------------------------------------------------------------------------------------------- masking functions
def hide_cells_completely_at_random(X: pd.DataFrame, hidden_fraction: float,
                                    seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Hide known cells of every record independently with probability hidden_fraction.

    Whether a cell is hidden depends on nothing in the data, so the hidden cells are missing completely at random
    (MCAR). Cells that are already missing stay missing and are not counted. Returns the masked records and the
    boolean matrix of the hidden cells.
    """
    rng = np.random.default_rng(seed)
    values = X.to_numpy(dtype=np.float64).copy()
    hidden = (rng.random(values.shape) < hidden_fraction) & ~np.isnan(values)
    values[hidden] = np.nan
    return pd.DataFrame(values, columns=X.columns, index=X.index), hidden


def hide_cells_with_real_missingness_patterns(X: pd.DataFrame, pattern_pool: pd.DataFrame, min_missing_fraction: float,
                                              seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Hide cells following the missingness patterns of real records.

    Every record receives the missing-value pattern of a record drawn at random (with replacement) from those rows
    of pattern_pool that miss at least min_missing_fraction of their attributes; its known cells in that pattern
    are hidden. The hidden cells thus follow the dataset's own patterns of unavailable counters. Returns the masked
    records and the boolean matrix of the hidden cells.
    """
    rng = np.random.default_rng(seed)
    pool_missing = pattern_pool.isna().to_numpy()
    candidates = np.where(pool_missing.mean(axis=1) >= min_missing_fraction)[0]
    if len(candidates) == 0:
        raise ValueError("no pool rows with that much missingness")
    values = X.to_numpy(dtype=np.float64).copy()
    patterns = pool_missing[rng.choice(candidates, len(values))]
    hidden = patterns & ~np.isnan(values)
    values[hidden] = np.nan
    return pd.DataFrame(values, columns=X.columns, index=X.index), hidden
