"""The preprocessing pipeline of Beikmohammadi et al., and a variant with a validation split of real rows (RQ1b).

Research question 1 of the thesis reproduces two published results on the Scania APS data; this folder belongs to
the second, Beikmohammadi et al. (Cluster Computing, 2026). Their public script DataPreprocessingCode.py prepares the
data in five steps, which this module re-implements:

  1. Drop every attribute with 10 % or more missing values in the training rows (142 of the 170 attributes remain
     in the authors' pipeline; 141 here, because the experiment scripts drop the constant attribute cd_000 first).
  2. Fill the remaining missing values with scikit-learn's IterativeImputer around a BayesianRidge regression
     (tolerance 1e-3, mean as the starting value, attributes in ascending order of missingness). The imputer works
     by chained equations: each attribute with missing values is in turn predicted by a Bayesian linear regression
     on all the other attributes, and the round over all attributes is repeated, up to max_iter rounds, until the
     filled-in values stop changing. The authors use up to 100 rounds; on the full training table the stopping rule
     is not met within 100 rounds, which take about four hours on one CPU core. The reported runs of this
     repository used 10 rounds (25 is the default of the tree-model script).
  3. SVM-SMOTE (sampling_strategy 0.5, k_neighbors 5, m_neighbors 10, out_step 0.5). SMOTE creates a synthetic
     minority row at a random point on the line between a minority row and one of its nearest minority neighbours.
     SVM-SMOTE starts only from failure rows near the class boundary, which it locates as the failure rows that are
     support vectors of an SVM fitted to both classes. The 10 nearest neighbours of each such row among all rows
     decide how it is used (the "danger" test of imbalanced-learn): a row whose 10 neighbours are all healthy is
     taken for noise and skipped; a row with half or more, but not all, healthy neighbours is "in danger", and new
     rows are interpolated between it and one of its 5 nearest failure neighbours; a row with fewer healthy
     neighbours is "safe", and new rows are extrapolated from it, placed on the side away from the failure neighbour
     at up to out_step (0.5) times the distance between the two. A sampling_strategy of 0.5 adds failure rows until
     there is one failure for every two healthy rows.
  4. Repeated edited nearest neighbours (RENN; sampling_strategy "majority", 3 neighbours, kind_sel "all", up to 100
     repetitions). Edited nearest neighbours removes the rows whose neighbourhood disagrees with their label. Here
     only healthy rows can be removed, and with kind_sel "all" a healthy row is removed unless all three of its
     nearest neighbours among the current rows (synthetic failures included) are healthy. RENN repeats this
     cleaning on the rows that are left until a pass removes nothing, for at most 100 passes. In the resulting
     training table, healthy rows close to failure rows are gone, a cleaning that held-out data never receive. On
     the full training table steps 3 and 4 give 57,136 healthy and 29,500 failure rows.
  5. A min-max scaler (each attribute mapped to [0, 1]) fitted on the resampled table and applied to every table.

The authors' TrainingCode.py then holds out a stratified 10 % of the resampled, scaled table as the validation set on
which the best epoch is chosen.

protocol="published" follows that order: imputation, resampling and scaling are fitted on all training rows, and the
validation rows are drawn afterwards from the resampled table. They therefore include synthetic SVM-SMOTE failures,
contain only the healthy rows that RENN kept, and took part in fitting the imputer and the scaler.
protocol="corrected", which the thesis calls the protocol with a validation split of real rows,
holds out the 10 % validation rows (real rows, stratified) before step 2. The imputer, the resampler and the scaler
are fitted on the remaining 90 %, and the validation rows are only imputed and scaled, so the choice of the best
epoch rests on real rows that took no part in fitting those steps. In both protocols the outer held-out rows (the
official test set for the Transformer port, a cross-validation fold for the tree models) are transformed with the
imputer and the scaler fitted on the training rows, never fitted on.

The results of the expensive steps are cached under cache/ (paths.CACHE_DIR), keyed by the rows and the settings,
so that several resampling variants and learners can share one imputed table. The cache can be deleted at any time.
The module has no command line; it is used by run_transformer_2026_port.py, evaluate_port_checkpoint.py and
run_tree_models_on_2026_preprocessing.py.
"""
from __future__ import annotations

import hashlib
import importlib
import time

import numpy as np
import pandas as pd

from experiments.common.paths import CACHE_DIR

MISSING_FRACTION_LIMIT = 0.10   # step 1: attributes with this fraction of missing values or more are dropped


def columns_below_missing_limit(X_train: pd.DataFrame, limit: float = MISSING_FRACTION_LIMIT) -> list[str]:
    """Columns of X_train whose fraction of missing values is below limit (step 1)."""
    missing_fraction = X_train.isna().mean()
    return list(missing_fraction[missing_fraction < limit].index)


def fit_bayesian_ridge_imputer(X: pd.DataFrame, imputer_rounds: int = 100, seed: int = 0, verbose=True):
    """IterativeImputer with BayesianRidge and the settings of Beikmohammadi et al. (step 2), fitted on X."""
    importlib.import_module("sklearn.experimental.enable_iterative_imputer")   # makes IterativeImputer importable
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge
    started = time.time()
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=imputer_rounds, tol=1e-3, initial_strategy="mean",
                               imputation_order="ascending", sample_posterior=False, random_state=seed)
    # float64 as in the original pandas-based code; float32 overflows in BayesianRidge
    imputer.fit(np.asarray(X, dtype=np.float64))
    if verbose:
        print(f"    imputer: {imputer.n_iter_} rounds in {time.time() - started:.0f}s", flush=True)
    return imputer


def _as_float64(table):
    return np.asarray(table, dtype=np.float64)


def _cache_file(prefix: str, cache_tag: str, key: str):
    return CACHE_DIR / ("_".join(part for part in (prefix, cache_tag, key) if part) + ".npz")


def impute_fold_as_in_2026_study(X_train: pd.DataFrame, X_held_out: pd.DataFrame, *, cache_tag: str,
                                 imputer_rounds: int = 25, seed: int = 0, verbose=True,
                                 X_additional: pd.DataFrame | None = None):
    """Steps 1 and 2 only, fitted on X_train and applied to X_train, X_held_out and, if given, X_additional.

    The columns are chosen and the imputer is fitted on X_train alone. X_additional is typically the official test
    set, so that a cross-validation fold can also score it with its own imputer. cache_tag names the fold in the
    cache file (e.g. "seed11_fold0").

    Returns (X_train_imputed, X_held_out_imputed, kept_columns, rounds_run, X_additional_imputed or None), the
    arrays as float32.
    """
    key = hashlib.md5((cache_tag + str(len(X_train)) + str(list(X_train.index[:100]))
                       + f"{imputer_rounds}{seed}").encode()).hexdigest()[:12]
    cache_file = _cache_file("imputed_fold_2026", cache_tag, key)
    if cache_file.exists():
        cached = np.load(cache_file, allow_pickle=True)
        additional = cached["X_additional"] if "X_additional" in cached.files else None
        return (cached["X_train"], cached["X_held_out"], list(cached["columns"]), int(cached["imputer_rounds"]),
                additional)
    columns = columns_below_missing_limit(X_train, MISSING_FRACTION_LIMIT)
    imputer = fit_bayesian_ridge_imputer(X_train[columns], imputer_rounds, seed, verbose)
    train_imputed = imputer.transform(_as_float64(X_train[columns])).astype(np.float32)
    held_out_imputed = imputer.transform(_as_float64(X_held_out[columns])).astype(np.float32)
    additional_imputed = (imputer.transform(_as_float64(X_additional[columns])).astype(np.float32)
                          if X_additional is not None else None)
    payload = dict(X_train=train_imputed, X_held_out=held_out_imputed, columns=np.array(columns),
                   imputer_rounds=np.array(imputer.n_iter_))
    if additional_imputed is not None:
        payload["X_additional"] = additional_imputed
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, **payload)
    return train_imputed, held_out_imputed, columns, int(imputer.n_iter_), additional_imputed


def resample_as_in_2026_study(X: np.ndarray, y: np.ndarray, seed: int = 0, use_renn=True, verbose=True,
                              use_smote=True):
    """Steps 3 and 4: SVM-SMOTE (unless use_smote is False), then RENN (unless use_renn is False)."""
    from imblearn.over_sampling import SVMSMOTE
    from imblearn.under_sampling import RepeatedEditedNearestNeighbours
    started = time.time()
    if use_smote:
        X_resampled, y_resampled = SVMSMOTE(sampling_strategy=0.5, k_neighbors=5, m_neighbors=10, out_step=0.5,
                                            random_state=seed).fit_resample(X, y)
    else:
        X_resampled, y_resampled = np.asarray(X), np.asarray(y)
    rows_after_smote = len(y_resampled)
    if use_renn:
        X_resampled, y_resampled = RepeatedEditedNearestNeighbours(sampling_strategy="majority", n_neighbors=3,
                                                                   max_iter=100, kind_sel="all").fit_resample(
                                                                       X_resampled, y_resampled)
    if verbose:
        print(f"    {'SVM-SMOTE' if use_smote else 'no SVM-SMOTE'}{' + RENN' if use_renn else ''}: "
              f"{len(y)} -> {rows_after_smote} -> {len(y_resampled)} rows (healthy {int((y_resampled == 0).sum())} / "
              f"failures {int(y_resampled.sum())}) in {time.time() - started:.0f}s", flush=True)
    return X_resampled, y_resampled


def prepare_fold_as_in_2026_study(X_train: pd.DataFrame, y_train: np.ndarray, X_held_out: pd.DataFrame, *,
                                  protocol="corrected", imputer_rounds=100, seed=0, use_renn=True,
                                  validation_fraction=0.1, cache_tag="", verbose=True, use_smote=True):
    """All five steps and the validation split, under the "published" or the "corrected" protocol.

    Returns a dict of arrays: X_fit, y_fit (the resampled and scaled rows the network is trained on), X_val, y_val
    (the validation rows used to choose the best epoch), X_held_out (the outer held-out rows, imputed and scaled),
    columns (the attributes kept), seconds and imputer_rounds (the rounds the imputer ran). use_smote and use_renn
    switch off steps 3 and 4 for the ablations. The result is cached; cache_tag names it in the cache file.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    key = hashlib.md5((cache_tag + str(len(X_train)) + str(int(y_train.sum())) + str(list(X_train.index[:50]))
                       + f"{protocol}{imputer_rounds}{seed}{use_renn}{validation_fraction}"
                       + f"{'' if use_smote else 'nosmote'}").encode()).hexdigest()[:12]
    cache_file = _cache_file("prepared_fold_2026", cache_tag, key)
    if cache_file.exists():
        cached = np.load(cache_file, allow_pickle=True)
        return {name: cached[name] for name in cached.files}
    started = time.time()
    columns = columns_below_missing_limit(X_train, MISSING_FRACTION_LIMIT)
    X_train_kept, X_held_out_kept = X_train[columns], X_held_out[columns]
    if protocol == "corrected":
        # real validation rows held out BEFORE imputation and resampling
        X_fit_raw, X_val_raw, y_fit_raw, y_val = train_test_split(X_train_kept, y_train, test_size=validation_fraction,
                                                                  stratify=y_train, random_state=seed)
        imputer = fit_bayesian_ridge_imputer(X_fit_raw, imputer_rounds, seed, verbose)
        fit_imputed = imputer.transform(_as_float64(X_fit_raw))
        val_imputed = imputer.transform(_as_float64(X_val_raw))
        held_out_imputed = imputer.transform(_as_float64(X_held_out_kept))
        X_resampled, y_resampled = resample_as_in_2026_study(fit_imputed, y_fit_raw, seed=seed, use_renn=use_renn,
                                                             verbose=verbose, use_smote=use_smote)
        scaler = MinMaxScaler().fit(X_resampled)
        prepared = dict(X_fit=scaler.transform(X_resampled), y_fit=y_resampled, X_val=scaler.transform(val_imputed),
                        y_val=y_val, X_held_out=scaler.transform(held_out_imputed))
    elif protocol == "published":
        imputer = fit_bayesian_ridge_imputer(X_train_kept, imputer_rounds, seed, verbose)
        train_imputed = imputer.transform(_as_float64(X_train_kept))
        held_out_imputed = imputer.transform(_as_float64(X_held_out_kept))
        X_resampled, y_resampled = resample_as_in_2026_study(train_imputed, y_train, seed=seed, use_renn=use_renn,
                                                             verbose=verbose, use_smote=use_smote)
        scaler = MinMaxScaler().fit(X_resampled)
        X_resampled = scaler.transform(X_resampled)
        # the validation rows are drawn from the resampled table, as in the authors' TrainingCode.py
        X_fit, X_val, y_fit, y_val = train_test_split(X_resampled, y_resampled, test_size=validation_fraction,
                                                      stratify=y_resampled, random_state=seed)
        prepared = dict(X_fit=X_fit, y_fit=y_fit, X_val=X_val, y_val=y_val,
                        X_held_out=scaler.transform(held_out_imputed))
    else:
        raise ValueError(protocol)
    prepared["columns"] = np.array(columns)
    prepared["seconds"] = np.array(time.time() - started)
    prepared["imputer_rounds"] = np.array(getattr(imputer, "n_iter_", -1))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, **prepared)
    if verbose:
        print(f"    prepared fold [{protocol} protocol] in {time.time() - started:.0f}s -> {cache_file.name}",
              flush=True)
    return prepared
