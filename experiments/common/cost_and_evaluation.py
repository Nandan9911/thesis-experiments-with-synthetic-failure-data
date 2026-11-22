"""The common evaluation protocol: cost curve, cost-minimising threshold, cross-validation and bootstrap interval.

Every classifier arm of RQ1 and RQ2 (run_experiment_arm.py), the hyperparameter search and the first-look baselines
are evaluated with `run_cross_validation`; the scripts of RQ1, RQ2 and RQ3 that produce their own predictions reuse
`cost_minimising_threshold`, `cost_curve` and `bootstrap_cost_confidence_interval`. The module has no command line.

The protocol, in the terms used throughout the thesis:

* Stratified k-fold cross-validation. The 60,000 training rows are shuffled and split into k parts (five unless
  stated otherwise), each with the same share of failures (1.67 %). Each part is held out once while a model is
  fitted on the other parts. With only 1,000 failures, an unstratified split could give folds with noticeably
  different numbers of failures, and their costs would not be comparable.
* Fitting inside the fold. Everything that learns from data, the imputer, the oversampler or generator, and the
  classifier, is fitted on the training part of the fold only. The held-out part stays real and untouched.
* Out-of-fold (OOF) predictions. Each held-out part is scored by the model that was fitted without it, so putting
  the k held-out parts together gives one failure probability for every training row from a model that never saw
  that row. Every choice in the thesis (hyperparameters, decision threshold, number of synthetic rows) is made on
  these OOF probabilities; the official 16,000-row test set is never used to choose anything.
* Decision threshold. A truck is flagged when its predicted failure probability is at least the threshold. The
  threshold is the one that minimises the cost on the OOF probabilities (see `cost_curve` for the candidates) and is
  frozen before the test set is scored.
* Test prediction. The test set is scored with the average of the k fold models' probabilities. This average is on
  the same scale as the OOF probabilities on which the threshold was chosen; a single model refitted on all training
  rows is not, and with cost-minimising thresholds around 1e-4 a threshold transfers poorly from one scale to another.
* Uncertainty of a test cost: a percentile bootstrap over the test rows with the decisions held fixed.

The cost is the challenge cost of data_loading.misclassification_cost: 500 per missed failure (false negative) plus
10 per false alarm (false positive).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from experiments.common.data_loading import (COST_OF_FALSE_ALARM, COST_OF_MISSED_FAILURE, TEST_SET_ROWS,
                                             misclassification_cost)

# augment(X_train, y_train, seed) -> (X_train with synthetic rows added, y_train extended accordingly)
Augmenter = Callable[[pd.DataFrame, np.ndarray, int], tuple[pd.DataFrame, np.ndarray]]
# preprocess(X_train, y_train) -> transform, where transform(X) returns the preprocessed copy of any table X
Preprocessor = Callable[[pd.DataFrame, np.ndarray], Callable[[pd.DataFrame], pd.DataFrame]]

# Candidate thresholds are the quantiles of the predicted probabilities at this many evenly spaced levels from 0 to 1,
# i.e. one every twentieth of a percentile.
QUANTILE_LEVELS = 2001


def cost_curve(probabilities: np.ndarray, y: np.ndarray, thresholds: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Cost of the decisions "flag the truck when its probability >= threshold" for a range of thresholds.

    `probabilities` holds predicted failure probabilities and y the true labels (1 = failure). Unless `thresholds` is
    given, the candidates are the quantiles of the probabilities at the 2,001 levels 0, 0.0005, 0.001, ..., 1,
    together with 0 and 1, with duplicates removed (at most 2,003 candidates, fewer when there are tied values).
    Quantiles follow the predictions wherever they lie: the cost-minimising thresholds of the boosted trees in this
    project are around 1e-5 to 1e-4, a region that a fixed grid such as 0.01, 0.02, ..., 0.99 never reaches. numpy's
    default (linear) interpolation can place a quantile between two observed probabilities; because a row is flagged
    when its probability is at least the threshold, such a candidate gives the same decisions as the next observed
    probability above it. The added 0 and 1 make sure that flagging every truck and (unless some probability is
    exactly 1) flagging none are among the candidates.

    Returns a DataFrame with one row per candidate threshold (in increasing order when the candidates are formed
    here) and the columns threshold, cost, fn, fp and tp.
    """
    y = np.asarray(y).astype(int)
    if thresholds is None:
        thresholds = np.unique(np.concatenate([np.quantile(probabilities, np.linspace(0, 1, QUANTILE_LEVELS)),
                                               [0.0, 1.0]]))
    # Sorted by decreasing probability, the rows flagged at any threshold form a prefix of this order, so cumulative
    # sums give the true and false positives of every threshold without going through the data again.
    order = np.argsort(-probabilities)
    sorted_probabilities, sorted_labels = probabilities[order], y[order]
    cumulative_true_positives = np.cumsum(sorted_labels)
    cumulative_false_positives = np.cumsum(1 - sorted_labels)
    total_failures = sorted_labels.sum()
    rows = []
    for threshold in thresholds:
        # number of rows with probability >= threshold (the negated sorted probabilities are increasing, as
        # searchsorted needs)
        flagged = int(np.searchsorted(-sorted_probabilities, -threshold, side="right"))
        tp = int(cumulative_true_positives[flagged - 1]) if flagged > 0 else 0
        fp = int(cumulative_false_positives[flagged - 1]) if flagged > 0 else 0
        fn = int(total_failures - tp)
        rows.append((float(threshold), COST_OF_MISSED_FAILURE * fn + COST_OF_FALSE_ALARM * fp, fn, fp, tp))
    return pd.DataFrame(rows, columns=["threshold", "cost", "fn", "fp", "tp"])


def cost_minimising_threshold(probabilities: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """The candidate threshold of `cost_curve` with the lowest cost on (probabilities, y), and that cost.

    When several thresholds reach the minimum, the largest of them is returned: among equally good choices it flags
    the fewest trucks, which is the conservative choice.
    """
    curve = cost_curve(probabilities, y)
    minimum_cost = curve["cost"].min()
    threshold = curve.loc[curve["cost"] == minimum_cost, "threshold"].max()
    return float(threshold), int(minimum_cost)


@dataclass
class CrossValidationResult:
    """What `run_cross_validation` produces for one arm and seed.

    out_of_fold_probabilities  failure probability of every training row, from the fold model that did not see it
    y                          true labels of the training rows
    fold_of_row                index (0 to k - 1) of the fold in which each row was held out
    seconds_per_fold           wall-clock seconds of each fold (preprocessing, augmentation, fitting, prediction); for a
                               fold restored from the fold cache, the time it took when it was computed
    fold_models                fitted classifier of each fold, kept so that the fold models can score the test set;
                               None for a fold restored from the fold cache
    fold_transforms            fitted preprocessing transform of each fold (None without preprocessing or for a
                               restored fold)
    fold_test_probabilities    test-set probabilities computed inside the fold loop (only when run_cross_validation
                               was given X_test), one vector per fold
    X_test                     the test table those probabilities belong to
    """
    name: str
    out_of_fold_probabilities: np.ndarray
    y: np.ndarray
    fold_of_row: np.ndarray
    seconds_per_fold: list = field(default_factory=list)
    fold_models: list = field(default_factory=list)
    fold_transforms: list = field(default_factory=list)
    fold_test_probabilities: list = field(default_factory=list)
    X_test: object = field(default=None, repr=False, compare=False)

    def fold_average_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        """Failure probabilities for the rows of X: the average over the fold models.

        Each fold model sees X after its own fold's preprocessing. If X is the very test table that was passed to
        run_cross_validation, the probabilities computed fold by fold are averaged instead (the same numbers, and
        the only possibility when some folds were restored from the fold cache and therefore have no model).
        """
        if (self.fold_test_probabilities and X is self.X_test
                and len(self.fold_test_probabilities) == len(self.fold_models)):
            return np.mean(self.fold_test_probabilities, axis=0)
        if any(model is None for model in self.fold_models):
            raise RuntimeError("fold models were restored from the fold cache; pass the cached test set")
        probabilities = []
        for model, transform in zip(self.fold_models, self.fold_transforms):
            X_for_model = transform(X) if transform is not None else X
            probabilities.append(model.predict_proba(X_for_model)[:, 1])
        return np.mean(probabilities, axis=0)

    def summary(self, threshold: Optional[float] = None) -> dict:
        """Out-of-fold cost at `threshold` (by default the cost-minimising threshold) and related figures.

        The keys are those stored in the result files: cost, tp, fn, fp, tn, recall and precision (as in
        misclassification_cost), name, threshold, fit_seconds_total (sum of seconds_per_fold), fold_costs and
        cost_scaled_to_test_set_size. fold_costs are the costs of the k held-out folds at the same, global threshold;
        they are the paired observations of the fold-level comparisons between arms. cost_scaled_to_test_set_size
        is the cost multiplied by 16,000 / (number of rows), which puts an out-of-fold cost on the scale of the test
        set; the slightly higher failure rate of the test file is ignored.
        """
        if threshold is None:
            threshold, _ = cost_minimising_threshold(self.out_of_fold_probabilities, self.y)
        summary = misclassification_cost(self.y, (self.out_of_fold_probabilities >= threshold).astype(int))
        summary.update({"name": self.name, "threshold": threshold,
                        "fit_seconds_total": float(np.sum(self.seconds_per_fold))})
        fold_costs = []
        for fold in np.unique(self.fold_of_row):
            in_fold = self.fold_of_row == fold
            fold_decisions = (self.out_of_fold_probabilities[in_fold] >= threshold).astype(int)
            fold_costs.append(misclassification_cost(self.y[in_fold], fold_decisions)["cost"])
        summary["fold_costs"] = fold_costs
        summary["cost_scaled_to_test_set_size"] = summary["cost"] * TEST_SET_ROWS / len(self.y)
        return summary


def run_cross_validation(name: str,
                         make_model: Callable[[int], object],
                         X: pd.DataFrame, y: np.ndarray,
                         n_splits: int = 5, seed: int = 11,
                         augment: Optional[Augmenter] = None,
                         preprocess: Optional[Preprocessor] = None,
                         verbose: bool = True, cache_dir=None,
                         X_test: Optional[pd.DataFrame] = None) -> CrossValidationResult:
    """Stratified k-fold cross-validation with out-of-fold probabilities, everything fitted inside the folds.

    name        label of the run, used in progress messages and in the names of the fold cache files
    make_model  make_model(seed) returns an unfitted classifier with fit and predict_proba
    X, y        training attributes and labels
    n_splits    number of folds k
    seed        fixes the split (a shuffled StratifiedKFold) and, as seed * 100 + fold, the random state of the
                augmenter and of the classifier in each fold, so that the folds differ from each other while every
                run with the same seed repeats exactly
    augment     augment(X_train, y_train, seed) adds synthetic rows to the training part of the fold (after the
                preprocessing); the held-out part never receives any
    preprocess  preprocess(X_train, y_train) is fitted on the training part of the fold and returns a transform that
                is applied to both parts (and later to the test set)
    verbose     print one line per fold with its time and the cost at the fold's own best threshold (progress
                information only)
    cache_dir, X_test
                the fold cache used by the cloud runs. With cache_dir, every finished fold stores its held-out row
                indices, its OOF probabilities, its time and, if X_test is given, its test probabilities in one .npz
                file per run, seed, k and fold; a rerun of the same arm and seed (for example after a cloud runtime
                was replaced half-way) restores the finished folds instead of refitting them. A stored fold is used
                only if its held-out indices equal those of the current split (and its test vector has the length of
                X_test); an unreadable file is ignored and the fold refitted. A restored fold has no model, so the test
                set must then be scored inside the fold loop, which is why X_test is passed in here. Without
                cache_dir, giving X_test only moves the scoring of the test set into the loop.

    Returns a CrossValidationResult.
    """
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out_of_fold_probabilities = np.zeros(len(y), dtype=np.float64)
    fold_of_row = np.zeros(len(y), dtype=np.int16)
    seconds_per_fold, fold_models, fold_transforms, fold_test_probabilities = [], [], [], []
    if cache_dir is not None:
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except Exception:  # noqa: BLE001
            cache_dir = None    # the cache only saves time: without a writable folder every fold is fitted
    for fold, (train_index, validation_index) in enumerate(folds.split(X, y)):
        fold_started = time.time()
        cache_file = None
        if cache_dir is not None:
            cache_file = os.path.join(cache_dir, f"fold_cache_{name}_seed{seed}_{n_splits}_folds_fold{fold}.npz")
        if cache_file is not None and os.path.exists(cache_file):
            try:
                cached = np.load(cache_file)
                if (np.array_equal(cached["validation_index"], validation_index)
                        and (X_test is None or len(cached["test_probabilities"]) == len(X_test))):
                    out_of_fold_probabilities[validation_index] = cached["out_of_fold_probabilities"]
                    fold_of_row[validation_index] = fold
                    seconds_per_fold.append(float(cached["seconds"]))
                    fold_models.append(None)
                    fold_transforms.append(None)
                    if X_test is not None:
                        fold_test_probabilities.append(cached["test_probabilities"])
                    if verbose:
                        print(f"  [{name}] fold {fold}: restored from the fold cache "
                              f"({seconds_per_fold[-1]:.0f}s when it was computed)", flush=True)
                    continue
            except Exception as error:  # noqa: BLE001
                print(f"  [{name}] fold {fold}: fold cache unreadable ({type(error).__name__}); refitting", flush=True)
        X_train, y_train = X.iloc[train_index], y[train_index]
        X_validation = X.iloc[validation_index]
        transform = None
        if preprocess is not None:
            transform = preprocess(X_train, y_train)
            X_train = transform(X_train)
            X_validation = transform(X_validation)
        if augment is not None:
            X_train, y_train = augment(X_train, y_train, seed * 100 + fold)
        model = make_model(seed * 100 + fold)
        model.fit(X_train, y_train)
        out_of_fold_probabilities[validation_index] = model.predict_proba(X_validation)[:, 1]
        fold_of_row[validation_index] = fold
        seconds_per_fold.append(time.time() - fold_started)
        fold_models.append(model)
        fold_transforms.append(transform)
        if X_test is not None:
            X_test_for_model = transform(X_test) if transform is not None else X_test
            fold_test_probabilities.append(model.predict_proba(X_test_for_model)[:, 1])
        if cache_file is not None:
            try:
                np.savez(cache_file, validation_index=validation_index,
                         out_of_fold_probabilities=out_of_fold_probabilities[validation_index],
                         seconds=seconds_per_fold[-1],
                         test_probabilities=fold_test_probabilities[-1] if X_test is not None else np.zeros(0))
            except Exception as error:  # noqa: BLE001
                print(f"  [{name}] fold {fold}: could not write the fold cache ({type(error).__name__}: {error})",
                      flush=True)
        if verbose:
            fold_threshold, fold_cost = cost_minimising_threshold(out_of_fold_probabilities[validation_index],
                                                                  y[validation_index])
            print(f"  [{name}] fold {fold}: {seconds_per_fold[-1]:.1f}s, cost {fold_cost} at the fold's own "
                  f"best threshold {fold_threshold:.4f}", flush=True)
    return CrossValidationResult(name=name, out_of_fold_probabilities=out_of_fold_probabilities, y=y,
                                 fold_of_row=fold_of_row, seconds_per_fold=seconds_per_fold, fold_models=fold_models,
                                 fold_transforms=fold_transforms, fold_test_probabilities=fold_test_probabilities,
                                 X_test=X_test)


def bootstrap_cost_confidence_interval(y: np.ndarray, y_pred: np.ndarray, n_boot: int = 2000,
                                       seed: int = 0) -> tuple[float, float]:
    """95 % percentile bootstrap interval of the cost of fixed decisions y_pred (1 = flag) against the labels y.

    The rows are resampled with replacement n_boot times, each resample as large as the data, and the cost of the
    unchanged decisions is computed on each; the interval runs from the 2.5th to the 97.5th percentile of these costs.
    It describes the uncertainty that comes from which trucks happen to be in the test set, not the variability of
    training. With 375 failures in the test set, every missed failure moves the cost by 500, and the interval is
    correspondingly wide.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    is_missed_failure = ((y == 1) & (y_pred == 0)).astype(int)
    is_false_alarm = ((y == 0) & (y_pred == 1)).astype(int)
    n = len(y)
    costs = np.empty(n_boot)
    for b in range(n_boot):
        resampled_rows = rng.integers(0, n, n)
        costs[b] = (COST_OF_MISSED_FAILURE * is_missed_failure[resampled_rows].sum()
                    + COST_OF_FALSE_ALARM * is_false_alarm[resampled_rows].sum())
    return float(np.percentile(costs, 2.5)), float(np.percentile(costs, 97.5))
