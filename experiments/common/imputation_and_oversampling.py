"""Median imputation and SMOTE-family oversampling, as building blocks of the cross-validation protocol.

Part of the common evaluation protocol (experiments/common). The classifier arms of RQ1 and RQ2 (experiment_arms.py) and
the synthetic-data diagnostics of RQ2 use these functions. Both return callables of the form that
cost_and_evaluation.run_cross_validation expects, so that they are fitted inside each training fold:

    preprocess(X_train, y_train) -> transform     fitted on the training part of a fold; transform(X) is then applied
                                                   to the training part, the held-out part and the test set
    augment(X_train, y_train, seed) -> (X, y)      applied to the training part of a fold only

Median imputation replaces each missing value by the median of its attribute over the observed values in the
training part of the fold, so the held-out rows never influence the fill values. It is needed by the learners that
reject missing values (scikit-learn's GradientBoostingClassifier) and by the SMOTE family, which interpolates between
rows and therefore needs complete rows; the random forest arms also run on median-imputed attributes.

The SMOTE family makes synthetic failure rows out of the real failure rows of the training part (all four methods as
implemented in imbalanced-learn):

    SMOTE             (Chawla et al., 2002) picks a failure row and one of its k nearest failure neighbours and
                      places a new row at a random point on the line segment between the two
    Borderline-SMOTE  (Han et al., 2005) does the same, but starts only from failure rows near the class boundary:
                      those for which half or more, but not all, of their nearest neighbours among all rows are healthy
    ADASYN            (He et al., 2008) makes more rows around the failure rows that are harder to learn, in proportion
                      to the share of healthy rows among their nearest neighbours
    SVM-SMOTE         (Nguyen et al., 2011) fits a support vector machine and generates rows around the failure rows
                      that are its support vectors, i.e. that lie near the class boundary

Each synthetic row lies on the line between two real failure rows (SVM-SMOTE can also step a little beyond one), so
the rows are interpolations of existing failure rows rather than samples from a learned model of the failures.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd


def median_imputation_preprocessor(add_missing_indicators: bool = False):
    """Preprocessor that fills missing values with the training-fold median of each attribute.

    With add_missing_indicators, the transform also appends one 0/1 column (float32, named "<attribute>_na") for each
    attribute that has at least one missing value in the training part of the fold, so that the information in the
    missingness pattern survives the imputation. (The ensemble arms add their indicators differently, see
    classifiers.MissingnessIndicatorFeatures.)
    """
    def preprocess(X_train: pd.DataFrame, y_train: np.ndarray):
        medians = X_train.median()
        columns_with_missing_values = [column for column in X_train.columns if X_train[column].isna().any()]

        def transform(X: pd.DataFrame) -> pd.DataFrame:
            imputed = X.fillna(medians)
            if add_missing_indicators:
                indicators = X[columns_with_missing_values].isna().astype(np.float32).add_suffix("_na")
                imputed = pd.concat([imputed, indicators], axis=1)
            return imputed
        return transform
    return preprocess


def smote_family_augmenter(kind="smote", sampling_strategy=0.5, k_neighbors=5, n_synthetic=None, verbose=True):
    """Augmenter that adds SMOTE-family failure rows to the training part of a fold.

    kind               "smote", "borderline_smote", "svm_smote" or "adasyn" (any other value also gives ADASYN)
    k_neighbors        number of nearest failure neighbours to interpolate towards (n_neighbors for ADASYN)
    n_synthetic        number of synthetic failure rows to add. If given, the sampling strategy becomes
                       {1: number of real failures + n_synthetic}, the target size of the failure class, so that the
                       arm adds as many rows as the generator arms (2,000) and the comparison between them is
                       count-matched. SMOTE, Borderline-SMOTE and SVM-SMOTE then add exactly n_synthetic rows; ADASYN
                       shares the rows out over the failure rows and rounds each share, so its total can differ
                       slightly.
    sampling_strategy  used only when n_synthetic is None and passed on to imbalanced-learn (a float is the ratio of
                       failures to healthy rows wanted after oversampling)
    verbose            print the row counts before and after

    The input must be complete (imputed). augment(X_train, y_train, seed) returns the real rows followed by the
    synthetic ones, as a new float32 DataFrame with the same columns and a fresh index, and the labels with the
    dtype of y_train.
    """
    from imblearn.over_sampling import ADASYN, SMOTE, SVMSMOTE, BorderlineSMOTE

    def augment(X_train: pd.DataFrame, y_train: np.ndarray, seed: int):
        started = time.time()
        columns = list(X_train.columns)
        if n_synthetic is not None:
            strategy = {1: int((y_train == 1).sum()) + int(n_synthetic)}
        else:
            strategy = sampling_strategy
        if kind == "smote":
            oversampler = SMOTE(sampling_strategy=strategy, k_neighbors=k_neighbors, random_state=seed)
        elif kind == "borderline_smote":
            oversampler = BorderlineSMOTE(sampling_strategy=strategy, k_neighbors=k_neighbors, random_state=seed)
        elif kind == "svm_smote":
            oversampler = SVMSMOTE(sampling_strategy=strategy, k_neighbors=k_neighbors, m_neighbors=10,
                                   random_state=seed)
        else:
            oversampler = ADASYN(sampling_strategy=strategy, n_neighbors=k_neighbors, random_state=seed)
        X_resampled, y_resampled = oversampler.fit_resample(X_train.to_numpy(np.float32), y_train)
        if verbose:
            print(f"    {kind}: {len(y_train)} -> {len(y_resampled)} rows (failures {int(y_resampled.sum())}) "
                  f"in {time.time() - started:.0f}s", flush=True)
        return pd.DataFrame(X_resampled, columns=columns), y_resampled.astype(y_train.dtype)
    return augment
