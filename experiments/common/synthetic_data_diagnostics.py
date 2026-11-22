"""Label-free statistics that compare synthetic failure rows with real ones: detectability, closeness, copying.

Part of the common evaluation protocol (experiments/common). The functions are used by
rq2_survey_of_methods/run_synthetic_data_diagnostics.py, which applies them to every generator of RQ1 and RQ2, and by
rq3_completion_and_cold_start/run_completion_memorisation_audit.py, which applies them to completed failure records.
They describe the generated rows only; the test set is not used.

Four kinds of statistic, each explained in its function:

* detectability AUC: how easily a classifier tells synthetic rows from real ones (a classifier two-sample test);
* distance to closest record (DCR): how close the synthetic rows lie to the real rows, measured against how close
  real rows lie to each other;
* memorisation statistics: whether the rows reveal which real rows the generator was fitted on;
* the mean column-wise 1-Wasserstein distance: how well each attribute's marginal distribution is reproduced.

All distances are Euclidean, computed after min-max scaling every attribute with the minimum and range of the real
rows the generator was fitted on, and with missing values set to 0, i.e. to the real rows' minimum. The runners also
compute the same statistics for real held-out failures (what an ideal generator would attain) and for copies of real
rows (pure memorisation), and the values are read against these two references.
"""
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors


def detectability_auc(real: pd.DataFrame, synthetic: pd.DataFrame, seed=11, fill=None):
    """Out-of-fold ROC AUC of a LightGBM trained to tell the synthetic rows `synthetic` (label 1) from the real rows
    `real` (label 0).

    The two tables are stacked, split by five-fold stratified cross-validation (shuffled with `seed`), and a small
    LightGBM (300 trees of 15 leaves, learning rate 0.05) scores every row from the fold in which it was held out.
    An AUC of 0.5 means that the classifier cannot tell the two apart; 1.0 means that it separates them perfectly. A
    LightGBM is used because the failure classifiers of the thesis are gradient-boosted trees as well: rows that such
    a model can single out are rows that a failure classifier can learn to treat differently from real failures.

    With the default fill=None, missing values stay missing and the classifier can use the missingness pattern, as
    the failure classifiers can. With fill given (a value or, typically, the per-attribute medians of the real
    failure rows), missing cells of both tables are filled first, so that the pattern cannot give the synthetic rows
    away and the AUC reflects the observed values only.
    """
    import lightgbm as lgb
    X = pd.concat([real, synthetic], ignore_index=True)
    y = np.r_[np.zeros(len(real)), np.ones(len(synthetic))]
    if fill is not None:
        X = X.fillna(fill)
    out_of_fold_scores = np.zeros(len(y))
    for train_index, validation_index in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, n_jobs=1, random_state=seed,
                                   verbose=-1)
        model.fit(X.iloc[train_index], y[train_index])
        out_of_fold_scores[validation_index] = model.predict_proba(X.iloc[validation_index])[:, 1]
    return float(roc_auc_score(y, out_of_fold_scores))


def _min_max_scaled(reference, other):
    """Both tables as arrays, min-max scaled with the per-attribute minimum and range of `reference` (a range of 0 is
    treated as 1) and with missing values set to 0. Values of `other` outside the reference range fall outside
    [0, 1]."""
    minimum, maximum = reference.min(), reference.max()
    value_range = (maximum - minimum).replace(0, 1)
    reference_scaled = ((reference - minimum) / value_range).fillna(0).values
    other_scaled = ((other - minimum) / value_range).fillna(0).values
    return reference_scaled, other_scaled


def distance_to_closest_record_statistics(real, synthetic):
    """Distance to closest record (DCR) of the synthetic rows `synthetic` with respect to the real rows `real`.

    For every synthetic row, the distance to its nearest real row is computed; as the yardstick, every real row's
    distance to its nearest other real row (leave-one-out: the nearest neighbour other than the row itself).
    Returns
      dcr                    median synthetic-to-real distance divided by the median real-to-real distance; about 1
                             means the synthetic rows are as far from the real rows as real rows are from each other,
                             well below 1 that they sit closer to real rows than real rows do (a sign of copying),
                             well above 1 that they lie away from the real data
      frac_closer            share of synthetic rows closer to a real row than the 5th percentile of the real-to-real
                             distances
      median_nn_real         median real-to-real (leave-one-out) nearest-neighbour distance
      median_nn_syn_to_real  median synthetic-to-real nearest-neighbour distance
    (The key names are those stored in the result files.)
    """
    real_scaled, synthetic_scaled = _min_max_scaled(real, synthetic)
    neighbours_among_real = NearestNeighbors(n_neighbors=2).fit(real_scaled)
    real_to_real = neighbours_among_real.kneighbors(real_scaled)[0][:, 1]   # [:, 0] is the row itself
    synthetic_to_real = NearestNeighbors(n_neighbors=1).fit(real_scaled).kneighbors(synthetic_scaled)[0][:, 0]
    return {"dcr": float(np.median(synthetic_to_real) / np.median(real_to_real)),
            "frac_closer": float((synthetic_to_real < np.percentile(real_to_real, 5)).mean()),
            "median_nn_real": float(np.median(real_to_real)),
            "median_nn_syn_to_real": float(np.median(synthetic_to_real))}


def memorisation_statistics(real_fit, real_heldout, synthetic):
    """Did the generator copy the rows it was fitted on? Three views, all in the min-max scaled space of the
    fitting rows `real_fit`, with real failure rows that the generator never saw (`real_heldout`) as the contrast.

    Returns
      mia_auc             a distance-based membership-inference attack: every real row, fitting or held-out, is
                          scored by its distance to the nearest synthetic row (closer = more likely to have been a
                          training row), and the AUC measures how well this score tells the fitting rows (members)
                          from the held-out rows (non-members). 0.5 means that the synthetic rows reveal nothing about
                          which real rows were used; 1.0 that they identify them perfectly.
      near_copy_frac      share of synthetic rows whose nearest fitting row is closer than 5 % of the median
                          leave-one-out nearest-neighbour distance among the fitting rows, i.e. near-duplicates of a
                          training row
      dcr_fit_vs_heldout  median distance of the synthetic rows to their nearest fitting row divided by their median
                          distance to the nearest held-out row; well below 1 means that they sit closer to what they
                          were fitted on. With more fitting rows than held-out rows, nearest neighbours are closer
                          among the fitting rows anyway, so the value is compared with the reference rows rather than
                          with 1.
    (The key names are those stored in the result files.)
    """
    fit_scaled, synthetic_scaled = _min_max_scaled(real_fit, synthetic)
    _, heldout_scaled = _min_max_scaled(real_fit, real_heldout)
    neighbours_among_synthetic = NearestNeighbors(n_neighbors=1).fit(synthetic_scaled)
    fit_to_synthetic = neighbours_among_synthetic.kneighbors(fit_scaled)[0][:, 0]
    heldout_to_synthetic = neighbours_among_synthetic.kneighbors(heldout_scaled)[0][:, 0]
    is_member = np.r_[np.ones(len(fit_to_synthetic)), np.zeros(len(heldout_to_synthetic))]
    membership_auc = float(roc_auc_score(is_member, -np.r_[fit_to_synthetic, heldout_to_synthetic]))
    fit_to_fit = NearestNeighbors(n_neighbors=2).fit(fit_scaled).kneighbors(fit_scaled)[0][:, 1]
    synthetic_to_fit = NearestNeighbors(n_neighbors=1).fit(fit_scaled).kneighbors(synthetic_scaled)[0][:, 0]
    synthetic_to_heldout = NearestNeighbors(n_neighbors=1).fit(heldout_scaled).kneighbors(synthetic_scaled)[0][:, 0]
    return {"mia_auc": membership_auc,
            "near_copy_frac": float((synthetic_to_fit < 0.05 * np.median(fit_to_fit)).mean()),
            "dcr_fit_vs_heldout": float(np.median(synthetic_to_fit) / max(np.median(synthetic_to_heldout), 1e-12))}


def mean_columnwise_wasserstein_distance(real, synthetic):
    """Mean over the attributes of the 1-Wasserstein distance between the real and the synthetic values.

    The 1-Wasserstein distance between two one-dimensional distributions is the area between their cumulative
    distribution functions (the "earth mover's distance"); computed on min-max scaled values it lies between 0
    (identical marginal distributions) and 1. Missing values enter as 0 after scaling, so an attribute whose share of
    missing values is wrong also adds to the distance. Only the marginals are compared: rows whose attributes are
    combined in an unrealistic way can still score well.
    """
    real_scaled, synthetic_scaled = _min_max_scaled(real, synthetic)
    # values of the synthetic rows outside the range of the real rows are clipped, so that a few outliers do not
    # dominate the distance
    real_scaled, synthetic_scaled = np.clip(real_scaled, 0, 1), np.clip(synthetic_scaled, 0, 1)
    return float(np.mean([wasserstein_distance(real_scaled[:, j], synthetic_scaled[:, j])
                          for j in range(real_scaled.shape[1])]))
