"""Factories for every classifier of the thesis, the tuned LightGBM settings, and the log-odds averaging ensemble.

Part of the common evaluation protocol (experiments/common). A factory takes a seed and optional keyword overrides and
returns an unfitted estimator with fit and predict_proba. The classifier arms of RQ1 and RQ2 (experiment_arms.py), the
synthetic-row count sweep and the tree models on the preprocessing pipeline of Beikmohammadi et al. (RQ1), and the
completion and cold-start experiments (RQ3) all build their classifiers through CLASSIFIER_FACTORIES:

    lightgbm, xgboost, catboost    gradient-boosted decision trees (each tree fitted to the errors of the trees
                                   before it) that handle missing values themselves
    random_forest, extra_trees     scikit-learn's random forest and extremely randomised trees
    histogram_gradient_boosting    scikit-learn's histogram-based gradient boosting
    sklearn_gradient_boosting      scikit-learn's classic GradientBoostingClassifier, which needs imputed attributes
    tabicl                         TabICLv2, a pretrained in-context model, as a bag of contexts (RQ2)
    ensemble                       the log-odds average of LightGBM, XGBoost and CatBoost (RQ2)

The settings written in the factories are fixed values, partly different from the libraries' own defaults. Only
LightGBM was tuned (hyperparameter_search.py, seed 11); load_tuned_hyperparameters returns the best settings of that
search, and every arm that trains a LightGBM uses them unchanged, so that differences between arms cannot come from
differently tuned learners.

The estimators that take a thread count (LightGBM, XGBoost, CatBoost and the two forests) use CPU_THREADS_PER_MODEL
threads: the value of the environment variable APS_CPU_THREADS_PER_MODEL when a queue runner shares one machine
between several jobs, otherwise every core of the machine.
"""
from __future__ import annotations

import json
import os

import numpy as np

from experiments.common.paths import HYPERPARAMETER_SEARCH_DIR

CPU_THREADS_PER_MODEL = max(1, int(os.environ.get("APS_CPU_THREADS_PER_MODEL") or (os.cpu_count() or 1)))


def load_tuned_hyperparameters(classifier: str, seed: int = 11) -> dict:
    """Best hyperparameters found by hyperparameter_search.py for this classifier and seed.

    Reads results/hyperparameter_search/<classifier>_seed<seed>_best_parameters.json. If no search was run for the
    classifier (for the thesis, only LightGBM on seed 11 was searched), the result is an empty dict, i.e. the factory's
    own settings.
    """
    best_parameters_file = HYPERPARAMETER_SEARCH_DIR / f"{classifier}_seed{seed}_best_parameters.json"
    if not best_parameters_file.exists():
        return {}
    with open(best_parameters_file) as handle:
        return json.load(handle)["best_params"]


def make_lightgbm(seed: int, **overrides):
    """LightGBM: 600 trees of up to 31 leaves (at least 20 rows per leaf) at learning rate 0.03, each tree fitted on
    80 % of the rows and 60 % of the attributes, with an L2 penalty of 1. The arms replace these values by the tuned
    ones; subsample_freq=1 stays and is what makes LightGBM draw a new row sample for every tree (without it the
    subsample setting has no effect). At every split LightGBM learns on which side the rows with a missing value go."""
    import lightgbm as lgb
    parameters = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0,
                      n_jobs=CPU_THREADS_PER_MODEL, random_state=seed, verbose=-1)
    parameters.update(overrides)
    return lgb.LGBMClassifier(**parameters)


def make_xgboost(seed: int, **overrides):
    """XGBoost with settings comparable to the LightGBM factory: 600 trees of depth 6 at learning rate 0.03, 80 % of
    the rows and 60 % of the attributes per tree, histogram-based split finding. Like LightGBM it learns a default
    direction for missing values at every split. No search was run for XGBoost."""
    import xgboost as xgb
    parameters = dict(n_estimators=600, learning_rate=0.03, max_depth=6, min_child_weight=1,
                      subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0, tree_method="hist",
                      n_jobs=CPU_THREADS_PER_MODEL, random_state=seed, eval_metric="logloss")
    parameters.update(overrides)
    return xgb.XGBClassifier(**parameters)


def make_catboost(seed: int, **overrides):
    """CatBoost: 800 symmetric trees of depth 6 at learning rate 0.05, L2 penalty 3, silent and without writing
    training files. By default CatBoost treats a missing value as smaller than every observed value."""
    from catboost import CatBoostClassifier
    parameters = dict(iterations=800, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
                      random_seed=seed, verbose=0, thread_count=CPU_THREADS_PER_MODEL, allow_writing_files=False)
    parameters.update(overrides)
    return CatBoostClassifier(**parameters)


def make_random_forest(seed: int, **overrides):
    """Random forest of 500 fully grown trees, each split choosing among sqrt(number of attributes) candidates. In
    the arms it is given median-imputed attributes."""
    from sklearn.ensemble import RandomForestClassifier
    parameters = dict(n_estimators=500, max_features="sqrt", min_samples_leaf=1, n_jobs=CPU_THREADS_PER_MODEL,
                      random_state=seed)
    parameters.update(overrides)
    return RandomForestClassifier(**parameters)


def make_extra_trees(seed: int, **overrides):
    """Extremely randomised trees: like the random forest, but each candidate split point is drawn at random instead
    of optimised, and every tree sees all rows (no bootstrap sample)."""
    from sklearn.ensemble import ExtraTreesClassifier
    parameters = dict(n_estimators=500, max_features="sqrt", min_samples_leaf=1, n_jobs=CPU_THREADS_PER_MODEL,
                      random_state=seed)
    parameters.update(overrides)
    return ExtraTreesClassifier(**parameters)


def make_histogram_gradient_boosting(seed: int, **overrides):
    """scikit-learn's HistGradientBoostingClassifier (missing values handled natively): at most 600 trees with up to
    31 leaves at learning rate 0.05, L2 penalty 1. scikit-learn's automatic early stopping is left on; with more than
    10,000 training rows it holds out 10 % of them and stops adding trees once the loss on them stops improving."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    parameters = dict(max_iter=600, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                      random_state=seed)
    parameters.update(overrides)
    return HistGradientBoostingClassifier(**parameters)


def make_sklearn_gradient_boosting(seed: int, **overrides):
    """scikit-learn's GradientBoostingClassifier, 300 trees of depth 3 at learning rate 0.1: the most literal "GBM" an
    author working in Python in 2019 would have used, which is why the reproduction of the study of Ranasinghe et al.
    includes it. It rejects missing values, so it is always paired with median imputation."""
    from sklearn.ensemble import GradientBoostingClassifier
    parameters = dict(n_estimators=300, learning_rate=0.1, max_depth=3, subsample=1.0, random_state=seed)
    parameters.update(overrides)
    return GradientBoostingClassifier(**parameters)


CLASSIFIER_FACTORIES = {"lightgbm": make_lightgbm, "xgboost": make_xgboost, "catboost": make_catboost,
                        "random_forest": make_random_forest, "extra_trees": make_extra_trees,
                        "histogram_gradient_boosting": make_histogram_gradient_boosting,
                        "sklearn_gradient_boosting": make_sklearn_gradient_boosting}


def make_bagged_tabicl(seed: int, **overrides):
    """TabICLv2 as a bag of contexts (see rq2_survey_of_methods/tabicl_classifier.py). The overrides n_bags, n_neg
    (healthy rows per context) and n_estimators are passed to bagged_tabicl_factory under these names, which are the
    names stored in the arm options."""
    from experiments.rq2_survey_of_methods.tabicl_classifier import bagged_tabicl_factory
    return bagged_tabicl_factory(**overrides)(seed)


CLASSIFIER_FACTORIES["tabicl"] = make_bagged_tabicl


# --------------------------------------------------------------------------------------------- heterogeneous ensemble
class MissingnessIndicatorFeatures:
    """Appends one 0/1 missingness indicator per attribute (named "miss_<attribute>") and keeps the attributes, with
    their missing values, as they are.

    The pattern of missing values is informative on APS: failure rows have about twice the missing rate of healthy
    rows. Learners that handle missing values natively can use it only through their split rules; the indicators
    make it available as features of its own. The transform is deterministic, so there is nothing to fit, and the
    same transform is applied to every table the ensemble sees. Unlike median_imputation_preprocessor with
    add_missing_indicators, it adds an indicator for every attribute and does not impute.
    """
    def transform(self, X):
        import pandas as pd
        indicators = X.isna().astype(np.int8)
        indicators.columns = [f"miss_{column}" for column in X.columns]
        return pd.concat([X, indicators], axis=1)


class LogOddsAveragingEnsemble:
    """A heterogeneous ensemble that averages its members' predictions in log-odds.

    The members (by default the tuned LightGBM and XGBoost and CatBoost with their factory settings) are fitted on
    the same rows. Each sees the attributes with their missing values, which all three handle natively, plus one
    missingness indicator per attribute if add_missingness_indicators is set. A TabICL member, if listed, sees the
    attributes only (its own preprocessing deals with missing values). Member i is given the seed seed + i.

    The ensemble probability is the logistic function of the mean of the members' log-odds log(p / (1 - p)), which
    is the same as taking the geometric mean of their odds. An arithmetic mean of probabilities would let the member
    with the largest probability decide each row: the cost-minimising thresholds of these models lie around 1e-5 to
    1e-4, and there one member's 1e-3 swamps another's 1e-5. In log-odds a factor of ten counts the same at every
    scale, so each member moves the ranking of the low-probability rows, where the threshold lies, by the same
    amount. Probabilities are clipped to [1e-7, 1 - 1e-7] first, so that a member that outputs exactly 0 or 1 cannot
    give an infinite log-odds. The decision threshold is then chosen on the out-of-fold ensemble output, as for
    every other arm.
    """
    def __init__(self, seed: int, lightgbm_parameters: dict, members=("lightgbm", "xgboost", "catboost"),
                 add_missingness_indicators=True, verbose=True, tabicl_parameters=None):
        self.seed, self.lightgbm_parameters, self.members = seed, lightgbm_parameters, members
        self.add_missingness_indicators, self.verbose = add_missingness_indicators, verbose
        self.tabicl_parameters = tabicl_parameters or {}
        self.indicator_features_ = MissingnessIndicatorFeatures() if add_missingness_indicators else None

    def _with_indicators(self, X):
        return self.indicator_features_.transform(X) if self.indicator_features_ is not None else X

    def fit(self, X, y):
        import time
        X_with_indicators = self._with_indicators(X)
        self.fitted_members_ = []
        self.member_sees_attributes_only_ = []
        for i, member in enumerate(self.members):
            started = time.time()
            if member == "lightgbm":
                estimator = make_lightgbm(self.seed + i, **self.lightgbm_parameters)
                X_member = X_with_indicators
            elif member == "xgboost":
                estimator = make_xgboost(self.seed + i)
                X_member = X_with_indicators
            elif member == "catboost":
                estimator = make_catboost(self.seed + i)
                X_member = X_with_indicators
            elif member == "tabicl":
                estimator = _TabICLEnsembleMember(self.seed + i, **self.tabicl_parameters)
                X_member = X
            else:
                raise ValueError(member)
            estimator.fit(X_member, y)
            self.fitted_members_.append(estimator)
            self.member_sees_attributes_only_.append(member == "tabicl")
            if self.verbose:
                print(f"      ensemble member {member}: {time.time() - started:.0f}s", flush=True)
        return self

    def predict_proba(self, X):
        X_with_indicators = self._with_indicators(X)
        clip = 1e-7
        member_log_odds = []
        for estimator, attributes_only in zip(self.fitted_members_, self.member_sees_attributes_only_):
            probabilities = np.clip(estimator.predict_proba(X if attributes_only else X_with_indicators)[:, 1],
                                    clip, 1 - clip)
            member_log_odds.append(np.log(probabilities / (1 - probabilities)))
        mean_log_odds = np.mean(member_log_odds, axis=0)
        ensemble_probabilities = 1 / (1 + np.exp(-mean_log_odds))
        return np.column_stack([1 - ensemble_probabilities, ensemble_probabilities])


class _TabICLEnsembleMember:
    """TabICLv2 as an ensemble member, bagged like the stand-alone arm: n_bags contexts (three in the arms), each with
    all failures of the training rows and n_neg randomly drawn healthy rows (n_neg is the number of healthy rows per
    context), and probabilities averaged over the bags (see rq2_survey_of_methods/tabicl_classifier.py). TabICLv2 is
    a pretrained model that classifies by in-context learning: it reads the labelled context rows together with the
    rows to classify and is not trained on this data. The keyword names are those of the arm options."""
    def __init__(self, seed, n_bags=3, n_neg=10000, n_estimators=2):
        from experiments.rq2_survey_of_methods.tabicl_classifier import bagged_tabicl_factory
        self.bagged_classifier = bagged_tabicl_factory(n_bags=n_bags, n_neg=n_neg, n_estimators=n_estimators)(seed)

    def fit(self, X, y):
        self.bagged_classifier.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.bagged_classifier.predict_proba(X)


def make_ensemble(seed: int, **overrides):
    """The log-odds averaging ensemble. Three overrides, under the names stored in the arm options, configure the
    ensemble itself:

    members        names of the member classifiers (default lightgbm, xgboost, catboost)
    use_mask       whether one missingness indicator per attribute is added (default yes); the stored name refers to
                   the 0/1 mask of missing cells, and it means add_missingness_indicators of LogOddsAveragingEnsemble
    tabicl_params  keyword arguments of a TabICL member (n_bags, n_neg = healthy rows per context, n_estimators)

    All other overrides are the LightGBM member's hyperparameters (the tuned set in the arms)."""
    members = tuple(overrides.pop("members", ("lightgbm", "xgboost", "catboost")))
    add_missingness_indicators = overrides.pop("use_mask", True)
    tabicl_parameters = overrides.pop("tabicl_params", None)
    return LogOddsAveragingEnsemble(seed, lightgbm_parameters=overrides, members=members,
                                    add_missingness_indicators=add_missingness_indicators,
                                    tabicl_parameters=tabicl_parameters)


CLASSIFIER_FACTORIES["ensemble"] = make_ensemble
