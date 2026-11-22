"""The first, approximate regeneration of the preprocessing pipeline of Beikmohammadi et al.: the imputer capped at five
rounds (RQ1b).

The authors' repository names the intermediate tables of their DataPreprocessingCode.py but does not contain them,
so re-scoring the released models first requires regenerating them. This script was the first attempt, within the
compute available at the time: it drops the attributes with 10 % or more missing values in the training file (28 of
170) and fills the remaining missing values with a BayesianRidge IterativeImputer fitted on all 60,000 training rows
(chained equations: each attribute is in turn regressed on the others, round after round), with the published
settings except that it stops after five rounds instead of running up to 100. It took about 15 minutes on two
cores. The five-round tables let the released models be scored (evaluate_shipped_models.py --tables five_round),
and on them all five models missed 358 to 369 of the 375 test failures. prepare_authors_inputs.py later
regenerated the inputs exactly, and run_scaler_attribution.sh shows what each deviation contributes.

    python -m experiments.rq1_transformer_2026_reproduction.authors_code.impute_five_rounds

Reads the raw training and test files from the authors' repository (their copies, without the licence header).
Writes train_imputed.csv, test_imputed.csv, train_label.csv and test_label.csv to
cache/authors_code_work/five_round_tables/. The console output, with the imputer's progress, is
results/rq1_transformer_2026_reproduction/authors_code/logs/impute_five_rounds.log.
"""
import importlib
import random
import time

import numpy as np
import pandas as pd

from experiments.rq1_transformer_2026_reproduction.authors_code.file_locations import (AUTHORS_REPOSITORY,
                                                                                    FIVE_ROUND_WORK_DIR)

SEED = 0


def main():
    importlib.import_module("sklearn.experimental.enable_iterative_imputer")   # makes IterativeImputer importable
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge
    random.seed(SEED)
    np.random.seed(SEED)
    training = pd.read_csv(AUTHORS_REPOSITORY / "aps_failure_training_set.csv", na_values="na")
    test = pd.read_csv(AUTHORS_REPOSITORY / "aps_failure_test_set.csv", na_values="na")
    for table in (training, test):
        table["class"] = table["class"].map({"neg": 0, "pos": 1})
    X_train = training.drop("class", axis=1)
    y_train = training["class"].values
    X_test = test.drop("class", axis=1)
    y_test = test["class"].values
    missing_fraction = X_train.isna().mean()
    dropped = missing_fraction.loc[missing_fraction >= 0.1].index.to_list()
    print("attributes dropped", len(dropped), "remaining", X_train.shape[1] - len(dropped), flush=True)
    print("attributes >80% missing", (missing_fraction > 0.8).sum(),
          "attributes >10% missing", (missing_fraction > 0.1).sum(), flush=True)
    X_train = X_train.drop(columns=dropped)
    X_test = X_test.drop(columns=dropped)
    print("max abs value in train", np.nanmax(np.abs(X_train.values)), flush=True)
    started = time.time()
    imputer = IterativeImputer(estimator=BayesianRidge(), sample_posterior=False, max_iter=5, tol=0.001,
                               initial_strategy="mean", imputation_order="ascending", skip_complete=False, verbose=2,
                               random_state=SEED)
    imputer.fit(X_train)
    print("imputer fit secs", time.time() - started, "n_iter_", imputer.n_iter_, flush=True)
    X_train_imputed = imputer.transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    FIVE_ROUND_WORK_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X_train_imputed, columns=X_train.columns).to_csv(FIVE_ROUND_WORK_DIR / "train_imputed.csv",
                                                                  index=False)
    pd.DataFrame(X_test_imputed, columns=X_train.columns).to_csv(FIVE_ROUND_WORK_DIR / "test_imputed.csv",
                                                                 index=False)
    pd.DataFrame({"class": y_train}).to_csv(FIVE_ROUND_WORK_DIR / "train_label.csv", index=False)
    pd.DataFrame({"class": y_test}).to_csv(FIVE_ROUND_WORK_DIR / "test_label.csv", index=False)
    print("done", time.time() - started, flush=True)


if __name__ == "__main__":
    main()
