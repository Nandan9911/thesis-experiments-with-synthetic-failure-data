"""Regenerate, exactly, the six input tables of TrainingCode.py released by Beikmohammadi et al. from the raw APS files
(RQ1b).

The authors' DataPreprocessingCode.py writes the tables that TrainingCode.py reads, but the repository does not
contain them, and the script itself no longer runs (it imports a TensorFlow wrapper module that has been removed
from TensorFlow). This script repeats its steps with the published settings, in the published order, without
importing it:

  1. drop the attributes with 10 % or more missing values in the training file (28 of 170; 142 remain);
  2. fill the missing values with an IterativeImputer around BayesianRidge (tolerance 1e-3, mean start, ascending
     order, seed 0) fitted on all 60,000 training rows, with up to --imputer-rounds rounds (100 as published). On
     this table the stopping rule is not met within 100 rounds, and the 100 rounds took about four hours on one core;
  3. SVM-SMOTE (sampling_strategy 0.5, k_neighbors 5, m_neighbors 10, out_step 0.5, seed 0) on the imputed training
     table, which adds synthetic failure rows near the class boundary until there is one failure for every two
     healthy rows;
  4. repeated edited nearest neighbours (RENN; majority class, 3 neighbours, kind_sel "all", up to 100
     repetitions), which removes healthy rows close to failure rows (steps 3 and 4 are explained in
     preprocessing_2026.py);
  5. a min-max scaler fitted on the resampled table and applied to the imputed training table, the resampled table
     and the imputed test table.

With 100 rounds the five released models score exactly the published mean cost of 3,440 on these tables
(evaluate_shipped_models.py --tables exact train_resampled.csv), and the retraining runs use them as inputs.

    python -m experiments.rq1_transformer_2026_reproduction.authors_code.prepare_authors_inputs --imputer-rounds 100

Reads aps_failure_training_set.csv and aps_failure_test_set.csv from --data-dir (data/ by default; the 20-line
licence header of the UCI copies is skipped, so the authors' copies without it work too). Writes, in the authors'
file names, into --authors-repository (where TrainingCode.py reads them):
  aps_failure_train_valid_set_cut_imputed_scaled.csv            and aps_failure_train_valid_set_label.csv
  aps_failure_train_valid_set_cut_imputed_resampled_scaled.csv  and aps_failure_train_valid_set_label_resampled.csv
  aps_failure_test_set_cut_imputed_scaled.csv                   and aps_failure_test_set_label.csv
and, into --work-dir, the unscaled tables that evaluate_shipped_models.py --tables exact uses: train_imputed.csv,
test_imputed.csv, train_resampled.csv and the label files train_label.csv, test_label.csv, train_resampled_label.csv.
The console output of the reported run is results/rq1_transformer_2026_reproduction/authors_code/logs/
prepare_authors_inputs.log (a copy is kept in shipped_models_exact_inputs/).
"""
import argparse
import importlib
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge

from experiments.common.paths import DATA_DIR
from experiments.rq1_transformer_2026_reproduction.authors_code.file_locations import AUTHORS_REPOSITORY, WORK_DIR

SEED = 0


class LeanBayesianRidge(BayesianRidge):
    """BayesianRidge without the stored posterior covariance. IterativeImputer keeps one fitted estimator per
    attribute and round (14,200 for 100 rounds); sigma_ is a 141 x 141 matrix per estimator (~160 KB) that
    transform() never reads when sample_posterior=False, and keeping it needs ~8 GB of memory. The imputed
    values are identical."""

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight)
        self.sigma_ = None
        return self


def read_raw_file(path):
    """A raw APS file as a DataFrame, skipping the licence header of the UCI copies if there is one."""
    with open(path) as handle:
        lines_above_header = next((number for number, line in enumerate(handle) if line.startswith("class")), 0)
    return pd.read_csv(path, skiprows=lines_above_header, na_values="na", low_memory=False)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--imputer-rounds", type=int, default=100, help="maximum rounds of the iterative imputer")
    parser.add_argument("--authors-repository", default=str(AUTHORS_REPOSITORY),
                        help="folder that receives the six input tables of TrainingCode.py (default: the authors' "
                             "repository, or the folder named by APS_AUTHORS_2026_REPOSITORY)")
    parser.add_argument("--work-dir", default=str(WORK_DIR), help="folder for the unscaled intermediate tables")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="folder with the two raw data files")
    return parser.parse_args()


def main():
    args = parse_arguments()
    random.seed(SEED)
    np.random.seed(SEED)
    importlib.import_module("sklearn.experimental.enable_iterative_imputer")   # makes IterativeImputer importable
    from sklearn.impute import IterativeImputer
    from sklearn.preprocessing import MinMaxScaler
    from imblearn.over_sampling import SVMSMOTE
    from imblearn.under_sampling import RepeatedEditedNearestNeighbours

    repository = Path(args.authors_repository)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    training = read_raw_file(data_dir / "aps_failure_training_set.csv")
    test = read_raw_file(data_dir / "aps_failure_test_set.csv")
    for table in (training, test):
        table["class"] = table["class"].map({"neg": 0, "pos": 1})
    X_train = training.drop("class", axis=1)
    y_train = training["class"].values
    X_test = test.drop("class", axis=1)
    y_test = test["class"].values
    missing_fraction = X_train.isna().mean()
    dropped = missing_fraction.loc[missing_fraction >= 0.1].index.to_list()
    X_train = X_train.drop(columns=dropped)
    X_test = X_test.drop(columns=dropped)
    print(f"dropped {len(dropped)} attributes, {X_train.shape[1]} remain", flush=True)

    started = time.time()
    imputer = IterativeImputer(estimator=LeanBayesianRidge(), sample_posterior=False, max_iter=args.imputer_rounds,
                               tol=0.001, n_nearest_features=None, initial_strategy="mean",
                               imputation_order="ascending", skip_complete=False, verbose=1, random_state=SEED)
    imputer.fit(X_train)
    X_train_imputed = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_train.columns)
    print(f"imputer: {imputer.n_iter_} rounds in {time.time() - started:.0f}s; negative cells in "
          f"{(X_train_imputed.min() < 0).sum()} attributes", flush=True)
    X_train_imputed.to_csv(work_dir / "train_imputed.csv", index=False)
    X_test_imputed.to_csv(work_dir / "test_imputed.csv", index=False)
    pd.DataFrame({"class": y_train}).to_csv(work_dir / "train_label.csv", index=False)
    pd.DataFrame({"class": y_test}).to_csv(work_dir / "test_label.csv", index=False)

    started = time.time()
    oversampler = SVMSMOTE(sampling_strategy=0.5, random_state=SEED, k_neighbors=5, m_neighbors=10,
                           svm_estimator=None, out_step=0.5)
    X_oversampled, y_oversampled = oversampler.fit_resample(X_train_imputed.values, y_train)
    print(f"SVM-SMOTE: {len(y_oversampled)} rows ({int(y_oversampled.sum())} positive) in {time.time() - started:.0f}s",
          flush=True)
    started = time.time()
    cleaner = RepeatedEditedNearestNeighbours(sampling_strategy="majority", n_neighbors=3, max_iter=100,
                                              kind_sel="all")
    X_resampled, y_resampled = cleaner.fit_resample(X_oversampled, y_oversampled)
    print(f"RENN: {len(y_resampled)} rows ({int(y_resampled.sum())} positive) in {time.time() - started:.0f}s",
          flush=True)
    X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    X_resampled.to_csv(work_dir / "train_resampled.csv", index=False)
    pd.DataFrame({"class": y_resampled}).to_csv(work_dir / "train_resampled_label.csv", index=False)

    scaler = MinMaxScaler().fit(X_resampled.values)    # fitted on the resampled table, as published

    def scaled(table):
        return pd.DataFrame(scaler.transform(table.values), columns=X_train.columns)

    scaled(X_train_imputed).to_csv(repository / "aps_failure_train_valid_set_cut_imputed_scaled.csv", index=False)
    pd.DataFrame({"class": y_train}).to_csv(repository / "aps_failure_train_valid_set_label.csv", index=False)
    scaled(X_resampled).to_csv(repository / "aps_failure_train_valid_set_cut_imputed_resampled_scaled.csv",
                               index=False)
    pd.DataFrame({"class": y_resampled}).to_csv(repository / "aps_failure_train_valid_set_label_resampled.csv",
                                                index=False)
    scaled(X_test_imputed).to_csv(repository / "aps_failure_test_set_cut_imputed_scaled.csv", index=False)
    pd.DataFrame({"class": y_test}).to_csv(repository / "aps_failure_test_set_label.csv", index=False)
    print("wrote the six input files of TrainingCode.py into", repository, flush=True)


if __name__ == "__main__":
    main()
