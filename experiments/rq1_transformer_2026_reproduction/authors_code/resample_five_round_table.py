"""SVM-SMOTE and RENN on the five-round imputed training table, as in the preprocessing pipeline of Beikmohammadi et al.
(RQ1b).

The published pipeline fits its min-max scaler on the resampled training table. To score the released models on the
five-round tables (impute_five_rounds.py) with the published scaler source, those tables are resampled here with the
authors' settings: SVM-SMOTE (sampling_strategy 0.5, k_neighbors 5, m_neighbors 10, out_step 0.5), which adds
synthetic failure rows near the class boundary until there is one failure for every two healthy rows, then repeated
edited nearest neighbours (RENN; majority class, 3 neighbours, kind_sel "all", up to 100 repetitions), which removes
healthy rows close to failure rows. Both steps are explained in preprocessing_2026.py (steps 3 and 4). This script
is step (ii) of the scaler attribution (run_scaler_attribution.sh) and takes several minutes, most of it in RENN.

    python -m experiments.rq1_transformer_2026_reproduction.authors_code.resample_five_round_table

Reads train_imputed.csv and train_label.csv from cache/authors_code_work/five_round_tables/ and writes
train_resampled.csv and train_resampled_label.csv next to them.
"""
import random
import time
from collections import Counter

import numpy as np
import pandas as pd

from experiments.rq1_transformer_2026_reproduction.authors_code.file_locations import FIVE_ROUND_WORK_DIR

SEED = 0


def main():
    from imblearn.over_sampling import SVMSMOTE
    from imblearn.under_sampling import RepeatedEditedNearestNeighbours
    random.seed(SEED)
    np.random.seed(SEED)
    X = pd.read_csv(FIVE_ROUND_WORK_DIR / "train_imputed.csv")
    y = pd.read_csv(FIVE_ROUND_WORK_DIR / "train_label.csv")["class"].values
    started = time.time()
    oversampler = SVMSMOTE(sampling_strategy=0.5, random_state=SEED, k_neighbors=5, m_neighbors=10, svm_estimator=None,
                           out_step=0.5)
    X_oversampled, y_oversampled = oversampler.fit_resample(X.values, y)
    print("after SVM-SMOTE", Counter(y_oversampled), "secs", round(time.time() - started), flush=True)
    started = time.time()
    cleaner = RepeatedEditedNearestNeighbours(sampling_strategy="majority", n_neighbors=3, max_iter=100, kind_sel="all")
    X_resampled, y_resampled = cleaner.fit_resample(X_oversampled, y_oversampled)
    print("after RENN", Counter(y_resampled), "secs", round(time.time() - started), flush=True)
    pd.DataFrame(X_resampled, columns=X.columns).to_csv(FIVE_ROUND_WORK_DIR / "train_resampled.csv", index=False)
    pd.DataFrame({"class": y_resampled}).to_csv(FIVE_ROUND_WORK_DIR / "train_resampled_label.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
