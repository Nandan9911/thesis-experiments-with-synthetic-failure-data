"""Folders and file names shared by the scripts that run against the repository released by Beikmohammadi et al. (RQ1b).

All scripts take their locations from this module, so they run from the repository root.

The authors' repository is third_party/cost-sensitive-transformer-model (experiments.common.paths), fetched at
commit 4db444a3958f0a5e95536e29b56206c497bb6d3d by third_party/fetch_third_party_code.sh. The environment variable
APS_AUTHORS_2026_REPOSITORY can name another folder instead, for example a per-run working copy that holds
TrainingCode.py and the six input tables: the authors' training code writes its CSV/ and Model/ folders into the
folder it runs in, so separate runs on one machine need separate copies.
"""
from __future__ import annotations

import os
from pathlib import Path

from experiments.common.paths import AUTHORS_2026_REPOSITORY_DIR, AUTHORS_CODE_RESULTS_DIR, CACHE_DIR

AUTHORS_REPOSITORY = Path(os.environ.get("APS_AUTHORS_2026_REPOSITORY") or AUTHORS_2026_REPOSITORY_DIR)

# Intermediate tables (imputed, resampled, labels), which can be deleted at any time: the exact regeneration of the
# authors' inputs (prepare_authors_inputs.py) at the top level, the earlier five-round approximation
# (impute_five_rounds.py, resample_five_round_table.py) in a subfolder.
WORK_DIR = CACHE_DIR / "authors_code_work"
FIVE_ROUND_WORK_DIR = WORK_DIR / "five_round_tables"

# outputs (the layout of results/rq1_transformer_2026_reproduction/authors_code/)
LOGS_DIR = AUTHORS_CODE_RESULTS_DIR / "logs"
SHIPPED_MODELS_FIVE_ROUND_TABLES_DIR = AUTHORS_CODE_RESULTS_DIR / "shipped_models_five_round_tables"
SHIPPED_MODELS_EXACT_INPUTS_DIR = AUTHORS_CODE_RESULTS_DIR / "shipped_models_exact_inputs"
RETRAINING_RUNS_DIR = AUTHORS_CODE_RESULTS_DIR / "retraining_runs"
# Boolean masks over the training and test negatives, saved as a dict with the keys
#   removed  (59,000 training negatives) removed by RENN when the published training table was built
#   hard_tr  (59,000 training negatives) hard by the one-shot rule
#   hard_te  (15,625 test negatives) hard by the same rule
# These are the groups computed, and the rule defined, in hard_negative_check_shipped_models.py.
HARD_NEGATIVE_MASKS_FILE = SHIPPED_MODELS_EXACT_INPUTS_DIR / "hard_negative_masks.npy"

# the six input tables of the authors' TrainingCode.py, kept in AUTHORS_REPOSITORY under the authors' names
TRAINING_TABLE_SCALED = "aps_failure_train_valid_set_cut_imputed_scaled.csv"
TRAINING_LABELS = "aps_failure_train_valid_set_label.csv"
RESAMPLED_TABLE_SCALED = "aps_failure_train_valid_set_cut_imputed_resampled_scaled.csv"
RESAMPLED_LABELS = "aps_failure_train_valid_set_label_resampled.csv"
TEST_TABLE_SCALED = "aps_failure_test_set_cut_imputed_scaled.csv"
TEST_LABELS = "aps_failure_test_set_label.csv"


def shipped_model_folders() -> list[Path]:
    """The five Keras models saved in the authors' repository (Model/...run1 to ...run5), in run order."""
    return sorted(AUTHORS_REPOSITORY.glob("Model/*run[1-5]"))
