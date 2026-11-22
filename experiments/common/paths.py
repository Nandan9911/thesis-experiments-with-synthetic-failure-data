"""Where everything lives on disk.

All scripts take their input and output locations from here, so the repository can be moved or cloned anywhere
without editing paths. The layout of results/ mirrors the layout of experiments/: one folder per research question,
a folder for the classifier arms that RQ1 and RQ2 share, and two folders for the work done before the arms (the
hyperparameter search and the preliminary checks).

The data files (data/) and the third-party code (third_party/) are not part of the repository; see the README files
in those folders for how to obtain them.
"""
from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# inputs
DATA_DIR = REPOSITORY_ROOT / "data"
TRAINING_CSV = DATA_DIR / "aps_failure_training_set.csv"
TEST_CSV = DATA_DIR / "aps_failure_test_set.csv"
THIRD_PARTY_DIR = REPOSITORY_ROOT / "third_party"
AUTHORS_2026_REPOSITORY_DIR = THIRD_PARTY_DIR / "cost-sensitive-transformer-model"  # code of Beikmohammadi et al.
CTAB_GAN_PLUS_REPOSITORY_DIR = THIRD_PARTY_DIR / "CTAB-GAN-Plus"                     # one of the RQ2 generators

# intermediate files that can be deleted at any time (imputed tables and similar)
CACHE_DIR = REPOSITORY_ROOT / "cache"

# outputs
RESULTS_DIR = REPOSITORY_ROOT / "results"
DATA_AUDIT_FILE = RESULTS_DIR / "data_audit.json"

HYPERPARAMETER_SEARCH_DIR = RESULTS_DIR / "hyperparameter_search"
PRELIMINARY_CHECKS_DIR = RESULTS_DIR / "preliminary_checks"

CLASSIFIER_ARMS_DIR = RESULTS_DIR / "classifier_arms"
ARM_RECORDS_FILE = CLASSIFIER_ARMS_DIR / "arm_records.jsonl"
ARM_PROBABILITIES_DIR = CLASSIFIER_ARMS_DIR / "predicted_probabilities"

CGAN_2019_RESULTS_DIR = RESULTS_DIR / "rq1_cgan_2019_reproduction"
TRANSFORMER_2026_RESULTS_DIR = RESULTS_DIR / "rq1_transformer_2026_reproduction"
PYTORCH_PORT_RESULTS_DIR = TRANSFORMER_2026_RESULTS_DIR / "pytorch_port"
AUTHORS_CODE_RESULTS_DIR = TRANSFORMER_2026_RESULTS_DIR / "authors_code"
SURVEY_RESULTS_DIR = RESULTS_DIR / "rq2_survey_of_methods"
COMPLETION_AND_COLD_START_RESULTS_DIR = RESULTS_DIR / "rq3_completion_and_cold_start"

# written by analysis/ from the files above
SUMMARY_TABLES_DIR = RESULTS_DIR / "summary_tables"
FIGURES_DIR = RESULTS_DIR / "figures"


def arm_probability_file(arm: str, seed: int, kind: str) -> Path:
    """Path of a saved probability vector of one classifier arm and seed.

    kind is one of
      "out_of_fold"      out-of-fold probabilities for the 60,000 training rows,
      "test"             probabilities for the 16,000 test rows (average of the five fold models),
      "test_full_refit"  probabilities for the test rows from one model refitted on all training rows.
    """
    if kind not in ("out_of_fold", "test", "test_full_refit"):
        raise ValueError(f"unknown probability kind: {kind}")
    return ARM_PROBABILITIES_DIR / arm / f"seed{seed}_{kind}.npy"
