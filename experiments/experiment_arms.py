"""The registry of experiment arms used in RQ1 and RQ2.

An *arm* is one complete recipe for training a failure classifier:

    classifier (and its hyperparameters)  x  preprocessing  x  augmentation

where augmentation means "add synthetic failure rows to the training data" (or, for the reference arms, add
nothing). Every arm is evaluated with exactly the same protocol by `run_experiment_arm.py`: five-fold
cross-validation on the 60,000 training rows, a decision threshold chosen on the out-of-fold predictions, and one
final evaluation on the official 16,000-row test set with that frozen threshold. This module only describes the arms
and is not run by itself; the arms of RQ1 (the conditional GAN of Ranasinghe et al.) and of RQ2 (the survey of other
generators and classifiers) are listed side by side so that they share one baseline.

The synthetic rows are always generated inside the training folds: each fold fits its own generator on the failures of
its own training part, so that no validation row ever influences the rows that are added, as the common evaluation
protocol requires (experiments/common/cost_and_evaluation.py).

The names follow the pattern

    <classifier>                      no synthetic rows (a reference arm)
    <classifier>_with_<what is added> synthetic or duplicated failure rows added

so `lightgbm_with_ctgan_rows` is the tuned LightGBM trained on the real rows plus 2,000 failure rows sampled from a
CTGAN fitted on the fold's real failures. The arms that use tuned hyperparameters read them from
results/hyperparameter_search/lightgbm_seed11_best_parameters.json (see classifiers.load_tuned_hyperparameters).

Some records in results/classifier_arms/arm_records.jsonl carry arm names that are not keys of ARMS:
`<arm>_3_folds` is an arm of this registry run by run_experiment_arm.py with --folds 3 (the stored ones are
lightgbm_tuned_3_folds), and `lightgbm_on_2026_preprocessing_<variant>` is the tuned LightGBM on the preprocessing of
the study of Beikmohammadi et al., written by
rq1_transformer_2026_reproduction/run_tree_models_on_2026_preprocessing.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.common.classifiers import CLASSIFIER_FACTORIES, load_tuned_hyperparameters
from experiments.common.imputation_and_oversampling import median_imputation_preprocessor, smote_family_augmenter
from experiments.rq1_cgan_2019_reproduction.conditional_gan_2019 import conditional_gan_augmenter

# Every generator arm adds the same number of synthetic failure rows, so that the arms are compared at a matched
# count. 2,000 is the count used in the study of Ranasinghe et al.
SYNTHETIC_ROWS = 2000

ARMS = {
    # ------------------------------------------------------------------------------ reference arms, no synthetic rows
    # The baseline of the whole thesis: LightGBM with the hyperparameters found by hyperparameter_search.py
    # (seed 11), on the raw attributes with their missing values.
    "lightgbm_tuned": {"classifier": "lightgbm", "use_tuned_hyperparameters": True},
    # No tuning was run for XGBoost, so this arm uses the fixed settings in classifiers.py.
    "xgboost_fixed_settings": {"classifier": "xgboost", "use_tuned_hyperparameters": False},
    # A random forest with fixed settings (500 trees), close to the learners of the 2016 challenge, and
    # scikit-learn's gradient boosting, the most literal "GBM" a Python user of 2019 would have reached for. Both
    # run on median-imputed attributes: scikit-learn's gradient boosting rejects missing values, and the random
    # forest was given the same input.
    "random_forest_fixed_settings": {"classifier": "random_forest", "use_tuned_hyperparameters": False,
                                     "preprocessing": "median_imputation",
                                     "classifier_parameters": {"n_estimators": 500}},
    "sklearn_gradient_boosting": {"classifier": "sklearn_gradient_boosting", "use_tuned_hyperparameters": False,
                                  "preprocessing": "median_imputation"},

    # --------------------------------- RQ1: the conditional GAN of Ranasinghe et al., generator fitted inside the folds
    # A conditional GAN trains a generator network to produce rows that a second network (the discriminator)
    # cannot tell from real failure rows; as in the study of Ranasinghe et al., both are told which of two k-means
    # clusters of the failure rows a row belongs to (see rq1_cgan_2019_reproduction/).
    "lightgbm_with_conditional_gan_2019_rows": {"classifier": "lightgbm", "use_tuned_hyperparameters": True,
                                                "augmentation": "conditional_gan_2019",
                                                "synthetic_rows": SYNTHETIC_ROWS},
    "random_forest_fixed_settings_with_conditional_gan_2019_rows": {
        "classifier": "random_forest", "use_tuned_hyperparameters": False, "preprocessing": "median_imputation",
        "classifier_parameters": {"n_estimators": 500}, "augmentation": "conditional_gan_2019",
        "synthetic_rows": SYNTHETIC_ROWS},
    "sklearn_gradient_boosting_with_conditional_gan_2019_rows": {
        "classifier": "sklearn_gradient_boosting", "use_tuned_hyperparameters": False,
        "preprocessing": "median_imputation", "augmentation": "conditional_gan_2019",
        "synthetic_rows": SYNTHETIC_ROWS},
    # The cGAN with the notebook's settings: standard scaling, the notebook's loss and no missing values
    # put back into the generated rows.
    "random_forest_with_conditional_gan_2019_literal_notebook_rows": {
        "classifier": "random_forest", "use_tuned_hyperparameters": False, "preprocessing": "median_imputation",
        "augmentation": "conditional_gan_2019", "synthetic_rows": SYNTHETIC_ROWS,
        "gan_input_scaling": "standard", "use_literal_notebook_loss": True, "reapply_missingness": False},
    # The duplication control: 2,000 copies of real training failures instead of generated ones. If a generator
    # cannot beat plain duplication, its rows add nothing beyond extra weight on the failure class.
    "lightgbm_with_duplicated_real_failures": {"classifier": "lightgbm", "use_tuned_hyperparameters": True,
                                               "augmentation": "duplicated_real_failures",
                                               "synthetic_rows": SYNTHETIC_ROWS},

    # ------------------------------------------------------------------- RQ2: classical oversampling, 2,000 rows each
    # The SMOTE family interpolates between neighbouring failure rows and cannot work with missing values, so these
    # arms run on median-imputed attributes (common/imputation_and_oversampling.py describes the four methods).
    "lightgbm_with_smote_rows": {"classifier": "lightgbm", "use_tuned_hyperparameters": True,
                                 "preprocessing": "median_imputation", "augmentation": "smote",
                                 "synthetic_rows": SYNTHETIC_ROWS},
    "lightgbm_with_borderline_smote_rows": {"classifier": "lightgbm", "use_tuned_hyperparameters": True,
                                            "preprocessing": "median_imputation", "augmentation": "borderline_smote",
                                            "synthetic_rows": SYNTHETIC_ROWS},
    "lightgbm_with_adasyn_rows": {"classifier": "lightgbm", "use_tuned_hyperparameters": True,
                                  "preprocessing": "median_imputation", "augmentation": "adasyn",
                                  "synthetic_rows": SYNTHETIC_ROWS},
    "lightgbm_with_svm_smote_rows": {"classifier": "lightgbm", "use_tuned_hyperparameters": True,
                                     "preprocessing": "median_imputation", "augmentation": "svm_smote",
                                     "synthetic_rows": SYNTHETIC_ROWS},

    # --------------------------------------------------------------------------- RQ2: statistical and deep generators
    # Each generator is fitted on the failure rows of the training fold only and adds 2,000 rows (see
    # rq2_survey_of_methods/tabular_generators.py). The Gaussian copula models each attribute's distribution
    # separately and ties them together through a multivariate normal distribution; TVAE is a variational
    # autoencoder and CTGAN a conditional GAN for tabular data; TabDDPM is a denoising diffusion model, which learns
    # to turn random noise into rows step by step; CTAB-GAN+ is a further conditional GAN for tabular data.
    "lightgbm_with_gaussian_copula_rows": {"classifier": "lightgbm", "use_tuned_hyperparameters": True,
                                           "augmentation": "gaussian_copula", "synthetic_rows": SYNTHETIC_ROWS},
    "lightgbm_with_tvae_rows": {"classifier": "lightgbm", "use_tuned_hyperparameters": True, "augmentation": "tvae",
                                "synthetic_rows": SYNTHETIC_ROWS, "generator_options": {"epochs": 300}},
    "lightgbm_with_ctgan_rows": {"classifier": "lightgbm", "use_tuned_hyperparameters": True, "augmentation": "ctgan",
                                 "synthetic_rows": SYNTHETIC_ROWS, "generator_options": {"epochs": 300}},
    "lightgbm_with_tabddpm_rows": {"classifier": "lightgbm", "use_tuned_hyperparameters": True,
                                   "augmentation": "tabddpm", "synthetic_rows": SYNTHETIC_ROWS,
                                   "generator_options": {"steps": 20000}},
    "lightgbm_with_ctab_gan_plus_rows": {"classifier": "lightgbm", "use_tuned_hyperparameters": True,
                                         "augmentation": "ctab_gan_plus", "synthetic_rows": SYNTHETIC_ROWS,
                                         "generator_options": {"epochs": 150}},
    # GReaT writes each row as a sentence and fine-tunes a language model on these sentences with LoRA (low-rank
    # adaptation: the pretrained weights stay frozen, and small low-rank matrices of rank lora_r added to the
    # attention projections are trained instead). A failure row is about 2,200 tokens long (Qwen writes every digit
    # as its own token), so the 3-billion-parameter model planned at first does not fit the 24 GB of memory of an
    # L4 (an NVIDIA data-centre GPU) at batch 8; Qwen2.5-0.5B at batch 2 does. This arm produced no usable rows
    # (every sampled row was cut off before its last attributes) and is reported in the thesis as a failed arm. The
    # cause has since been fixed in tabular_generators.py, but the corrected version was not rerun.
    "lightgbm_with_great_language_model_rows": {"classifier": "lightgbm", "use_tuned_hyperparameters": True,
                                                "augmentation": "great", "synthetic_rows": SYNTHETIC_ROWS,
                                                "generator_options": {"llm": "Qwen/Qwen2.5-0.5B", "epochs": 10,
                                                                      "lora_r": 8, "batch_size": 2}},

    # ------------------------------------------------------------------- RQ2: Forest-Flow and TabEBM, 2,000 rows each
    # Forest-Flow generates rows by flow matching with gradient-boosted trees: at each of a number of noise levels,
    # tree ensembles learn the direction that moves noisy rows towards real ones, and sampling follows these
    # directions from pure noise to a row (see rq2_survey_of_methods/forest_flow_generator.py). It is run in three
    # sizes. The reduced profile (10 noise levels, 10 duplicates per row, 20 trees of depth 5) is the one that was
    # run on three seeds. The middle and paper profiles need about a day of an L4 per seed and were not completed
    # within the compute budget.
    "lightgbm_with_forest_flow_rows_reduced_profile": {"classifier": "lightgbm", "use_tuned_hyperparameters": True,
                                                       "augmentation": "forest_flow", "synthetic_rows": SYNTHETIC_ROWS},
    "lightgbm_with_forest_flow_rows_middle_profile": {"classifier": "lightgbm", "use_tuned_hyperparameters": True,
                                                      "augmentation": "forest_flow", "synthetic_rows": SYNTHETIC_ROWS,
                                                      "noise_levels": 50, "duplicates_per_row": 20,
                                                      "trees_per_regressor": 100, "tree_depth": 7,
                                                      "use_gpu_histograms": True},
    "lightgbm_with_forest_flow_rows_paper_profile": {"classifier": "lightgbm", "use_tuned_hyperparameters": True,
                                                     "augmentation": "forest_flow", "synthetic_rows": SYNTHETIC_ROWS,
                                                     "noise_levels": 50, "duplicates_per_row": 100,
                                                     "trees_per_regressor": 100, "tree_depth": 7},
    # TabEBM turns TabPFN, a pretrained transformer that classifies tabular data in context without being trained
    # on it, into an energy-based model of the failure class and samples from it with Langevin dynamics (gradient
    # steps on the energy plus random noise), see rq2_survey_of_methods/tabebm_generator.py. It needs a GPU in
    # practice, since every sampling step runs the TabPFN model.
    "lightgbm_with_tabebm_rows": {"classifier": "lightgbm", "use_tuned_hyperparameters": True, "augmentation": "tabebm",
                                  "synthetic_rows": SYNTHETIC_ROWS},

    # ---------------------------------------------------------------- RQ2: arms that change the learner, not the data
    # The missingness indicators are one extra 0/1 column per attribute that says whether the value was missing
    # (the option use_mask, see classifiers.make_ensemble). On APS the pattern is informative: failure rows have
    # about twice the missing rate of healthy rows. The first arm is an "ensemble" with LightGBM as its only
    # member, i.e. the tuned LightGBM on the attributes plus the indicators.
    "lightgbm_tuned_with_missingness_indicators": {"classifier": "ensemble", "use_tuned_hyperparameters": True,
                                                   "classifier_parameters": {"members": ["lightgbm"],
                                                                             "use_mask": True}},
    # The ensemble averages the log-odds of a tuned LightGBM, an XGBoost and a CatBoost model (see
    # classifiers.LogOddsAveragingEnsemble for why log-odds).
    "ensemble_without_missingness_indicators": {"classifier": "ensemble", "use_tuned_hyperparameters": True,
                                                "classifier_parameters": {"use_mask": False}},
    "ensemble_with_missingness_indicators": {"classifier": "ensemble", "use_tuned_hyperparameters": True},
    # TabICLv2 is a pretrained tabular foundation model that classifies by "in-context learning": it reads
    # labelled example rows (the context) together with the rows to classify, without any training. 48,000 context
    # rows do not fit in memory, so each fold model is a bag of three contexts, each with all training failures and
    # a fresh random sample of 10,000 healthy rows (n_neg, the healthy rows per context).
    "tabicl_bagged_contexts": {"classifier": "tabicl", "use_tuned_hyperparameters": False,
                               "classifier_parameters": {"n_bags": 3, "n_neg": 10000, "n_estimators": 2}},
    "ensemble_with_missingness_indicators_and_tabicl": {
        "classifier": "ensemble", "use_tuned_hyperparameters": True,
        "classifier_parameters": {"members": ["lightgbm", "xgboost", "catboost", "tabicl"], "use_mask": True,
                                  "tabicl_params": {"n_bags": 3, "n_neg": 10000, "n_estimators": 2}}},
}


def duplicated_real_failures_augmenter(synthetic_rows: int):
    """Augmenter that appends `synthetic_rows` copies of real training failures, drawn with replacement. It is the
    duplication control for every generator: same number of added rows, no new information."""
    def augment(X_train: pd.DataFrame, y_train: np.ndarray, seed: int):
        rng = np.random.default_rng(seed)
        failures = X_train[y_train == 1]
        copies = failures.iloc[rng.integers(0, len(failures), synthetic_rows)]
        return (pd.concat([X_train, copies], ignore_index=True),
                np.concatenate([y_train, np.ones(synthetic_rows, dtype=y_train.dtype)]))
    return augment


def build_arm(name: str, options: dict):
    """Turn the options of one arm into the three callables that the evaluation needs.

    The options (a dict of ARMS, possibly changed with --override) are
      classifier                 key of classifiers.CLASSIFIER_FACTORIES (default lightgbm)
      use_tuned_hyperparameters  start from the tuned LightGBM settings (default True; for the ensemble they go
                                 to its LightGBM member, for a classifier without a search they are empty)
      classifier_parameters      further keyword arguments for the factory, applied on top
      preprocessing              none, median_imputation or median_imputation_with_indicators
      augmentation               none, conditional_gan_2019, smote, borderline_smote, adasyn, svm_smote,
                                 gaussian_copula, tvae, ctgan, tabddpm, ctab_gan_plus, great, forest_flow,
                                 tabebm or duplicated_real_failures
      synthetic_rows             number of failure rows added (default 2,000)
      reapply_missingness        put missing values into the generated rows, which the generators produce
                                 without any (default True; each generator documents how it does this)
      gan_training_steps, gan_input_scaling, use_literal_notebook_loss
                                 settings of the conditional GAN of Ranasinghe et al. (defaults 30,000, minmax, False)
      sampling_strategy          passed to the SMOTE family, where it has no effect because synthetic_rows
                                 fixes the number of rows
      generator_options          keyword arguments of the RQ2 generators (epochs, steps, llm, lora_r, batch_size)
      noise_levels, duplicates_per_row, trees_per_regressor, tree_depth, condition_on_clusters, use_gpu_histograms
                                 Forest-Flow settings (defaults 10, 10, 20, 5, True, False: the reduced profile);
                                 forest_flow_generator.py's docstring gives the generator's own names for them
      langevin_steps             number of Langevin sampling steps of TabEBM (default 200)

    Returns (make_classifier, preprocess, augment, description):
      make_classifier(seed)            -> an unfitted classifier,
      preprocess(X_train, y_train)     -> a transform fitted on the training fold (or None),
      augment(X_train, y_train, seed)  -> the training fold with synthetic rows added (or None),
      description                      -> a dict that is stored with the results.
    """
    classifier_name = options.get("classifier", "lightgbm")
    if options.get("use_tuned_hyperparameters", True):
        # the ensemble uses the tuned LightGBM settings for its LightGBM member
        tuned_for = "lightgbm" if classifier_name == "ensemble" else classifier_name
        classifier_parameters = dict(load_tuned_hyperparameters(tuned_for))
    else:
        classifier_parameters = {}
    classifier_parameters.update(options.get("classifier_parameters", {}))
    factory = CLASSIFIER_FACTORIES[classifier_name]

    def make_classifier(seed):
        return factory(seed, **classifier_parameters)

    preprocessing = options.get("preprocessing", "none")
    if preprocessing == "none":
        preprocess = None
    elif preprocessing == "median_imputation":
        preprocess = median_imputation_preprocessor()
    elif preprocessing == "median_imputation_with_indicators":
        preprocess = median_imputation_preprocessor(add_missing_indicators=True)
    else:
        raise ValueError(f"unknown preprocessing: {preprocessing}")

    augmentation = options.get("augmentation", "none")
    synthetic_rows = options.get("synthetic_rows", SYNTHETIC_ROWS)
    if augmentation == "none":
        augment = None
    elif augmentation == "conditional_gan_2019":
        augment = conditional_gan_augmenter(n_synthetic=synthetic_rows,
                                            reapply_missingness=options.get("reapply_missingness", True),
                                            steps=options.get("gan_training_steps", 30000),
                                            scaling=options.get("gan_input_scaling", "minmax"),
                                            literal_loss=options.get("use_literal_notebook_loss", False))
    elif augmentation in ("smote", "borderline_smote", "adasyn", "svm_smote"):
        augment = smote_family_augmenter(kind=augmentation, sampling_strategy=options.get("sampling_strategy", 0.5),
                                         n_synthetic=synthetic_rows)
    elif augmentation in ("gaussian_copula", "tvae", "ctgan", "tabddpm", "ctab_gan_plus", "great"):
        # imported here so that the deep-learning dependencies are needed only by these arms
        from experiments.rq2_survey_of_methods.tabular_generators import survey_generator_augmenter
        augment = survey_generator_augmenter(augmentation, n_synthetic=synthetic_rows,
                                             reapply_missingness=options.get("reapply_missingness", True),
                                             **options.get("generator_options", {}))
    elif augmentation == "forest_flow":
        from experiments.rq2_survey_of_methods.forest_flow_generator import forest_flow_augmenter
        augment = forest_flow_augmenter(n_synthetic=synthetic_rows, n_t=options.get("noise_levels", 10),
                                        duplicate_K=options.get("duplicates_per_row", 10),
                                        n_estimators=options.get("trees_per_regressor", 20),
                                        max_depth=options.get("tree_depth", 5),
                                        condition_on_clusters=options.get("condition_on_clusters", True),
                                        gpu_hist=options.get("use_gpu_histograms", False))
    elif augmentation == "tabebm":
        from experiments.rq2_survey_of_methods.tabebm_generator import tabebm_augmenter
        augment = tabebm_augmenter(n_synthetic=synthetic_rows, sgld_steps=options.get("langevin_steps", 200))
    elif augmentation == "duplicated_real_failures":
        augment = duplicated_real_failures_augmenter(synthetic_rows)
    else:
        raise ValueError(f"unknown augmentation: {augmentation}")

    description = {"arm": name, "classifier": classifier_name, "classifier_parameters": classifier_parameters,
                   "preprocessing": preprocessing, "augmentation": augmentation, "arm_options": options}
    return make_classifier, preprocess, augment, description
