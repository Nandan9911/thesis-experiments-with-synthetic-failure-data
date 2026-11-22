"""How realistic, how detectable and how close to copies are the synthetic failure rows of each generator?

Part of RQ2, and of RQ1a for the conditional GAN of Ranasinghe et al. The classifier arms show whether synthetic rows
lower the cost; this script looks at the rows themselves. Each generator is fitted once, on the 800 real failure rows in
the training part of fold 0 (five stratified folds of the 60,000 training rows, seed 11), and asked for 2,000 rows. The
seeds are those that the arms use in that fold (1100 for fitting, 1101 for sampling), and every generator runs with the
default settings of its class, which are the settings of its arm except for GReaT, whose arm uses a smaller language
model and batch size. The rows are compared with the fitting rows and with the 200 real failures of the validation part
of fold 0 (the held-out failures).

    python -m experiments.rq2_survey_of_methods.run_synthetic_data_diagnostics \\
        --generators conditional_gan_2019 duplicated_real_failures forest_flow gaussian_copula \\
                     smote borderline_smote adasyn svm_smote
    python -m experiments.rq2_survey_of_methods.run_synthetic_data_diagnostics \\
        --generators tabebm ctgan tvae tabddpm ctab_gan_plus          # these need a GPU in practice

Reads the training file (and, for ctab_gan_plus, the CTAB-GAN+ repository in third_party/). Writes
results/rq2_survey_of_methods/synthetic_data_diagnostics.json. The file is merged, so a run for some generators
keeps the entries of all others; the top-level n_syn is the row count of the latest run.

Generators (--generators)
  conditional_gan_2019                     the conditional GAN of Ranasinghe et al. with min-max inputs and a single
                                           sigmoid (the arms' default)
  conditional_gan_2019_literal_notebook    the cGAN with the notebook's settings: the scaling, loss and batch drawing of
                                           the notebook, without the real missingness put back into its rows
  conditional_gan_2019_without_missingness the default GAN (min-max inputs, single sigmoid) without the real
                                           missingness put back
  duplicated_real_failures                 2,000 fitting failures drawn with replacement: the duplication control
  forest_flow, gaussian_copula, tvae, ctgan, tabddpm, ctab_gan_plus, great, tabebm
                                           the generators of RQ2 (rq2_survey_of_methods/*_generator.py and
                                           tabular_generators.py)
  smote, borderline_smote, adasyn, svm_smote
                                           the SMOTE family. SMOTE makes a new failure row by interpolating between
                                           a failure row and one of its nearest failure neighbours; Borderline-SMOTE,
                                           SVM-SMOTE and ADASYN concentrate the new rows near the healthy class. Like
                                           the arms, they run on median-imputed attributes (the medians of the
                                           fitting failures), here on the 800 fitting failures plus the first 5,000
                                           healthy rows of the training part.

Diagnostics (keys of the JSON file). Each statistic is defined in experiments/common/synthetic_data_diagnostics.py;
here the real rows are the 800 fitting failures, and the 200 held-out failures are the real rows that the generator
never saw.
  Reference entries:
    reference_real_heldout      the 200 held-out real failures scored as if they were synthetic: the values an
                                ideal generator would reach (detect_auc, dcr, frac_closer, median_nn_*, colwise_w1)
    reference_duplicates_memorisation
                                the memorisation statistics of 2,000 copies of fitting failures: the values of
                                pure copying (mia_auc, near_copy_frac, dcr_fit_vs_heldout)
  For every generator:
    detect_auc          detectability AUC (detectability_auc): how well a LightGBM classifier tells the synthetic
                        rows from the fitting failures, from 0.5 (not at all) to 1.0 (every row recognised). Missing
                        values stay missing (NaN), as the classifiers of the arms see them.
    detect_auc_nomask   the same after filling the missing cells of both sets with the medians of the fitting
                        failures, so that the missingness pattern cannot give the rows away
    detect_auc_rounded  the same with the synthetic values rounded to whole numbers (the real counters are whole
                        numbers), to see whether fractional values alone give the rows away
    dcr, median_nn_syn_to_real, median_nn_real, frac_closer
                        distance to closest record relative to the spacing of the real rows, the two medians it is
                        the ratio of, and the share of synthetic rows very close to a real row
                        (distance_to_closest_record_statistics)
    mia_auc, near_copy_frac, dcr_fit_vs_heldout
                        membership-inference AUC, share of near-copies of fitting failures, and closeness to the
                        fitting failures relative to the held-out ones (memorisation_statistics)
    colwise_w1          mean column-wise 1-Wasserstein distance between the real and the synthetic values
                        (mean_columnwise_wasserstein_distance); 0 means identical marginal distributions
    missing_frac_syn, missing_frac_real    share of cells that are missing
    zero_frac_syn, zero_frac_real          share of cells that are exactly 0 (missing cells count as non-zero)
    integer_frac_syn, integer_frac_real    share of cells holding a whole number (missing cells count as not whole)
    seconds             time for fitting, sampling and computing the diagnostics
  Top level: n_real_fit (800), n_real_heldout (200) and n_syn (synthetic rows per generator).
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from experiments.common.data_loading import CONSTANT_ATTRIBUTE, load_training_set
from experiments.common.paths import SURVEY_RESULTS_DIR
from experiments.common.synthetic_data_diagnostics import (detectability_auc, distance_to_closest_record_statistics,
                                                           mean_columnwise_wasserstein_distance,
                                                           memorisation_statistics)
from experiments.rq1_cgan_2019_reproduction.conditional_gan_2019 import ConditionalGan2019

DIAGNOSTICS_FILE = SURVEY_RESULTS_DIR / "synthetic_data_diagnostics.json"
DIAGNOSED_GENERATORS = ["conditional_gan_2019", "conditional_gan_2019_literal_notebook",
                        "conditional_gan_2019_without_missingness", "duplicated_real_failures", "forest_flow",
                        "gaussian_copula", "tvae", "ctgan", "tabddpm", "ctab_gan_plus", "great", "tabebm",
                        "smote", "borderline_smote", "adasyn", "svm_smote"]
# the seeds the arms use in fold 0 of seed 11: a generator gets seed * 100 + fold for fitting and one more for
# sampling
FIT_SEED, SAMPLE_SEED = 1100, 1101


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generators", nargs="+", choices=DIAGNOSED_GENERATORS,
                        default=["conditional_gan_2019", "conditional_gan_2019_literal_notebook",
                                 "duplicated_real_failures"])
    parser.add_argument("--gan-training-steps", type=int, default=30000,
                        help="training steps of the conditional GAN of Ranasinghe et al. (30,000, as in the notebook)")
    parser.add_argument("--synthetic-rows", type=int, default=2000, help="rows sampled from every generator")
    return parser.parse_args()


def synthetic_failure_rows(generator_name, real_fitting_failures, X_train_part, y_train_part, n_rows,
                           gan_training_steps):
    """Fit one generator on the real fitting failures and return n_rows synthetic failure rows."""
    if generator_name == "conditional_gan_2019":
        return ConditionalGan2019(steps=gan_training_steps, seed=FIT_SEED).fit(real_fitting_failures).sample(
            n_rows, reapply_missingness=True, seed=SAMPLE_SEED)
    if generator_name == "conditional_gan_2019_literal_notebook":
        return ConditionalGan2019(steps=gan_training_steps, seed=FIT_SEED, scaling="standard",
                                  literal_loss=True).fit(real_fitting_failures).sample(
            n_rows, reapply_missingness=False, seed=SAMPLE_SEED)
    if generator_name == "conditional_gan_2019_without_missingness":
        return ConditionalGan2019(steps=gan_training_steps, seed=FIT_SEED).fit(real_fitting_failures).sample(
            n_rows, reapply_missingness=False, seed=SAMPLE_SEED)
    if generator_name == "forest_flow":
        from experiments.rq2_survey_of_methods.forest_flow_generator import ForestFlowFailureGenerator
        return ForestFlowFailureGenerator(seed=FIT_SEED).fit(real_fitting_failures).sample(
            n_rows, reapply_missingness=True, seed=SAMPLE_SEED)
    if generator_name == "duplicated_real_failures":
        rng = np.random.default_rng(FIT_SEED)
        return real_fitting_failures.iloc[rng.integers(0, len(real_fitting_failures), n_rows)].reset_index(drop=True)
    if generator_name in ("gaussian_copula", "tvae", "ctgan", "tabddpm", "ctab_gan_plus", "great"):
        from experiments.rq2_survey_of_methods.tabular_generators import SURVEY_GENERATORS
        return SURVEY_GENERATORS[generator_name](seed=FIT_SEED).fit(real_fitting_failures).sample(
            n_rows, reapply_missingness=True, seed=SAMPLE_SEED)
    if generator_name == "tabebm":
        from experiments.rq2_survey_of_methods.tabebm_generator import TabEBMFailureGenerator
        return TabEBMFailureGenerator(seed=FIT_SEED).fit(real_fitting_failures).sample(
            n_rows, reapply_missingness=True, seed=SAMPLE_SEED)
    if generator_name in ("smote", "borderline_smote", "adasyn", "svm_smote"):
        from experiments.common.imputation_and_oversampling import smote_family_augmenter
        failure_medians = real_fitting_failures.median()
        X_two_classes = real_fitting_failures.fillna(failure_medians)
        y_two_classes = np.ones(len(X_two_classes), dtype=int)
        healthy = X_train_part[y_train_part == 0].iloc[:5000].fillna(failure_medians)
        X_two_classes = pd.concat([X_two_classes, healthy], ignore_index=True)
        y_two_classes = np.concatenate([y_two_classes, np.zeros(len(healthy), dtype=int)])
        X_augmented, _ = smote_family_augmenter(kind=generator_name, n_synthetic=n_rows, verbose=False)(
            X_two_classes, y_two_classes, FIT_SEED)
        return X_augmented.iloc[len(X_two_classes):].reset_index(drop=True)   # the appended rows are the new ones
    raise ValueError(generator_name)


def main():
    args = parse_arguments()
    X, y = load_training_set()
    X = X.drop(columns=[CONSTANT_ATTRIBUTE])
    folds = StratifiedKFold(5, shuffle=True, random_state=11)
    train_rows, validation_rows = next(iter(folds.split(X, y)))
    real_fitting_failures = X.iloc[train_rows][y[train_rows] == 1].reset_index(drop=True)
    real_heldout_failures = X.iloc[validation_rows][y[validation_rows] == 1].reset_index(drop=True)
    diagnostics = {"n_real_fit": int(len(real_fitting_failures)), "n_real_heldout": int(len(real_heldout_failures)),
                   "n_syn": args.synthetic_rows}
    # what an ideal generator would score: real held-out failures against the real fitting failures
    diagnostics["reference_real_heldout"] = {
        "detect_auc": detectability_auc(real_fitting_failures, real_heldout_failures),
        **distance_to_closest_record_statistics(real_fitting_failures, real_heldout_failures),
        "colwise_w1": mean_columnwise_wasserstein_distance(real_fitting_failures, real_heldout_failures)}
    # what pure copying scores
    rng = np.random.default_rng(FIT_SEED)
    duplicates = real_fitting_failures.iloc[rng.integers(0, len(real_fitting_failures), args.synthetic_rows)]
    duplicates = duplicates.reset_index(drop=True)
    diagnostics["reference_duplicates_memorisation"] = memorisation_statistics(real_fitting_failures,
                                                                               real_heldout_failures, duplicates)
    for generator_name in args.generators:
        started = time.time()
        synthetic = synthetic_failure_rows(generator_name, real_fitting_failures, X.iloc[train_rows], y[train_rows],
                                           args.synthetic_rows, args.gan_training_steps)
        synthetic = synthetic[real_fitting_failures.columns]
        failure_medians = real_fitting_failures.median()
        result = {
            "detect_auc": detectability_auc(real_fitting_failures, synthetic),
            "detect_auc_nomask": detectability_auc(real_fitting_failures, synthetic, fill=failure_medians),
            **distance_to_closest_record_statistics(real_fitting_failures, synthetic),
            **memorisation_statistics(real_fitting_failures, real_heldout_failures, synthetic),
            "colwise_w1": mean_columnwise_wasserstein_distance(real_fitting_failures, synthetic),
            "missing_frac_syn": float(synthetic.isna().mean().mean()),
            "missing_frac_real": float(real_fitting_failures.isna().mean().mean()),
            "zero_frac_syn": float((synthetic == 0).values.mean()),
            "zero_frac_real": float((real_fitting_failures == 0).values.mean()),
            "integer_frac_syn": float(np.nanmean(np.isclose(synthetic.values, np.round(synthetic.values)))),
            "integer_frac_real": float(np.nanmean(np.isclose(real_fitting_failures.values,
                                                             np.round(real_fitting_failures.values)))),
            "detect_auc_rounded": detectability_auc(real_fitting_failures, synthetic.round()),
            "seconds": time.time() - started}
        diagnostics[generator_name] = result
        print(generator_name, {key: (round(value, 3) if isinstance(value, float) else value)
                               for key, value in result.items()}, flush=True)
    DIAGNOSTICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if DIAGNOSTICS_FILE.exists():             # merge: a run for one generator must not drop the others
        with open(DIAGNOSTICS_FILE) as handle:
            previous = json.load(handle)
        previous.update(diagnostics)
        diagnostics = previous
    with open(DIAGNOSTICS_FILE, "w") as handle:
        json.dump(diagnostics, handle, indent=1)
    print("reference (real held-out)",
          {key: round(value, 3) for key, value in diagnostics["reference_real_heldout"].items()})


if __name__ == "__main__":
    main()
