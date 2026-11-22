"""First look at the data: three gradient-boosting libraries with fixed, untuned settings.

This was the first thing run in the project, before any tuning and before any arm of RQ1 or RQ2, to see roughly
what cost a plain gradient-boosted tree model reaches on APS and how much the decision threshold matters. The
"fixed settings" of the configuration names *_fixed_settings are the settings of the factories in classifiers.py
(for LightGBM and XGBoost 600 trees at learning rate 0.03, for example), not the libraries' own defaults;
lightgbm_scale_pos_weight_30 is the same LightGBM with the failure rows weighted 30 times more heavily. Each
configuration is evaluated with five-fold stratified cross-validation on the 60,000 training rows (seed 11): every
training row is predicted by the fold model that did not see it (its out-of-fold prediction), and the cost is taken
at the threshold that minimises it on these predictions (500 per missed failure, 10 per false alarm). The test set
is not used.

Besides the cost at the best threshold, every record lists the smallest number of false alarms needed to keep the
missed failures at exactly 10, 20, 30, 40 and 60 ("fp_at_fn", searched over the candidate thresholds of
cost_and_evaluation.cost_curve; null when no candidate gives exactly that number), which shows how steep the
trade-off between the two kinds of error is.

    python -m experiments.quick_baseline_check                              # all four configurations
    python -m experiments.quick_baseline_check lightgbm_fixed_settings    # only the named ones

Appends one JSON line per configuration to results/preliminary_checks/quick_baseline.jsonl as it goes, so partial
progress is never lost, and saves the out-of-fold probabilities next to it as
quick_baseline_<configuration>_out_of_fold.npy.
"""
import json
import sys
import time

import numpy as np

from experiments.common.classifiers import make_histogram_gradient_boosting, make_lightgbm, make_xgboost
from experiments.common.cost_and_evaluation import cost_curve, run_cross_validation
from experiments.common.data_loading import CONSTANT_ATTRIBUTE, load_training_set
from experiments.common.paths import PRELIMINARY_CHECKS_DIR

CONFIGURATIONS = [
    ("lightgbm_fixed_settings", make_lightgbm, {}),
    # scale_pos_weight multiplies the weight of every failure row in the training loss; 30 is a rough first guess at
    # a useful up-weighting of the rare class
    ("lightgbm_scale_pos_weight_30", make_lightgbm, {"scale_pos_weight": 30}),
    ("xgboost_fixed_settings", make_xgboost, {}),
    ("histogram_gradient_boosting_fixed_settings", make_histogram_gradient_boosting, {}),
]


def main():
    selected = sys.argv[1:]
    X, y = load_training_set()
    X = X.drop(columns=[CONSTANT_ATTRIBUTE])
    PRELIMINARY_CHECKS_DIR.mkdir(parents=True, exist_ok=True)
    for name, factory, parameters in CONFIGURATIONS:
        if selected and name not in selected:
            continue
        started = time.time()
        def make_model(seed, factory=factory, parameters=parameters):
            return factory(seed, **parameters)

        result = run_cross_validation(name, make_model, X, y, n_splits=5, seed=11)
        summary = result.summary()
        summary["elapsed"] = time.time() - started
        curve = cost_curve(result.out_of_fold_probabilities, result.y)
        summary["fp_at_fn"] = {int(fn): int(curve.loc[curve.fn == fn, "fp"].min()) if (curve.fn == fn).any() else None
                               for fn in [10, 20, 30, 40, 60]}
        np.save(PRELIMINARY_CHECKS_DIR / f"quick_baseline_{name}_out_of_fold.npy", result.out_of_fold_probabilities)
        print(json.dumps({key: value for key, value in summary.items() if key != "fold_costs"}), flush=True)
        with open(PRELIMINARY_CHECKS_DIR / "quick_baseline.jsonl", "a") as handle:
            handle.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
