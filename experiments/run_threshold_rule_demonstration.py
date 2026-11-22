"""What the decision rules of the earlier phase of the project cost: one model per learner, four threshold rules.

A preliminary check that motivates the threshold choice of the common evaluation protocol (cost_and_evaluation.py). The
earlier phase of the project is the exploratory work done before the common evaluation protocol was fixed; in it the
test set was scored many times, and the notebook attributed to the study of Ranasinghe et al. was obtained. This script
reproduces the decision rule of that phase rather than the protocol: each of three exploratory learners is trained once
on a stratified 80 % of the 60,000 training rows, with missing values set to 0 as the notebooks of the earlier phase
did; the threshold is chosen on the remaining 20 % under four rules; and the official test set is scored under each
rule. The rules are

    linear_0.01   the best threshold on the grid 0.01, 0.02, ..., 0.99 (the grid of the evaluation code of the
                  earlier phase)
    all_values    the best threshold among all distinct predicted probabilities of the validation rows, which reaches
                  1e-5 and below
    fixed_0.5     the default cut-off 0.5, chosen without looking at any data
    bayes_0.0196  Elkan's cost-optimal cut-off. Elkan (2001) showed that if the predicted probabilities are
                  calibrated (of all trucks given a probability of 2 %, about 2 % fail), flagging a truck whenever
                  P(failure) is at least c_FP / (c_FP + c_FN) minimises the expected cost; with c_FP = 10 for a false
                  alarm and c_FN = 500 for a missed failure this is 10 / (10 + 500) = 0.0196. Boosted trees trained on
                  a rare class are not calibrated at the extremes, so the rule cannot be applied blindly.

On the validation rows, ties between thresholds go to the smallest one. Two diagnostics that use the test labels are
added for each learner: the lowest test cost of any threshold on the quantile grid of cost_and_evaluation.cost_curve
("oracle") and the smallest number of false alarms with at most six missed failures ("min_fp_at_fn6"). This script
scores the test set more than once by design; its numbers measure how much of a reported cost depends on the
threshold rule and are not results of any arm.

    python -m experiments.run_threshold_rule_demonstration

Writes results/preliminary_checks/threshold_rule_demonstration.json: for each learner the seconds to fit and predict
(fit_seconds), the test confusion counts and cost under each rule together with the threshold and, for the two
searched rules, the validation cost (val_cost), and the two diagnostics.
"""
import json
import time

import numpy as np
from sklearn.model_selection import train_test_split

from experiments.common.cost_and_evaluation import cost_curve
from experiments.common.data_loading import (CONSTANT_ATTRIBUTE, COST_OF_FALSE_ALARM, COST_OF_MISSED_FAILURE,
                                             load_test_set, load_training_set, misclassification_cost)
from experiments.common.paths import PRELIMINARY_CHECKS_DIR

OUTPUT_FILE = PRELIMINARY_CHECKS_DIR / "threshold_rule_demonstration.json"


def learners():
    """The three exploratory learners of the earlier phase, under the names used as keys in the output file."""
    import lightgbm as lgb
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    yield "sklearn GBM (default: 100 trees, depth 3)", GradientBoostingClassifier(random_state=11)
    yield ("RF (300 trees, depth 20, balanced_subsample)",
           RandomForestClassifier(n_estimators=300, max_depth=20, class_weight="balanced_subsample", n_jobs=2,
                                  random_state=11))
    yield "LightGBM (500 trees, defaults)", lgb.LGBMClassifier(n_estimators=500, n_jobs=2, random_state=11, verbose=-1)


def best_threshold_on_grid(probabilities, labels, candidate_thresholds):
    """The candidate threshold with the lowest cost on (probabilities, labels), and that cost; ties go to the first
    candidate, i.e. the smallest threshold of an increasing grid."""
    costs = [misclassification_cost(labels, (probabilities >= threshold).astype(int))["cost"]
             for threshold in candidate_thresholds]
    best = int(np.argmin(costs))
    return float(candidate_thresholds[best]), int(costs[best])


def main():
    X, y = load_training_set()
    X = X.drop(columns=[CONSTANT_ATTRIBUTE]).fillna(0.0)
    X_test, y_test = load_test_set()
    X_test = X_test.drop(columns=[CONSTANT_ATTRIBUTE]).fillna(0.0)
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, stratify=y, random_state=11)

    results = {}
    for name, model in learners():
        started = time.time()
        model.fit(X_train, y_train)
        validation_probabilities = model.predict_proba(X_validation)[:, 1]
        test_probabilities = model.predict_proba(X_test)[:, 1]
        learner_results = {"fit_seconds": time.time() - started}
        searched_rules = {"linear_0.01": np.arange(0.01, 1.0, 0.01), "all_values": np.unique(validation_probabilities)}
        for rule, candidate_thresholds in searched_rules.items():
            threshold, validation_cost = best_threshold_on_grid(validation_probabilities, y_validation,
                                                                candidate_thresholds)
            test_result = misclassification_cost(y_test, (test_probabilities >= threshold).astype(int))
            test_result.update({"threshold": threshold, "val_cost": validation_cost})
            learner_results[rule] = test_result
        fixed_rules = {"fixed_0.5": 0.5,
                       "bayes_0.0196": COST_OF_FALSE_ALARM / (COST_OF_MISSED_FAILURE + COST_OF_FALSE_ALARM)}
        for rule, threshold in fixed_rules.items():
            test_result = misclassification_cost(y_test, (test_probabilities >= threshold).astype(int))
            test_result["threshold"] = threshold
            learner_results[rule] = test_result
        # diagnostics on the test labels
        test_curve = cost_curve(test_probabilities, y_test)
        learner_results["oracle"] = {"cost": int(test_curve.cost.min()),
                                     "threshold": float(test_curve.loc[test_curve.cost.idxmin(), "threshold"])}
        at_most_six_missed = test_curve[test_curve.fn <= 6]
        learner_results["min_fp_at_fn6"] = int(at_most_six_missed.fp.min()) if len(at_most_six_missed) else None
        results[name] = learner_results
        print(name, {rule: (value["cost"], value.get("fn"), value.get("fp")) for rule, value in learner_results.items()
                     if isinstance(value, dict) and "cost" in value}, flush=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
