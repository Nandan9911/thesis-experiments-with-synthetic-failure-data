"""Memorisation audit of the completion engines (RQ3): do completed records copy the failures the engine was fitted on?

A completion engine fitted on real failures could fill a partial record by copying a similar training failure
instead of inferring the missing values. The completion benchmark (run_completion_benchmark.py) stores accuracy and
the classifier's view, but not the completed records, so this script repeats one cell of it: fold --fold of the
five-fold split of the 1,000 training failures with seed --seed (fold 0 of seed 11 by default), every known cell of
the 200 held-out failures hidden with probability --hidden-fraction (0.5, the mask random_50_percent), and every
engine fitted on the 800 fitting failures. The engines and the masked records are the same as in that cell of the
benchmark; the stochastic engines draw --draws completions per record (5 by default, the benchmark used 20).

The point completions take the place of synthetic rows in the statistics of the synthetic-data diagnostics, which
are defined in experiments/common/synthetic_data_diagnostics.py, with the fitting failures as the real rows:
  dcr, median_nn_syn_to_real, median_nn_real, frac_closer
                          distance to closest record relative to the spacing of the fitting failures, the two medians
                          it is the ratio of, and the share of very close records
                          (distance_to_closest_record_statistics)
  mia_auc, near_copy_frac, dcr_fit_vs_heldout
                          membership-inference AUC, share of near-copies of fitting failures, and closeness to the
                          fitting failures relative to the held-out ones (memorisation_statistics)
  seconds                 time to fit the engine and complete the records
The held-out set contains the original of every completed record, and a completed record keeps all its known cells,
so its distance to the held-out failures is short by construction. That makes mia_auc and dcr_fit_vs_heldout
uninformative here; dcr, frac_closer and near_copy_frac are the measures to read. The reference entry "real held-out
failures (reference)" scores the 200 real held-out failures in place of completed records; for its
membership-inference part, the first 100 held-out failures act as the non-members and the other 100 as the records
under test.

An engine that fails (for example TabPFN when its model weights cannot be downloaded) is recorded as skipped, with
the error message.

    python -m experiments.rq3_completion_and_cold_start.run_completion_memorisation_audit \\
        --engines median k_nearest_neighbours chained_equations unmasking_trees forest_flow tabpfn \\
        --override '{"forest_flow": {"gpu_hist": true}, "unmasking_trees": {"xgboost_kwargs": {"device": "cuda"}}}'

Reads the training file. Writes results/rq3_completion_and_cold_start/completion_memorisation_audit.json (rewritten
after every engine), with the run settings under "setting": seed, fold, hide (the hidden fraction), k (the draws per
record) and override (the engine options by engine name).
"""
import argparse
import json
import time

import numpy as np
from sklearn.model_selection import KFold

from experiments.common.data_loading import CONSTANT_ATTRIBUTE, load_training_set
from experiments.common.paths import COMPLETION_AND_COLD_START_RESULTS_DIR
from experiments.common.synthetic_data_diagnostics import distance_to_closest_record_statistics, memorisation_statistics
from experiments.rq3_completion_and_cold_start.completion_engines import (COMPLETION_ENGINES,
                                                                          hide_cells_completely_at_random)

AUDIT_FILE = COMPLETION_AND_COLD_START_RESULTS_DIR / "completion_memorisation_audit.json"


def write_audit(audit):
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_FILE, "w") as handle:
        json.dump(audit, handle, indent=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engines", nargs="+", choices=sorted(COMPLETION_ENGINES),
                        default=["median", "k_nearest_neighbours", "chained_equations", "unmasking_trees",
                                 "forest_flow", "tabpfn"])
    parser.add_argument("--draws", type=int, default=5, help="completions drawn per record by the stochastic engines")
    parser.add_argument("--seed", type=int, default=11, help="seed of the five-fold split of the failures")
    parser.add_argument("--fold", type=int, default=0, help="which fold of that split to use (0-4)")
    parser.add_argument("--hidden-fraction", type=float, default=0.5,
                        help="probability with which each known cell of a held-out failure is hidden")
    parser.add_argument("--override", default="{}",
                        help='JSON object of engine options by engine name, '
                             'e.g. {"unmasking_trees": {"xgboost_kwargs": {"device": "cuda"}}}')
    args = parser.parse_args()
    X, y = load_training_set()
    X = X.drop(columns=[CONSTANT_ATTRIBUTE])
    failure_rows = np.where(y == 1)[0]
    fitting, held_out = list(KFold(5, shuffle=True, random_state=args.seed).split(failure_rows))[args.fold]
    fitting_failures = X.iloc[failure_rows[fitting]].reset_index(drop=True)
    heldout_failures = X.iloc[failure_rows[held_out]].reset_index(drop=True)
    X_masked, _ = hide_cells_completely_at_random(heldout_failures, args.hidden_fraction,
                                                  seed=args.seed * 10 + args.fold)
    audit = {"setting": {"seed": args.seed, "fold": args.fold, "hide": args.hidden_fraction, "k": args.draws,
                         "override": json.loads(args.override)}}
    reference = distance_to_closest_record_statistics(fitting_failures, heldout_failures)
    reference.update(memorisation_statistics(fitting_failures, heldout_failures.iloc[:100],
                                             heldout_failures.iloc[100:]))
    audit["real held-out failures (reference)"] = reference
    print("reference", {key: round(value, 3) for key, value in reference.items()}, flush=True)
    for name in args.engines:
        started = time.time()
        try:
            engine = COMPLETION_ENGINES[name](seed=args.seed + args.fold, verbose=False,
                                              **json.loads(args.override).get(name, {})).fit(fitting_failures)
            point, _, _ = engine.complete(X_masked, draws=args.draws)     # the audit uses the point completions
        except Exception as error:  # noqa: BLE001
            audit[name] = {"skipped": f"{type(error).__name__}: {str(error)[:160]}"}
            print(name, "skipped:", audit[name]["skipped"], flush=True)
            write_audit(audit)
            continue
        result = distance_to_closest_record_statistics(fitting_failures, point)
        result.update(memorisation_statistics(fitting_failures, heldout_failures, point))
        result["seconds"] = time.time() - started
        audit[name] = result
        print(name, {key: round(value, 3) for key, value in result.items()}, flush=True)
        write_audit(audit)


if __name__ == "__main__":
    main()
