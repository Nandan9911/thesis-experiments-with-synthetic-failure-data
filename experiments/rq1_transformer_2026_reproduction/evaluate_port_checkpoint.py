"""Score a saved training state of the PyTorch port of the cost-sensitive Transformer of Beikmohammadi et al. on the
official test set, offline (RQ1b).

run_transformer_2026_port.py scores the test set only when a run ends. This script does the same for a run that was
stopped earlier, from its checkpoint file. It was needed once: seed 0 of the protocol with a validation split of real
rows (--protocol corrected), continued towards 4,000 epochs under the run name corrected_protocol_extended, was stopped
by the compute budget after 3,250 epochs (the checkpoint stores the 0-based index of the last finished epoch, 3,249).

A checkpoint holds the current weights, the weights of the best validation epoch and the validation probabilities at
that epoch, so the same three numbers as at the end of a run can be produced: the best-validation weights at the
published 0.5 cut-off and at the threshold that minimises the cost on the validation rows, and the last saved
weights at 0.5. The preprocessing is rebuilt with the run's own settings (protocol, imputer rounds, seed and the
ablation switches), so that the test rows pass through exactly the transformations the network was trained with
and the validation labels line up with the stored probabilities; with the same cache folder the run's cached tables
are reused. The architecture is the published one.

Example (the reported evaluation, from the repository root):

    python -m experiments.rq1_transformer_2026_reproduction.evaluate_port_checkpoint \\
        --checkpoint <checkpoint folder>/corrected_protocol_extended_seed0.pt --protocol corrected \\
        --imputer-rounds 10 --seed 0

Reads the checkpoint and the two data files. Writes, in results/rq1_transformer_2026_reproduction/pytorch_port/,
<run-name>_seed<k>.json (ckpt: the checkpoint's file name, epoch_saved, best_epoch, best_val_cost, stopped,
protocol, seed, test_best_ckpt, val_threshold, test_last_saved_epoch),
<run-name>_seed<k>_test_probabilities_best_checkpoint.npy and
<run-name>_seed<k>_test_probabilities_last_checkpoint.npy.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.common.cost_and_evaluation import bootstrap_cost_confidence_interval, cost_minimising_threshold
from experiments.common.data_loading import (CONSTANT_ATTRIBUTE, load_test_set, load_training_set,
                                             misclassification_cost)
from experiments.common.paths import PYTORCH_PORT_RESULTS_DIR
from experiments.rq1_transformer_2026_reproduction.preprocessing_2026 import prepare_fold_as_in_2026_study
from experiments.rq1_transformer_2026_reproduction.transformer_2026_pytorch_port import (Transformer2026Port,
                                                                                        predict_proba)

PUBLISHED_ARCHITECTURE = dict(head_size=256, n_heads=4, ff_dim=4, n_blocks=4, mlp_units=(128, 64), dropout=0.25,
                              mlp_dropout=0.4)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="checkpoint file written by run_transformer_2026_port.py")
    parser.add_argument("--protocol", default="corrected", choices=["published", "corrected"])
    parser.add_argument("--imputer-rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default="corrected_protocol_extended",
                        help="name of the output files <run-name>_seed<k>...")
    parser.add_argument("--no-renn", action="store_true")
    parser.add_argument("--no-smote", action="store_true")
    return parser.parse_args()


def main():
    args = parse_arguments()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    X, y = load_training_set()
    X = X.drop(columns=[CONSTANT_ATTRIBUTE])
    X_test, y_test = load_test_set()
    X_test = X_test.drop(columns=[CONSTANT_ATTRIBUTE])
    started = time.time()
    prepared = prepare_fold_as_in_2026_study(X, y, X_test, protocol=args.protocol, imputer_rounds=args.imputer_rounds,
                                             seed=args.seed, use_renn=not args.no_renn, cache_tag="full_training_set",
                                             use_smote=not args.no_smote)
    X_test_prepared = prepared["X_held_out"]
    print(f"preprocessing {time.time() - started:.0f}s; attributes {X_test_prepared.shape[1]}", flush=True)
    model = Transformer2026Port(X_test_prepared.shape[1], **PUBLISHED_ARCHITECTURE)
    # the key names are those of the reported result file
    record = {"ckpt": Path(args.checkpoint).name, "epoch_saved": int(checkpoint["epoch"]),
              "best_epoch": int(checkpoint["best_epoch"]), "best_val_cost": float(checkpoint["best_cost"]),
              "stopped": checkpoint.get("stopped"), "protocol": args.protocol, "seed": args.seed}
    model.load_state_dict(checkpoint["best_state"])
    test_probabilities_best = predict_proba(model, X_test_prepared)
    val_threshold, _ = cost_minimising_threshold(np.asarray(checkpoint["best_pv"]), prepared["y_val"])
    record["test_best_ckpt"] = {
        "at_0.5": misclassification_cost(y_test, (test_probabilities_best >= 0.5).astype(int)),
        "at_val_threshold": misclassification_cost(y_test, (test_probabilities_best >= val_threshold).astype(int))}
    record["test_best_ckpt"]["at_0.5"]["ci95"] = bootstrap_cost_confidence_interval(
        y_test, (test_probabilities_best >= 0.5).astype(int))
    record["val_threshold"] = float(val_threshold)
    model.load_state_dict(checkpoint["model"])
    test_probabilities_last = predict_proba(model, X_test_prepared)
    record["test_last_saved_epoch"] = {
        "at_0.5": misclassification_cost(y_test, (test_probabilities_last >= 0.5).astype(int))}
    output_stem = f"{args.run_name}_seed{args.seed}"
    PYTORCH_PORT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PYTORCH_PORT_RESULTS_DIR / f"{output_stem}_test_probabilities_best_checkpoint.npy", test_probabilities_best)
    np.save(PYTORCH_PORT_RESULTS_DIR / f"{output_stem}_test_probabilities_last_checkpoint.npy", test_probabilities_last)
    with open(PYTORCH_PORT_RESULTS_DIR / f"{output_stem}.json", "w") as handle:
        json.dump(record, handle, indent=1, default=float)
    print(json.dumps({key: record[key] for key in ("epoch_saved", "best_epoch", "best_val_cost")}),
          "| best checkpoint at 0.5:", record["test_best_ckpt"]["at_0.5"]["cost"],
          "| at the validation threshold:", record["test_best_ckpt"]["at_val_threshold"]["cost"],
          "| last saved epoch at 0.5:", record["test_last_saved_epoch"]["at_0.5"]["cost"], flush=True)


if __name__ == "__main__":
    main()
