"""Train the PyTorch port of the cost-sensitive Transformer of Beikmohammadi et al. under the published protocol or the
protocol with a validation split of real rows (RQ1b).

The study of Beikmohammadi et al. (Cluster Computing) reports a test cost of 3,440 for a cost-sensitive
Transformer. Besides re-scoring and retraining the authors' own Keras code (authors_code/), the thesis trains a
PyTorch port of the network (transformer_2026_pytorch_port.py) under two protocols, to measure how much the order of
the preprocessing and the validation split affects the result:

  published  the authors' order: impute, resample (SVM-SMOTE, then RENN) and scale the whole training table, then
             hold out 10 % of the resampled table for validation (preprocessing_2026.py explains the steps);
  corrected  the protocol with a validation split of real rows: hold out 10 %
             real training rows first, fit the imputation, the resampling and the scaling on the other 90 %, and use
             the held-out real rows for validation.


In both, the network is trained on the remaining rows with the authors' loss and settings, and the epoch with the
lowest validation cost at the 0.5 cut-off is kept (the authors' checkpoint rule). The official test set is scored
once at the end, by the best-validation weights at the published 0.5 cut-off and at the threshold that minimises
the cost on the validation rows, and by the last epoch's weights at 0.5.

Ablations of the resampling run under the published protocol with --no-renn, --no-smote, or both.

The reported runs (on Colab GPUs, 500 epochs, 10 imputer rounds; seeds 0, 1, 2 for the protocol with a validation split
of real rows and 0, 1 for the others) were, from the repository root:

    python -m experiments.rq1_transformer_2026_reproduction.run_transformer_2026_port --protocol corrected \\
        --epochs 500 --patience 100000 --imputer-rounds 10 --seed 0 --run-name corrected_protocol \\
        --checkpoint-dir <checkpoint folder> --checkpoint-every 50
    python -m experiments.rq1_transformer_2026_reproduction.run_transformer_2026_port --protocol published \\
        --epochs 500 --patience 100000 --imputer-rounds 10 --seed 0 --run-name published_protocol \\
        --checkpoint-dir <checkpoint folder> --checkpoint-every 50
    ... the same with --no-renn (run name published_protocol_without_renn), --no-smote
        (published_protocol_without_svm_smote) and --no-smote --no-renn (published_protocol_without_resampling)

Seed 0 of the protocol with a validation split of real rows was then continued towards 4,000 epochs from its checkpoint,
under its own run name (the checkpoint file is copied to the new name first, so that the run resumes from it):

    cp <checkpoint folder>/corrected_protocol_seed0.pt <checkpoint folder>/corrected_protocol_extended_seed0.pt
    python -m experiments.rq1_transformer_2026_reproduction.run_transformer_2026_port --protocol corrected \\
        --epochs 4000 --patience 100000 --imputer-rounds 10 --seed 0 --run-name corrected_protocol_extended \\
        --checkpoint-dir <checkpoint folder> --checkpoint-every 50

(It was stopped by the compute budget before the end and scored offline with evaluate_port_checkpoint.py.)
A quick check on a CPU:

    python -m experiments.rq1_transformer_2026_reproduction.run_transformer_2026_port --protocol corrected \\
        --epochs 2 --imputer-rounds 2 --smoke-test-rows 3000

Reads the two data files. Writes, in results/rq1_transformer_2026_reproduction/pytorch_port/, <run-name>_seed<k>.json
(the settings under "args", timings, the per-epoch history, the validation result at the best epoch, and the test
results under "test_best_ckpt" and "test_final_epoch"), <run-name>_seed<k>_test_probabilities_best_checkpoint.npy
and <run-name>_seed<k>_test_probabilities_final_epoch.npy. With --checkpoint-dir the training state is kept in
<checkpoint-dir>/<run-name>_seed<k>.pt. The preprocessed tables are cached in cache/.
"""
from __future__ import annotations

import argparse
import copy
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
from experiments.rq1_transformer_2026_reproduction.transformer_2026_pytorch_port import predict_proba, train_transformer


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", default="corrected", choices=["published", "corrected"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=250, help="stop after this many epochs without a new best")
    parser.add_argument("--batch-size", type=int, default=72)
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Adam learning rate")
    parser.add_argument("--pos-weight", type=float, default=0.95, help="focal loss weight of the positive term")
    parser.add_argument("--loss-mode", default="package", choices=["package", "paper"],
                        help="focal loss as in the focal_loss package (the authors' code) or as in the paper")
    parser.add_argument("--amp", action="store_true", help="mixed precision (GPU only)")
    parser.add_argument("--gamma", type=float, default=1.5, help="focal loss exponent")
    parser.add_argument("--head-size", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=4)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--mlp-units", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--mlp-dropout", type=float, default=0.4)
    parser.add_argument("--imputer-rounds", type=int, default=100,
                        help="maximum rounds of the iterative imputer (100 as published; the reported runs used 10)")
    parser.add_argument("--no-renn", action="store_true", help="ablation: skip RENN")
    parser.add_argument("--no-smote", action="store_true",
                        help="ablation: skip SVM-SMOTE (RENN still applied unless --no-renn)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-minutes", type=float, default=None, help="stop training after this many minutes")
    parser.add_argument("--subsample-train", type=int, default=None, help="pilot only: cap the number of training rows")
    parser.add_argument("--run-name", default=None,
                        help="name of the output files <run-name>_seed<k>... (default: <protocol>_protocol)")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="save the training state to <checkpoint-dir>/<run-name>_seed<k>.pt every "
                             "--checkpoint-every epochs and resume from it when the same command is run again")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--initialisation", default="keras", choices=["keras", "torch"],
                        help="weight initialisation: Keras defaults (as in the authors' model) or PyTorch defaults")
    parser.add_argument("--adam-epsilon", type=float, default=1e-7,
                        help="Adam epsilon (Keras default 1e-7; PyTorch default 1e-8)")
    parser.add_argument("--collapse-patience", type=int, default=100,
                        help="stop a run whose validation predictions have been constant for this many epochs "
                             "(a collapsed network); 0 = never")
    parser.add_argument("--smoke-test-rows", type=int, default=None,
                        help="quick check only: stratified sample of this many training rows before preprocessing; "
                             "the output files are named smoke_test_<run-name>...")
    return parser.parse_args()


def main():
    args = parse_arguments()
    args.run_name = args.run_name or f"{args.protocol}_protocol"
    if args.smoke_test_rows:
        args.run_name = f"smoke_test_{args.run_name}"    # never overwrite the files of a real run
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} protocol={args.protocol}", flush=True)

    X, y = load_training_set()
    X = X.drop(columns=[CONSTANT_ATTRIBUTE])
    X_test, y_test = load_test_set()
    X_test = X_test.drop(columns=[CONSTANT_ATTRIBUTE])
    cache_tag = "full_training_set"
    if args.smoke_test_rows:
        from sklearn.model_selection import train_test_split
        X, _, y, _ = train_test_split(X, y, train_size=args.smoke_test_rows, stratify=y, random_state=args.seed)
        cache_tag = f"smoke_test_{args.smoke_test_rows}_rows"
    started = time.time()
    prepared = prepare_fold_as_in_2026_study(X, y, X_test, protocol=args.protocol, imputer_rounds=args.imputer_rounds,
                                             seed=args.seed, use_renn=not args.no_renn, cache_tag=cache_tag,
                                             use_smote=not args.no_smote)
    preparation_seconds = time.time() - started
    X_fit, y_fit, X_val, y_val = prepared["X_fit"], prepared["y_fit"], prepared["X_val"], prepared["y_val"]
    X_test_prepared = prepared["X_held_out"]
    if args.subsample_train:
        rng = np.random.default_rng(args.seed)
        kept = rng.choice(len(y_fit), args.subsample_train, replace=False)
        X_fit, y_fit = X_fit[kept], y_fit[kept]
    print(f"training rows {len(y_fit)} (failures {int(y_fit.sum())}), validation rows {len(y_val)} "
          f"(failures {int(y_val.sum())}), attributes {X_fit.shape[1]}, preprocessing {preparation_seconds:.0f}s",
          flush=True)

    architecture = dict(head_size=args.head_size, n_heads=args.n_heads, ff_dim=args.ff_dim, n_blocks=args.n_blocks,
                        mlp_units=tuple(args.mlp_units), dropout=args.dropout, mlp_dropout=args.mlp_dropout)
    checkpoint_path = (str(Path(args.checkpoint_dir) / f"{args.run_name}_seed{args.seed}.pt")
                       if args.checkpoint_dir else None)
    training_started = time.time()
    model, history, val_probabilities_best, best_epoch, final_state = train_transformer(
        X_fit, y_fit, X_val, y_val, epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
        learning_rate=args.learning_rate, pos_weight=args.pos_weight, gamma=args.gamma, loss_mode=args.loss_mode,
        architecture=architecture, seed=args.seed, device=device, max_minutes=args.max_minutes, amp=args.amp,
        checkpoint_path=checkpoint_path, checkpoint_every=args.checkpoint_every,
        collapse_patience=(args.collapse_patience or None), initialisation=args.initialisation,
        adam_epsilon=args.adam_epsilon)
    training_seconds = time.time() - training_started
    # the best-validation weights (the authors' stated rule) and the last epoch's weights (what their code scores)
    test_probabilities = predict_proba(model, X_test_prepared, device)
    val_threshold, _ = cost_minimising_threshold(val_probabilities_best, y_val)
    final_model = copy.deepcopy(model)
    final_model.load_state_dict(final_state)
    test_probabilities_final = predict_proba(final_model, X_test_prepared, device)
    # the key names below are those of the reported result files
    record = {"args": vars(args), "device": device, "prep_seconds": preparation_seconds,
              "train_seconds": training_seconds, "epochs_run": len(history), "best_epoch": best_epoch,
              "history": history,
              "val_best": {"cost_at_0.5": misclassification_cost(y_val, (val_probabilities_best >= 0.5).astype(int)),
                           "threshold_opt": val_threshold},
              "test_best_ckpt": {"at_0.5": misclassification_cost(y_test, (test_probabilities >= 0.5).astype(int)),
                                 "at_val_threshold": misclassification_cost(
                                     y_test, (test_probabilities >= val_threshold).astype(int))},
              "test_final_epoch": {"at_0.5": misclassification_cost(y_test,
                                                                    (test_probabilities_final >= 0.5).astype(int))}}
    record["test_best_ckpt"]["at_0.5"]["ci95"] = bootstrap_cost_confidence_interval(
        y_test, (test_probabilities >= 0.5).astype(int))
    output_stem = f"{args.run_name}_seed{args.seed}"
    PYTORCH_PORT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PYTORCH_PORT_RESULTS_DIR / f"{output_stem}_test_probabilities_best_checkpoint.npy", test_probabilities)
    np.save(PYTORCH_PORT_RESULTS_DIR / f"{output_stem}_test_probabilities_final_epoch.npy", test_probabilities_final)
    with open(PYTORCH_PORT_RESULTS_DIR / f"{output_stem}.json", "w") as handle:
        json.dump(record, handle, indent=1, default=float)
    print("TEST (best checkpoint, threshold 0.5):", record["test_best_ckpt"]["at_0.5"], flush=True)
    print("TEST (best checkpoint, validation-optimal threshold):", record["test_best_ckpt"]["at_val_threshold"],
          flush=True)


if __name__ == "__main__":
    main()
