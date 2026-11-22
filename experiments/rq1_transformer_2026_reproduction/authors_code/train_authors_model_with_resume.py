"""The training code released by Beikmohammadi et al., run from scratch with checkpoints, a test-cost trace and
test-inclusion controls.

Part of RQ1b. The five models released by Beikmohammadi et al. reproduce the published test cost of 3,440 on the exactly
regenerated inputs (prepare_authors_inputs.py, evaluate_shipped_models.py). The question here is whether the released
training code reaches a similar cost when it is run again. This script runs the authors' training code, unchanged
in substance, on those inputs: it executes the model, loss, metric and evaluation code of their TrainingCode.py
(read from the authors' repository) and repeats their maincode step by step, with the same seeds, the same six input
tables, the same stratified 90/10 split of the resampled table into training and validation rows (random_state =
seed), the same network, focal loss, Adam optimiser, batch size, number of epochs and ModelCheckpoint callback, which
keeps the model of the epoch with the lowest validation cost. Only TensorBoard logging is left out. The additions:

* Checkpoints (--checkpoint-dir, --checkpoint-every): the full model and the epoch counter are saved every N epochs,
  and the script resumes from them when it is started again with the same arguments (the authors' CSV log is
  appended to). This is what lets a run of thousands of epochs survive the end of a Colab session.
* A test-cost trace (--test-every): every N epochs the official test set is scored at the published 0.5 cut-off and
  the cost, missed failures (FN) and false alarms (FP) are appended to test_trace.jsonl. The trace is a diagnostic;
  the published protocol scores the test set once, at the end of a run.
* Staging (--stop-epoch, --max-hours): training stops after a given epoch or time budget, with --num_epochs (and
  therefore the authors' file names) kept at the published 8,000; a later run with a larger --stop-epoch continues
  from the checkpoint. Every stage ends with the final evaluation below.
* Test-inclusion controls (--contamination-fraction, --contamination-seed): a stratified fraction of the official
  test rows (imputed and scaled exactly like the published inputs) is appended to the resampled table before the
  90/10 split, so that those test rows take part in training and validation like any other row. The trace and the
  final evaluation then report the included ("seen") and the excluded ("unseen") test rows separately. A control
  shows how a model trained with the authors' own procedure scores test rows that were part of its training data and
  test rows that were not. A fraction of 0 (the default) leaves the authors' pipeline unchanged.
* Speed (--steps-per-execution, --xla): Keras runs several batches per compiled call instead of returning to Python
  after every batch (the model is small, so that overhead dominated the epoch time), and XLA compiles the training
  step, fusing the many small element-wise operations of the attention blocks (1.8 times faster per step on an A100
  GPU). Neither changes the model, the loss, the optimiser or the order of the updates; with XLA, floating-point
  rounding can differ at about the 1e-6 relative level, as it already does between GPU generations.

At the end of a run or stage the script repeats the authors' own final evaluation: the final-epoch model scores the
training and the test rows at the 0.5 cut-off (a row is flagged when its probability exceeds 0.5). Because the paper
does not say whether its costs come from the final-epoch model or from the best-validation checkpoint, the
checkpoint that the authors' callback kept is scored as well (the epoch with the lowest validation cost in the CSV
log; after a resume the callback restarts its record, so the log decides). Finally the authors' test() is called,
which prints the same costs and draws plots; it is skipped if it cannot run on the machine.

The flags --seed_number, --learning_rate, --pos_weight, --gamma, --num_epochs, --batch_size, --head_size, --num_heads,
--ff_dim, --num_transformer_blocks, --mlp_dropout, --dropout and --mlp_units are those of the authors' TrainingCode.py,
with the same names and defaults.

The reported runs were made on Colab GPUs through cloud_runs/, one checkpoint folder per run: the retraining runs with
the authors' seeds 0 to 4, the controls with half of the test rows (seeds 0, 1, 2) and with all of them (seed 0), and
two 300-epoch verification runs (a retraining run and a control run with half of the test rows, seed 0). Every other run
first went to 1,000 epochs; the retraining runs of seeds 0 and 1 and seed 0 of the control runs with half of the test
rows in the training data were then extended towards 8,000 (the retraining run of seed 1 was stopped at 5,950 by the
compute budget). For example, from the repository root:

    APS_AUTHORS_2026_REPOSITORY=<working copy of the authors' repository> \\
    python -m experiments.rq1_transformer_2026_reproduction.authors_code.train_authors_model_with_resume \\
        --seed_number 0 --num_epochs 8000 --stop-epoch 1000 --checkpoint-dir <checkpoint folder> \\
        --checkpoint-every 50 --test-every 50
    ... --contamination-fraction 0.5   (half of the test rows in the training table)
    ... --contamination-fraction 1.0   (all of them)

The authors' code reads the six input tables from, and writes its CSV/ and Model/ folders into, the folder it runs
in, so the script changes into the authors' repository (APS_AUTHORS_2026_REPOSITORY if set; a working copy with
TrainingCode.py and the six tables keeps parallel runs apart).

Writes into the checkpoint folder (a relative --checkpoint-dir is taken from the current folder; the default is ckpt/
inside the authors' repository):
  state.json                            the last finished epoch and the authors' run name (resumed from)
  last_model.keras                      the full model at that epoch (resumed from)
  training_log.csv                      a copy of the authors' per-epoch CSV log
  test_trace.jsonl                      the test-cost trace (with "seen" and "unseen" parts for a control)
  contaminating_test_rows.npy           indices of the test rows added to the training table (empty for a run without
                                        test rows)
  final_evaluation.json                 the final evaluation of the last stage (train, test, epochs_run, and, where
                                        applicable, contamination and test_best_ckpt)
  test_probabilities_final.npy          final-epoch probabilities for the 16,000 test rows
  training_probabilities_final.npy      final-epoch probabilities for the 60,000 training rows (used by the
                                        hard-negative comparison for the retrained models)
  test_probabilities_best_checkpoint.npy  probabilities of the best-validation checkpoint for the test rows
final_evaluation.json is overwritten at the end of every stage. Before a run is extended beyond a stage, the cloud job
wrapper keeps a copy as final_evaluation_after_<epochs>_epochs.json, named after the epochs_run field inside it. The
run folders in results/rq1_transformer_2026_reproduction/authors_code/retraining_runs/ hold the files above except
last_model.keras.

Environment: APS_AUTHORS_2026_REPOSITORY (see above), APS_STEPS_PER_EXECUTION and XLA_JIT (defaults of
--steps-per-execution and --xla); TF_USE_LEGACY_KERAS is set to 1, since the authors wrote for Keras 2.
"""
import argparse
import glob
import json
import os
import re
import shutil
import time
import types
from pathlib import Path

# the authors wrote for Keras 2 (checkpoint paths without extension, SavedModel folders)
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import pandas as pd

from experiments.common.data_loading import COST_OF_FALSE_ALARM, COST_OF_MISSED_FAILURE
from experiments.rq1_transformer_2026_reproduction.authors_code.file_locations import (
    AUTHORS_REPOSITORY, RESAMPLED_LABELS, RESAMPLED_TABLE_SCALED, TEST_LABELS, TEST_TABLE_SCALED, TRAINING_LABELS,
    TRAINING_TABLE_SCALED)


def load_authors_module():
    """Execute the authors' TrainingCode.py minus its __main__ block, so that build_model, cost_sensitive_metric_dl,
    cost_sensitive_score and test() are their code verbatim."""
    source = (AUTHORS_REPOSITORY / "TrainingCode.py").read_text()
    source = source.split("if __name__ == '__main__':")[0]
    # drop the import lines of packages that no longer exist or are not installed (e.g. the removed
    # tensorflow.keras.wrappers, HF transformers, plotly, scikit-plot); none of them is used by the
    # model, the loss or the metric
    kept_lines = []
    for line in source.splitlines():
        if line.startswith(("import ", "from ")) and "(" not in line:
            try:
                exec(line, {})
            except Exception:  # noqa: BLE001
                print(f"    skipping authors' import (not available here): {line.strip()}", flush=True)
                continue
        kept_lines.append(line)
    source = "\n".join(kept_lines)
    module = types.ModuleType("authors_training_code")
    module.__file__ = str(AUTHORS_REPOSITORY / "TrainingCode.py")
    exec(compile(source, str(AUTHORS_REPOSITORY / "TrainingCode.py"), "exec"), module.__dict__)
    return module


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    authors = parser.add_argument_group("the authors' arguments (names and defaults as in their TrainingCode.py)")
    authors.add_argument("--seed_number", type=int, default=0)
    authors.add_argument("--learning_rate", type=float, default=0.0005)
    authors.add_argument("--pos_weight", type=float, default=0.95)
    authors.add_argument("--gamma", type=float, default=1.5)
    authors.add_argument("--num_epochs", type=int, default=8000)
    authors.add_argument("--batch_size", type=int, default=72)
    authors.add_argument("--head_size", type=int, default=256)
    authors.add_argument("--num_heads", type=int, default=4)
    authors.add_argument("--ff_dim", type=int, default=4)
    authors.add_argument("--num_transformer_blocks", type=int, default=4)
    authors.add_argument("--mlp_dropout", type=float, default=0.4)
    authors.add_argument("--dropout", type=float, default=0.25)
    authors.add_argument("--mlp_units", type=str, default="128_64")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="folder for the checkpoint and the outputs (default: ckpt/ in the authors' repository)")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--test-every", type=int, default=0,
                        help="score the official test set every N epochs, for the trace only "
                             "(0 = never; the authors' script scores the test set once, at the end)")
    parser.add_argument("--max-hours", type=float, default=None,
                        help="stop training after this many hours (checkpoint, save, run the final evaluation); "
                             "for deadline-bounded runs")
    parser.add_argument("--stop-epoch", type=int, default=None,
                        help="stop after this epoch while keeping --num_epochs (and so the authors' file names) at the "
                             "published 8000; rerun with a larger value to continue")
    parser.add_argument("--contamination-fraction", type=float, default=0.0,
                        help="test-inclusion control: add this stratified fraction of the official test rows (imputed "
                             "and scaled exactly like the published inputs) to the table before the 90/10 split, so "
                             "that they are part of training like any other row. The final evaluation then reports the "
                             "included and the excluded test rows separately. 0 = the authors' pipeline unchanged; "
                             "1 = all test rows.")
    parser.add_argument("--contamination-seed", type=int, default=None,
                        help="seed of the draw of test rows (default: --seed_number)")
    parser.add_argument("--steps-per-execution", type=int,
                        default=int(os.environ.get("APS_STEPS_PER_EXECUTION", "32")),
                        help="training batches per compiled Keras call (1 = the plain Keras loop)")
    parser.add_argument("--xla", type=int, default=int(os.environ.get("XLA_JIT", "1")),
                        help="1 = compile the training step with XLA (jit_compile=True), 0 = off")
    return parser.parse_args()


def split_evaluation(probabilities, y, rows):
    """Errors at the 0.5 cut-off on the test rows selected by the boolean mask rows (the key names are those of the
    reported files)."""
    decisions = (probabilities > 0.5).astype(int)
    fn = int(((y == 1) & (decisions == 0) & rows).sum())
    fp = int(((y == 0) & (decisions == 1) & rows).sum())
    n_failures = int(((y == 1) & rows).sum())
    n_healthy = int(((y == 0) & rows).sum())
    return {"n": int(rows.sum()), "fn": fn, "fp": fp, "fn_rate": fn / max(n_failures, 1),
            "fp_rate": fp / max(n_healthy, 1), "cost_at_0.5": COST_OF_MISSED_FAILURE * fn + COST_OF_FALSE_ALARM * fp}


def write_json(path, content, indent=None):
    with open(path, "w") as handle:
        json.dump(content, handle, indent=indent)


def main():
    args = parse_arguments()
    # resolved before the change of folder below, so that a relative path means the folder the command was run in
    checkpoint_dir = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else AUTHORS_REPOSITORY / "ckpt"
    os.chdir(AUTHORS_REPOSITORY)        # the authors' code uses relative CSV paths
    import random
    import tensorflow as tf
    from focal_loss import BinaryFocalLoss
    from sklearn.model_selection import train_test_split
    from tensorflow import keras
    authors_code = load_authors_module()

    seed = args.seed_number
    mlp_units = [int(units) for units in args.mlp_units.split("_")]
    # the authors' run name ("fname" in TrainingCode.py), which names their CSV log and their model folders
    run_file_name = ("mlp" + args.mlp_units + "blocks" + str(args.num_transformer_blocks)
                     + "heads" + str(args.num_heads) + "size" + str(args.head_size) + "ffdim" + str(args.ff_dim)
                     + "mlpdropout" + str(args.mlp_dropout) + "dropout" + str(args.dropout)
                     + "alpha" + str(args.pos_weight) + "gamma" + str(args.gamma) + "lr" + str(args.learning_rate)
                     + "batch" + str(args.batch_size) + "epochs" + str(args.num_epochs) + "run")
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Inputs: the resampled training table and the test table, imputed and scaled as published (the authors' files).
    X_resampled = pd.read_csv(RESAMPLED_TABLE_SCALED)
    y_resampled = pd.read_csv(RESAMPLED_LABELS)["class"].ravel()
    X_test = pd.read_csv(TEST_TABLE_SCALED)
    y_test = pd.read_csv(TEST_LABELS)["class"].ravel()
    # Test-inclusion control: a stratified share of the test rows is appended to the resampled table before the split.
    contaminating_rows = np.array([], dtype=int)
    if args.contamination_fraction > 0:
        contamination_seed = args.contamination_seed if args.contamination_seed is not None else seed
        if args.contamination_fraction >= 1:          # every official test row
            contaminating_rows = np.arange(len(y_test))
        else:
            from sklearn.model_selection import StratifiedShuffleSplit
            splitter = StratifiedShuffleSplit(n_splits=1, train_size=args.contamination_fraction,
                                              random_state=contamination_seed)
            contaminating_rows = np.sort(next(splitter.split(np.zeros(len(y_test)), y_test))[0])
        X_resampled = pd.concat([X_resampled,
                                 X_test.iloc[contaminating_rows].set_axis(X_resampled.columns, axis=1)],
                                ignore_index=True)
        y_resampled = np.concatenate([y_resampled, y_test[contaminating_rows]])
        print(f"CONTROL: {len(contaminating_rows)} official test rows ({args.contamination_fraction:.0%}, "
              f"{int(y_test[contaminating_rows].sum())} positives) added to the table before the 90/10 split",
              flush=True)
    # the authors' stratified 90/10 split into training and validation rows, and their input shape (rows, attributes, 1)
    X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.1, stratify=y_resampled,
                                                      random_state=seed)
    X_train = np.array(X_train).reshape((len(X_train), X_train.shape[1], 1))
    X_val = np.array(X_val).reshape((len(X_val), X_val.shape[1], 1))
    X_test_array = np.array(X_test).reshape((len(X_test), X_test.shape[1], 1))

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    np.save(checkpoint_dir / "contaminating_test_rows.npy", contaminating_rows)
    test_row_seen = np.zeros(len(y_test), dtype=bool)
    test_row_seen[contaminating_rows] = True
    # Build the authors' model, or resume it, with its optimiser state, from the last checkpoint of this run.
    state_file = checkpoint_dir / "state.json"
    model_file = checkpoint_dir / "last_model.keras"
    initial_epoch = 0
    if state_file.exists() and model_file.exists():
        with open(state_file) as handle:
            initial_epoch = json.load(handle)["epoch"]
        model = keras.models.load_model(model_file, custom_objects={
            "cost_sensitive_metric_dl": authors_code.cost_sensitive_metric_dl, "BinaryFocalLoss": BinaryFocalLoss})
        print(f"resumed from epoch {initial_epoch}", flush=True)
        # steps_per_execution is not stored in the saved model; the optimiser state is kept, nothing is recompiled
        if args.steps_per_execution > 1:
            try:
                model._configure_steps_per_execution(args.steps_per_execution)
            except Exception as error:  # noqa: BLE001
                print(f"steps_per_execution not applied after resume ({error}); continuing with the plain loop",
                      flush=True)
        # the jit setting is a property of the compiled model; setting it rebuilds the train function only
        try:
            model.jit_compile = bool(args.xla)
        except Exception as error:  # noqa: BLE001
            print(f"jit_compile not applied after resume ({error}); continuing as saved", flush=True)
    else:
        model = authors_code.build_model(input_shape=X_train.shape[1:], head_size=args.head_size,
                                         num_heads=args.num_heads, ff_dim=args.ff_dim,
                                         num_transformer_blocks=args.num_transformer_blocks, mlp_units=mlp_units,
                                         mlp_dropout=args.mlp_dropout, dropout=args.dropout)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                      loss=BinaryFocalLoss(pos_weight=args.pos_weight, gamma=args.gamma),
                      metrics=[authors_code.cost_sensitive_metric_dl],
                      steps_per_execution=max(1, args.steps_per_execution), jit_compile=bool(args.xla))
    print(f"parameters: {model.count_params()}  steps_per_execution: {args.steps_per_execution}  "
          f"xla: {bool(args.xla)}", flush=True)

    # Callbacks of this script: checkpoints for resuming, staging (stop epoch, time budget) and the test-cost trace.
    training_started = time.time()
    authors_csv_log = Path("./CSV") / (run_file_name + ".csv")

    def save_checkpoint(model_to_save, epoch):
        model_to_save.save(model_file)
        write_json(state_file, {"epoch": epoch, "fname": run_file_name})

    class CheckpointStopAndTrace(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            finished_epochs = epoch + 1
            budget_reached = args.max_hours is not None and (time.time() - training_started) / 3600 > args.max_hours
            stop_epoch_reached = args.stop_epoch is not None and finished_epochs >= args.stop_epoch
            if budget_reached or stop_epoch_reached:
                reason = f"time budget of {args.max_hours} h" if budget_reached else f"stop epoch {args.stop_epoch}"
                print(f"  {reason} reached at epoch {finished_epochs}; stopping", flush=True)
                self.model.stop_training = True
                save_checkpoint(self.model, finished_epochs)
            if finished_epochs % args.checkpoint_every == 0 or finished_epochs == args.num_epochs:
                save_checkpoint(self.model, finished_epochs)
                if authors_csv_log.exists():
                    shutil.copy(authors_csv_log, checkpoint_dir / "training_log.csv")
            if args.test_every and finished_epochs % args.test_every == 0:
                probabilities = self.model.predict(X_test_array, batch_size=256, verbose=0).ravel()
                decisions = (probabilities > 0.5).astype(int)
                fn = int(((y_test == 1) & (decisions == 0)).sum())
                fp = int(((y_test == 0) & (decisions == 1)).sum())
                trace_entry = {"epoch": finished_epochs,
                               "test_cost_at_0.5": COST_OF_MISSED_FAILURE * fn + COST_OF_FALSE_ALARM * fp,
                               "fn": fn, "fp": fp, "time": time.time()}
                if len(contaminating_rows):
                    trace_entry["seen"] = split_evaluation(probabilities, y_test, test_row_seen)
                    trace_entry["unseen"] = split_evaluation(probabilities, y_test, ~test_row_seen)
                with open(checkpoint_dir / "test_trace.jsonl", "a") as handle:
                    handle.write(json.dumps(trace_entry) + "\n")
                print(f"  [diagnostic] epoch {finished_epochs}: official test cost at 0.5 = "
                      f"{COST_OF_MISSED_FAILURE * fn + COST_OF_FALSE_ALARM * fp} (FN {fn}, FP {fp})", flush=True)

    # The authors' callbacks: ModelCheckpoint keeps the model of the epoch with the lowest validation cost, and
    # CSVLogger writes their per-epoch log.
    os.makedirs("./CSV", exist_ok=True)
    os.makedirs("./Model", exist_ok=True)
    keras_3 = str(getattr(keras, "__version__", "2")).startswith("3")
    # Keras 3 insists on an extension; the authors used Keras 2 SavedModel folders
    extension = ".keras" if keras_3 else ""
    best_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="./Model/" + run_file_name + "epoch{epoch:01d}" + extension, monitor="val_cost_sensitive_metric_dl",
        verbose=0, save_best_only=True, save_weights_only=False, mode="min", save_freq="epoch")
    csv_logger = keras.callbacks.CSVLogger(str(authors_csv_log), separator=",", append=initial_epoch > 0)
    # Training, from the first epoch or from the resumed one, until --num_epochs or until a callback stops it.
    if args.stop_epoch is not None and initial_epoch >= args.stop_epoch:
        print(f"already at epoch {initial_epoch} >= stop epoch {args.stop_epoch}; evaluating only", flush=True)
    else:
        model.fit(X_train, y_train, epochs=args.num_epochs, initial_epoch=initial_epoch, batch_size=args.batch_size,
                  validation_data=(X_val, y_val),
                  callbacks=[csv_logger, best_checkpoint_callback, CheckpointStopAndTrace()], verbose=2)
    model.save("./Model/" + run_file_name + extension)

    # final evaluation as in the authors' test(): training and test cost at the 0.5 cut-off
    X_training_rows = pd.read_csv(TRAINING_TABLE_SCALED)
    y_training_rows = pd.read_csv(TRAINING_LABELS)["class"].ravel()
    training_probabilities = model.predict(
        np.array(X_training_rows).reshape((len(X_training_rows), X_training_rows.shape[1], 1)),
        batch_size=256, verbose=0).ravel()
    test_probabilities = model.predict(X_test_array, batch_size=256, verbose=0).ravel()
    evaluation = {}
    for name, y, probabilities in (("train", y_training_rows, training_probabilities),
                                   ("test", y_test, test_probabilities)):
        decisions = (probabilities > 0.5).astype(int)
        fn = int(((y == 1) & (decisions == 0)).sum())
        fp = int(((y == 0) & (decisions == 1)).sum())
        evaluation[name] = {"cost_at_0.5": COST_OF_MISSED_FAILURE * fn + COST_OF_FALSE_ALARM * fp, "fn": fn, "fp": fp}
        # the wording of the authors' own message
        print(f"{name} set: Based on standard threshold: Total cost is: "
              f"{COST_OF_MISSED_FAILURE * fn + COST_OF_FALSE_ALARM * fp} (FN {fn}, FP {fp})", flush=True)
    if state_file.exists():
        with open(state_file) as handle:
            evaluation["epochs_run"] = int(json.load(handle)["epoch"])
    else:
        evaluation["epochs_run"] = args.num_epochs
    if len(contaminating_rows):
        evaluation["contamination"] = {
            "frac": args.contamination_fraction, "n_included": int(len(contaminating_rows)),
            "included_test_rows": split_evaluation(test_probabilities, y_test, test_row_seen),
            "excluded_test_rows": split_evaluation(test_probabilities, y_test, ~test_row_seen)}
        print("CONTROL  included test rows:", evaluation["contamination"]["included_test_rows"], flush=True)
        print("CONTROL  excluded test rows:", evaluation["contamination"]["excluded_test_rows"], flush=True)
    # The best-validation checkpoint that the authors' ModelCheckpoint callback kept (save_best_only on the validation
    # cost metric). The paper does not say whether the reported costs come from it or from the final-epoch model, so
    # both are scored.
    checkpoint_folders = sorted(glob.glob("./Model/" + run_file_name + "epoch*"),
                                key=lambda path: int(re.findall(r"epoch(\d+)", path)[-1]))
    best_checkpoint = None
    try:   # the global minimum of the logged validation metric (the callback restarts its 'best' after a resume)
        authors_log = pd.read_csv(authors_csv_log)
        # CSVLogger counts the epochs from 0, the checkpoint names from 1
        best_epoch = int(authors_log.loc[authors_log["val_cost_sensitive_metric_dl"].idxmin(), "epoch"]) + 1
        candidate = "./Model/" + run_file_name + f"epoch{best_epoch}" + extension
        best_checkpoint = candidate if os.path.exists(candidate) else None
    except Exception:  # noqa: BLE001
        pass
    if best_checkpoint is None and checkpoint_folders:
        best_checkpoint = checkpoint_folders[-1]
    if best_checkpoint:
        try:
            best_model = keras.models.load_model(best_checkpoint, compile=False, custom_objects={
                "cost_sensitive_metric_dl": authors_code.cost_sensitive_metric_dl, "BinaryFocalLoss": BinaryFocalLoss})
            best_probabilities = best_model.predict(X_test_array, batch_size=256, verbose=0).ravel()
            decisions = (best_probabilities > 0.5).astype(int)
            fn = int(((y_test == 1) & (decisions == 0)).sum())
            fp = int(((y_test == 0) & (decisions == 1)).sum())
            evaluation["test_best_ckpt"] = {"epoch": int(re.findall(r"epoch(\d+)", best_checkpoint)[-1]),
                                            "cost_at_0.5": COST_OF_MISSED_FAILURE * fn + COST_OF_FALSE_ALARM * fp,
                                            "fn": fn, "fp": fp}
            if len(contaminating_rows):
                evaluation["test_best_ckpt"]["included_test_rows"] = split_evaluation(
                    best_probabilities, y_test, test_row_seen)
                evaluation["test_best_ckpt"]["excluded_test_rows"] = split_evaluation(
                    best_probabilities, y_test, ~test_row_seen)
            np.save(checkpoint_dir / "test_probabilities_best_checkpoint.npy", best_probabilities)
            print(f"best-validation checkpoint (epoch {evaluation['test_best_ckpt']['epoch']}): test cost at 0.5 = "
                  f"{COST_OF_MISSED_FAILURE * fn + COST_OF_FALSE_ALARM * fp} (FN {fn}, FP {fp})", flush=True)
        except Exception as error:  # noqa: BLE001
            print("could not score the best-validation checkpoint:", error, flush=True)
    # Outputs of the stage (see the module docstring), then the authors' own test().
    write_json(checkpoint_dir / "final_evaluation.json", evaluation, indent=1)
    np.save(checkpoint_dir / "test_probabilities_final.npy", test_probabilities)
    np.save(checkpoint_dir / "training_probabilities_final.npy", training_probabilities)
    if not keras_3:
        try:
            authors_code.test(run_file_name)    # the authors' own final evaluation (prints the same costs, plus plots)
        except Exception as error:  # noqa: BLE001  (plotting back-ends vary between machines)
            print("authors' test() could not run here:", error, flush=True)


if __name__ == "__main__":
    main()
