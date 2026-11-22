#!/usr/bin/env python3
"""Merge the results of the Colab queue sessions, from a downloaded copy of the store, into results/.

The queue sessions (cloud_runs/queue_runner.py) leave their results in the store folder on Google Drive: every
session copies the result files of its jobs to results/<session>/, in the layout of the repository's results/
folder, and the retraining runs of the code released by Beikmohammadi et al. keep their outputs in ckpt/<run>/ (with
their training logs also in runs/<run>/). After downloading the store folder (for example with Google Drive for desktop,
or as a ZIP file from drive.google.com), this script reports what the store holds that results/ lacks and, with --apply,
adds it:

    python cloud_runs/merge_cloud_results.py --store <downloaded store folder>              # dry run: report only
    python cloud_runs/merge_cloud_results.py --store <downloaded store folder> --apply      # copy and append
    python cloud_runs/merge_cloud_results.py --store <folder> --results-dir <other results folder> --apply

Nothing that is already in results/ is overwritten or changed: a record whose key is present is skipped, a file
that exists is kept, and a JSON summary only gains the entries it does not have, so the earlier runs stay the
reference and the sessions fill the gaps. Files whose path contains "smoke_test" are ignored. What is merged:

  classifier_arms/arm_records.jsonl of each session
      -> results/classifier_arms/arm_records.jsonl: the records of (arm, seed) pairs that results/ lacks, marked with
         "source": "colab:<session>" (the records of a cloud session carry that field). A pair recorded more than
         once is taken from the first session, in alphabetical order, that also holds its probability files.
  classifier_arms/predicted_probabilities/<arm>/seed<k>_<kind>.npy
      -> the same path, for the records merged above only; probabilities of other reruns are not copied.
  classifier_arms/verification_reruns.jsonl
      -> results/classifier_arms/verification_reruns.jsonl (records that are not there yet).
  rq3_completion_and_cold_start/cold_start_runs*.jsonl
      -> results/rq3_completion_and_cold_start/cold_start_runs.jsonl, keyed on (generator, seed, labelled_failures),
         with the test probabilities cold_start_test_probabilities/<generator>_seed<k>_<n>_failures.npy of the
         merged records.
  rq3_completion_and_cold_start/completion_benchmark*.jsonl
      -> results/rq3_completion_and_cold_start/completion_benchmark.jsonl, keyed on (fold, mask, engine).
  any other *.jsonl file -> the same path, records that are not there yet (compared as whole records).
  rq2_survey_of_methods/synthetic_data_diagnostics.json, rq3_completion_and_cold_start/completion_memorisation_audit.json
      -> the entries (generators, engines) that the file in results/ does not have.
  every other file in one of the result folders (the PyTorch port's result files, the prototype demonstration, ...)
      -> the same path, if absent. Files outside those folders (for example an old-style record file at the top of a
         session folder) are listed and left alone.
  ckpt/<run>/ of every retraining run of the authors' code (a folder with a state.json)
      -> results/rq1_transformer_2026_reproduction/authors_code/retraining_runs/<run>/: every file except the model
         checkpoint last_model.keras; training_log.csv is taken from runs/<run>/CSV/ when the checkpoint folder has
         none. A folder with the file names of the earlier training script (final_eval.json) is reported and skipped:
         those runs are in results/ already.

Not merged: the job logs (logs/<job>.log; the console output of the prototype demonstration, for example, is the log
of the job prototype_demonstration), the other checkpoints (the folds of the arms, the PyTorch port's .pt files,
which evaluate_port_checkpoint.py reads directly) and the best-validation model folders in runs/<run>/Model/.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# The tools in cloud_runs/ are run as scripts; the repository root goes on the import path so that the result
# locations come from experiments/common/paths.py like everywhere else.
sys.path.insert(0, str(REPOSITORY_ROOT))
from experiments.common.paths import (ARM_PROBABILITIES_DIR, ARM_RECORDS_FILE, AUTHORS_CODE_RESULTS_DIR,  # noqa: E402
                                      CGAN_2019_RESULTS_DIR, CLASSIFIER_ARMS_DIR, COMPLETION_AND_COLD_START_RESULTS_DIR,
                                      HYPERPARAMETER_SEARCH_DIR, PRELIMINARY_CHECKS_DIR, RESULTS_DIR, SURVEY_RESULTS_DIR,
                                      TRANSFORMER_2026_RESULTS_DIR)

# locations relative to results/ (the same in results/ and in every results/<session>/ of the store)
ARM_RECORDS = ARM_RECORDS_FILE.relative_to(RESULTS_DIR)
ARM_PROBABILITIES = ARM_PROBABILITIES_DIR.relative_to(RESULTS_DIR)
VERIFICATION_RERUNS = (CLASSIFIER_ARMS_DIR / "verification_reruns.jsonl").relative_to(RESULTS_DIR)
COLD_START_RUNS = (COMPLETION_AND_COLD_START_RESULTS_DIR / "cold_start_runs.jsonl").relative_to(RESULTS_DIR)
COLD_START_PROBABILITIES = (COMPLETION_AND_COLD_START_RESULTS_DIR / "cold_start_test_probabilities").relative_to(
    RESULTS_DIR)
COMPLETION_BENCHMARK = (COMPLETION_AND_COLD_START_RESULTS_DIR / "completion_benchmark.jsonl").relative_to(RESULTS_DIR)
SUMMARY_FILES = [(SURVEY_RESULTS_DIR / "synthetic_data_diagnostics.json").relative_to(RESULTS_DIR),
                 (COMPLETION_AND_COLD_START_RESULTS_DIR / "completion_memorisation_audit.json").relative_to(RESULTS_DIR)]
RETRAINING_RUNS = (AUTHORS_CODE_RESULTS_DIR / "retraining_runs").relative_to(RESULTS_DIR)
# the folders of results/ that experiment scripts write into; other files of a session folder are not merged
RESULT_FOLDERS = {folder.relative_to(RESULTS_DIR).parts[0] for folder in (
    HYPERPARAMETER_SEARCH_DIR, PRELIMINARY_CHECKS_DIR, CLASSIFIER_ARMS_DIR, CGAN_2019_RESULTS_DIR,
    TRANSFORMER_2026_RESULTS_DIR, SURVEY_RESULTS_DIR, COMPLETION_AND_COLD_START_RESULTS_DIR)}
NOT_MERGED_FROM_CHECKPOINTS = {"last_model.keras"}


def read_records(path: Path) -> list[dict]:
    """The records of a JSONL file (unreadable lines are skipped)."""
    records = []
    with open(path) as handle:
        for line in handle:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"  skipped an unreadable line in {path}")
    return records


def whole_record_key(record: dict) -> str:
    return json.dumps(record, sort_keys=True)


def arm_key(record: dict):
    return (record.get("arm"), record.get("seed"))


def cold_start_key(record: dict):
    return (record.get("generator"), record.get("seed"), record.get("labelled_failures"))


def completion_key(record: dict):
    return (record.get("fold"), record.get("mask"), record.get("engine"))


class Merger:
    """Collects the planned actions (description, function) and carries them out with --apply."""

    def __init__(self, store: Path, results_dir: Path):
        self.store = store
        self.results_dir = results_dir
        self.sessions = sorted(path for path in (store / "results").glob("*") if path.is_dir())
        self.actions: list[tuple[str, object]] = []
        self.handled_files: set[Path] = set()     # session files that a specific rule has dealt with

    def session_files(self, pattern: str) -> list[tuple[str, Path]]:
        """(session, path) of the files that match pattern (relative to results/) in every session folder."""
        found = []
        for session_dir in self.sessions:
            for path in sorted(session_dir.glob(pattern)):
                if path.is_file() and "smoke_test" not in str(path.relative_to(session_dir)):
                    found.append((session_dir.name, path))
        return found

    def plan_copy(self, source: Path, destination: Path, description: str) -> bool:
        """Plan the copy of one file unless the destination exists; returns whether it was planned."""
        if destination.exists():
            return False

        def copy(source=source, destination=destination):
            if not destination.exists():                  # checked again: never overwrite
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        self.actions.append((description, copy))
        return True

    def plan_append(self, destination: Path, records: list[dict], description: str) -> None:
        def append(destination=destination, records=records):
            destination.parent.mkdir(parents=True, exist_ok=True)
            with open(destination, "a") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")
        self.actions.append((description, append))

    def merge_records(self, pattern: str, destination_relative: Path, key, label: str) -> list[dict]:
        """Records of the session files matching pattern whose key the destination file lacks; returns them."""
        destination = self.results_dir / destination_relative
        present = {key(record) for record in read_records(destination)} if destination.exists() else set()
        new, duplicates = {}, 0
        for session, path in self.session_files(pattern):
            self.handled_files.add(path)
            for record in read_records(path):
                record_key = key(record)
                if record_key in present:
                    duplicates += 1
                elif record_key not in new:
                    new[record_key] = (session, record)
        if duplicates:
            print(f"  {label}: {duplicates} records already in {destination_relative}, kept as they are")
        if new:
            self.plan_append(destination, [record for _, record in new.values()],
                             f"append {len(new)} records to {destination_relative}")
        return [dict(record, _session=session) for session, record in new.values()]

    # ---- the rules ---------------------------------------------------------------------------------------------
    def merge_arm_records(self) -> None:
        destination = self.results_dir / ARM_RECORDS
        present = {arm_key(record) for record in read_records(destination)} if destination.exists() else set()
        candidates, already_present = {}, set()      # (arm, seed) -> [(session, record), ...] in session order
        for session, path in self.session_files(str(ARM_RECORDS)):
            self.handled_files.add(path)
            for record in read_records(path):
                if str(record.get("arm", "")).startswith("smoke_test"):
                    continue
                key = arm_key(record)
                if key in present:
                    already_present.add(key)
                else:
                    candidates.setdefault(key, []).append((session, record))
        new, recorded_twice = {}, set()
        for (arm, seed), entries in candidates.items():
            # the record and its probability files must come from the same session: prefer one that has the files
            with_probabilities = [(session, record) for session, record in entries
                                  if (self.store / "results" / session / ARM_PROBABILITIES / arm
                                      / f"seed{seed}_out_of_fold.npy").exists()]
            session, record = (with_probabilities or entries)[0]
            new[(arm, seed)] = dict(record, source=f"colab:{session}")
            if len(entries) > 1:
                recorded_twice.add((arm, seed))
        if already_present:
            listed = sorted(already_present, key=str)
            print(f"  {len(listed)} (arm, seed) records already in results, kept as they are: {listed[:8]}"
                  f"{' ...' if len(listed) > 8 else ''}")
        if recorded_twice:
            print(f"  recorded more than once (the record of the first session that holds its probability files is "
                  f"taken): {sorted(recorded_twice, key=str)}")
        for (arm, seed), record in sorted(new.items(), key=str):
            test = record.get("test", {})
            print(f"  new arm record {arm} seed {seed}: out-of-fold cost {record.get('out_of_fold', {}).get('cost')}, "
                  f"test cost {test.get('cost')} (FN {test.get('fn')}, FP {test.get('fp')})  [{record['source']}]")
        if not new:
            return
        self.plan_append(destination, [new[key] for key in sorted(new, key=str)],
                         f"append {len(new)} records to {ARM_RECORDS}")
        for (arm, seed), record in sorted(new.items(), key=str):
            session_dir = self.store / "results" / record["source"].split(":", 1)[1]
            for kind in ("out_of_fold", "test", "test_full_refit"):
                relative = ARM_PROBABILITIES / arm / f"seed{seed}_{kind}.npy"
                if (session_dir / relative).exists():
                    self.handled_files.add(session_dir / relative)
                    self.plan_copy(session_dir / relative, self.results_dir / relative, f"copy {relative}")

    def merge_cold_start_runs(self) -> None:
        new = self.merge_records(f"{COLD_START_RUNS.parent}/cold_start_runs*.jsonl", COLD_START_RUNS, cold_start_key,
                                 "cold start")
        for record in new:
            test = record.get("test", {})
            print(f"  new cold-start record {record.get('generator')} seed {record.get('seed')}, "
                  f"{record.get('labelled_failures')} labelled failures: test cost {test.get('cost')}  "
                  f"[{record['_session']}]")
            name = f"{record['generator']}_seed{record['seed']}_{record['labelled_failures']}_failures.npy"
            source = self.store / "results" / record["_session"] / COLD_START_PROBABILITIES / name
            if source.exists():
                self.handled_files.add(source)
                self.plan_copy(source, self.results_dir / COLD_START_PROBABILITIES / name,
                               f"copy {COLD_START_PROBABILITIES / name}")

    def merge_completion_benchmark(self) -> None:
        new = self.merge_records(f"{COMPLETION_BENCHMARK.parent}/completion_benchmark*.jsonl", COMPLETION_BENCHMARK,
                                 completion_key, "completion benchmark")
        for record in new:
            print(f"  new completion record fold {record.get('fold')}, {record.get('mask')}, {record.get('engine')}: "
                  f"NRMSE {record.get('nrmse')}  [{record['_session']}]")

    def merge_verification_reruns(self) -> None:
        for record in self.merge_records(str(VERIFICATION_RERUNS), VERIFICATION_RERUNS, whole_record_key,
                                         "verification reruns"):
            print(f"  verification rerun {record.get('arm')} seed {record.get('seed')}: out-of-fold cost "
                  f"{record.get('out_of_fold', {}).get('cost')}, test cost {record.get('test', {}).get('cost')}  "
                  f"[{record['_session']}]")

    def merge_summary_files(self) -> None:
        for relative in SUMMARY_FILES:
            destination = self.results_dir / relative
            current = json.loads(destination.read_text()) if destination.exists() else {}
            additions = {}
            for session, path in self.session_files(str(relative)):
                self.handled_files.add(path)
                for key, value in json.loads(path.read_text()).items():
                    if key not in current and key not in additions:
                        additions[key] = value
                        print(f"  new entry '{key}' for {relative}  [{session}]")
            if additions:
                def add_entries(destination=destination, additions=additions):
                    content = json.loads(destination.read_text()) if destination.exists() else {}
                    for key, value in additions.items():
                        content.setdefault(key, value)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(json.dumps(content, indent=1))
                self.actions.append((f"add {len(additions)} entries to {relative}", add_entries))

    def merge_other_files(self) -> None:
        """Every session file that no rule above has dealt with: JSONL records by whole record, files if absent."""
        skipped, outside_layout, without_record = 0, [], []
        for session, path in self.session_files("**/*"):
            if path in self.handled_files:
                continue
            relative = path.relative_to(self.store / "results" / session)
            if len(relative.parts) < 2 or relative.parts[0] not in RESULT_FOLDERS:
                outside_layout.append(f"{session}/{relative}")
                continue
            if relative.is_relative_to(ARM_PROBABILITIES) or relative.is_relative_to(COLD_START_PROBABILITIES):
                if not (self.results_dir / relative).exists():
                    without_record.append(f"{session}/{relative}")
                continue          # copied together with their records only
            if path.suffix == ".jsonl":
                self.merge_records(str(relative), relative, whole_record_key, str(relative))
            elif not self.plan_copy(path, self.results_dir / relative, f"copy {relative}  [{session}]"):
                skipped += 1
        if skipped:
            print(f"  {skipped} other files already in results, kept as they are")
        if without_record:
            print(f"  probability files without a merged record, not copied: {without_record[:8]}"
                  f"{' ...' if len(without_record) > 8 else ''}")
        if outside_layout:
            print(f"  not in the layout of results/, so not merged: {outside_layout[:8]}"
                  f"{' ...' if len(outside_layout) > 8 else ''}")

    def merge_retraining_runs(self) -> None:
        for checkpoint_dir in sorted(path for path in (self.store / "ckpt").glob("*") if (path / "state.json").exists()):
            run = checkpoint_dir.name
            if (checkpoint_dir / "final_eval.json").exists():      # the file names of the earlier training script
                print(f"  retraining run {run}: written by an earlier version of the training script; not merged")
                continue
            destination_dir = self.results_dir / RETRAINING_RUNS / run
            files = {path.name: path for path in sorted(checkpoint_dir.iterdir())
                     if path.is_file() and path.name not in NOT_MERGED_FROM_CHECKPOINTS}
            authors_logs = sorted((self.store / "runs" / run / "CSV").glob("*.csv"))
            if "training_log.csv" not in files and authors_logs:
                files["training_log.csv"] = authors_logs[-1]
            print(f"  retraining run {run}: {self.describe_retraining_run(checkpoint_dir)}")
            planned = sum(self.plan_copy(source, destination_dir / name, f"copy {RETRAINING_RUNS / run / name}")
                          for name, source in sorted(files.items()))
            if planned < len(files):
                print(f"      {len(files) - planned} of its {len(files)} files already in results, kept as they are")

    @staticmethod
    def describe_retraining_run(checkpoint_dir: Path) -> str:
        parts = []
        state = json.loads((checkpoint_dir / "state.json").read_text())
        parts.append(f"epochs run {state.get('epoch')}")
        evaluation_file = checkpoint_dir / "final_evaluation.json"
        if evaluation_file.exists():
            evaluation = json.loads(evaluation_file.read_text())
            test = evaluation.get("test", {})
            parts.append(f"final evaluation after {evaluation.get('epochs_run')} epochs: test cost at 0.5 "
                         f"{test.get('cost_at_0.5')} (FN {test.get('fn')}, FP {test.get('fp')})")
            control = evaluation.get("contamination")
            if control:
                seen, unseen = control["included_test_rows"], control["excluded_test_rows"]
                parts.append(f"control {control['frac']:.0%}: false-alarm rate {seen['fp_rate']:.4%} on the test rows "
                             f"in training, {unseen['fp_rate']:.4%} on the others")
        trace_file = checkpoint_dir / "test_trace.jsonl"
        if trace_file.exists():
            trace = read_records(trace_file)
            if trace:
                parts.append(f"test trace of {len(trace)} points, last {trace[-1].get('test_cost_at_0.5')} at epoch "
                             f"{trace[-1].get('epoch')} (a diagnostic only)")
        return "; ".join(parts)

    def run(self, apply: bool) -> None:
        print(f"store {self.store}: sessions {[path.name for path in self.sessions]}")
        print(f"merging into {self.results_dir}")
        self.merge_arm_records()
        self.merge_verification_reruns()
        self.merge_cold_start_runs()
        self.merge_completion_benchmark()
        self.merge_summary_files()
        self.merge_other_files()
        self.merge_retraining_runs()
        print("\nactions:" if self.actions else "\nnothing new to merge")
        for description, action in self.actions:
            print("  -", description)
            if apply:
                action()
        if self.actions and not apply:
            print("(dry run; add --apply to carry them out)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", required=True, help="the downloaded store folder (it holds results/, ckpt/, runs/)")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR),
                        help="the results folder to merge into (default: results/ of this repository)")
    parser.add_argument("--apply", action="store_true", help="carry out the actions (default: report them only)")
    args = parser.parse_args()
    store = Path(args.store)
    if not (store / "results").is_dir() and not (store / "ckpt").is_dir():
        raise SystemExit(f"{store} does not look like a store folder (no results/ or ckpt/ in it)")
    Merger(store, Path(args.results_dir)).run(args.apply)


if __name__ == "__main__":
    main()
