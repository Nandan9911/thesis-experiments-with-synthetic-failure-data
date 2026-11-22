#!/usr/bin/env python3
"""Job queue for the long experiments of the thesis, shared by several Google Colab sessions through Google Drive.

Most experiments of the thesis ran on a two-core CPU machine. The jobs that need a GPU (the retraining of the
cost-sensitive Transformer of Beikmohammadi et al. and its PyTorch port in RQ1b, the neural generators and the
in-context classifier of RQ2, most of RQ3) and a few long CPU jobs ran on Google Colab instead, in several sessions at
the same time (runtimes with an A100 GPU, an L4 GPU or CPUs only). A Colab runtime ends after at most 24 hours, and
earlier when it is disconnected, while single jobs ran for more than a day (an 8,000-epoch run of the authors' training
code). The only storage that every runtime can see is a folder in Google Drive, so the sessions were coordinated through
files in that folder, called the store below. Every session runs this script, which repeatedly takes the next job it may
run from the job list, runs it as a subprocess and records the outcome in the store. Long jobs save checkpoints in the
store, so a job whose runtime ends is taken up again by the next session that can run it and continues from its last
checkpoint.

    python cloud_runs/queue_runner.py --store /content/drive/MyDrive/aps_thesis --session l4-1 --lanes gpu_light,cpu --slots 2
    python cloud_runs/queue_runner.py --store /content/drive/MyDrive/aps_thesis --status

Run it from the repository root (the notebooks in cloud_runs/colab/ do so). Jobs run in the repository root as well,
so their commands read as they would on a local machine (python -m experiments....). cloud_runs/README.md explains
how a session is started; cloud_runs/build_job_list.py writes the job list of the thesis runs.

Store layout
  job_list.jsonl                    optional; when present it is used instead of cloud_runs/job_list.jsonl, so that
                                    jobs can be added or changed while sessions run
  inputs/authors_exact_inputs.zip   the six input tables of the training code released by Beikmohammadi et al. (made by
                                    the job prepare_authors_exact_inputs, or uploaded); the retraining jobs wait for it
  inputs/data/                      optional copies of the two UCI data files (otherwise each session downloads them)
  state/done/<job>.json             finished jobs
  state/failed/<job>.json           number of failed attempts of a job and its last exit code
  state/locks/<job>.<session>.json  claim and heartbeat of a session that runs the job
  logs/<job>.log                    console output of every attempt of a job, copied every minute while it runs
  logs/environment_<session>.log    output of setup_environment.sh
  logs/queue_finished_<session>.txt, logs/waiting_<session>.txt   see "Ending a session" below
  results/<session>/                result files of the session's jobs, in the layout of the repository's results/
                                    folder; cloud_runs/merge_cloud_results.py merges them into results/
  ckpt/<job or run>/                checkpoints: the finished folds of an arm, the state of a Transformer run
  runs/<run>/                       training logs and best-validation models of the authors' retraining runs
  control/                          files that steer running sessions (see below)

Job fields (one JSON object per line of the job list; empty lines and lines that start with # are skipped)
  name                    unique; names the job's state, lock and log files and, for most jobs, its checkpoint folder
  lane                    a lane or a list of lanes (default "cpu"); a session runs only jobs of its own lanes
  priority                lower numbers run first (default 100); equal priorities in alphabetical order of the names
  command                 shell command, run in the repository root after the placeholders {STORE}, {JOB},
                          {SESSION}, {SEED} and {FULL_TRAINING_EPOCHS} have been replaced
  weight                  job slots the job occupies (default 1)
  max_attempts            how often a failing job is started (default 2)
  needs_jobs              names of jobs that must be finished first
  needs_file              a path under the store that must exist first
  done_when_arm_recorded  [arm, seed]: the job counts as finished as soon as a record of that arm and seed exists,
                          in results/classifier_arms/arm_records.jsonl of the repository or in the arm records of any
                          session in the store
  seed                    the job's seed (used only for {SEED})
  threads                 CPU threads for the job (default: the machine's cores divided by the session's slots)

Lanes group the jobs by the hardware they need; a session is started with the lanes its runtime can serve. The job
list of the thesis uses gpu_anchor (the four runs continued to the full epoch budget, taken first by the first two
A100 sessions), gpu_heavy (the other Transformer runs; A100), gpu_light (generators, TabICL, completion engines and
the cold-start curves that need a GPU; L4), cpu (tree arms, classical oversamplers and other CPU work; a CPU
runtime, or spare slots of an L4 session) and extra (reruns and additions at the end, run one at a time on an L4).

Claiming a job. Google Drive's file system offers no exclusive file creation: two sessions that create the same file
within Drive's synchronisation delay both succeed. Every session therefore writes its own lock file and decides a
claim only after listing the lock files of the job twice, CLAIM_WAIT_SECONDS apart; the earliest claim wins, equal
claim times go to the alphabetically first session. A running job refreshes its lock (the heartbeat) every
HEARTBEAT_SECONDS and checks the claim again, and a copy that finds an earlier claim of another session stops itself.
A lock whose heartbeat is older than STALE_LOCK_SECONDS belongs to a runtime that has ended; the job can then be
claimed again. Jobs are started in priority order; when the next job that this session could start does not fit
into its free slots, the runner waits for slots rather than start a lighter job of lower priority.

Resumption. A job that fails is started again until it has used max_attempts attempts. A job whose runtime ends
leaves a stale lock and is claimed again later, by any session with a matching lane. The jobs resume on their own
from the store: an experiment arm caches every finished fold in ckpt/<job>/ (experiments/run_experiment_arm.py, which
the environment variables APS_CLOUD_STORE and APS_JOB_NAME switch on), and the Transformer runs keep their
checkpoints in ckpt/<run>/. A job stopped through a control file or by stopping the runner stays pending and is not
counted as a failed attempt.

Control files in <store>/control/, read by every runner once per pass (about once a minute):
  kill_<job>             stop that job in whichever session runs it; it stays pending and is claimed again later
  kill_<job>__<session>  the same, acted on by that session only
  drain_<session>        claim no new jobs in that session and exit when its running jobs have finished
  drain_all              the same for every session
  config_<session>.json  e.g. {"slots": 2, "lanes": ["gpu_heavy"], "threads_per_job": 3}: changes the slots, the lanes
                         and the CPU threads of later jobs of a running session (read again whenever the file changes)
  keep_<session>         read by the notebook, not by the runner: keeps the runtime after the queue has finished

Result files. Every job runs in the repository, so its outputs land in results/ as when the scripts are run by hand.
When the job ends, every file under results/ that changed while it ran is copied to the same relative path under
<store>/results/<session>/. Record files (*.jsonl) are not copied whole, because a fresh runtime starts from the
repository's copy of them and would overwrite the records of an earlier runtime: only the lines added while the job
ran are appended, and only if the store's copy lacks them. The arm runner, the cold-start study and the completion
benchmark do not need that step: the runner points their record files at the session's folder in the store
(APS_ARM_RECORDS_FILE, APS_COLD_START_RECORDS_FILE, APS_COMPLETION_RECORDS_FILE), so their records reach the store as
soon as they are written.

Environment of a job: APS_CLOUD_STORE, APS_JOB_NAME and APS_SESSION_NAME, the three record files above, and one CPU
thread budget (APS_CPU_THREADS_PER_MODEL, read by experiments/common/classifiers.py, and the OpenMP, MKL and OpenBLAS
thread counts), so that the concurrent jobs of a session do not compete for the cores.

Ending a session. A runner exits when nothing is left that its lanes could run for --idle-exit-minutes, or when it
has been drained. A runner that finds fresh locks of its own session written by another machine (the session's
previous runtime is still alive, for example a notebook opened again without deleting the old runtime) waits up to
SAME_SESSION_GIVE_UP_SECONDS for them to go stale, writing logs/waiting_<session>.txt, and then exits, so that the
same job never runs twice into the same checkpoint folder. In all three cases the runner writes
logs/queue_finished_<session>.txt and the local marker cache/queue_session_finished.txt, which the last cell of the
notebook reads before it releases the runtime.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

CLOUD_RUNS_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = CLOUD_RUNS_DIR.parent
# The tools in cloud_runs/ are run as scripts; the repository root goes on the import path so that the result
# locations come from experiments/common/paths.py like everywhere else.
sys.path.insert(0, str(REPOSITORY_ROOT))
from experiments.common.paths import (ARM_RECORDS_FILE, CACHE_DIR, COMPLETION_AND_COLD_START_RESULTS_DIR,  # noqa: E402
                                      RESULTS_DIR)

JOB_LIST_FILE = CLOUD_RUNS_DIR / "job_list.jsonl"
LOCAL_JOB_LOGS_DIR = CACHE_DIR / "queue_job_logs"          # working copies of the job logs on the runtime's disk
FINISHED_MARKER = CACHE_DIR / "queue_session_finished.txt"  # read by cloud_runs/colab/colab_bootstrap.py

# record files that a session writes straight into its folder in the store, relative to results/
ARM_RECORDS_RELATIVE = ARM_RECORDS_FILE.relative_to(RESULTS_DIR)
COLD_START_RECORDS_RELATIVE = (COMPLETION_AND_COLD_START_RESULTS_DIR / "cold_start_runs.jsonl").relative_to(RESULTS_DIR)
COMPLETION_RECORDS_RELATIVE = (COMPLETION_AND_COLD_START_RESULTS_DIR / "completion_benchmark.jsonl").relative_to(
    RESULTS_DIR)

HEARTBEAT_SECONDS = 60                  # how often a running job refreshes its lock
STALE_LOCK_SECONDS = 25 * 60            # a lock older than this belongs to a runtime that has ended
CLAIM_WAIT_SECONDS = 30                 # wait between the lock listings of a claim (Drive shows other sessions'
                                        # files only after some seconds)
SAME_SESSION_FRESH_SECONDS = 5 * 60     # a lock of this session refreshed this recently by another machine means
                                        # that the session's previous runtime is still running there
SAME_SESSION_GIVE_UP_SECONDS = 20 * 60  # ... and if it still runs after this long, the new runtime exits
HOST = socket.gethostname()             # one Colab runtime = one host name

DEFAULT_LANE = "cpu"
DEFAULT_PRIORITY = 100
DEFAULT_MAX_ATTEMPTS = 2


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def read_json(path: Path, default=None):
    """Content of a JSON file, or default when it is missing or unreadable (Drive can show half-synchronised files)."""
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return default


def write_json(path: Path, content) -> None:
    """Write a JSON file through a temporary file and a rename, so that no reader sees it half-written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp{uuid.uuid4().hex[:6]}")
    temporary.write_text(json.dumps(content, indent=1))
    os.replace(temporary, path)


def read_record_lines(path: Path) -> list[str]:
    """The non-empty lines of a JSONL file, stripped (an empty list if the file cannot be read)."""
    try:
        with open(path) as handle:
            return [line.strip() for line in handle if line.strip()]
    except OSError:
        return []


def retry_on_drive_error(function, *arguments, attempts: int = 4, wait_seconds: float = 20):
    """Call function(*arguments), trying again after a pause when it raises (Drive has short outages)."""
    for attempt in range(1, attempts + 1):
        try:
            return function(*arguments)
        except Exception as error:  # noqa: BLE001
            log(f"{getattr(function, '__name__', 'write')} failed ({type(error).__name__}: {error}); attempt {attempt}")
            time.sleep(wait_seconds)
    return None


def recorded_arm_seed_pairs(store: Path) -> set[tuple[str, int]]:
    """(arm, seed) pairs that have a record: the repository's own arm records and those of every session in the store."""
    record_files = [ARM_RECORDS_FILE] + sorted((store / "results").glob(f"*/{ARM_RECORDS_RELATIVE}"))
    pairs = set()
    for path in record_files:
        for line in read_record_lines(path):
            try:
                record = json.loads(line)
                pairs.add((record.get("arm"), int(record.get("seed", -1))))
            except (ValueError, TypeError):
                continue
    return pairs


def results_snapshot() -> tuple[dict[Path, float], dict[Path, set[str]]]:
    """Modification time of every file under results/ and the lines of every record file, taken when a job starts."""
    modification_times, record_lines = {}, {}
    if RESULTS_DIR.exists():
        for path in RESULTS_DIR.rglob("*"):
            if path.is_file():
                modification_times[path] = path.stat().st_mtime
                if path.suffix == ".jsonl":
                    record_lines[path] = set(read_record_lines(path))
    return modification_times, record_lines


def append_new_record_lines(local_file: Path, lines_before_job: set[str], store_file: Path) -> None:
    """Append to store_file the lines that local_file gained during the job and that store_file does not hold yet."""
    added_lines = [line for line in read_record_lines(local_file) if line not in lines_before_job]
    if not added_lines:
        return
    present = set(read_record_lines(store_file))
    missing = [line for line in added_lines if line not in present]
    if missing:
        store_file.parent.mkdir(parents=True, exist_ok=True)
        with open(store_file, "a") as handle:
            handle.write("".join(line + "\n" for line in missing))


class QueueRunner:
    """The queue loop of one session: claims jobs from the job list, runs them and records their outcome."""

    def __init__(self, store: Path, lanes: list[str], slots: int, session: str, full_training_epochs: int,
                 idle_exit_minutes: float, threads_per_job: int | None = None):
        self.store = store
        self.lanes = lanes
        self.slots = slots
        self.session = session
        self.full_training_epochs = full_training_epochs
        self.idle_exit_minutes = idle_exit_minutes
        self.threads_per_job_setting = threads_per_job
        self.session_results_dir = store / "results" / session
        self.running: dict[str, dict] = {}      # job name -> process, job, start time, "stopped" flag
        self.running_lock = threading.Lock()
        self.claim_times: dict[str, float] = {}
        self.stop_requested = False
        self.draining = False
        self.config_file_mtime = None
        self.idle_since = None

    # ---- the job list ------------------------------------------------------------------------------------------
    def load_jobs(self) -> list[dict]:
        """The jobs of the store's job list if there is one, else of cloud_runs/job_list.jsonl, in priority order."""
        job_list = self.store / "job_list.jsonl"
        if not job_list.exists():
            job_list = JOB_LIST_FILE
        jobs = []
        for line in job_list.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                jobs.append(json.loads(line))
        return sorted(jobs, key=lambda job: (job.get("priority", DEFAULT_PRIORITY), job["name"]))

    @staticmethod
    def job_lanes(job: dict) -> list[str]:
        lane = job.get("lane", DEFAULT_LANE)
        return lane if isinstance(lane, list) else [lane]

    # ---- state -------------------------------------------------------------------------------------------------
    def is_done(self, job: dict, recorded_arms: set, write_marker: bool = True) -> bool:
        done_file = self.store / "state" / "done" / f"{job['name']}.json"
        if done_file.exists():
            return True
        arm_and_seed = job.get("done_when_arm_recorded")
        if arm_and_seed and (arm_and_seed[0], int(arm_and_seed[1])) in recorded_arms:
            if write_marker:
                write_json(done_file, {"session": "record-found", "time": time.time()})
            return True
        return False

    def can_start(self, job: dict, recorded_arms: set, check_lanes: bool = True,
                  write_marker: bool = True) -> tuple[bool, str]:
        """Whether this session may start the job now, and if not, why."""
        if check_lanes and not set(self.job_lanes(job)) & set(self.lanes):
            return False, "other lanes"
        if self.is_done(job, recorded_arms, write_marker):
            return False, "done"
        failed = read_json(self.store / "state" / "failed" / f"{job['name']}.json", {})
        if failed.get("attempts", 0) >= job.get("max_attempts", DEFAULT_MAX_ATTEMPTS):
            return False, "failed"
        needed_file = job.get("needs_file")
        if needed_file and not (self.store / needed_file).exists():
            return False, f"needs {needed_file}"
        for needed_job in job.get("needs_jobs", []):
            if not (self.store / "state" / "done" / f"{needed_job}.json").exists():
                return False, f"needs {needed_job}"
        if self.fresh_lock_of_another_session(job["name"]):
            return False, "locked"
        return True, "ok"

    # ---- locks (see "Claiming a job" in the module docstring) ---------------------------------------------------
    def lock_files(self, job_name: str) -> list[tuple[Path, dict]]:
        """(path, content) of every readable lock file of a job, one per session that has claimed it."""
        found = []
        for path in (self.store / "state" / "locks").glob(f"{job_name}.*.json"):
            content = read_json(path)
            if isinstance(content, dict) and content.get("session"):
                found.append((path, content))
        return found

    def fresh_lock_of_another_session(self, job_name: str) -> dict | None:
        for _, lock in self.lock_files(job_name):
            if lock.get("session") != self.session and time.time() - lock.get("heartbeat", 0) < STALE_LOCK_SECONDS:
                return lock
        return None

    def own_lock_path(self, job_name: str) -> Path:
        return self.store / "state" / "locks" / f"{job_name}.{self.session}.json"

    def write_lock(self, job_name: str, claimed: float, pid: int | None = None) -> None:
        lock = {"session": self.session, "heartbeat": time.time(), "claimed": claimed, "host": HOST}
        if pid:
            lock["pid"] = pid
        write_json(self.own_lock_path(job_name), lock)

    def release_lock(self, job_name: str) -> None:
        try:
            self.own_lock_path(job_name).unlink()
        except Exception:  # noqa: BLE001  (already gone, or Drive unavailable: the lock goes stale on its own)
            pass

    def lost_claim(self, job_name: str, claimed: float) -> bool:
        """True when another session holds a fresh lock with an earlier claim (equal times: the first session name)."""
        for _, lock in self.lock_files(job_name):
            if lock.get("session") == self.session or time.time() - lock.get("heartbeat", 0) >= STALE_LOCK_SECONDS:
                continue
            other_claim = lock.get("claimed", 0)
            if other_claim < claimed or (other_claim == claimed and str(lock.get("session")) < self.session):
                return True
        return False

    def claim(self, job: dict) -> bool:
        name = job["name"]
        claimed = time.time()
        self.write_lock(name, claimed)
        for _ in range(2):      # two listings, CLAIM_WAIT_SECONDS apart
            time.sleep(CLAIM_WAIT_SECONDS)
            if self.lost_claim(name, claimed):
                self.release_lock(name)
                log(f"claim of {name} lost to another session")
                return False
        self.claim_times[name] = claimed
        return True

    # ---- running a job -----------------------------------------------------------------------------------------
    def expand_placeholders(self, job: dict, text: str) -> str:
        return (text.replace("{STORE}", str(self.store)).replace("{JOB}", job["name"])
                .replace("{SESSION}", self.session).replace("{SEED}", str(job.get("seed", 0)))
                .replace("{FULL_TRAINING_EPOCHS}", str(self.full_training_epochs)))

    def threads_per_job(self) -> int:
        """CPU threads a job may use: --threads-per-job or the config file if set, else the cores shared between the
        slots (three jobs that each start 12 OpenMP threads on 12 cores crawl; three jobs with 4 threads each do not)."""
        if self.threads_per_job_setting:
            return max(1, int(self.threads_per_job_setting))
        return max(1, (os.cpu_count() or 2) // max(1, self.slots))

    def job_environment(self, job: dict) -> dict[str, str]:
        threads = str(job.get("threads") or self.threads_per_job())
        return {"APS_CLOUD_STORE": str(self.store), "APS_JOB_NAME": job["name"], "APS_SESSION_NAME": self.session,
                "APS_ARM_RECORDS_FILE": str(self.session_results_dir / ARM_RECORDS_RELATIVE),
                "APS_COLD_START_RECORDS_FILE": str(self.session_results_dir / COLD_START_RECORDS_RELATIVE),
                "APS_COMPLETION_RECORDS_FILE": str(self.session_results_dir / COMPLETION_RECORDS_RELATIVE),
                "APS_CPU_THREADS_PER_MODEL": threads, "OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads,
                "OPENBLAS_NUM_THREADS": threads}

    def stop_process(self, entry: dict) -> None:
        """Stop a running job (its whole process group); it stays pending and is not counted as a failed attempt."""
        entry["stopped"] = True
        try:
            os.killpg(os.getpgid(entry["process"].pid), signal.SIGTERM)
        except Exception:  # noqa: BLE001  (the job may have ended in the meantime)
            pass

    def heartbeat(self, name: str, process: subprocess.Popen, started: float, local_log: Path,
                  store_log: Path) -> None:
        """Refresh the lock, give way to an earlier claim of another session, and copy the log to the store."""
        try:      # a Drive error must not end the thread that watches the job
            claimed = self.claim_times.get(name, started)
            self.write_lock(name, claimed, pid=process.pid)
            if self.lost_claim(name, claimed):
                log(f"DUPLICATE {name}: another session holds an earlier claim; stopping this copy")
                with self.running_lock:
                    entry = self.running[name]
                self.stop_process(entry)
            shutil.copyfile(local_log, store_log)
        except Exception as error:  # noqa: BLE001
            log(f"heartbeat of {name} failed ({type(error).__name__}: {error}); trying again in a minute")

    def run_job(self, job: dict) -> None:
        name = job["name"]
        started = time.time()
        LOCAL_JOB_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        local_log = LOCAL_JOB_LOGS_DIR / f"{name}.log"
        store_log = self.store / "logs" / f"{name}.log"
        if not local_log.exists() and store_log.exists():
            try:      # a fresh runtime continues the log of the earlier attempts instead of replacing it
                shutil.copyfile(store_log, local_log)
            except Exception:  # noqa: BLE001
                pass
        environment = dict(os.environ)
        environment.update(self.job_environment(job))
        command = self.expand_placeholders(job, job["command"])
        log(f"START {name}: {command[:160]}{'...' if len(command) > 160 else ''}")
        snapshot = results_snapshot()
        with open(local_log, "a") as log_handle:
            log_handle.write(f"\n===== {time.strftime('%F %T')} session {self.session} =====\n{command}\n")
            log_handle.flush()
            # a session of its own, so that a stop reaches every process the command starts
            process = subprocess.Popen(command, shell=True, cwd=str(REPOSITORY_ROOT), env=environment,
                                       stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True)
            with self.running_lock:
                self.running[name] = {"process": process, "job": job, "started": started, "stopped": False}
            last_heartbeat = 0.0
            while process.poll() is None:
                time.sleep(5)
                if time.time() - last_heartbeat > HEARTBEAT_SECONDS:
                    self.heartbeat(name, process, started, local_log, store_log)
                    last_heartbeat = time.time()
        exit_code = process.returncode
        try:
            shutil.copyfile(local_log, store_log)
        except Exception:  # noqa: BLE001  (the log was copied at the last heartbeat)
            pass
        copied = self.copy_new_results(snapshot)
        with self.running_lock:
            stopped = bool(self.running.get(name, {}).get("stopped"))
            self.running.pop(name, None)
        hours = (time.time() - started) / 3600
        if stopped:
            log(f"STOPPED {name} after {hours:.2f} h; it stays pending")
        elif exit_code == 0:
            retry_on_drive_error(write_json, self.store / "state" / "done" / f"{name}.json",
                                 {"session": self.session, "time": time.time(), "seconds": time.time() - started,
                                  "files_copied": copied})
            log(f"DONE  {name} in {hours:.2f} h ({copied} result files)")
        else:
            failed_file = self.store / "state" / "failed" / f"{name}.json"
            failed = read_json(failed_file, {"attempts": 0})
            failed.update({"attempts": failed.get("attempts", 0) + 1, "last_exit": exit_code, "session": self.session,
                           "time": time.time()})
            retry_on_drive_error(write_json, failed_file, failed)
            log(f"FAIL  {name} exit {exit_code} (attempt {failed['attempts']}); see logs/{name}.log")
        self.release_lock(name)

    def copy_new_results(self, snapshot: tuple[dict[Path, float], dict[Path, set[str]]]) -> int:
        """Copy the files under results/ that changed since the snapshot into the session's folder in the store
        (record files: only their new lines, see "Result files" in the module docstring). Returns the number of files."""
        modification_times, record_lines = snapshot
        copied = 0
        for path in sorted(RESULTS_DIR.rglob("*")):
            if not path.is_file():
                continue
            if path in modification_times and path.stat().st_mtime <= modification_times[path]:
                continue
            relative = path.relative_to(RESULTS_DIR)
            destination = self.session_results_dir / relative
            for _ in range(3):
                try:
                    if path.suffix == ".jsonl":
                        append_new_record_lines(path, record_lines.get(path, set()), destination)
                    else:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, destination)
                    copied += 1
                    break
                except Exception as error:  # noqa: BLE001
                    log(f"copy of {relative} failed ({type(error).__name__}); trying again")
                    time.sleep(20)
        return copied

    def slots_in_use(self) -> float:
        with self.running_lock:
            return sum(entry["job"].get("weight", 1) for entry in self.running.values())

    # ---- control files (see the module docstring) ---------------------------------------------------------------
    def apply_control_files(self) -> None:
        control_dir = self.store / "control"
        if not control_dir.exists():
            return
        for path in list(control_dir.glob("kill_*")):
            target = path.name[len("kill_"):]
            if "__" in target:                    # kill_<job>__<session>: only that session acts
                target, session = target.split("__", 1)
                if session != self.session:
                    continue
            with self.running_lock:
                entry = self.running.get(target)
            if entry is None:
                continue
            self.stop_process(entry)
            log(f"STOP  {target} (control file {path.name})")
            try:
                path.unlink()
            except Exception:  # noqa: BLE001
                pass
        if not self.draining and ((control_dir / f"drain_{self.session}").exists()
                                  or (control_dir / "drain_all").exists()):
            self.draining = True
            log("draining: no new jobs will be claimed; exiting when the running jobs have finished")
        config_file = control_dir / f"config_{self.session}.json"
        if config_file.exists():
            try:
                modification_time = config_file.stat().st_mtime
                if modification_time != self.config_file_mtime:
                    self.config_file_mtime = modification_time
                    config = read_json(config_file, {})
                    if config.get("slots") is not None:
                        self.slots = int(config["slots"])
                    if config.get("lanes"):
                        self.lanes = list(config["lanes"])
                    if config.get("threads_per_job") is not None:
                        self.threads_per_job_setting = int(config["threads_per_job"])
                    log(f"config: {self.slots} slots, lanes {self.lanes}, {self.threads_per_job()} threads per job")
            except Exception as error:  # noqa: BLE001
                log(f"config file ignored: {error}")

    # ---- the main loop -----------------------------------------------------------------------------------------
    def prepare_store_folders(self) -> None:
        for folder in ("state/done", "state/locks", "state/failed", "logs", "ckpt", "runs"):
            (self.store / folder).mkdir(parents=True, exist_ok=True)
        for relative in (ARM_RECORDS_RELATIVE, COLD_START_RECORDS_RELATIVE, COMPLETION_RECORDS_RELATIVE):
            (self.session_results_dir / relative).parent.mkdir(parents=True, exist_ok=True)

    def locks_of_this_session_from_other_machines(self) -> list[str]:
        """Fresh lock files of this session written by another host: the session's previous runtime still runs."""
        found = []
        for path in (self.store / "state" / "locks").glob(f"*.{self.session}.json"):
            lock = read_json(path)
            if lock and lock.get("host") != HOST and time.time() - lock.get("heartbeat", 0) < SAME_SESSION_FRESH_SECONDS:
                found.append(path.name)
        return found

    def mark_finished(self, reason: str) -> None:
        message = f"{time.strftime('%Y-%m-%d %H:%M:%S')} session {self.session} on {HOST}: {reason}\n"
        try:
            FINISHED_MARKER.parent.mkdir(parents=True, exist_ok=True)
            FINISHED_MARKER.write_text(message)
            (self.store / "logs" / f"queue_finished_{self.session}.txt").write_text(message)
        except Exception as error:  # noqa: BLE001
            log(f"could not write the finished marker: {error}")

    def run(self) -> None:
        self.prepare_store_folders()
        FINISHED_MARKER.unlink(missing_ok=True)
        waited_seconds = 0
        notice = self.store / "logs" / f"waiting_{self.session}.txt"
        while not self.stop_requested:
            busy = self.locks_of_this_session_from_other_machines()
            if not busy:
                notice.unlink(missing_ok=True)
                break
            if waited_seconds >= SAME_SESSION_GIVE_UP_SECONDS:
                # the old runtime is alive and working: this one would only use compute units while it waits
                reason = (f"another runtime of session {self.session} is still running ({', '.join(busy[:4])}); "
                          f"this runtime stops so that nothing runs twice")
                log(f"{reason}. To move the session to a fresh runtime, open the old notebook, use Runtime > "
                    f"Disconnect and delete runtime, then Run all here again.")
                notice.unlink(missing_ok=True)
                self.mark_finished(reason)
                return
            if waited_seconds % 300 == 0:
                log(f"session {self.session} still has live locks from another machine ({', '.join(busy[:4])}); "
                    f"waiting so that nothing runs twice. Delete the old runtime (Runtime > Disconnect and delete "
                    f"runtime) if it is still open.")
                try:
                    notice.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {HOST} waiting for {', '.join(busy[:4])}\n")
                except OSError:
                    pass
            time.sleep(60)
            waited_seconds += 60
        while not self.stop_requested:
            try:
                finished = self.run_once()
            except Exception as error:  # noqa: BLE001  (mostly Drive errors; the next pass usually works)
                log(f"queue loop error ({type(error).__name__}: {error}); trying again in 60 s")
                time.sleep(60)
                continue
            if finished:
                return
        self.stop_running_jobs()

    def run_once(self) -> bool:
        """One pass over the job list: start the jobs that can start and fit, then decide whether to exit (True)."""
        self.apply_control_files()
        recorded_arms = recorded_arm_seed_pairs(self.store)
        started = 0
        for job in self.load_jobs():
            if self.draining:
                break
            with self.running_lock:
                if job["name"] in self.running:
                    continue
            if not self.can_start(job, recorded_arms)[0]:
                continue
            if self.slots_in_use() + job.get("weight", 1) > self.slots:
                break       # wait for free slots rather than start a lighter job of lower priority
            if self.claim(job):
                threading.Thread(target=self.run_job, args=(job,), daemon=True).start()
                started += 1
                time.sleep(3)
        pending = [job["name"] for job in self.load_jobs()
                   if job["name"] not in self.running and self.can_start(job, recorded_arms)[0]]
        if self.draining and not self.running:
            log("drained; exiting")
            self.mark_finished("drained")
            return True
        if not self.running and not pending:
            self.idle_since = self.idle_since or time.time()
            if (time.time() - self.idle_since) / 60 >= self.idle_exit_minutes:
                log(f"nothing left to run for this session's lanes {self.lanes}; exiting")
                self.mark_finished(f"nothing left to run for lanes {self.lanes}")
                return True
        else:
            self.idle_since = None
        if started == 0:
            time.sleep(60)
        return False

    def stop_running_jobs(self) -> None:
        with self.running_lock:
            entries = list(self.running.items())
        for _, entry in entries:
            self.stop_process(entry)

    # ---- status ------------------------------------------------------------------------------------------------
    def print_status(self) -> None:
        """Print every job of the job list with its priority, lanes and state, then a count of the states."""
        recorded_arms = recorded_arm_seed_pairs(self.store)
        rows = []
        for job in self.load_jobs():
            name = job["name"]
            if self.is_done(job, recorded_arms, write_marker=False):
                state = "done"
            else:
                fresh_locks = [lock for _, lock in self.lock_files(name)
                               if time.time() - lock.get("heartbeat", 0) < STALE_LOCK_SECONDS]
                failed = read_json(self.store / "state" / "failed" / f"{name}.json", {})
                if fresh_locks:
                    state = "running in " + "+".join(sorted({str(lock.get("session")) for lock in fresh_locks}))
                elif failed.get("attempts", 0) >= job.get("max_attempts", DEFAULT_MAX_ATTEMPTS):
                    state = f"FAILED x{failed['attempts']}"
                else:
                    startable, reason = self.can_start(job, recorded_arms, check_lanes=False, write_marker=False)
                    state = "pending" if startable else f"waiting ({reason})"
            rows.append((job.get("priority", DEFAULT_PRIORITY), name, "/".join(self.job_lanes(job)), state))
        width = max((len(row[1]) for row in rows), default=4)
        for priority, name, lanes, state in rows:
            print(f"{priority:4d}  {name:{width}s}  {lanes:20s}  {state}")
        counts: dict[str, int] = {}
        for *_, state in rows:
            key = state.split(" ")[0] if not state.startswith("waiting") else "waiting"
            counts[key] = counts.get(key, 0) + 1
        print(f"{len(rows)} jobs: " + ", ".join(f"{count} {key}" for key, count in sorted(counts.items())))


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", required=True,
                        help="the shared store folder (on Colab a folder in the mounted Google Drive)")
    parser.add_argument("--lanes", default=DEFAULT_LANE,
                        help="comma-separated lanes whose jobs this session runs (default: cpu)")
    parser.add_argument("--slots", type=int, default=1,
                        help="job slots of this session; a job occupies as many slots as its weight (default: 1)")
    parser.add_argument("--session", default=None,
                        help="name of the session, e.g. a100-1; names its result folder in the store and its control "
                             "files (default: the lanes and a random suffix)")
    parser.add_argument("--full-training-epochs", type=int, default=8000,
                        help="value of the placeholder {FULL_TRAINING_EPOCHS}: the epoch budget of the retraining runs "
                             "that were continued to the published budget (default: 8000, as in the study of "
                             "Beikmohammadi et al.)")
    parser.add_argument("--threads-per-job", type=int, default=None,
                        help="CPU threads of each job (default: the machine's cores divided by the slots)")
    parser.add_argument("--idle-exit-minutes", type=float, default=30,
                        help="when nothing is left to run, keep looking for new jobs this long before exiting")
    parser.add_argument("--status", action="store_true", help="print the state of every job and exit")
    return parser.parse_args()


def main():
    args = parse_arguments()
    store = Path(args.store)
    lanes = [lane.strip() for lane in args.lanes.split(",") if lane.strip()]
    session = args.session or f"{'-'.join(lanes)}-{uuid.uuid4().hex[:4]}"
    runner = QueueRunner(store, lanes, args.slots, session, args.full_training_epochs, args.idle_exit_minutes,
                         args.threads_per_job)
    if args.status:
        runner.print_status()
        return
    log(f"session {session}: lanes {lanes}, {args.slots} slots, store {store}")

    def stop_runner(*_):
        runner.stop_requested = True
        runner.stop_running_jobs()

    signal.signal(signal.SIGTERM, stop_runner)
    signal.signal(signal.SIGINT, stop_runner)
    runner.run()


if __name__ == "__main__":
    main()
