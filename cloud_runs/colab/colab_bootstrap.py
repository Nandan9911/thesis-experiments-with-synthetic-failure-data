"""Start-up code of the Colab notebooks that run the queue sessions (see cloud_runs/README.md).

The notebook of a session (written by build_colab_notebooks.py) mounts Google Drive, puts a fresh copy of this
repository on the runtime's local disk, imports this module from it and calls

    settings = colab_bootstrap.prepare_session(SESSION, STORE)

which looks the session up in SESSIONS, prints what the runtime offers (GPU, cores, memory, disk), sets the
environment variables that the experiments read, starts the NVIDIA MPS daemon on GPU runtimes and reports which
inputs the store already holds. The notebook then runs cloud_runs/setup_environment.sh and the queue runner with the
session's lanes and slots. Its last cell calls release_runtime_if_queue_finished(...), which hands the runtime back
to Colab once the runner has reported that the session is finished (nothing left for its lanes, drained, or another
runtime of the same session still alive), so that a session that finishes at night stops using compute units
instead of idling until somebody disconnects it.

SESSIONS are the sessions of the thesis runs: three A100 sessions for the Transformer runs, two L4 sessions for the
GPU work of RQ2 and RQ3, and one CPU session. Nothing here runs on import, so build_colab_notebooks.py can read
SESSIONS outside Colab.
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass

# written by cloud_runs/queue_runner.py when the session has finished, relative to the repository root
FINISHED_MARKER = os.path.join("cache", "queue_session_finished.txt")


@dataclass(frozen=True)
class SessionSettings:
    runtime: str                # Colab runtime type: "A100", "L4" or "CPU" (Runtime > Change runtime type)
    lanes: str                  # comma-separated lanes, passed to queue_runner.py --lanes
    slots: int                  # job slots, passed to --slots
    idle_exit_minutes: int      # passed to --idle-exit-minutes

    @property
    def has_gpu(self) -> bool:
        return self.runtime != "CPU"


SESSIONS = {
    "a100-1": SessionSettings("A100", "gpu_anchor,gpu_heavy", 2, 20),
    "a100-2": SessionSettings("A100", "gpu_anchor,gpu_heavy", 2, 20),
    "a100-3": SessionSettings("A100", "gpu_heavy", 2, 20),
    # two slots per L4: three GPU jobs at once did not fit into its memory
    "l4-1": SessionSettings("L4", "gpu_light,cpu", 2, 30),
    "l4-2": SessionSettings("L4", "gpu_light", 2, 30),
    "cpu-1": SessionSettings("CPU", "cpu", 4, 120),
}


def prepare_session(session: str, store: str) -> SessionSettings:
    """Set up the environment of this runtime for the queue session `session` with the store folder `store`."""
    settings = SESSIONS[session]
    os.makedirs(f"{store}/inputs", exist_ok=True)
    environment = dict(APS_CLOUD_STORE=store, APS_SESSION_NAME=session, TF_FORCE_GPU_ALLOW_GROWTH="true",
                       TABPFN_ALLOW_CPU_LARGE_DATASET="1", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    if settings.has_gpu:
        # TabICL classifies on the GPU, 4,000 query rows at a time (its defaults are the CPU and 500 rows; see
        # experiments/rq2_survey_of_methods/tabicl_classifier.py)
        environment.update(APS_TABICL_DEVICE="cuda", APS_TABICL_QUERY_CHUNK_ROWS="4000")
    os.environ.update(environment)
    print(f"session {session}: {settings.runtime} runtime, lanes {settings.lanes}, {settings.slots} slots")
    print(subprocess.run("nvidia-smi -L || echo 'CPU runtime'; nproc; free -g | head -2; df -h /content | tail -1",
                         shell=True, capture_output=True, text=True).stdout)
    for relative_path, if_missing in (
            ("inputs/authors_exact_inputs.zip", "made by the job prepare_authors_exact_inputs; the retraining jobs wait"),
            ("inputs/data/aps_failure_training_set.csv", "downloaded from the UCI repository instead"),
            ("inputs/data/aps_failure_test_set.csv", "downloaded from the UCI repository instead")):
        present = os.path.exists(f"{store}/{relative_path}")
        print(f"{'present' if present else 'missing'}  {relative_path}{'' if present else f' ({if_missing})'}")
    # NVIDIA MPS lets the concurrent jobs of a session run their small GPU kernels side by side instead of taking turns
    # on the GPU (two or three training processes under MPS finished 25-30 % more epochs per hour than without it)
    if settings.has_gpu and os.path.exists("/opt/bin/nvidia-cuda-mps-control"):
        os.environ.update(CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps", CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log")
        os.makedirs("/tmp/nvidia-mps", exist_ok=True)
        os.makedirs("/tmp/nvidia-log", exist_ok=True)
        result = subprocess.run("/opt/bin/nvidia-cuda-mps-control -d", shell=True, capture_output=True, text=True)
        print("MPS daemon:", "started" if result.returncode == 0
              else f"not started ({(result.stderr or result.stdout).strip()[:200]})")
    return settings


def release_runtime_if_queue_finished(session: str, store: str, repository_dir: str) -> None:
    """Release this runtime if the queue runner has left its finished marker; keep it otherwise, or when the store
    holds control/keep_<session>."""
    marker = os.path.join(repository_dir, FINISHED_MARKER)
    if not os.path.exists(marker) or os.path.exists(f"{store}/control/keep_{session}"):
        print(f"runtime kept: the queue did not report that it finished (or control/keep_{session} exists)")
        return
    with open(marker) as handle:
        print(handle.read().strip(), "-> releasing this runtime so that it stops using compute units")
    from google.colab import drive, runtime
    try:
        drive.flush_and_unmount()          # every result file reaches Drive before the runtime goes away
    except Exception as error:  # noqa: BLE001
        print("flush:", error)
        time.sleep(60)
    runtime.unassign()
