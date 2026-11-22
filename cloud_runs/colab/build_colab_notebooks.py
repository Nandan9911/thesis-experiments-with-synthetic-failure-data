#!/usr/bin/env python3
"""Write the Colab notebook of each queue session, queue_session_<session>.ipynb (see cloud_runs/README.md).

All sessions use the same notebook; only the session name and the runtime type stored in the notebook's metadata
differ. The sessions, with their runtime type, lanes, job slots and idle limit, are listed in colab_bootstrap.py.
A notebook has five cells:
  1. instructions;
  2. the settings (session, store folder, repository address), the Google Drive mount, a fresh copy of the
     repository on the runtime's disk (cloned from GitHub, or copied from <store>/repository/), and
     colab_bootstrap.prepare_session();
  3. cloud_runs/setup_environment.sh;
  4. the queue runner with the session's lanes, slots and idle limit;
  5. colab_bootstrap.release_runtime_if_queue_finished(), which hands the runtime back once the queue has finished.
The repository address is a placeholder that the reader fills in. Mounting Drive and "Run all" remain manual steps:
Colab asks the account owner to allow the Drive access in a dialogue.

    python cloud_runs/colab/build_colab_notebooks.py --sessions l4-1           # the example kept in this folder
    python cloud_runs/colab/build_colab_notebooks.py --output-dir <folder>     # the notebooks of every session

The output depends only on this script and on the session table, so the example notebook can be compared with a
freshly written one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from colab_bootstrap import SESSIONS

HERE = Path(__file__).resolve().parent
REPOSITORY_URL_PLACEHOLDER = "https://github.com/<account>/scania-aps-synthetic-failure-data.git"

INSTRUCTIONS = """# Experiment queue of the thesis: session `{session}`

This notebook runs one session of the job queue in `cloud_runs/` of the repository (see `cloud_runs/README.md`). Before
*Run all*:

1. *Runtime → Change runtime type*: {runtime_choice}.
2. In the first code cell, replace the placeholder in `REPOSITORY_URL` by the address of the repository on GitHub (for
   a private repository, an address that contains an access token), or leave it as it is and put a copy of the
   repository into the store folder as `repository/`.
3. Change `STORE` if the store folder is not `My Drive/aps_thesis`.

Then *Run all*, and allow the access to Google Drive when Colab asks for it. The first code cell mounts Drive and
puts a fresh copy of the repository on the runtime, the second installs the environment (its output also goes to
`logs/environment_{session}.log` in the store), the third runs the queue until nothing is left for this session, and
the last one releases the runtime so that it stops using compute units.

To move the session to a fresh runtime (a runtime lives at most 24 hours), use *Runtime → Disconnect and delete
runtime* in the notebook that runs it now, then open this notebook and *Run all* again. The jobs continue from their
checkpoints in the store. If the old runtime is still alive, the new one waits up to 20 minutes and then gives up, so
that nothing runs twice.

The state of the queue can be read from any notebook or terminal that sees the store:
`python cloud_runs/queue_runner.py --store <store folder> --status`."""

SETTINGS_CELL = """SESSION = "{session}"   # {session_names} (see cloud_runs/colab/colab_bootstrap.py)
STORE = "/content/drive/MyDrive/aps_thesis"   # the shared store folder in Google Drive
# FILL IN the address of the repository on GitHub, or leave the placeholder and copy the repository to STORE/repository
REPOSITORY_URL = "{repository_url}"
REPOSITORY_DIR = "/content/scania-aps-synthetic-failure-data"

import os, shutil, subprocess, sys
from google.colab import drive
drive.mount("/content/drive")
os.chdir("/content")
shutil.rmtree(REPOSITORY_DIR, ignore_errors=True)   # a fresh copy of the repository for every Run all
if "<account>" not in REPOSITORY_URL:
    subprocess.run(["git", "clone", "--quiet", REPOSITORY_URL, REPOSITORY_DIR], check=True)
elif os.path.isdir(f"{{STORE}}/repository"):
    shutil.copytree(f"{{STORE}}/repository", REPOSITORY_DIR)
else:
    raise RuntimeError(f"fill in REPOSITORY_URL, or copy the repository to {{STORE}}/repository")
os.chdir(REPOSITORY_DIR)                             # the following cells run in the repository
sys.path.insert(0, f"{{REPOSITORY_DIR}}/cloud_runs/colab")
import colab_bootstrap
settings = colab_bootstrap.prepare_session(SESSION, STORE)
LANES, SLOTS, IDLE_EXIT_MINUTES = settings.lanes, settings.slots, settings.idle_exit_minutes"""

ENVIRONMENT_CELL = """# packages, data, the external repositories and the exact inputs (full output: logs/environment_<session>.log in the store)
!bash cloud_runs/setup_environment.sh 2>&1 | tail -60"""

QUEUE_CELL = """# the queue: runs this session's jobs until nothing is left for IDLE_EXIT_MINUTES or the session is drained
!python cloud_runs/queue_runner.py --store "{STORE}" --session "{SESSION}" --lanes "{LANES}" --slots {SLOTS} --idle-exit-minutes {IDLE_EXIT_MINUTES}"""

RELEASE_CELL = """# hands the runtime back to Colab once the queue has finished for this session
colab_bootstrap.release_runtime_if_queue_finished(SESSION, STORE, REPOSITORY_DIR)"""

RUNTIME_CHOICES = {"A100": "A100 GPU, high RAM", "L4": "L4 GPU, high RAM", "CPU": "CPU, high RAM"}


def cell(kind: str, source: str) -> dict:
    content = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        content.update({"execution_count": None, "outputs": []})
    return content


def build_notebook(session: str) -> dict:
    settings = SESSIONS[session]
    colab_metadata = {"provenance": [], "machine_shape": "hm"}         # "hm": a high-RAM machine
    if settings.has_gpu:
        colab_metadata["gpuType"] = settings.runtime
    metadata = {"colab": colab_metadata, "kernelspec": {"display_name": "Python 3", "name": "python3"},
                "language_info": {"name": "python"}}
    if settings.has_gpu:
        metadata["accelerator"] = "GPU"
    cells = [cell("markdown", INSTRUCTIONS.format(session=session, runtime_choice=RUNTIME_CHOICES[settings.runtime])),
             cell("code", SETTINGS_CELL.format(session=session, session_names=", ".join(SESSIONS),
                                               repository_url=REPOSITORY_URL_PLACEHOLDER)),
             cell("code", ENVIRONMENT_CELL), cell("code", QUEUE_CELL), cell("code", RELEASE_CELL)]
    return {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sessions", nargs="+", choices=list(SESSIONS), default=list(SESSIONS),
                        help="sessions to write a notebook for (default: all)")
    parser.add_argument("--output-dir", default=str(HERE), help="folder for the notebooks (default: this folder)")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for session in args.sessions:
        path = output_dir / f"queue_session_{session.replace('-', '_')}.ipynb"
        path.write_text(json.dumps(build_notebook(session), indent=1, ensure_ascii=False) + "\n")
        print("wrote", path)


if __name__ == "__main__":
    main()
