#!/usr/bin/env bash
# Step 0 (README, section 5): check that the two UCI files in data/ are the ones used for the thesis.
# Prints the audit and writes results/data_audit.json; compare the checksums with data/README.md. Takes a few seconds.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m experiments.common.data_loading
