#!/usr/bin/env bash
# Step 7 (README, section 5): statistics, summary tables and figures from everything in results/.
# Writes results/summary_tables/ (CSV and Markdown) and results/figures/ (PNG). Takes well under a minute and
# trains nothing.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m analysis.run_all_summaries
