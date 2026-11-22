"""Write every summary table and figure of the thesis from the result files.

Run from the repository root:

    python -m analysis.run_all_summaries

This calls the four summarise_* modules in turn. Together they read the files under results/ (and the class labels
of the two data files in data/), and write results/summary_tables/<name>.csv and .md and results/figures/<name>.png;
existing files of the same name are overwritten, nothing else under results/ is touched, and no model is trained.
A full run takes about 15 seconds, most of it in summarise_classifier_arms, which computes two cost curves for each
arm and seed record.
"""
from __future__ import annotations

from analysis import (summarise_classifier_arms, summarise_completion_and_cold_start, summarise_supporting_results,
                      summarise_transformer_2026)


def main() -> None:
    summarise_classifier_arms.main()
    summarise_transformer_2026.main()
    summarise_completion_and_cold_start.main()
    summarise_supporting_results.main()


if __name__ == "__main__":
    main()
