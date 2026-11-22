"""Analysis of the experiment results: the statistics, summary tables and figures behind the thesis.

The experiments (package experiments/) write their raw results to results/: one record per classifier arm and seed,
saved predicted probabilities, training traces, benchmark records. This package reads those files and turns them
into the tables and figures that the thesis reports, so that every number in the thesis can be traced back to a
result file. It trains no model; the test labels are used only to describe predictions.

    python -m analysis.run_all_summaries          # everything, from the repository root

writes results/summary_tables/<name>.csv (unrounded numbers) and <name>.md (the table rounded as in the thesis,
with a caption and an explanation of its columns), and results/figures/<name>.png.

Modules:
  load_results                         readers for the result files; one row per (arm, seed) with diagnostics
  paired_statistics                    corrected resampled t-test, Wilcoxon test, Holm adjustment, evidence tiers
  summarise_classifier_arms            RQ1 and RQ2: classifier arms, paired comparisons, the survey table
  summarise_transformer_2026           RQ1b: reproduction of the study of Beikmohammadi et al.
  summarise_completion_and_cold_start  RQ3: completion benchmark, memorisation audit, cold-start study
  summarise_supporting_results         data audit, first-look baselines, hyperparameter search, threshold rules,
                                       TabICL pilot, synthetic-data diagnostics, synthetic-row count sweep
  write_outputs                        CSV, Markdown and PNG writing and number formatting
  run_all_summaries                    runs the four summarise_* modules
"""
