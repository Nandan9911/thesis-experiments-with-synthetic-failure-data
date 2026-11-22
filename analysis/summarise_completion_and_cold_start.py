"""RQ3: summary tables and figures of the completion prototype: the completion benchmark, the memorisation audit
and the cold-start study.

The prototype fills in the missing attributes of a partially observed failure record ("completion") with one of six
engines, each fitted on real failure records only (experiments/rq3_completion_and_cold_start/completion_engines.py
describes them in detail): the median of the fitting failures; k-nearest neighbours (the mean over the ten most
similar fitting records); chained equations (scikit-learn's IterativeImputer, each attribute regressed on the others
with Bayesian ridge regression, drawing from the predictive distribution); Unmasking Trees (gradient-boosted trees
that learn to predict each attribute from the still visible ones, with attributes masked in random orders);
Forest-Flow inpainting (the tree-based diffusion model of Forest-Diffusion in its imputation mode, which generates the
missing cells while the known cells are held fixed; the diffusion variant, not the flow-matching generator of RQ2);
and TabPFN (a pretrained in-context model, one regression per missing attribute with the fitting records as context).
Every engine returns a point completion and an 80 % interval for each filled cell.

  completion benchmark   five folds over the 1,000 training failures: the engines are fitted on the 800 failures of
                         four folds and complete the 200 held-out failures after some of their observed cells have
                         been hidden, either completely at random (MCAR: 20 %, 50 % or 80 % of the observed cells,
                         independently of their values) or with the dataset's own missingness patterns (the
                         missing-value mask of a real record lacking at least 30 % of its attributes);
  memorisation audit     whether completed records sit unusually close to the fitting failures (copies rather
                         than completions), in one cell of the benchmark;
  cold start             the classifier cost when only 25 to 1,000 labelled failures are available for training,
                         with and without 2,000 synthetic failure rows from several generators. Two of them use a
                         completion engine as a generator ("inpainting"): real failures are drawn with replacement,
                         part of their observed cells is hidden, and the engine fitted on the same failures
                         completes them.

Run from the repository root (also part of `python -m analysis.run_all_summaries`):

    python -m analysis.summarise_completion_and_cold_start

Reads results/rq3_completion_and_cold_start/completion_benchmark.jsonl, completion_memorisation_audit.json and
cold_start_runs.jsonl. Writes to results/summary_tables/:

  rq3_completion_benchmark          accuracy, interval coverage and the classifier's view of the completed records
  rq3_completion_memorisation_audit distances of completed records to the fitting failures
  rq3_cold_start_test_costs         median test cost per generator and number of labelled failures

and to results/figures/: rq3_completion_benchmark and rq3_cold_start_curves.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from analysis.load_results import (COLD_START_RUNS_FILE, COMPLETION_BENCHMARK_FILE, COMPLETION_MEMORISATION_AUDIT_FILE,
                                   read_json, read_jsonl)
from analysis.write_outputs import (MISSING, fixed_decimals, is_missing, save_figure, signed_thousands, thousands,
                                    write_summary_table)

ENGINE_LABELS = {
    "median": "Median of the training failures",
    "k_nearest_neighbours": "k-nearest neighbours",
    "chained_equations": "Chained equations (Bayesian ridge)",
    "unmasking_trees": "Unmasking Trees",
    "forest_flow": "Forest-Flow completer",
    "tabpfn": "TabPFN",
}
MASK_LABELS = {
    "random_20_percent": "20 % of observed cells hidden at random",
    "random_50_percent": "50 % hidden at random",
    "random_80_percent": "80 % hidden at random",
    "real_patterns_30_percent": "real missingness patterns (>= 30 % missing)",
}
SHARE_HIDDEN_AT_RANDOM = {"random_20_percent": 0.2, "random_50_percent": 0.5, "random_80_percent": 0.8}
MEMORISATION_REFERENCE = "real held-out failures (reference)"       # key of the reference row in the audit file

COLD_START_BASELINE = "no_synthetic_rows"
COLD_START_GENERATOR_LABELS = {
    "no_synthetic_rows": "No synthetic rows",
    "duplicated_real_failures": "Duplicated real failures",
    "smote": "SMOTE",
    "conditional_gan_2019": "cGAN of Ranasinghe et al.",
    "tabebm": "TabEBM",
    "forest_flow": "Forest-Flow",
    "forest_flow_inpainting": "Forest-Flow inpainting",
    "unmasking_trees_inpainting": "Unmasking Trees inpainting",
}
# Fields of a benchmark record that are identifiers or settings rather than measurements.
BENCHMARK_NON_MEASUREMENTS = ("fold", "mask", "engine", "options")


def _in_order(values, preferred_order) -> list:
    """The distinct values, those named in `preferred_order` first and in that order, any others after them."""
    present = list(dict.fromkeys(values))
    ordered = [value for value in preferred_order if value in present]
    return ordered + [value for value in present if value not in ordered]


# ---------------------------------------------------------------------------------------------- completion benchmark
def load_completion_benchmark() -> pd.DataFrame:
    """One row per (fold, mask, engine) of completion_benchmark.jsonl."""
    return pd.DataFrame(read_jsonl(COMPLETION_BENCHMARK_FILE))


def write_completion_benchmark_table(benchmark: pd.DataFrame) -> None:
    """Medians over the five folds of every measurement, per mask and engine.

    The measurements are those of run_completion_benchmark.py, on the hidden cells of the 200 held-out failures:
    nrmse (root mean squared error in log1p space, each attribute divided by its standard deviation among the
    fitting failures, so 1.0 is about as wrong as always predicting the typical value; attributes that hardly vary
    among the fitting failures are left out, see n_hidden_informative), mae_log (mean absolute error in log1p space),
    cover80 (share of hidden cells inside the engine's 80 % interval; a well-calibrated engine gets about 0.80) and
    width80 (mean width of that interval in log1p units), and the view of a classifier: the tuned LightGBM, trained
    on the fitting folds' real rows and every healthy row, applied to the true, the masked and the completed records,
    as the mean failure probability (clf_prob_*) and the share flagged at the LightGBM arm's OOF threshold
    (clf_flag_*). A completion that restores the failure signature keeps clf_flag_completed close to clf_flag_true."""
    measurements = [column for column in benchmark.columns if column not in BENCHMARK_NON_MEASUREMENTS]
    medians = benchmark.groupby(["mask", "engine"])[measurements].median().reset_index()
    masks = _in_order(medians["mask"], MASK_LABELS)
    engines = _in_order(medians["engine"], ENGINE_LABELS)
    medians["mask"] = pd.Categorical(medians["mask"], masks, ordered=True)
    medians["engine"] = pd.Categorical(medians["engine"], engines, ordered=True)
    medians = medians.sort_values(["mask", "engine"]).reset_index(drop=True)
    medians["mask"] = medians["mask"].astype(str)
    medians["engine"] = medians["engine"].astype(str)
    sections = []
    for mask in masks:
        rows = medians[medians["mask"] == mask]
        readable = pd.DataFrame({
            "Engine": [ENGINE_LABELS.get(engine, engine) for engine in rows.engine],
            "NRMSE": rows.nrmse.map(fixed_decimals),
            "MAE (log)": rows.mae_log.map(fixed_decimals),
            "cover80": rows.cover80.map(fixed_decimals),
            "width80": rows.width80.map(fixed_decimals),
            "flag: true": rows.clf_flag_true.map(fixed_decimals),
            "flag: masked": rows.clf_flag_masked.map(fixed_decimals),
            "flag: completed": rows.clf_flag_completed.map(fixed_decimals),
            "s / fold": rows.seconds.map(thousands),
        })
        sections.append((f"{MASK_LABELS.get(mask, mask)} ({mask})", readable))
    folds = benchmark.fold.nunique()
    write_summary_table(
        "rq3_completion_benchmark", medians, sections,
        title="RQ3: completion benchmark",
        caption=f"""
        Medians over {folds} folds of the benchmark records in
        results/rq3_completion_and_cold_start/completion_benchmark.jsonl (one record per fold, mask and engine).
        In each fold the engines are fitted on 800 real failures and complete the 200 held-out failures after the
        mask has hidden part of their observed cells; all accuracy measures are on the hidden cells. The
        classifier columns show whether the completed records still look like failures to the tuned LightGBM.""",
        column_notes={
            "NRMSE": "root mean squared error in log1p space, per attribute divided by its standard deviation among "
                     "the fitting failures (CSV: nrmse; cells used: n_hidden_informative of n_hidden_cells).",
            "MAE (log)": "mean absolute error in log1p space (CSV: mae_log).",
            "cover80": "share of hidden cells inside the engine's 80 % interval (CSV: cover80).",
            "width80": "mean width of the 80 % interval in log1p units (CSV: width80).",
            "flag: true / masked / completed": "share of the 200 held-out failures flagged by the classifier as "
                                               "failures in their true, masked and completed form (CSV: "
                                               "clf_flag_true, clf_flag_masked, clf_flag_completed; mean "
                                               "probabilities clf_prob_*; the threshold clf_threshold).",
            "s / fold": "seconds per fold for the completion step (CSV: seconds).",
            "CSV only": "n_records (held-out failures per fold).",
        })


def plot_completion_benchmark(benchmark: pd.DataFrame) -> None:
    """NRMSE and share flagged against the share of cells hidden at random, per engine (median over folds, with the
    range over folds as a band in the left panel)."""
    hidden_at_random = benchmark[benchmark["mask"].isin(SHARE_HIDDEN_AT_RANDOM)].copy()
    hidden_at_random["share_hidden"] = hidden_at_random["mask"].map(SHARE_HIDDEN_AT_RANDOM)
    figure, (error_axes, flag_axes) = plt.subplots(1, 2, figsize=(12, 4.8))
    for engine in _in_order(hidden_at_random.engine, ENGINE_LABELS):
        rows = hidden_at_random[hidden_at_random.engine == engine]
        error = rows.groupby("share_hidden")["nrmse"].agg(["median", "min", "max"]).reset_index()
        label = ENGINE_LABELS.get(engine, engine)
        error_axes.plot(error.share_hidden, error["median"], "-o", ms=3, label=label)
        error_axes.fill_between(error.share_hidden, error["min"], error["max"], alpha=0.12)
        flagged = rows.groupby("share_hidden")["clf_flag_completed"].median().reset_index()
        flag_axes.plot(flagged.share_hidden, flagged.clf_flag_completed, "-o", ms=3, label=label)
    flag_axes.axhline(hidden_at_random.clf_flag_true.median(), color="k", lw=0.8, ls="--", label="true records")
    masked = hidden_at_random.groupby("share_hidden")["clf_flag_masked"].median().reset_index()
    flag_axes.plot(masked.share_hidden, masked.clf_flag_masked, "k:", label="masked records (not completed)")
    error_axes.set_xlabel("share of observed cells hidden at random")
    error_axes.set_ylabel("NRMSE on the hidden cells (log1p space)")
    error_axes.legend(fontsize=8, frameon=False)
    flag_axes.set_xlabel("share of observed cells hidden at random")
    flag_axes.set_ylabel("share flagged as failure by the classifier")
    flag_axes.legend(fontsize=8, frameon=False)
    figure.suptitle("RQ3 completion benchmark (median over folds)")
    save_figure(figure, "rq3_completion_benchmark")


# ---------------------------------------------------------------------------------------------- memorisation audit
def write_memorisation_audit_table() -> None:
    """Distances of completed records to the fitting failures (completion_memorisation_audit.json).

    The audit repeats one cell of the benchmark (its settings are in the file) and keeps the completed records. Its
    measures are those of experiments/common/synthetic_data_diagnostics.py, where they are defined, with the
    completed records in place of synthetic rows and the 800 fitting failures as the real rows. The reference row
    treats the real held-out failures as if they were completions. mia_auc and dcr_fit_vs_heldout are kept in the
    CSV but not tabulated: for a completed record the held-out set contains its own original, which makes them hard
    to read."""
    audit = read_json(COMPLETION_MEMORISATION_AUDIT_FILE)
    setting = audit.get("setting", {})
    rows, skipped = [], []
    engines = [key for key in audit if key not in ("setting", MEMORISATION_REFERENCE)]
    for key in [MEMORISATION_REFERENCE] + _in_order(engines, ENGINE_LABELS):
        entry = audit[key]
        label = "Real held-out failures (reference)" if key == MEMORISATION_REFERENCE else ENGINE_LABELS.get(key, key)
        if "skipped" in entry:
            skipped.append(f"{label}: {entry['skipped']}")
            rows.append({"records": key, "label": label})
            continue
        rows.append({"records": key, "label": label, **{name: entry.get(name) for name in (
            "dcr", "frac_closer", "near_copy_frac", "mia_auc", "dcr_fit_vs_heldout", "median_nn_real",
            "median_nn_syn_to_real", "seconds")}})
    table = pd.DataFrame(rows)
    readable = pd.DataFrame({"Records": table.label,
                             "Relative NN distance": table.dcr.map(fixed_decimals),
                             "Share closer": table.frac_closer.map(fixed_decimals),
                             "Near copies": table.near_copy_frac.map(fixed_decimals)})
    skipped_note = f" Not run: {'; '.join(skipped)}." if skipped else ""
    write_summary_table(
        "rq3_completion_memorisation_audit", table, readable,
        title="RQ3: memorisation audit of the completion engines",
        caption=f"""
        One cell of the completion benchmark repeated with the completed records kept: seed {setting.get('seed')},
        fold {setting.get('fold')}, {100 * setting.get('hide', float('nan')):.0f} % of the observed cells of the
        200 held-out failures hidden at random, every engine fitted on the 800 fitting failures. Distances are
        measured in the min-max scaled space of the fitting failures; the measures are defined in
        experiments/common/synthetic_data_diagnostics.py. Source:
        results/rq3_completion_and_cold_start/completion_memorisation_audit.json.{skipped_note}""",
        column_notes={
            "Relative NN distance": "median distance of a completed record to its nearest fitting failure, divided "
                                    "by the fitting failures' median leave-one-out nearest-neighbour distance (CSV: "
                                    "dcr).",
            "Share closer": "share of completed records closer to a fitting failure than the 5th percentile of the "
                            "fitting failures' leave-one-out nearest-neighbour distances (CSV: frac_closer).",
            "Near copies": "share of completed records whose nearest fitting failure is closer than 5 % of the "
                           "fitting failures' median leave-one-out nearest-neighbour distance (CSV: near_copy_frac).",
            "CSV only": "records (key in the result file), mia_auc, dcr_fit_vs_heldout, median_nn_real, "
                        "median_nn_syn_to_real (the two medians behind dcr) and seconds.",
        })


# ---------------------------------------------------------------------------------------------- cold start
def load_cold_start_runs() -> pd.DataFrame:
    """One row per (generator, seed, number of labelled failures) of cold_start_runs.jsonl, with the OOF cost scaled
    to the 16,000 rows of the test set and the test cost at the OOF threshold."""
    rows = []
    for record in read_jsonl(COLD_START_RUNS_FILE):
        rows.append({"generator": record["generator"], "seed": record["seed"],
                     "labelled_failures": record["labelled_failures"],
                     "oof_cost_scaled_to_test_set_size": record["out_of_fold"]["cost_scaled_to_test_set_size"],
                     "test_cost": record["test"]["cost"]})
    return pd.DataFrame(rows)


def write_cold_start_table(runs: pd.DataFrame) -> None:
    """Median over seeds of the OOF and test cost per generator and number of labelled failures, and the difference
    of the median test cost from the run without synthetic rows at the same number of failures."""
    medians = runs.groupby(["generator", "labelled_failures"]).agg(
        oof_cost_scaled_to_test_set_size=("oof_cost_scaled_to_test_set_size", "median"),
        test_cost=("test_cost", "median"), seeds=("seed", "nunique")).reset_index()
    baseline = medians[medians.generator == COLD_START_BASELINE].set_index("labelled_failures").test_cost
    medians["test_cost_difference_from_no_synthetic_rows"] = [
        row.test_cost - baseline[row.labelled_failures] if row.labelled_failures in baseline.index else float("nan")
        for row in medians.itertuples()]
    generators = _in_order(medians.generator, COLD_START_GENERATOR_LABELS)
    levels = sorted(medians.labelled_failures.unique())
    medians["generator"] = pd.Categorical(medians.generator, generators, ordered=True)
    medians = medians.sort_values(["generator", "labelled_failures"]).reset_index(drop=True)
    medians["generator"] = medians.generator.astype(str)
    readable_rows = []
    for generator in generators:
        cells = {"Generator": COLD_START_GENERATOR_LABELS.get(generator, generator)}
        for level in levels:
            match = medians[(medians.generator == generator) & (medians.labelled_failures == level)]
            cells[f"{thousands(level)} failures"] = _cold_start_cell(match, generator)
        readable_rows.append(cells)
    seed_count_note = _seed_count_note(medians)
    write_summary_table(
        "rq3_cold_start_test_costs", medians, pd.DataFrame(readable_rows),
        title="RQ3 cold start: test cost with few labelled failures",
        caption=f"""
        Median test cost over seeds (fold-ensemble predictions at the OOF threshold) when only the first n of a
        fixed random order of the training failures are kept in each training fold (nested subsets), all healthy
        rows are kept, and a generator fitted on those n failures adds 2,000 synthetic failure rows. Validation
        folds and the test set are untouched real data. In brackets: difference from the median without synthetic
        rows at the same n (negative is cheaper). Source:
        results/rq3_completion_and_cold_start/cold_start_runs.jsonl.
        {seed_count_note}""",
        column_notes={
            "n failures": "labelled failures available in each training fold (CSV: labelled_failures).",
            "Cells": "median test cost (CSV: test_cost) and difference from no synthetic rows (CSV: "
                     "test_cost_difference_from_no_synthetic_rows).",
            "CSV only": "oof_cost_scaled_to_test_set_size (median OOF cost scaled to the 16,000 rows of the test "
                        "set), seeds.",
        })


def _seed_count_note(medians: pd.DataFrame) -> str:
    """A sentence naming the cells whose median is over another number of seeds than most cells, e.g. 'Unmasking
    Trees inpainting has one seed at 100, 250 and 1,000 failures; every other cell has three seeds.'"""
    usual = int(medians.seeds.mode().iloc[0])
    clauses = []
    for generator in dict.fromkeys(medians.generator):
        unusual = medians[(medians.generator == generator) & (medians.seeds != usual)]
        for seeds, cells in unusual.groupby("seeds"):
            levels = [thousands(level) for level in sorted(cells.labelled_failures)]
            clauses.append(f"{COLD_START_GENERATOR_LABELS.get(generator, generator)} has {_seeds_in_words(seeds)} at "
                           f"{_listed(levels)} failures")
    if not clauses:
        return f"Every cell has {_seeds_in_words(usual)}."
    return f"{'; '.join(clauses)}; every other cell has {_seeds_in_words(usual)}."


def _seeds_in_words(count: int) -> str:
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    return f"{words.get(count, count)} seed{'' if count == 1 else 's'}"


def _listed(items: list[str]) -> str:
    """'100', '100 and 250' or '100, 250 and 1,000'."""
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _cold_start_cell(match: pd.DataFrame, generator: str) -> str:
    if len(match) == 0:
        return MISSING
    row = match.iloc[0]
    cell = thousands(row.test_cost)
    difference = row.test_cost_difference_from_no_synthetic_rows
    if generator != COLD_START_BASELINE and not is_missing(difference):
        cell += f" ({signed_thousands(difference)})"
    return cell


def plot_cold_start_curves(runs: pd.DataFrame) -> None:
    """Median test cost against the number of labelled failures (log scale), with the range over seeds as a band."""
    figure, axes = plt.subplots(figsize=(10, 5.5))
    for generator in _in_order(runs.generator, COLD_START_GENERATOR_LABELS):
        test_costs = runs[runs.generator == generator].groupby("labelled_failures")["test_cost"]
        summary = test_costs.agg(["median", "min", "max"]).reset_index()
        is_baseline = generator == COLD_START_BASELINE
        axes.plot(summary.labelled_failures, summary["median"], "-o", ms=3, lw=2.0 if is_baseline else 1.1,
                  color="k" if is_baseline else None, label=COLD_START_GENERATOR_LABELS.get(generator, generator))
        axes.fill_between(summary.labelled_failures, summary["min"], summary["max"], alpha=0.1)
    axes.set_xscale("log")
    axes.set_xlabel("labelled failures available in each training fold (log scale)")
    axes.set_ylabel("test cost (fold ensemble at the OOF threshold)")
    axes.set_title("RQ3 cold start: median over seeds, band = range over seeds")
    axes.legend(fontsize=8, frameon=False, ncol=2)
    save_figure(figure, "rq3_cold_start_curves")


def main() -> None:
    print("Completion prototype and cold start (RQ3)")
    benchmark = load_completion_benchmark()
    write_completion_benchmark_table(benchmark)
    plot_completion_benchmark(benchmark)
    write_memorisation_audit_table()
    cold_start_runs = load_cold_start_runs()
    write_cold_start_table(cold_start_runs)
    plot_cold_start_curves(cold_start_runs)


if __name__ == "__main__":
    main()
