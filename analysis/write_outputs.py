"""Writing the summary tables and figures: locations, number formatting and the Markdown layout.

Every summary table is written twice into results/summary_tables/:

  <name>.csv   the numbers at full precision, one column per quantity, for checking and for further analysis;
  <name>.md    the same table as the thesis shows it, rounded the way the thesis rounds it (costs with a thousands
               separator and no decimals, p-values with two decimals, ...), preceded by a short caption that names
               the result files it comes from and followed by an explanation of every column.

Figures are written to results/figures/<name>.png. This module computes nothing; the summarise_* modules decide
what goes into each table and figure.
"""
from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg")       # write files only; no display is needed (or available on a compute server)
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.common.paths import FIGURES_DIR, SUMMARY_TABLES_DIR  # noqa: E402

MISSING = "--"              # how an empty cell is printed, as in the thesis tables


def is_missing(value) -> bool:
    """True for None and NaN; False for every other value, including strings and lists."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):         # lists and arrays: pd.isna returns an array
        return False


def thousands(value, decimals: int = 0) -> str:
    """A number with a thousands separator, e.g. 9350.0 -> '9,350' (how the thesis prints costs)."""
    if is_missing(value):
        return MISSING
    return f"{value:,.{decimals}f}"


def signed_thousands(value) -> str:
    """A difference with an explicit sign, e.g. 217.6 -> '+218', -1240.2 -> '-1,240'."""
    if is_missing(value):
        return MISSING
    return f"{value:+,.0f}"


def fixed_decimals(value, decimals: int = 2) -> str:
    if is_missing(value):
        return MISSING
    return f"{value:.{decimals}f}"


def scientific(value, digits: int = 2) -> str:
    """Thresholds span several orders of magnitude, so they are printed as 4.61e-05."""
    if is_missing(value):
        return MISSING
    return f"{value:.{digits}e}"


def cost_with_errors(cost, missed_failures, false_alarms) -> str:
    """'9,020 (10 / 402)': a cost followed by the missed failures and false alarms behind it."""
    if is_missing(cost):
        return MISSING
    return f"{thousands(cost)} ({thousands(missed_failures)} / {thousands(false_alarms)})"


def bracketed_interval(low, high) -> str:
    """'[6,110, 12,290]': an interval or a range."""
    if is_missing(low) or is_missing(high):
        return MISSING
    return f"[{thousands(low)}, {thousands(high)}]"


def median_with_range(median, low, high) -> str:
    """'9,350 [8,970, 10,360]': median over seeds with the smallest and largest value."""
    if is_missing(median):
        return MISSING
    return f"{thousands(median)} {bracketed_interval(low, high)}"


def percentage(share, decimals: int = 1) -> str:
    """A share between 0 and 1 printed as a percentage, e.g. 0.375 -> '37.5 %'."""
    if is_missing(share):
        return MISSING
    return f"{100 * share:.{decimals}f} %"


def _markdown_cell(value) -> str:
    text = MISSING if is_missing(value) else str(value)
    return text.replace("|", "\\|")


def markdown_table_lines(table: pd.DataFrame, left_aligned_columns=1) -> list[str]:
    """The lines of a Markdown table. Columns of names are left-aligned, columns of numbers right-aligned;
    `left_aligned_columns` is either the number of leading name columns or a list of the name columns."""
    if isinstance(left_aligned_columns, int):
        name_columns = set(table.columns[:left_aligned_columns])
    else:
        name_columns = set(left_aligned_columns)
    header = "| " + " | ".join(str(column) for column in table.columns) + " |"
    alignments = [":---" if column in name_columns else "---:" for column in table.columns]
    lines = [header, "| " + " | ".join(alignments) + " |"]
    for row in table.itertuples(index=False):
        lines.append("| " + " | ".join(_markdown_cell(value) for value in row) + " |")
    return lines


def write_summary_table(name: str, full_precision: pd.DataFrame, readable, title: str, caption: str,
                        column_notes: dict[str, str] | None = None, left_aligned_columns=1) -> None:
    """Write results/summary_tables/<name>.csv and <name>.md.

    full_precision        the CSV content, numbers unrounded.
    readable              the Markdown table, already formatted as strings; either one DataFrame or a list of
                          (subheading, DataFrame) pairs when the table is shown in sections.
    caption               a few sentences on what the table shows and where its numbers come from.
    column_notes          {column: explanation}; the explanation names the CSV columns behind each Markdown column.
    left_aligned_columns  see markdown_table_lines.
    """
    SUMMARY_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    full_precision.to_csv(SUMMARY_TABLES_DIR / f"{name}.csv", index=False)
    lines = [f"# {title}", "", textwrap.dedent(caption).strip(), ""]
    sections = readable if isinstance(readable, list) else [(None, readable)]
    for subheading, frame in sections:
        if subheading:
            lines += [f"## {subheading}", ""]
        lines += markdown_table_lines(frame, left_aligned_columns) + [""]
    if column_notes:
        lines += ["Columns:", ""]
        lines += [f"- {column}: {textwrap.dedent(note).strip()}" for column, note in column_notes.items()]
        lines.append("")
    lines.append(f"Unrounded values: `{name}.csv` in this folder.")
    (SUMMARY_TABLES_DIR / f"{name}.md").write_text("\n".join(lines) + "\n")
    print(f"  table   results/summary_tables/{name}.csv and .md")


def save_figure(figure, name: str) -> None:
    """Write results/figures/<name>.png and close the figure."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  figure  results/figures/{name}.png")
