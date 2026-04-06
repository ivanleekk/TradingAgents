#!/usr/bin/env python3
"""
Plot Event Study Figures From Saved Equity Curves
=================================================
Regenerates event-study plots directly from equity-curve CSV files in results/
without rerunning any backtest.

Default inputs:
  - results/bl_equity_curves.csv
  - results/endowus_systematic_equity_curves.csv

Default output:
  - results/event_studies/*.png
  - results/event_studies/event_study_plots_only_report.md
"""

import argparse
import os
from typing import Any, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

EVENTS = {
    "COVID-19 Crash & Rebound": ("2020-02-01", "2020-05-31"),
    "The Emerging Market Divergence": ("2021-08-01", "2021-12-31"),
    "The Inflation Print Shock": ("2021-11-01", "2022-01-31"),
    "Russia-Ukraine Invasion": ("2022-02-01", "2022-04-30"),
    "The BOJ Yield Curve Surprise": ("2022-12-01", "2023-01-31"),
    "Regional Banking Crisis": ("2023-03-01", "2023-05-31"),
}

EQUITY_ASSETS = {
    "0P0001AF7U.SI",
    "SPY",
    "^990100-USD-STRD",
    "DE000SLA4YD9.SG",
    "0P0001AF7Z.SI",
    "0P0001EF2T.SI",
    "EIMI.L",
}

FIXED_INCOME_ASSETS = {
    "0P0000KYEE.SI",
    "AGGG.L",
    "IE0002461055.IR",
    "0P0001EQUE.SI",
    "0P0001CC3M",
    "PEBIX",
    "0P0001DWI0.SI",
}

# Keep exclusions aligned with the main event-study pipeline.
EXCLUDED_STRATEGY_PATTERNS = (
    "human-bl",
    "zero-view",
    "endowus-60/40-norebalanceifmissing",
    "endowus-60/40-buyhold",
)


def is_excluded_strategy(strategy_name: str) -> bool:
    name_lc = str(strategy_name).lower()
    return any(pat in name_lc for pat in EXCLUDED_STRATEGY_PATTERNS)


def load_equity_curves(
    bl_curves_path: str,
    endowus_curves_path: str,
) -> pd.DataFrame:
    if os.path.exists(bl_curves_path):
        df_bl = pd.read_csv(bl_curves_path, index_col=0, parse_dates=True)
    else:
        df_bl = pd.DataFrame()

    if os.path.exists(endowus_curves_path):
        df_endowus = pd.read_csv(endowus_curves_path, index_col=0, parse_dates=True)
    else:
        df_endowus = pd.DataFrame()

    if df_bl.empty and df_endowus.empty:
        raise FileNotFoundError(
            "No equity curve CSV found. Expected at least one of: "
            f"{bl_curves_path}, {endowus_curves_path}"
        )

    if not df_bl.empty and not df_endowus.empty:
        df = df_bl.join(df_endowus, how="outer", rsuffix="_endowus")
        for col in df_endowus.columns:
            merge_col = f"{col}_endowus"
            if merge_col in df.columns:
                df[col] = df[merge_col]
                df = df.drop(columns=[merge_col])
    elif not df_endowus.empty:
        df = df_endowus
    else:
        df = df_bl

    df = df.sort_index().ffill()

    excluded_cols = [c for c in df.columns if is_excluded_strategy(c)]
    if excluded_cols:
        df = df.drop(columns=excluded_cols)

    return df


def compute_normalized_window(
    equity_curves_df: pd.DataFrame,
    event_start: str,
    event_end: str,
    pad_months: int = 5,
) -> pd.DataFrame:
    event_start_dt = pd.to_datetime(event_start)
    start_date = event_start_dt - pd.DateOffset(months=pad_months)
    end_date = pd.to_datetime(event_end) + pd.DateOffset(months=pad_months)

    mask = (equity_curves_df.index >= start_date) & (equity_curves_df.index <= end_date)
    window_df = equity_curves_df.loc[mask].copy()

    if window_df.empty:
        return pd.DataFrame()

    norm_window = window_df.copy() * np.nan
    for strategy in window_df.columns:
        valid_data = window_df[strategy].dropna()
        if valid_data.empty:
            continue

        base_value_raw: Any = valid_data.asof(event_start_dt)
        if isinstance(base_value_raw, pd.Series):
            if base_value_raw.empty:
                continue
            base_value_raw = base_value_raw.iloc[-1]

        if base_value_raw is None:
            continue

        try:
            base_value = float(base_value_raw)
        except (TypeError, ValueError):
            continue

        if not np.isfinite(base_value) or base_value == 0.0:
            continue

        norm_window[strategy] = (window_df[strategy] / base_value) * 100.0

    # Keep only strategies that have at least one valid point.
    keep_cols = [c for c in norm_window.columns if norm_window[c].notna().any()]
    return norm_window[keep_cols]


def should_plot(strategy: str, filter_curves: Optional[list[str]]) -> bool:
    if is_excluded_strategy(strategy):
        return False
    if filter_curves is None:
        return True
    return any(token in strategy for token in filter_curves)


def plot_event(
    event_name: str,
    norm_window: pd.DataFrame,
    output_dir: str,
    filter_curves: Optional[list[str]] = None,
) -> Optional[str]:
    if norm_window.empty:
        return None

    plt.figure(figsize=(10, 11))

    event_start_dt = pd.to_datetime(EVENTS[event_name][0])

    for strategy in norm_window.columns:
        if not should_plot(strategy, filter_curves):
            continue

        if "Endowus_Actual" in strategy:
            plt.plot(
                norm_window.index,
                norm_window[strategy],
                label=strategy.replace("Endowus_Actual_", "Actual "),
                ls="--",
                lw=1.5,
                alpha=0.7,
            )
        elif strategy == "Endowus-60/40":
            plt.plot(
                norm_window.index,
                norm_window[strategy],
                label="Proxy 60/40 Benchmark",
                color="black",
                lw=2.5,
                ls="-",
            )
        elif strategy == "Systematic-BL":
            plt.plot(
                norm_window.index,
                norm_window[strategy],
                label="Systematic Baseline",
                color="red",
                lw=2.5,
                ls="-",
            )
        elif "LLM-BL" in strategy:
            plt.plot(
                norm_window.index,
                norm_window[strategy],
                label=strategy,
                lw=2.0,
                ls="-",
            )
        else:
            plt.plot(
                norm_window.index,
                norm_window[strategy],
                label=strategy,
                color="black",
                lw=1.5,
                ls="-",
            )

    event_start_num = float(mdates.date2num(event_start_dt.to_pydatetime()))
    plt.axvline(
        x=event_start_num,
        color="black",
        linestyle="--",
        lw=1.5,
        alpha=0.6,
        label="Event Start (t=0)",
    )

    plt.title(
        f"{event_name} (Normalized to 100 at t=0)", fontsize=14, fontweight="bold"
    )
    plt.ylabel("Normalized Value")
    plt.xlabel("Date")

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        legend_cols = min(3, max(1, len(by_label)))
        plt.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.62),
            ncol=legend_cols,
            frameon=False,
        )

    plt.grid(True, alpha=0.3)
    plt.tight_layout(rect=(0.0, 0.02, 1.0, 1.0))

    filename = event_name.replace(" ", "_").replace("-", "_").lower() + ".png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    return filepath


def write_plots_only_report(event_plots: dict[str, str], output_dir: str) -> str:
    report_path = os.path.join(output_dir, "event_study_plots_only_report.md")
    with open(report_path, "w") as f:
        f.write("# Stress Testing & Event Studies Plots-Only Report\n\n")
        f.write(
            "This report contains event-study figures regenerated from saved equity curves.\n\n"
        )

        for event, dates in EVENTS.items():
            f.write(f"## {event} ({dates[0]} to {dates[1]})\n\n")
            img_path = event_plots.get(event)
            if img_path:
                img_name = os.path.basename(img_path)
                f.write(f"![{event}]({img_name})\n\n")
            else:
                f.write("_No data available for this event window._\n\n")
            f.write("---\n\n")

    return report_path


def plot_combined_curves(
    curves_df: pd.DataFrame,
    output_path: str,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 8))

    for column in curves_df.columns:
        if "Endowus_Actual" in column:
            ax.plot(
                curves_df.index,
                curves_df[column],
                label=column.replace("Endowus_Actual_", "Actual "),
                ls="--",
                lw=1.5,
                alpha=0.7,
            )
        elif column == "Endowus-60/40":
            ax.plot(
                curves_df.index,
                curves_df[column],
                label="Proxy 60/40 Benchmark",
                color="black",
                lw=2.5,
                ls="-",
            )
        elif column == "Systematic-BL":
            ax.plot(
                curves_df.index,
                curves_df[column],
                label="Systematic Baseline",
                color="red",
                lw=2.5,
                ls="-",
            )
        elif "LLM-BL" in column:
            ax.plot(curves_df.index, curves_df[column], label=column, lw=2.0, ls="-")
        else:
            ax.plot(curves_df.index, curves_df[column], label=column, lw=1.5, ls="-")

    ax.set_title(
        "Systematic Baseline & AI BL Portfolios vs Actual Endowus Flagship Funds (2020-2024)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("Portfolio Value (Base $10,000)")
    ax.set_xlabel("Date")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend_cols = 1 if len(labels) <= 2 else 2
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=legend_cols,
            framealpha=0.9,
        )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"Saved: {output_path}")


def load_weights_logs(weights_raw_dir: str) -> dict[str, pd.DataFrame]:
    logs: dict[str, pd.DataFrame] = {}

    manifest_path = os.path.join(weights_raw_dir, "raw_weights_manifest.csv")
    if os.path.exists(manifest_path):
        manifest = pd.read_csv(manifest_path)
        required_cols = {"Strategy", "File"}
        if required_cols.issubset(set(manifest.columns)):
            for _, row in manifest.iterrows():
                strategy = str(row["Strategy"]).strip()
                if not strategy or is_excluded_strategy(strategy):
                    continue

                file_name = str(row["File"]).strip()
                if not file_name:
                    continue

                file_path = os.path.join(weights_raw_dir, file_name)
                if not os.path.exists(file_path):
                    continue

                df = pd.read_csv(file_path, index_col=0, parse_dates=True).sort_index()
                if not df.empty:
                    logs[strategy] = df

    # Fallback path: load any *_raw_weights.csv not already loaded.
    loaded_files = {f"{name.replace('/', '_')}_raw_weights.csv" for name in logs.keys()}
    for file_name in sorted(os.listdir(weights_raw_dir)):
        if not file_name.endswith("_raw_weights.csv"):
            continue
        if file_name in loaded_files:
            continue

        strategy = file_name.replace("_raw_weights.csv", "").replace("_", "/")
        if is_excluded_strategy(strategy):
            continue

        file_path = os.path.join(weights_raw_dir, file_name)
        df = pd.read_csv(file_path, index_col=0, parse_dates=True).sort_index()
        if not df.empty:
            logs[strategy] = df

    return logs


def plot_composition_comparison(
    weights_logs: dict[str, pd.DataFrame],
    output_path: str,
) -> None:
    valid_logs = {
        name: w.copy()
        for name, w in weights_logs.items()
        if isinstance(w, pd.DataFrame) and not w.empty
    }
    if not valid_logs:
        print("Skipped composition plot: no weights logs available.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    all_assets = sorted(
        {
            col
            for df in valid_logs.values()
            for col in df.columns
            if isinstance(col, str) and col.strip()
        }
    )
    if not all_assets:
        print("Skipped composition plot: no allocation columns available.")
        return

    strategy_names = list(valid_logs.keys())
    n = len(strategy_names)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(18 if ncols == 2 else 12, max(4 * nrows, 5)),
        sharex=True,
        sharey=True,
    )
    axes_arr = np.atleast_1d(axes).flatten()
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(all_assets))]

    percent_formatter = FuncFormatter(lambda v, _: f"{v:.0f}%")

    for i, name in enumerate(strategy_names):
        ax = axes_arr[i]
        df = valid_logs[name].sort_index().copy()
        df = df.reindex(columns=all_assets, fill_value=0.0)
        df = df.clip(lower=0.0)

        row_sums = df.sum(axis=1)
        nonzero = row_sums > 1e-10
        if nonzero.any():
            df.loc[nonzero] = df.loc[nonzero].div(row_sums[nonzero], axis=0)

        x = mdates.date2num(pd.to_datetime(df.index).to_pydatetime())
        y = [df[c].to_numpy(dtype=float) * 100.0 for c in all_assets]
        ax.stackplot(x, y, labels=all_assets, colors=colors, alpha=0.9)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(percent_formatter)
        ax.grid(True, alpha=0.2)
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    for j in range(n, len(axes_arr)):
        axes_arr[j].axis("off")

    fig.suptitle(
        "Portfolio Composition Over Time Comparison", fontsize=15, fontweight="bold"
    )
    fig.supxlabel("Date")
    fig.supylabel("Allocation")
    fig.autofmt_xdate(rotation=45)

    handles, labels = axes_arr[0].get_legend_handles_labels()
    if handles:
        legend_cols = min(len(all_assets), 6)
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=legend_cols,
            fontsize=8,
            frameon=True,
        )

        legend_rows = (
            int(np.ceil(len(all_assets) / legend_cols)) if legend_cols > 0 else 1
        )
        bottom_margin = min(0.08 + 0.03 * legend_rows, 0.18)
    else:
        bottom_margin = 0.08

    plt.subplots_adjust(bottom=bottom_margin, top=0.90, hspace=0.35)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_asset_class_allocation_comparison(
    weights_logs: dict[str, pd.DataFrame],
    output_path: str,
) -> None:
    valid_logs = {
        name: w.copy()
        for name, w in weights_logs.items()
        if isinstance(w, pd.DataFrame) and not w.empty
    }
    if not valid_logs:
        print("Skipped asset-class plot: no weights logs available.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    strategy_names = list(valid_logs.keys())
    n = len(strategy_names)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(18 if ncols == 2 else 12, max(4 * nrows, 5)),
        sharex=True,
        sharey=True,
    )
    axes_arr = np.atleast_1d(axes).flatten()

    percent_formatter = FuncFormatter(lambda v, _: f"{v:.0f}%")

    for i, name in enumerate(strategy_names):
        ax = axes_arr[i]
        df = valid_logs[name].sort_index().copy().fillna(0.0)

        if df.empty:
            ax.set_title(name, fontsize=11, fontweight="bold")
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.2)
            continue

        df = df.clip(lower=0.0)
        total = df.sum(axis=1)

        eq_cols = [c for c in df.columns if c in EQUITY_ASSETS]
        fi_cols = [c for c in df.columns if c in FIXED_INCOME_ASSETS]

        eq_value = (
            df[eq_cols].sum(axis=1) if eq_cols else pd.Series(0.0, index=df.index)
        )
        fi_value = (
            df[fi_cols].sum(axis=1) if fi_cols else pd.Series(0.0, index=df.index)
        )

        denom = total.replace(0.0, np.nan)
        eq_pct = (eq_value / denom * 100.0).fillna(0.0)
        fi_pct = (fi_value / denom * 100.0).fillna(0.0)

        ax.plot(df.index, eq_pct, label="Equity", lw=1.8, color="tab:blue")
        ax.plot(df.index, fi_pct, label="Fixed Income", lw=1.8, color="tab:orange")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(percent_formatter)
        ax.grid(True, alpha=0.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    for j in range(n, len(axes_arr)):
        axes_arr[j].axis("off")

    fig.suptitle(
        "Portfolio Asset-Class Allocation Over Time", fontsize=15, fontweight="bold"
    )
    fig.supxlabel("Date")
    fig.supylabel("Allocation")
    fig.autofmt_xdate(rotation=45)

    handles, labels = axes_arr[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=2,
            fontsize=9,
            frameon=True,
        )

    plt.subplots_adjust(bottom=0.15, top=0.92, hspace=0.4)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate event-study and selected full-period plots from saved "
            "equity-curve/weights CSV files"
        )
    )
    parser.add_argument(
        "--bl-curves",
        default="results/bl_equity_curves.csv",
        help="Path to BL equity curves CSV",
    )
    parser.add_argument(
        "--endowus-curves",
        default="results/endowus_systematic_equity_curves.csv",
        help="Path to Endowus/systematic equity curves CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="results/event_studies",
        help="Directory to save regenerated plots",
    )
    parser.add_argument(
        "--weights-raw-dir",
        default="results/weights_raw",
        help="Directory containing *_raw_weights.csv and raw_weights_manifest.csv",
    )
    parser.add_argument(
        "--comparison-plot-path",
        default="results/endowus_systematic_comparison_plot.png",
        help="Output path for full-period comparison plot",
    )
    parser.add_argument(
        "--composition-plot-path",
        default="results/endowus_systematic_composition_comparison.png",
        help="Output path for composition comparison plot",
    )
    parser.add_argument(
        "--asset-class-plot-path",
        default="results/endowus_systematic_asset_class_allocation.png",
        help="Output path for asset-class allocation plot",
    )
    parser.add_argument(
        "--filter-curves",
        nargs="*",
        default=None,
        help="Optional list of substrings to include only matching curve names",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    curves_df = load_equity_curves(args.bl_curves, args.endowus_curves)

    event_plots: dict[str, str] = {}
    for event_name, (start, end) in EVENTS.items():
        print(f"Plotting {event_name}...")
        norm_window = compute_normalized_window(curves_df, start, end)
        img_path = plot_event(
            event_name,
            norm_window,
            args.output_dir,
            filter_curves=args.filter_curves,
        )
        if img_path is not None:
            event_plots[event_name] = img_path
            print(f"  Saved: {img_path}")
        else:
            print("  Skipped (no data in window)")

    report_path = write_plots_only_report(event_plots, args.output_dir)
    print(f"Generated plots-only report at {report_path}")

    # Additional full-period outputs requested from the systematic baseline flow.
    plot_combined_curves(curves_df, args.comparison_plot_path)

    weights_logs = load_weights_logs(args.weights_raw_dir)
    plot_composition_comparison(weights_logs, args.composition_plot_path)
    plot_asset_class_allocation_comparison(weights_logs, args.asset_class_plot_path)


if __name__ == "__main__":
    main()
