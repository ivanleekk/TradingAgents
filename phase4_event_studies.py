#!/usr/bin/env python3
"""
Stress Testing & Event Studies
======================================
Evaluates the Black-Litterman portfolios during specific market shocks.

Events:
  - COVID-19 Crash & Rebound: Feb 1, 2020 - May 31, 2020
  - Inflation Print Shock: Nov 1, 2021 - Jan 31, 2022
  - Russia-Ukraine Invasion: Feb 1, 2022 - Apr 30, 2022
  - Regional Banking Crisis: Mar 1, 2023 - May 31, 2023
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore")

EVENTS = {
    "COVID-19 Crash & Rebound": ("2020-02-01", "2020-05-31"),
    "The Inflation Print Shock": ("2021-11-01", "2022-01-31"),
    "Russia-Ukraine Invasion": ("2022-02-01", "2022-04-30"),
    "Regional Banking Crisis": ("2023-03-01", "2023-05-31"),
}

def analyze_event(event_name, start_date, end_date, equity_curves_df):
    """
    Analyzes an event window and returns metrics for all strategies within that window.
    """
    # Filter the equity curves for the event window
    mask = (equity_curves_df.index >= pd.to_datetime(start_date)) & (equity_curves_df.index <= pd.to_datetime(end_date))
    window_df = equity_curves_df.loc[mask].copy()

    if window_df.empty:
        print(f"No data for {event_name} ({start_date} to {end_date})")
        return None, window_df

    # Normalize window so everything starts at 100 for easy comparison
    start_values = window_df.iloc[0]
    norm_window = (window_df / start_values) * 100

    metrics = []

    for strategy in norm_window.columns:
        curve = norm_window[strategy]

        # Calculate isolated window return
        total_return = (curve.iloc[-1] / curve.iloc[0]) - 1

        # Calculate max drawdown in window
        rolling_max = curve.cummax()
        drawdowns = (curve - rolling_max) / rolling_max
        max_dd = drawdowns.min()

        metrics.append({
            "Event": event_name,
            "Strategy": strategy,
            "Return": f"{total_return:.2%}",
            "Max Drawdown": f"{max_dd:.2%}",
            "_return": total_return,
            "_max_dd": max_dd
        })

    return pd.DataFrame(metrics), norm_window

def plot_event(event_name, norm_window, output_dir):
    """Plots the normalized equity curves for a specific event window."""
    plt.figure(figsize=(10, 6))

    # Define base styles for the legacy standard portfolios if they appear
    base_styles = {
        "Zero-View": {"color": "#9E9E9E", "lw": 2.0, "ls": "-"},
        "Human-BL": {"color": "#2196F3", "lw": 2.0, "ls": "-"},
    }

    for strategy in norm_window.columns:
        if "Endowus_Actual" in strategy:
            # Render the actual funds as dashed lines with lower alpha so they form a "background" spectrum
            plt.plot(norm_window.index, norm_window[strategy], label=strategy.replace("Endowus_Actual_", "Actual "), ls="--", lw=1.5, alpha=0.7)
        elif strategy == "Endowus-60/40":
            # Our proxy benchmark
            plt.plot(norm_window.index, norm_window[strategy], label="Proxy 60/40 Benchmark", color="black", lw=2.5, ls="-")
        elif strategy == "Systematic-BL":
            # Our systematic baseline
            plt.plot(norm_window.index, norm_window[strategy], label="Systematic Baseline", color="red", lw=2.5, ls="-")
        elif "LLM-BL" in strategy:
            # Our AI portfolios
            plt.plot(norm_window.index, norm_window[strategy], label=strategy, lw=2.0, ls="-")
        else:
            # Legacy or unknown strategies
            s = base_styles.get(strategy, {"color": "black", "lw": 1.5, "ls": "-"})
            plt.plot(norm_window.index, norm_window[strategy], label=strategy, **s)

    plt.title(f"{event_name} (Normalized to 100)", fontsize=14, fontweight="bold")
    plt.ylabel("Normalized Value")
    plt.xlabel("Date")

    # Formatting x-axis
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)

    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = event_name.replace(" ", "_").replace("-", "_").lower() + ".png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150)
    plt.close()
    return filepath

def generate_markdown_report(all_metrics_df, event_plots, output_dir):
    """Generates a comprehensive Markdown report."""
    report_path = os.path.join(output_dir, "event_study_report.md")

    with open(report_path, "w") as f:
        f.write("# Stress Testing & Event Studies Report\n\n")
        f.write("This report evaluates the performance of the Black-Litterman portfolio variations during specific historical market shocks.\n\n")

        for event, dates in EVENTS.items():
            f.write(f"## {event} ({dates[0]} to {dates[1]})\n\n")

            # Write image
            img_path = event_plots.get(event)
            if img_path:
                img_name = os.path.basename(img_path)
                f.write(f"![{event}]({img_name})\n\n")

            # Write table
            event_metrics = all_metrics_df[all_metrics_df["Event"] == event]
            if not event_metrics.empty:
                display_cols = ["Strategy", "Return", "Max Drawdown"]
                markdown_table = event_metrics[display_cols].to_markdown(index=False)
                f.write(markdown_table + "\n\n")

            f.write("---\n\n")

    print(f"Generated Markdown report at {report_path}")

def main():
    print("=" * 60)
    print("  Stress Testing & Event Studies Engine")
    print("=" * 60)

    # First load standard BL equity curves if available
    curves_path = "results/bl_equity_curves.csv"
    if os.path.exists(curves_path):
        df_bl = pd.read_csv(curves_path, index_col=0, parse_dates=True)
    else:
        df_bl = pd.DataFrame()

    # Load endowus systematic curves if available
    endowus_path = "results/endowus_systematic_equity_curves.csv"
    if os.path.exists(endowus_path):
        df_endowus = pd.read_csv(endowus_path, index_col=0, parse_dates=True)
    else:
        df_endowus = pd.DataFrame()

    if df_bl.empty and df_endowus.empty:
        print(f"Error: Could not find any curves data. Run bl_portfolio_engine.py or systematic_bl_baseline.py first.")
        return

    # Combine dataframes for plotting everything
    if not df_bl.empty and not df_endowus.empty:
        # Align indexes and combine columns, dropping overlapping duplicates
        df = df_bl.join(df_endowus, how="outer", rsuffix="_endowus")

        # If there are overlapping columns, prioritize the endowus ones as they have the updated logic
        for col in df_endowus.columns:
            if f"{col}_endowus" in df.columns:
                df[col] = df[f"{col}_endowus"]
                df = df.drop(columns=[f"{col}_endowus"])
    elif not df_endowus.empty:
        df = df_endowus
    else:
        df = df_bl

    output_dir = "results/event_studies"
    os.makedirs(output_dir, exist_ok=True)

    all_metrics = []
    event_plots = {}

    for event_name, dates in EVENTS.items():
        print(f"Analyzing {event_name}...")
        metrics_df, norm_window = analyze_event(event_name, dates[0], dates[1], df)

        if metrics_df is not None:
            all_metrics.append(metrics_df)
            img_path = plot_event(event_name, norm_window, output_dir)
            event_plots[event_name] = img_path

    if all_metrics:
        final_metrics_df = pd.concat(all_metrics, ignore_index=True)
        csv_path = os.path.join(output_dir, "event_study_metrics.csv")
        final_metrics_df.to_csv(csv_path, index=False)
        print(f"Saved metrics to {csv_path}")

        generate_markdown_report(final_metrics_df, event_plots, output_dir)

if __name__ == "__main__":
    main()
