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

import pandas_datareader.data as web
import statsmodels.api as sm
from datetime import datetime

RISK_FREE_RATE = 0.04
WEEKS_PER_YEAR = 52

EVENTS = {
    "COVID-19 Crash & Rebound": ("2020-02-01", "2020-05-31"),
    "The Emerging Market Divergence": ("2021-08-01", "2021-12-31"),
    "The Inflation Print Shock": ("2021-11-01", "2022-01-31"),
    "Russia-Ukraine Invasion": ("2022-02-01", "2022-04-30"),
    "The BOJ Yield Curve Surprise": ("2022-12-01", "2023-01-31"),
    "Regional Banking Crisis": ("2023-03-01", "2023-05-31"),
}


def analyze_event(event_name, base_start, base_end, equity_curves_df, ff_data=None):
    """
    Analyzes an expanded event window (2 months prior to 5 months post)
    and computes standard metrics + CAPM, FF3, and FF5 alphas.
    Normalizes curves to 100 at the actual event start date.
    """
    event_start_dt = pd.to_datetime(base_start)
    start_date = event_start_dt - pd.DateOffset(months=5)
    end_date = pd.to_datetime(base_end) + pd.DateOffset(months=5)

    # Filter the equity curves for the expanded event window
    mask = (equity_curves_df.index >= start_date) & (equity_curves_df.index <= end_date)
    window_df = equity_curves_df.loc[mask].copy()
    if window_df.empty:
        print(f"No data for {event_name} ({start_date.date()} to {end_date.date()})")
        return None, window_df

    # Normalize each strategy to its value on the event start date.
    norm_window = window_df.copy() * np.nan
    for strategy in window_df.columns:
        valid_data = window_df[strategy].dropna()
        if valid_data.empty:
            continue

        # Fetch the value on or immediately preceding the event start date
        base_value = valid_data.asof(event_start_dt)

        # Fallback: if the strategy starts after the event, use its first valid value
        if pd.isna(base_value) or base_value == 0:
            base_value = valid_data.iloc[0]
            if pd.isna(base_value) or base_value == 0:
                continue

        norm_window[strategy] = (window_df[strategy] / base_value) * 100

    metrics = []

    for strategy in norm_window.columns:
        curve = norm_window[strategy]
        valid_curve = curve.dropna()
        if valid_curve.empty:
            continue

        # Calculate return from the event start date to the end of the window
        total_return = (valid_curve.iloc[-1] / 100.0) - 1

        # Calculate max drawdown in window
        rolling_max = valid_curve.cummax()
        drawdowns = (valid_curve - rolling_max) / rolling_max
        max_dd = drawdowns.min()

        # Compute annualized return using elapsed calendar time in the window
        elapsed_days = (valid_curve.index[-1] - valid_curve.index[0]).days
        elapsed_years = elapsed_days / 365.25 if elapsed_days > 0 else np.nan
        ann_return = (
            (1 + total_return) ** (1 / elapsed_years) - 1
            if pd.notna(elapsed_years) and elapsed_years > 0
            else np.nan
        )

        # Event-window Sharpe: annualized from weekly-aligned returns
        sharpe = np.nan
        strategy_returns = valid_curve.pct_change().dropna()
        if len(strategy_returns) > 1:
            sharpe_returns = strategy_returns
            if ff_data is not None:
                ff_idx = strategy_returns.index.intersection(ff_data.index)
                if len(ff_idx) > 1:
                    sharpe_returns = strategy_returns.loc[ff_idx]

            if len(sharpe_returns) > 1:
                excess = sharpe_returns - (RISK_FREE_RATE / WEEKS_PER_YEAR)
                vol = excess.std()
                if vol > 0:
                    sharpe = (excess.mean() / vol) * np.sqrt(WEEKS_PER_YEAR)
                else:
                    sharpe = 0.0

        calmar = (
            ann_return / abs(max_dd)
            if pd.notna(ann_return) and pd.notna(max_dd) and max_dd != 0
            else np.nan
        )

        # Compute Alpha Models if Fama-French data is available
        capm_alpha, ff3_alpha, ff5_alpha = np.nan, np.nan, np.nan

        if ff_data is not None and len(valid_curve) > 2:
            # We need returns for regressions (normalization scalar does not affect pct_change)
            str_returns = valid_curve.pct_change().dropna()

            # Align with weekly FF data
            common_idx = str_returns.index.intersection(ff_data.index)
            if len(common_idx) > 5:  # Need sufficient datapoints
                y = (
                    str_returns.loc[common_idx] * 100
                )  # convert to percentage to match FF data scale
                ff = ff_data.loc[common_idx]

                # Excess return
                y_ex = y - ff["RF"]

                # CAPM: Regress on Mkt-RF
                try:
                    X_capm = sm.add_constant(ff["Mkt-RF"])
                    capm_model = sm.OLS(y_ex, X_capm).fit()
                    capm_alpha = capm_model.params.get("const", np.nan) * 52
                except:
                    pass

                # FF3: Mkt-RF, SMB, HML
                try:
                    X_ff3 = sm.add_constant(ff[["Mkt-RF", "SMB", "HML"]])
                    ff3_model = sm.OLS(y_ex, X_ff3).fit()
                    ff3_alpha = ff3_model.params.get("const", np.nan) * 52
                except:
                    pass

                # FF5: Mkt-RF, SMB, HML, RMW, CMA
                try:
                    X_ff5 = sm.add_constant(ff[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]])
                    ff5_model = sm.OLS(y_ex, X_ff5).fit()
                    ff5_alpha = ff5_model.params.get("const", np.nan) * 52
                except:
                    pass

        metrics.append(
            {
                "Event": event_name,
                "Strategy": strategy,
                "Return": f"{total_return:.2%}",
                "Sharpe Ratio": f"{sharpe:.3f}" if pd.notna(sharpe) else "N/A",
                "Max Drawdown": f"{max_dd:.2%}",
                "Calmar Ratio": f"{calmar:.3f}" if pd.notna(calmar) else "N/A",
                "CAPM Alpha (Ann)": (
                    f"{capm_alpha:.2f}%" if pd.notna(capm_alpha) else "N/A"
                ),
                "FF3 Alpha (Ann)": (
                    f"{ff3_alpha:.2f}%" if pd.notna(ff3_alpha) else "N/A"
                ),
                "FF5 Alpha (Ann)": (
                    f"{ff5_alpha:.2f}%" if pd.notna(ff5_alpha) else "N/A"
                ),
                "_return": total_return,
                "_max_dd": max_dd,
            }
        )

    return pd.DataFrame(metrics), norm_window


def load_fama_french_data(start, end):
    """Fetches Fama-French 5-Factor daily data, compounds it to weekly (Monday), and caches it."""
    cache_path = "data/ff5_factors_weekly.csv"
    if os.path.exists(cache_path):
        ff_weekly = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        # Verify it covers our needed range
        if (
            not ff_weekly.empty
            and ff_weekly.index[0] <= pd.to_datetime(start)
            and ff_weekly.index[-1] >= pd.to_datetime(end)
        ):
            return ff_weekly

    print("Fetching and compounding Fama-French 5-Factor data to weekly...")
    try:
        # F-F_Research_Data_5_Factors_2x3_daily is standard for FF5
        ff_dict = web.DataReader(
            "F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start, end
        )
        ff_daily = ff_dict[0]

        # FF data is in percentages. Convert to decimals, add 1 for compounding
        ff_daily = (ff_daily / 100.0) + 1.0

        # Resample to W-MON. Use product to geometrically link daily returns, then subtract 1
        ff_weekly = ff_daily.resample("W-MON", label="left", closed="left").prod() - 1.0

        # Convert back to percentages to match standard FF format used in regressions
        ff_weekly = ff_weekly * 100.0

        os.makedirs("data", exist_ok=True)
        ff_weekly.to_csv(cache_path)
        return ff_weekly
    except Exception as e:
        print(f"Error fetching/compounding FF data: {e}")
        return None


def plot_event(event_name, norm_window, output_dir):
    """Plots the normalized equity curves with a vertical line at the event start."""
    plt.figure(figsize=(10, 6))

    # Look up the event start date from the global EVENTS dict
    event_start_str = EVENTS[event_name][0]
    event_start_dt = pd.to_datetime(event_start_str)

    # Define base styles for the legacy standard portfolios if they appear
    base_styles = {
        "Zero-View": {"color": "#9E9E9E", "lw": 2.0, "ls": "-"},
        "Human-BL": {"color": "#2196F3", "lw": 2.0, "ls": "-"},
    }

    for strategy in norm_window.columns:
        if "Endowus_Actual" in strategy:
            # Render the actual funds as dashed lines with lower alpha so they form a "background" spectrum
            plt.plot(
                norm_window.index,
                norm_window[strategy],
                label=strategy.replace("Endowus_Actual_", "Actual "),
                ls="--",
                lw=1.5,
                alpha=0.7,
            )
        elif strategy == "Endowus-60/40":
            # Our proxy benchmark
            plt.plot(
                norm_window.index,
                norm_window[strategy],
                label="Proxy 60/40 Benchmark",
                color="black",
                lw=2.5,
                ls="-",
            )
        elif strategy == "Systematic-BL":
            # Our systematic baseline
            plt.plot(
                norm_window.index,
                norm_window[strategy],
                label="Systematic Baseline",
                color="red",
                lw=2.5,
                ls="-",
            )
        elif "LLM-BL" in strategy:
            # Our AI portfolios
            plt.plot(
                norm_window.index, norm_window[strategy], label=strategy, lw=2.0, ls="-"
            )
        else:
            # Legacy or unknown strategies
            s = base_styles.get(strategy, {"color": "black", "lw": 1.5, "ls": "-"})
            plt.plot(norm_window.index, norm_window[strategy], label=strategy, **s)

    # --- NEW: Add vertical line for event start (t=0) ---
    plt.axvline(
        x=event_start_dt,
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

    # Formatting x-axis
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)

    # Deduplicate legend labels (in case of multiple identical labels)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="best")

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
        f.write(
            "This report evaluates the performance of the Black-Litterman portfolio variations during specific historical market shocks.\n\n"
        )

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
                display_cols = [
                    "Strategy",
                    "Return",
                    "Sharpe Ratio",
                    "Max Drawdown",
                    "Calmar Ratio",
                    "CAPM Alpha (Ann)",
                    "FF3 Alpha (Ann)",
                    "FF5 Alpha (Ann)",
                ]
                # Only display alpha columns if they exist in the dataframe (i.e., we successfully computed them)
                available_cols = [c for c in display_cols if c in event_metrics.columns]
                markdown_table = event_metrics[available_cols].to_markdown(index=False)
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
        print(
            f"Error: Could not find any curves data. Run bl_portfolio_engine.py or systematic_bl_baseline.py first."
        )
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

    # Fill missing values from prior observations after aligning series.
    df = df.sort_index().ffill()

    output_dir = "results/event_studies"
    os.makedirs(output_dir, exist_ok=True)

    # Load Fama-French data for the whole window + padding
    min_date = pd.to_datetime(min([d[0] for d in EVENTS.values()])) - pd.DateOffset(
        months=1
    )
    max_date = pd.to_datetime(max([d[1] for d in EVENTS.values()])) + pd.DateOffset(
        months=1
    )
    ff_data = load_fama_french_data(
        min_date.strftime("%Y-%m-%d"), max_date.strftime("%Y-%m-%d")
    )

    all_metrics = []
    event_plots = {}

    for event_name, dates in EVENTS.items():
        print(f"Analyzing {event_name}...")
        metrics_df, norm_window = analyze_event(
            event_name, dates[0], dates[1], df, ff_data=ff_data
        )
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
