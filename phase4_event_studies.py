#!/usr/bin/env python3
"""
Stress Testing & Event Studies
======================================
Evaluates the Black-Litterman portfolios during specific market shocks.

Events:
  - COVID-19 Crash & Rebound: Feb 1, 2020 - May 31, 2020
  - The Emerging Market Divergence: Aug 1, 2021 - Dec 31, 2021
  - The Inflation Print Shock: Nov 1, 2021 - Jan 31, 2022
  - Russia-Ukraine Invasion: Feb 1, 2022 - Apr 30, 2022
  - The BOJ Yield Curve Surprise: Dec 1, 2022 - Jan 31, 2023
  - Regional Banking Crisis: Mar 1, 2023 - May 31, 2023
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas_datareader.data as web
import statsmodels.api as sm
from datetime import datetime

warnings.filterwarnings("ignore")

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

EXCLUDED_STRATEGY_PATTERNS = (
    "human-bl",
    "zero-view",
    "endowus-60/40-norebalanceifmissing",
    "endowus-60/40-buyhold",
)


def is_excluded_strategy(strategy_name: str) -> bool:
    strategy_name_lc = str(strategy_name).lower()
    return any(pat in strategy_name_lc for pat in EXCLUDED_STRATEGY_PATTERNS)


def analyze_event(
    event_name,
    base_start,
    base_end,
    equity_curves_df,
    ff_data_weekly=None,
    ff_data_monthly=None,
):
    """
    Analyzes an expanded event window dynamically handling both Weekly and Monthly equity curves.
    """
    event_start_dt = pd.to_datetime(base_start)
    start_date = event_start_dt - pd.DateOffset(months=5)
    end_date = pd.to_datetime(base_end) + pd.DateOffset(months=5)

    visible_cols = [
        col for col in equity_curves_df.columns if not is_excluded_strategy(col)
    ]
    if not visible_cols:
        return None, pd.DataFrame()

    mask = (equity_curves_df.index >= start_date) & (equity_curves_df.index <= end_date)
    window_df = equity_curves_df.loc[mask, visible_cols].copy()

    if window_df.empty:
        return None, window_df

    norm_window = window_df.copy() * np.nan
    for strategy in window_df.columns:
        valid_data = window_df[strategy].dropna()
        if valid_data.empty:
            continue

        base_value = valid_data.asof(event_start_dt)
        if pd.isna(base_value) or base_value == 0:
            base_value = valid_data.iloc[0]
            if pd.isna(base_value) or base_value == 0:
                continue

        norm_window[strategy] = (window_df[strategy] / base_value) * 100

    metrics = []

    for strategy in norm_window.columns:
        valid_curve = norm_window[strategy].dropna()
        if len(valid_curve) < 2:
            continue

        # 1. FREQUENCY DETECTION
        # Calculate the median days between data points
        median_days = valid_curve.index.to_series().diff().dt.days.median()

        if median_days > 20:  # Monthly frequency
            ann_factor = 12
            ff_data = ff_data_monthly
        else:  # Weekly frequency
            ann_factor = 52
            ff_data = ff_data_weekly

        # Return & Drawdown
        total_return = (valid_curve.iloc[-1] / 100.0) - 1
        rolling_max = valid_curve.cummax()
        max_dd = ((valid_curve - rolling_max) / rolling_max).min()

        elapsed_days = (valid_curve.index[-1] - valid_curve.index[0]).days
        elapsed_years = elapsed_days / 365.25 if elapsed_days > 0 else np.nan
        ann_return = (
            ((1 + total_return) ** (1 / elapsed_years) - 1)
            if elapsed_years > 0
            else np.nan
        )

        # 2. DYNAMIC SHARPE RATIO
        sharpe = np.nan
        str_returns = valid_curve.pct_change().dropna()

        if len(str_returns) > 1:
            excess = str_returns - (0.04 / ann_factor)  # Assuming 4% Risk-Free Rate
            vol = excess.std()
            if vol > 0:
                sharpe = (excess.mean() / vol) * np.sqrt(ann_factor)
            else:
                sharpe = 0.0

        calmar = (
            ann_return / abs(max_dd) if pd.notna(ann_return) and max_dd != 0 else np.nan
        )

        # 3. DYNAMIC ALPHA REGRESSION
        capm_alpha, ff3_alpha, ff5_alpha = np.nan, np.nan, np.nan

        if ff_data is not None and len(str_returns) > 2:
            # To handle end-of-month vs start-of-month indexing quirks, convert to period
            if ann_factor == 12:
                # Align on Year-Month for monthly curves
                str_returns.index = str_returns.index.to_period("M")
                ff_aligned = ff_data.copy()
                if not isinstance(ff_aligned.index, pd.PeriodIndex):
                    ff_aligned.index = pd.to_datetime(
                        ff_aligned.index.astype(str)
                    ).to_period("M")
            else:
                ff_aligned = ff_data

            common_idx = str_returns.index.intersection(ff_aligned.index)

            if len(common_idx) > 3:  # Need minimum data points for OLS
                y = str_returns.loc[common_idx] * 100
                ff = ff_aligned.loc[common_idx]
                y_ex = y - ff["RF"]

                try:
                    X_capm = sm.add_constant(ff["Mkt-RF"])
                    capm_alpha = (
                        sm.OLS(y_ex, X_capm).fit().params.get("const", np.nan)
                        * ann_factor
                    )
                except:
                    pass

                try:
                    X_ff3 = sm.add_constant(ff[["Mkt-RF", "SMB", "HML"]])
                    ff3_alpha = (
                        sm.OLS(y_ex, X_ff3).fit().params.get("const", np.nan)
                        * ann_factor
                    )
                except:
                    pass

                try:
                    X_ff5 = sm.add_constant(ff[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]])
                    ff5_alpha = (
                        sm.OLS(y_ex, X_ff5).fit().params.get("const", np.nan)
                        * ann_factor
                    )
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
            }
        )

    return pd.DataFrame(metrics), norm_window


def load_fama_french_data_weekly(start, end):
    """Fetches Fama-French 5-Factor daily data, compounds it to weekly (Monday), and caches it."""
    cache_path = "data/ff5_factors_weekly.csv"
    if os.path.exists(cache_path):
        ff_weekly = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if (
            not ff_weekly.empty
            and ff_weekly.index[0] <= pd.to_datetime(start)
            and ff_weekly.index[-1] >= pd.to_datetime(end)
        ):
            return ff_weekly

    print("Fetching and compounding Fama-French 5-Factor data to weekly...")
    try:
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
        print(f"Error fetching/compounding FF weekly data: {e}")
        return None


def load_fama_french_data_monthly(start, end):
    """Fetches Fama-French 5-Factor monthly data and caches it."""
    cache_path = "data/ff5_factors_monthly.csv"
    if os.path.exists(cache_path):
        ff_monthly = pd.read_csv(cache_path, index_col=0)
        # Convert to PeriodIndex for robust monthly alignment
        ff_monthly.index = pd.to_datetime(ff_monthly.index.astype(str)).to_period("M")
        return ff_monthly

    print("Fetching Fama-French 5-Factor monthly data...")
    try:
        # Standard monthly 5-factor dataset
        ff_dict = web.DataReader(
            "F-F_Research_Data_5_Factors_2x3", "famafrench", start, end
        )
        ff_monthly = ff_dict[0]

        os.makedirs("data", exist_ok=True)
        ff_monthly.to_csv(cache_path)
        return ff_monthly
    except Exception as e:
        print(f"Error fetching FF monthly data: {e}")
        return None


def plot_event(event_name, norm_window, output_dir):
    """Plots the normalized equity curves with a vertical line at the event start.
    Added: filter_curves (list of substrings or exact names) to select which curves to plot.
    """

    def _should_plot(strategy, filter_curves):
        if is_excluded_strategy(strategy):
            return False
        if filter_curves is None:
            return True
        for f in filter_curves:
            if f in strategy:
                return True
        return False

    plt.figure(figsize=(10, 11))

    event_start_str = EVENTS[event_name][0]
    event_start_dt = pd.to_datetime(event_start_str)

    base_styles = {}

    filter_curves = getattr(plot_event, "filter_curves", None)

    for strategy in norm_window.columns:
        if not _should_plot(strategy, filter_curves):
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
                norm_window.index, norm_window[strategy], label=strategy, lw=2.0, ls="-"
            )
        else:
            s = base_styles.get(strategy, {"color": "black", "lw": 1.5, "ls": "-"})
            plt.plot(norm_window.index, norm_window[strategy], label=strategy, **s)

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

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
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
                available_cols = [c for c in display_cols if c in event_metrics.columns]
                markdown_table = event_metrics[available_cols].to_markdown(index=False)
                f.write(markdown_table + "\n\n")

            f.write("---\n\n")

    print(f"Generated Markdown report at {report_path}")


def main():
    print("=" * 60)
    print("  Stress Testing & Event Studies Engine")
    print("=" * 60)

    curves_path = "results/bl_equity_curves.csv"
    if os.path.exists(curves_path):
        df_bl = pd.read_csv(curves_path, index_col=0, parse_dates=True)
    else:
        df_bl = pd.DataFrame()

    endowus_path = "results/endowus_systematic_equity_curves.csv"
    if os.path.exists(endowus_path):
        df_endowus = pd.read_csv(endowus_path, index_col=0, parse_dates=True)
    else:
        df_endowus = pd.DataFrame()

    if df_bl.empty and df_endowus.empty:
        print(
            "Error: Could not find any curves data. Run bl_portfolio_engine.py or systematic_bl_baseline.py first."
        )
        return

    if not df_bl.empty and not df_endowus.empty:
        df = df_bl.join(df_endowus, how="outer", rsuffix="_endowus")
        for col in df_endowus.columns:
            if f"{col}_endowus" in df.columns:
                df[col] = df[f"{col}_endowus"]
                df = df.drop(columns=[f"{col}_endowus"])
    elif not df_endowus.empty:
        df = df_endowus
    else:
        df = df_bl

    df = df.sort_index().ffill()
    excluded_cols = [col for col in df.columns if is_excluded_strategy(col)]
    if excluded_cols:
        df = df.drop(columns=excluded_cols)

    output_dir = "results/event_studies"
    os.makedirs(output_dir, exist_ok=True)

    min_date = pd.to_datetime(min([d[0] for d in EVENTS.values()])) - pd.DateOffset(
        months=1
    )
    max_date = pd.to_datetime(max([d[1] for d in EVENTS.values()])) + pd.DateOffset(
        months=1
    )

    # Load both frequencies of Fama-French data
    ff_weekly = load_fama_french_data_weekly(
        min_date.strftime("%Y-%m-%d"), max_date.strftime("%Y-%m-%d")
    )
    ff_monthly = load_fama_french_data_monthly(
        min_date.strftime("%Y-%m-%d"), max_date.strftime("%Y-%m-%d")
    )

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filter-curves",
        nargs="*",
        default=None,
        help="List of substrings or names to filter which curves to plot (case-sensitive)",
    )
    args, _ = parser.parse_known_args()
    filter_curves = args.filter_curves

    all_metrics = []
    event_plots = {}

    for event_name, dates in EVENTS.items():
        print(f"Analyzing {event_name}...")
        metrics_df, norm_window = analyze_event(
            event_name,
            dates[0],
            dates[1],
            df,
            ff_data_weekly=ff_weekly,
            ff_data_monthly=ff_monthly,
        )

        if metrics_df is not None:
            all_metrics.append(metrics_df)
            setattr(plot_event, "filter_curves", filter_curves)
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
