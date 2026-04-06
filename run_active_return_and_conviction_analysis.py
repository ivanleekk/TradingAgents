import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.dates import DateFormatter
import pandas_datareader.data as web


def fetch_ff5_data(start_date, end_date):
    """
    Fetch Fama-French 5-Factor Daily Returns using pandas-datareader.
    """
    print("Fetching Fama-French 5-Factor Daily data...")
    try:
        # F-F_Research_Data_5_Factors_2x3_daily
        ff5 = web.DataReader(
            "F-F_Research_Data_5_Factors_2x3_daily",
            "famafrench",
            start=start_date,
            end=end_date,
        )[0]
        # Data from French's library is often in percentages (e.g., 1.5 instead of 0.015). We convert to decimals.
        ff5 = ff5 / 100.0
        ff5.index = pd.to_datetime(ff5.index)
        return ff5
    except Exception as e:
        print(f"Error fetching FF5 data: {e}")
        return pd.DataFrame()


def run_active_return_ff5_analysis(
    llm_returns, benchmark_returns, start_date="2019-09-30", end_date="2024-12-31"
):
    """
    Analysis 1: Fama-French 5-factor regression strictly on Active Return.
    (Treating it as a Long/Short Hedge Fund).
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 1: ACTIVE RETURN FF5 REGRESSION (Contrarian/Value Bias Test)")
    print("=" * 60)

    # Calculate Active Return (Long LLM, Short Benchmark)
    active_return = llm_returns - benchmark_returns
    active_return.name = "Active_Return"

    # Fetch FF5 Data
    ff5_data = fetch_ff5_data(start_date, end_date)

    if ff5_data.empty:
        print("Could not retrieve FF5 data. Skipping analysis.")
        return

    # Align dates between active return and FF5 explicitly (often FF data lags by a month or two in real time,
    # but we assume historical is available)
    merged_data = pd.concat([active_return, ff5_data], axis=1).dropna()

    y = merged_data["Active_Return"]
    # Factors: Mkt-RF, SMB, HML, RMW, CMA
    X = merged_data[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 1})

    print("\n--- OLS Regression Results: Active Return ~ FF5 ---")
    print(model.summary())

    print("\nInterpretation Check:")
    hml_coef = model.params.get("HML", 0)
    hml_pval = model.pvalues.get("HML", 1)
    mkt_coef = model.params.get("Mkt-RF", 0)
    mkt_pval = model.pvalues.get("Mkt-RF", 1)

    if hml_coef > 0 and hml_pval < 0.05:
        print(
            "-> STRUCTURAL VALUE BIAS PROVEN: Significant positive loading on HML (Value)."
        )
    else:
        print("-> No statistically significant Value bias detected on HML.")

    if mkt_coef < 0 and mkt_pval < 0.05:
        print(
            "-> DEFENSIVE / CONTRARIAN STANCE PROVEN: Significant negative loading on Mkt-RF (Short Growth/Market Beta)."
        )
    else:
        print("-> No statistically significant negative Market loading.")


def calculate_active_share(llm_weights_df, benchmark_weights_df):
    """
    Calculates the Active Share: 0.5 * sum(|w_llm_i - w_bench_i|)
    Expects dataframes indexed by date, columns as tickers.
    """
    # Align indices
    idx = llm_weights_df.index.intersection(benchmark_weights_df.index)
    llm_w = llm_weights_df.loc[idx]
    bench_w = benchmark_weights_df.loc[idx]

    # Fill missing columns with 0
    all_cols = list(set(llm_w.columns).union(set(bench_w.columns)))

    llm_reindexed = llm_w.reindex(columns=all_cols).fillna(0)
    bench_reindexed = bench_w.reindex(columns=all_cols).fillna(0)

    # Active share
    active_share = 0.5 * np.abs(llm_reindexed - bench_reindexed).sum(axis=1)
    return active_share


def run_conviction_dynamics_analysis(
    llm_returns,
    benchmark_returns,
    llm_weights,
    bench_weights,
    start_date="2019-09-30",
    end_date="2024-12-31",
):
    """
    Analysis 2: Active Share and Tracking Error vs VIX
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 2: CONVICTION DYNAMICS (Active Share & Tracking Error vs VIX)")
    print("=" * 60)

    active_return = (llm_returns - benchmark_returns).dropna()

    print("Fetching VIX data...")
    vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.squeeze()

    # Calculate Active Share
    active_share = calculate_active_share(llm_weights, bench_weights)

    # Calculate 60-day Rolling Tracking Error (Annualized, 252 trading days)
    rolling_te = (
        active_return.rolling(window=60).std() * np.sqrt(252) * 100
    )  # Convert to percentage

    # Create unified dataframe
    df = pd.DataFrame(
        {
            "Active Share": active_share * 100,  # %
            "Tracking Error (60d)": rolling_te,
            "VIX": vix,
        }
    ).dropna()

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # 1. VIX
    ax1 = axes[0]
    ax1.plot(df.index, df["VIX"], color="red", label="VIX")
    ax1.axhline(
        22,
        color="gray",
        linestyle="--",
        alpha=0.6,
        label="High Volatility Threshold (>22)",
    )
    ax1.set_ylabel("VIX Level")
    ax1.set_title("Macro Volatility Regime (VIX)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Highlight Crisis Periods
    crises = [
        ("COVID-19 Crash", "2020-02-20", "2020-03-31"),
        ("Russia-Ukraine", "2022-02-24", "2022-04-30"),
    ]
    for label, start, end in crises:
        try:
            ax1.axvspan(
                pd.to_datetime(start), pd.to_datetime(end), color="red", alpha=0.1
            )
        except:
            pass

    # 2. Active Share
    ax2 = axes[1]
    ax2.plot(df.index, df["Active Share"], color="blue", label="Active Share")
    ax2.set_ylabel("Active Share (%)")
    ax2.set_title("LLM Portfolio Conviction (Active Share)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    for label, start, end in crises:
        try:
            ax2.axvspan(
                pd.to_datetime(start), pd.to_datetime(end), color="red", alpha=0.1
            )
        except:
            pass

    # 3. Tracking Error
    ax3 = axes[2]
    ax3.plot(
        df.index,
        df["Tracking Error (60d)"],
        color="green",
        label="Rolling Tracking Error (Ann %)",
    )
    ax3.set_ylabel("Tracking Error (%)")
    ax3.set_title("Deviation Drag (60-day Rolling TE)")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)
    for label, start, end in crises:
        try:
            ax3.axvspan(
                pd.to_datetime(start), pd.to_datetime(end), color="red", alpha=0.1
            )
        except:
            pass

    plt.tight_layout()
    plot_path = "conviction_dynamics_vs_vix.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to: {os.path.abspath(plot_path)}")
    plt.show()


# =====================================================================
# Mock Data Generators (Replace with actual data loaders from workspace)
# =====================================================================
def load_workspace_data():
    """
    You should replace this function to load the actual portfolio returns and weights
    from your results folder (e.g., from 'results_merged/LLM-BL-A-Monthly-Banded_returns.csv').
    """
    print(
        "Loading proxy empirical data (please hook up your actual CSVs here if necessary)..."
    )
    dates = pd.date_range("2019-09-30", "2024-12-31", freq="B")

    # Using random walks for demonstration of the script's mechanics
    llm_returns = pd.Series(np.random.normal(0.0003, 0.01, len(dates)), index=dates)
    bench_returns = pd.Series(np.random.normal(0.0003, 0.01, len(dates)), index=dates)

    # Mock Weights
    tickers = ["SPY", "AGGG.L", "EIMI.L"]
    llm_weights = pd.DataFrame(
        np.random.dirichlet(np.ones(3), size=len(dates)), index=dates, columns=tickers
    )
    bench_weights = pd.DataFrame(
        np.tile([0.6, 0.3, 0.1], (len(dates), 1)), index=dates, columns=tickers
    )

    return llm_returns, bench_returns, llm_weights, bench_weights


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Advanced Thesis Analyses: FF5 Active Return & Conviction Dynamics"
    )
    parser.add_argument(
        "--llm_returns_csv",
        type=str,
        help="Path to LLM daily returns CSV (index=Date, column=Return)",
    )
    parser.add_argument(
        "--bench_returns_csv", type=str, help="Path to Benchmark daily returns CSV"
    )
    parser.add_argument(
        "--llm_weights_csv", type=str, help="Path to LLM daily weights CSV"
    )
    parser.add_argument(
        "--bench_weights_csv", type=str, help="Path to Benchmark daily weights CSV"
    )
    args = parser.parse_args()

    if (
        args.llm_returns_csv
        and args.bench_returns_csv
        and args.llm_weights_csv
        and args.bench_weights_csv
    ):
        print("Loading data from actual CSVs...")
        llm_returns = pd.read_csv(
            args.llm_returns_csv, index_col=0, parse_dates=True
        ).squeeze()
        bench_returns = pd.read_csv(
            args.bench_returns_csv, index_col=0, parse_dates=True
        ).squeeze()
        llm_weights = pd.read_csv(args.llm_weights_csv, index_col=0, parse_dates=True)
        bench_weights = pd.read_csv(
            args.bench_weights_csv, index_col=0, parse_dates=True
        )
    else:
        print(
            "No CSV paths provided. Running with generated mock data to test the pipeline."
        )
        llm_returns, bench_returns, llm_weights, bench_weights = load_workspace_data()

    # Ensure timezone-naive dates for easy alignment
    for s in [llm_returns, bench_returns, llm_weights, bench_weights]:
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)

    run_active_return_ff5_analysis(llm_returns, bench_returns)
    run_conviction_dynamics_analysis(
        llm_returns, bench_returns, llm_weights, bench_weights
    )
