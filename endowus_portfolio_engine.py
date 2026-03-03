#!/usr/bin/env python3
"""
Endowus Black-Litterman Portfolio Engine
======================================
Evaluates multiple AI-driven portfolio strategies using Black-Litterman allocation
against a static 60/40 benchmark representing the Endowus Flagship Fund.

Strategies:
  - Endowus-60/40 : Static benchmark weights (rebalanced weekly)
  - LLM-BL-A   : LLM Macro/Fundamentals variation
  - LLM-BL-B   : LLM News variation
  - LLM-BL-C   : LLM Technicals variation
  - LLM-BL-D   : Full Debate variation

Rebalancing: Weekly on Mondays
Transaction cost: 0.1% on every buy/sell dollar value
Starting equity: $10,000
"""

import glob
import os
import warnings
import json

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yfinance as yf
from pypfopt import BlackLittermanModel, risk_models, expected_returns, EfficientFrontier

warnings.filterwarnings("ignore")

DATA_DIR = "data"
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"
INITIAL_EQUITY = 10_000.0
TRANSACTION_COST = 0.001
LOOKBACK_WEEKS = 52
RISK_FREE_RATE = 0.04
WEEKS_PER_YEAR = 52

# Endowus Flagship Target Weights
ENDOWUS_WEIGHTS = {
    "URTH": 0.285, # Global Developed Equity
    "SPY": 0.174,  # US S&P 500
    "EEM": 0.093,  # Emerging Markets Equity
    "VPL": 0.048,  # Pacific Basin Small/Mid Cap
    "BNDW": 0.230, # Global Aggregate Bond
    "AGG": 0.070,  # US Aggregate Bond
    "EMB": 0.060,  # Emerging Markets Government Bond
    "BSV": 0.040,  # Short-Term Global Bond
}

def load_bl_signal_csvs(data_dir: str) -> pd.DataFrame:
    """Load JSON signals outputted by the LLM trader for the Endowus universe."""
    files = glob.glob(os.path.join(data_dir, "*_decisions.csv"))
    if not files:
        print(f"No decision CSVs found in '{data_dir}', returning empty signals")
        return pd.DataFrame()

    frames = []
    for fp in sorted(files):
        ticker = os.path.basename(fp).split("_decisions.csv")[0]
        try:
            df = pd.read_csv(fp, parse_dates=["test_date"])
            df = df.rename(columns={"test_date": "Date"})
            df["Ticker"] = ticker

            target_returns = []
            confidences = []

            import re
            for idx, row in df.iterrows():
                decision_str = str(row['decision']).strip()

                # Strip markdown blocks if present
                if decision_str.startswith("```json"):
                    decision_str = decision_str[7:]
                if decision_str.startswith("```"):
                    decision_str = decision_str[3:]
                if decision_str.endswith("```"):
                    decision_str = decision_str[:-3]
                decision_str = decision_str.strip()

                try:
                    data = json.loads(decision_str)
                    ret_str = str(data.get("Target_Return_30d", "")).replace('%', '')

                    if not ret_str:
                        target_returns.append(None)
                    else:
                        ret = float(ret_str) / 100.0
                        annual_ret = (1 + ret)**(365/30) - 1
                        target_returns.append(annual_ret)

                    conf = float(data.get("Confidence_Score", 5))
                    confidences.append(conf)
                except Exception as e:
                    target_returns.append(None)
                    confidences.append(5.0)

            df['target_return'] = target_returns
            df['confidence'] = confidences
            frames.append(df[["Date", "Ticker", "target_return", "confidence"]])
        except Exception as e:
            print(f"Error parsing {fp}: {e}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)

def to_weekly_monday(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("W-MON", label="left", closed="left").last()

def download_prices(tickers: list[str], start: str, end: str):
    print(f"Downloading prices for {len(tickers)} tickers …")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        close_df = raw["Close"]
        open_df = raw["Open"]
    else:
        close_df = raw[["Close"]].rename(columns={"Close": tickers[0]})
        open_df = raw[["Open"]].rename(columns={"Open": tickers[0]})

    close_df.index = pd.to_datetime(close_df.index)
    open_df.index = pd.to_datetime(open_df.index)
    return close_df.sort_index().dropna(how="all"), open_df.sort_index().dropna(how="all")

def get_execution_prices(daily_open: pd.DataFrame, weekly_dates: pd.DatetimeIndex) -> pd.DataFrame:
    trading_days = daily_open.index
    rows = {}
    for monday in weekly_dates:
        future = trading_days[trading_days > monday]
        if len(future) > 0:
            rows[monday] = daily_open.loc[future[0]]
        else:
            on_or_after = trading_days[trading_days >= monday]
            if len(on_or_after) > 0:
                rows[monday] = daily_open.loc[on_or_after[0]]
    exec_df = pd.DataFrame(rows).T.sort_index()
    exec_df.index.name = "Date"
    return exec_df

def compute_bl_weights(
    active_set: list[str],
    price_history: pd.DataFrame,
    views: dict,
    confidences: dict
) -> dict[str, float]:
    """
    Computes Black-Litterman weights.
    Here, the "market prior" is explicitly set to the Endowus target weights.
    """
    sub = price_history[active_set].ffill().bfill().dropna(axis=0, how="all")

    # Normalize the target weights for the active set
    target_weights_dict = {t: ENDOWUS_WEIGHTS.get(t, 0.0) for t in active_set}
    total_w = sum(target_weights_dict.values())
    if total_w > 0:
        target_weights = np.array([target_weights_dict[t]/total_w for t in active_set])
        fallback_weights = {t: target_weights_dict[t]/total_w for t in active_set}
    else:
        # Extreme fallback
        fallback_weights = {t: 1.0/len(active_set) for t in active_set}
        target_weights = np.array(list(fallback_weights.values()))

    if sub.shape[0] < 10 or len(active_set) < 2:
        return fallback_weights

    try:
        # Calculate covariance matrix
        S = risk_models.CovarianceShrinkage(sub, frequency=252).ledoit_wolf()

        # Calculate market-implied returns (prior) assuming Endowus is the equilibrium
        delta = 2.5 # Risk aversion parameter
        # market implied returns pi = delta * S * w
        market_prior = delta * S.dot(target_weights)

        # Format views. If LLM gave None, fallback to the calculated market prior
        Q = pd.Series({
            t: views.get(t) if views.get(t) is not None else market_prior.get(t, 0.05)
            for t in active_set
        })

        # Construct confidence matrix Omega (diagonal)
        # Scale uncertainty inversely to confidence (1-10)
        omega = np.diag([S.loc[t, t] * (1.1 - (confidences.get(t, 5) / 10.0)) for t in active_set])

        # Black-Litterman Model
        bl = BlackLittermanModel(S, pi=market_prior, absolute_views=Q, omega=omega)
        bl_returns = bl.bl_returns()
        bl_cov = bl.bl_cov()

        # Optimize for max sharpe
        ef = EfficientFrontier(bl_returns, bl_cov)
        try:
            ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
        except:
            ef = EfficientFrontier(bl_returns, bl_cov)
            ef.min_volatility()

        cleaned = ef.clean_weights()
        weights = {t: w for t, w in cleaned.items() if w > 1e-6}

        return weights if weights else fallback_weights

    except Exception as e:
        print(f"[DEBUG] BL Math failed for {active_set}: {e}")
        return fallback_weights

class Portfolio:
    def __init__(self, initial_equity: float):
        self.cash = initial_equity
        self.shares: dict[str, float] = {}

    def total_equity(self, prices: dict[str, float]) -> float:
        mv = sum(self.shares.get(t, 0.0) * prices.get(t, 0.0) for t in self.shares)
        return self.cash + mv

    def liquidate(self, tickers_to_sell: list[str], prices: dict[str, float]) -> float:
        proceeds = 0.0
        for t in tickers_to_sell:
            qty = self.shares.pop(t, 0.0)
            if qty > 0:
                gross = qty * prices[t]
                fee = gross * TRANSACTION_COST
                net = gross - fee
                self.cash += net
                proceeds += gross
        return proceeds

    def buy_target_weights(self, weights: dict[str, float], prices: dict[str, float], total_equity: float):
        target_tickers = set(weights.keys())
        current_tickers = set(t for t, q in self.shares.items() if q > 0)

        to_sell = list(current_tickers - target_tickers)
        self.liquidate(to_sell, prices)

        target_dollars = {t: total_equity * w for t, w in weights.items()}

        for t in target_tickers:
            current_val = self.shares.get(t, 0.0) * prices.get(t, 0.0)
            desired_val = target_dollars.get(t, 0.0)
            if current_val > desired_val + 1e-6:
                excess_shares = (current_val - desired_val) / prices[t]
                gross = excess_shares * prices[t]
                fee = gross * TRANSACTION_COST
                self.shares[t] = self.shares.get(t, 0.0) - excess_shares
                self.cash += gross - fee

        for t in target_tickers:
            if prices.get(t, 0.0) <= 0:
                continue
            current_val = self.shares.get(t, 0.0) * prices.get(t, 0.0)
            desired_val = target_dollars.get(t, 0.0)
            if desired_val > current_val + 1e-6:
                spend = min(desired_val - current_val, self.cash)
                if spend <= 0: continue
                gross = spend
                fee = gross * TRANSACTION_COST
                net_spend = gross + fee
                net_spend = min(net_spend, self.cash)
                actual_gross = net_spend / (1 + TRANSACTION_COST)
                self.shares[t] = self.shares.get(t, 0.0) + actual_gross / prices[t]
                self.cash -= net_spend

def run_endowus_backtest(
    strategy: str,
    signals_df: pd.DataFrame,
    weekly_prices: pd.DataFrame,
    weekly_exec_prices: pd.DataFrame,
    daily_prices: pd.DataFrame
) -> tuple[pd.Series, pd.DataFrame]:

    port = Portfolio(INITIAL_EQUITY)
    equity_curve = {}
    weights_log = {}

    tickers = list(ENDOWUS_WEIGHTS.keys())

    dates = weekly_prices.index.intersection(weekly_exec_prices.index)
    if not signals_df.empty:
        signal_dates = pd.DatetimeIndex(signals_df['Date'].unique())
        dates = dates.intersection(signal_dates)

    for date in dates:
        price_row = weekly_prices.loc[date]
        exec_row = weekly_exec_prices.loc[date]

        val_prices = {t: float(price_row[t]) for t in tickers if pd.notna(price_row.get(t)) and float(price_row.get(t, 0)) > 0}
        exec_prices = {t: float(exec_row[t]) for t in tickers if pd.notna(exec_row.get(t)) and float(exec_row.get(t, 0)) > 0}

        for t in tickers:
            if t not in exec_prices and t in val_prices:
                exec_prices[t] = val_prices[t]

        active_set = list(exec_prices.keys())

        views = {}
        confidences = {}

        if strategy == "Endowus-60/40":
            # Just use static target weights, normalized for active assets
            target_weights_dict = {t: ENDOWUS_WEIGHTS.get(t, 0.0) for t in active_set}
            total_w = sum(target_weights_dict.values())
            weights = {t: target_weights_dict[t]/total_w for t in active_set} if total_w > 0 else {}
        else:
            # LLM strategies: fetch from signals and run Black-Litterman
            if not signals_df.empty:
                day_signals = signals_df[signals_df['Date'] == date]
                for _, row in day_signals.iterrows():
                    t = row['Ticker']
                    if t in active_set:
                        views[t] = row['target_return']
                        confidences[t] = row['confidence']

            lookback_start = date - pd.Timedelta(weeks=LOOKBACK_WEEKS)
            history = daily_prices.loc[lookback_start:date, :]

            weights = compute_bl_weights(active_set, history, views, confidences)

        te = port.total_equity(val_prices)
        port.buy_target_weights(weights, exec_prices, te)
        equity_curve[date] = port.total_equity(val_prices)
        weights_log[date] = weights

    weights_df = pd.DataFrame(weights_log).T.sort_index().fillna(0.0)
    weights_df.index.name = "Date"
    return pd.Series(equity_curve, name=strategy).sort_index(), weights_df

def compute_metrics(equity: pd.Series, name: str = "") -> dict:
    eq = equity.dropna()
    if len(eq) < 2: return {}

    weekly_returns = eq.pct_change().dropna()
    total_return = (eq.iloc[-1] / eq.iloc[0]) - 1
    n_years = len(eq) / WEEKS_PER_YEAR
    ann_return = (1 + total_return) ** (1 / n_years) - 1

    excess = weekly_returns - RISK_FREE_RATE / WEEKS_PER_YEAR
    sharpe = ((excess.mean() / excess.std()) * np.sqrt(WEEKS_PER_YEAR) if excess.std() > 0 else 0.0)

    rolling_max = eq.cummax()
    drawdowns = (eq - rolling_max) / rolling_max
    max_dd = drawdowns.min()

    calmar = ann_return / abs(max_dd) if max_dd != 0 else np.nan

    return {
        "Strategy": name or equity.name,
        "Ann. Return": f"{ann_return:.2%}",
        "Sharpe Ratio": f"{sharpe:.3f}",
        "Max Drawdown": f"{max_dd:.2%}",
        "Calmar Ratio": f"{calmar:.3f}"
    }

def main():
    print("=" * 60)
    print("  Endowus Black-Litterman Portfolio Engine")
    print("=" * 60)

    tickers = list(ENDOWUS_WEIGHTS.keys())

    daily_prices, daily_open = download_prices(tickers, START_DATE, END_DATE)

    valid_tickers = [t for t in tickers if t in daily_prices.columns]
    daily_prices = daily_prices[valid_tickers]
    daily_open = daily_open[[t for t in valid_tickers if t in daily_open.columns]]

    weekly_prices = to_weekly_monday(daily_prices)
    common_dates = weekly_prices.index
    weekly_exec_prices = get_execution_prices(daily_open, common_dates)

    signals_A = load_bl_signal_csvs("results_endowus_A")
    signals_B = load_bl_signal_csvs("results_endowus_B")
    signals_C = load_bl_signal_csvs("results_endowus_C")
    signals_D = load_bl_signal_csvs("results_endowus_D")

    curves = {}
    weights_logs = {}

    strategies = [
        ("Endowus-60/40", pd.DataFrame()),
        ("LLM-BL-A", signals_A),
        ("LLM-BL-B", signals_B),
        ("LLM-BL-C", signals_C),
        ("LLM-BL-D", signals_D)
    ]

    for name, sigs in strategies:
        print(f"Running {name}...")
        if name.startswith("LLM") and sigs.empty:
            print(f"  Skipping {name} - no signal data.")
            continue

        curve, wlog = run_endowus_backtest(name, sigs, weekly_prices, weekly_exec_prices, daily_prices)
        curves[name] = curve
        weights_logs[name] = wlog
        print(f"  Final equity: ${curve.iloc[-1]:,.2f}")

    print("\n[Metrics Summary]")
    print("=" * 70)
    metrics_list = [compute_metrics(c, n) for n, c in curves.items()]
    summary = pd.DataFrame(metrics_list)
    if not summary.empty:
        print(summary.to_string(index=False))
    print("=" * 70)

    os.makedirs("results", exist_ok=True)
    summary.to_csv("results/endowus_bl_performance_summary.csv", index=False)
    pd.DataFrame(curves).to_csv("results/endowus_bl_equity_curves.csv")

    print("Backtest complete. Results saved to results/ folder.")

if __name__ == "__main__":
    main()
