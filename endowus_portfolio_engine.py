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
from pypfopt import (
    BlackLittermanModel,
    risk_models,
    expected_returns,
    EfficientFrontier,
)

warnings.filterwarnings("ignore")

DATA_DIR = "data"
START_DATE = "2019-01-01"
END_DATE = "2024-12-31"
INITIAL_EQUITY = 10_000.0
TRANSACTION_COST = 0.001
LOOKBACK_WEEKS = 52
RISK_FREE_RATE = 0.04
WEEKS_PER_YEAR = 52

# Fallback proxy weights if raw Endowus composition cannot be loaded.
DEFAULT_ENDOWUS_WEIGHTS = {
    "0P0001AF7U.SI": 0.195,  # Dimensional Global Core Equity Fund
    "0P0000KYEE.SI": 0.1,  # PIMCO GIS Global Bond Fund SGD-Hedged
    "SPY": 0.09,  # iShares US Index Fund (IE) S&P 500
    "^990100-USD-STRD": 0.09,  # iShares Developed World Index Fund (IE)
    "DE000SLA4YD9.SG": 0.084,  # Amundi Prime USA Fund
    "AGGG.L": 0.08,  # Amundi Core Global Aggregate Bond SGD-Hedged
    "IE0002461055.IR": 0.07,  # PIMCO GIS Income Fund SGD-Hedged
    "0P0001AF7Z.SI": 0.06,  # Dimensional Emerging Markets Large Cap Core Equity Fund
    "0P0001EQUE.SI": 0.05,  # Dimensional Global Core Fixed Income Fund SGD-Hedged
    "0P0001EF2T.SI": 0.048,  # Dimensional Pacific Basin Small Companies Fund
    "0P0001CC3M": 0.04,  # iShares Global Aggregate 1-5 Year Bond Index Fund (IE) SGD-Hedged
    "PEBIX": 0.04,  # iShares Emerging Markets Government Bond Index Fund (IE)
    "EIMI.L": 0.033,  # Amundi Core MSCI Emerging Markets Fund
    "0P0001DWI0.SI": 0.02,  # PIMCO GIS Emerging Markets Bond Fund SGD-Hedged
}

ENDOWUS_WEIGHTS = {
    "0P0001AF7U.SI": 0.195,  # Dimensional Global Core Equity Fund
    "0P0000KYEE.SI": 0.1,  # PIMCO GIS Global Bond Fund SGD-Hedged
    "SPY": 0.09,  # iShares US Index Fund (IE) S&P 500
    "^990100-USD-STRD": 0.09,  # iShares Developed World Index Fund (IE)
    "DE000SLA4YD9.SG": 0.084,  # Amundi Prime USA Fund
    "AGGG.L": 0.08,  # Amundi Core Global Aggregate Bond SGD-Hedged
    "IE0002461055.IR": 0.07,  # PIMCO GIS Income Fund SGD-Hedged
    "0P0001AF7Z.SI": 0.06,  # Dimensional Emerging Markets Large Cap Core Equity Fund
    "0P0001EQUE.SI": 0.05,  # Dimensional Global Core Fixed Income Fund SGD-Hedged
    "0P0001EF2T.SI": 0.048,  # Dimensional Pacific Basin Small Companies Fund
    "0P0001CC3M": 0.04,  # iShares Global Aggregate 1-5 Year Bond Index Fund (IE) SGD-Hedged
    "PEBIX": 0.04,  # iShares Emerging Markets Government Bond Index Fund (IE)
    "EIMI.L": 0.033,  # Amundi Core MSCI Emerging Markets Fund
    "0P0001DWI0.SI": 0.02,  # PIMCO GIS Emerging Markets Bond Fund SGD-Hedged
}

# Adjustable class-level allocation target used by optimization outputs.
TARGET_EQUITY_ALLOCATION = 0.80

# Default class map for the fallback ETF universe.
EQUITY_ASSETS = ["URTH", "SPY", "EEM", "VPL"]
FIXED_INCOME_ASSETS = ["BNDW", "AGG", "EMB", "BSV"]


def infer_periods_per_year(index: pd.Index) -> int:
    """Infer sampling frequency from the time index for annualization."""
    if len(index) < 3:
        return WEEKS_PER_YEAR

    diffs = index.to_series().diff().dropna().dt.days
    if diffs.empty:
        return WEEKS_PER_YEAR

    median_days = float(diffs.median())
    if median_days <= 10:
        return 52
    if median_days <= 40:
        return 12
    if median_days <= 120:
        return 4
    return 1


def align_and_rebase_curves(
    curves: dict[str, pd.Series], base_value: float = INITIAL_EQUITY
) -> tuple[dict[str, pd.Series], pd.Timestamp | None]:
    """Trim all curves to a common start date and rebase to the same base value."""
    valid_curves = {
        name: s.sort_index().dropna()
        for name, s in curves.items()
        if isinstance(s, pd.Series) and not s.dropna().empty
    }
    if not valid_curves:
        return {}, None

    common_start = max(s.index.min() for s in valid_curves.values())

    aligned: dict[str, pd.Series] = {}
    for name, s in valid_curves.items():
        trimmed = s[s.index >= common_start].dropna()
        if trimmed.empty:
            continue

        first_val = float(trimmed.iloc[0])
        if first_val > 0:
            trimmed = (trimmed / first_val) * base_value
        trimmed.name = name
        aligned[name] = trimmed

    return aligned, common_start


def enforce_class_allocation_targets(
    weights: dict[str, float],
    active_set: list[str],
    equity_assets: list[str] | None = None,
    fixed_income_assets: list[str] | None = None,
    target_equity_allocation: float = TARGET_EQUITY_ALLOCATION,
) -> dict[str, float]:
    """Rescale long-only weights so class totals match target equity/fixed-income split."""
    if not active_set:
        return {}

    eq_assets = set(equity_assets or EQUITY_ASSETS)
    fi_assets = set(fixed_income_assets or FIXED_INCOME_ASSETS)

    base = {t: max(float(weights.get(t, 0.0)), 0.0) for t in active_set}
    base_sum = sum(base.values())
    if base_sum <= 0:
        base = {t: 1.0 / len(active_set) for t in active_set}
    else:
        base = {t: w / base_sum for t, w in base.items()}

    eq_tickers = [t for t in active_set if t in eq_assets]
    fi_tickers = [t for t in active_set if t in fi_assets]
    other_tickers = [t for t in active_set if t not in eq_assets and t not in fi_assets]

    if not eq_tickers and not fi_tickers:
        return base

    target_eq = min(max(float(target_equity_allocation), 0.0), 1.0)
    target_fi = 1.0 - target_eq

    if not eq_tickers:
        target_eq, target_fi = 0.0, 1.0
    if not fi_tickers:
        target_eq, target_fi = 1.0, 0.0

    adjusted = {t: 0.0 for t in active_set}

    eq_sum = sum(base[t] for t in eq_tickers)
    fi_sum = sum(base[t] for t in fi_tickers)

    if eq_tickers:
        if eq_sum > 0:
            for t in eq_tickers:
                adjusted[t] = (base[t] / eq_sum) * target_eq
        else:
            for t in eq_tickers:
                adjusted[t] = target_eq / len(eq_tickers)

    if fi_tickers:
        if fi_sum > 0:
            for t in fi_tickers:
                adjusted[t] = (base[t] / fi_sum) * target_fi
        else:
            for t in fi_tickers:
                adjusted[t] = target_fi / len(fi_tickers)

    for t in other_tickers:
        adjusted[t] = 0.0

    total = sum(adjusted.values())
    if total > 0:
        adjusted = {t: w / total for t, w in adjusted.items()}

    return adjusted


def load_endowus_6040_weights(
    raw_json_path: str = "data/endowus_raw.json",
    id_to_yahoo_csv_path: str = "data/endowus_yfinance/instrument_id_to_yahoo.csv",
) -> dict[str, float]:
    """Load 60|40 portfolio target weights and Yahoo symbols from Endowus raw data."""
    if not os.path.exists(raw_json_path):
        print(
            f"[WARN] Endowus raw data not found at {raw_json_path}; using fallback weights."
        )
        return DEFAULT_ENDOWUS_WEIGHTS.copy()

    if not os.path.exists(id_to_yahoo_csv_path):
        print(
            f"[WARN] Instrument to Yahoo mapping not found at {id_to_yahoo_csv_path}; "
            "using fallback weights."
        )
        return DEFAULT_ENDOWUS_WEIGHTS.copy()

    try:
        with open(raw_json_path, "r") as f:
            all_portfolios = json.load(f)

        portfolio_6040 = next(
            (p for p in all_portfolios if p.get("shortName") == "60 | 40"), None
        )
        if portfolio_6040 is None:
            print("[WARN] Could not find shortName '60 | 40'; using fallback weights.")
            return DEFAULT_ENDOWUS_WEIGHTS.copy()

        funds = portfolio_6040.get("portfolioUnderlyingPage", {}).get(
            "underlyingFundsCards", []
        )
        if not funds:
            print("[WARN] 60|40 underlyingFundsCards empty; using fallback weights.")
            return DEFAULT_ENDOWUS_WEIGHTS.copy()

        mapping_df = pd.read_csv(id_to_yahoo_csv_path)
        id_to_symbol = {
            str(row["instrumentId"]): str(row["yahooSymbol"])
            for _, row in mapping_df.iterrows()
            if str(row.get("yahooSymbol", "")).strip()
            and str(row.get("status", "")).startswith("resolved")
        }

        weights: dict[str, float] = {}
        dropped = 0
        for fund in funds:
            instrument_id = str(fund.get("instrumentId", "")).strip()
            target_weight = float(fund.get("targetWeight", 0.0))
            symbol = id_to_symbol.get(instrument_id)

            if not symbol or target_weight <= 0:
                dropped += 1
                continue

            weights[symbol] = weights.get(symbol, 0.0) + target_weight

        total_weight = sum(weights.values())
        if not weights or total_weight <= 0:
            print("[WARN] 60|40 produced no mapped symbols; using fallback weights.")
            return DEFAULT_ENDOWUS_WEIGHTS.copy()

        normalized_weights = {t: w / total_weight for t, w in weights.items()}
        print(
            f"Loaded Endowus 60|40 universe: {len(normalized_weights)} tickers "
            f"(dropped {dropped} unmapped funds)."
        )
        return normalized_weights

    except Exception as e:
        print(f"[WARN] Failed loading 60|40 composition ({e}); using fallback weights.")
        return DEFAULT_ENDOWUS_WEIGHTS.copy()


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
                decision_str = str(row["decision"]).strip()

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
                    ret_str = str(data.get("Target_Return_30d", "")).replace("%", "")

                    if not ret_str:
                        target_returns.append(None)
                    else:
                        ret = float(ret_str) / 100.0
                        annual_ret = (1 + ret) ** (365 / 30) - 1
                        target_returns.append(annual_ret)

                    conf = float(data.get("Confidence_Score", 5))
                    confidences.append(conf)
                except Exception as e:
                    target_returns.append(None)
                    confidences.append(5.0)

            df["target_return"] = target_returns
            df["confidence"] = confidences
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
    return close_df.sort_index().dropna(how="all"), open_df.sort_index().dropna(
        how="all"
    )


def get_execution_prices(
    daily_open: pd.DataFrame, weekly_dates: pd.DatetimeIndex
) -> pd.DataFrame:
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


def get_latest_price_on_or_before(
    data: pd.DataFrame, ticker: str, as_of: pd.Timestamp
) -> float | None:
    """Return latest non-null price at or before as_of for ticker."""
    if ticker not in data.columns:
        return None

    hist = data.loc[:as_of, ticker].dropna()
    if hist.empty:
        return None
    return float(hist.iloc[-1])


def compute_bl_weights(
    active_set: list[str], price_history: pd.DataFrame, views: dict, confidences: dict
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
        target_weights = np.array(
            [target_weights_dict[t] / total_w for t in active_set]
        )
        fallback_weights = {t: target_weights_dict[t] / total_w for t in active_set}
    else:
        # Extreme fallback
        fallback_weights = {t: 1.0 / len(active_set) for t in active_set}
        target_weights = np.array(list(fallback_weights.values()))

    fallback_weights = enforce_class_allocation_targets(fallback_weights, active_set)

    if sub.shape[0] < 10 or len(active_set) < 2:
        return fallback_weights

    try:
        # Calculate covariance matrix
        S = risk_models.CovarianceShrinkage(sub, frequency=252).ledoit_wolf()

        # Calculate market-implied returns (prior) assuming Endowus is the equilibrium
        delta = 2.5  # Risk aversion parameter
        # market implied returns pi = delta * S * w
        market_prior = delta * S.dot(target_weights)

        # Format views. If LLM gave None, fallback to the calculated market prior
        Q = pd.Series(
            {
                t: (
                    views.get(t)
                    if views.get(t) is not None
                    else market_prior.get(t, 0.05)
                )
                for t in active_set
            }
        )

        # Construct confidence matrix Omega (diagonal)
        # Scale uncertainty inversely to confidence (1-10)
        omega = np.diag(
            [S.loc[t, t] * (1.1 - (confidences.get(t, 5) / 10.0)) for t in active_set]
        )

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
        weights = {str(t): float(w) for t, w in cleaned.items() if w > 1e-6}
        if not weights:
            return fallback_weights

        return enforce_class_allocation_targets(weights, active_set)

    except Exception as e:
        print(f"[DEBUG] BL Math failed for {active_set}: {e}")
        return fallback_weights


class Portfolio:
    def __init__(
        self, initial_equity: float, transaction_cost: float = TRANSACTION_COST
    ):
        self.cash = initial_equity
        self.shares: dict[str, float] = {}
        self.transaction_cost = float(transaction_cost)

    def total_equity(self, prices: dict[str, float]) -> float:
        mv = sum(self.shares.get(t, 0.0) * prices.get(t, 0.0) for t in self.shares)
        return self.cash + mv

    def current_weights(self, prices: dict[str, float]) -> dict[str, float]:
        total = self.total_equity(prices)
        if total <= 0:
            return {}

        weights: dict[str, float] = {}
        for t, qty in self.shares.items():
            px = prices.get(t, 0.0)
            if qty <= 0 or px <= 0:
                continue
            w = (qty * px) / total
            if w > 1e-8:
                weights[t] = w
        return weights

    def liquidate(self, tickers_to_sell: list[str], prices: dict[str, float]) -> float:
        proceeds = 0.0
        for t in tickers_to_sell:
            qty = self.shares.pop(t, 0.0)
            if qty > 0:
                gross = qty * prices[t]
                fee = gross * self.transaction_cost
                net = gross - fee
                self.cash += net
                proceeds += gross
        return proceeds

    def buy_target_weights(
        self, weights: dict[str, float], prices: dict[str, float], total_equity: float
    ):
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
                fee = gross * self.transaction_cost
                self.shares[t] = self.shares.get(t, 0.0) - excess_shares
                self.cash += gross - fee

        for t in target_tickers:
            if prices.get(t, 0.0) <= 0:
                continue
            current_val = self.shares.get(t, 0.0) * prices.get(t, 0.0)
            desired_val = target_dollars.get(t, 0.0)
            if desired_val > current_val + 1e-6:
                spend = min(desired_val - current_val, self.cash)
                if spend <= 0:
                    continue
                gross = spend
                fee = gross * self.transaction_cost
                net_spend = gross + fee
                net_spend = min(net_spend, self.cash)
                actual_gross = net_spend / (1 + self.transaction_cost)
                self.shares[t] = self.shares.get(t, 0.0) + actual_gross / prices[t]
                self.cash -= net_spend


def run_endowus_backtest(
    strategy: str,
    signals_df: pd.DataFrame,
    weekly_prices: pd.DataFrame,
    weekly_exec_prices: pd.DataFrame,
    daily_prices: pd.DataFrame,
    rebalance_policy: str = "weekly",
    log_realized_weights: bool = False,
    transaction_cost: float = TRANSACTION_COST,
) -> tuple[pd.Series, pd.DataFrame]:

    port = Portfolio(INITIAL_EQUITY, transaction_cost)
    equity_curve = {}
    weights_log = {}
    last_target_weights: dict[str, float] = {}
    has_rebalanced_once = False

    tickers = list(ENDOWUS_WEIGHTS.keys())

    dates = weekly_prices.index.intersection(weekly_exec_prices.index)
    if not signals_df.empty:
        signal_dates = pd.DatetimeIndex(signals_df["Date"].unique())
        dates = dates.intersection(signal_dates)

    for date in dates:
        val_prices: dict[str, float] = {}
        exec_prices: dict[str, float] = {}

        for t in tickers:
            val_p = get_latest_price_on_or_before(weekly_prices, t, date)
            if val_p is None or val_p <= 0:
                continue
            val_prices[t] = val_p

            exec_p = get_latest_price_on_or_before(weekly_exec_prices, t, date)
            if exec_p is None or exec_p <= 0:
                # Fallback to valuation price if execution series has no value yet.
                exec_p = val_p
            exec_prices[t] = exec_p

        active_set = list(exec_prices.keys())
        full_universe_available = len(active_set) == len(tickers)

        if rebalance_policy == "weekly":
            should_rebalance = True
        elif rebalance_policy == "full-universe-only":
            should_rebalance = full_universe_available
        elif rebalance_policy == "buy-and-hold":
            should_rebalance = (not has_rebalanced_once) and full_universe_available
        else:
            raise ValueError(
                "rebalance_policy must be one of: weekly, full-universe-only, buy-and-hold"
            )

        if should_rebalance:
            views = {}
            confidences = {}

            if strategy.startswith("Endowus-60/40"):
                # Static target weights, normalized for assets currently tradable.
                target_weights_dict = {
                    t: ENDOWUS_WEIGHTS.get(t, 0.0) for t in active_set
                }
                total_w = sum(target_weights_dict.values())
                weights = (
                    {t: target_weights_dict[t] / total_w for t in active_set}
                    if total_w > 0
                    else {}
                )
                weights = enforce_class_allocation_targets(weights, active_set)
            else:
                # LLM strategies: fetch from signals and run Black-Litterman.
                if not signals_df.empty:
                    day_signals = signals_df[signals_df["Date"] == date]
                    for _, row in day_signals.iterrows():
                        t = row["Ticker"]
                        if t in active_set:
                            views[t] = row["target_return"]
                            confidences[t] = row["confidence"]

                lookback_start = date - pd.Timedelta(weeks=LOOKBACK_WEEKS)
                history = daily_prices.loc[lookback_start:date, :]

                weights = compute_bl_weights(active_set, history, views, confidences)

            te = port.total_equity(val_prices)
            port.buy_target_weights(weights, exec_prices, te)
            last_target_weights = weights.copy()
            if rebalance_policy == "buy-and-hold":
                has_rebalanced_once = True

        equity_curve[date] = port.total_equity(val_prices)

        if log_realized_weights:
            weights_log[date] = port.current_weights(val_prices)
        elif should_rebalance:
            weights_log[date] = last_target_weights
        elif last_target_weights:
            weights_log[date] = last_target_weights
        else:
            weights_log[date] = port.current_weights(val_prices)

    weights_df = pd.DataFrame(weights_log).T.sort_index().fillna(0.0)
    weights_df.index.name = "Date"
    return pd.Series(equity_curve, name=strategy).sort_index(), weights_df


def compute_metrics(equity: pd.Series, name: str = "") -> dict:
    eq = equity.dropna()
    if len(eq) < 2:
        return {}

    periodic_returns = eq.pct_change().dropna()
    periods_per_year = infer_periods_per_year(eq.index)

    total_return = (eq.iloc[-1] / eq.iloc[0]) - 1
    elapsed_days = (eq.index[-1] - eq.index[0]).days
    if elapsed_days > 0:
        n_years = elapsed_days / 365.25
    else:
        n_years = len(eq) / periods_per_year

    ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    period_rf = (1 + RISK_FREE_RATE) ** (1 / periods_per_year) - 1
    excess = periodic_returns - period_rf
    sharpe = (
        (excess.mean() / excess.std()) * np.sqrt(periods_per_year)
        if excess.std() > 0
        else 0.0
    )

    rolling_max = eq.cummax()
    drawdowns = (eq - rolling_max) / rolling_max
    max_dd = drawdowns.min()

    calmar = ann_return / abs(max_dd) if max_dd != 0 else np.nan

    return {
        "Strategy": name or equity.name,
        "Ann. Return": f"{ann_return:.2%}",
        "Sharpe Ratio": f"{sharpe:.3f}",
        "Max Drawdown": f"{max_dd:.2%}",
        "Calmar Ratio": f"{calmar:.3f}",
    }


def load_endowus_historical(curves: dict, metrics_list: list):
    """Loads actual Endowus historical NAVs and exact metrics from the provided JSON dump."""
    json_path = os.path.join("data", "endowus_raw.json")
    if not os.path.exists(json_path):
        print("Historical Endowus data not found.")
        return

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        for portfolio in data:
            name = f"Endowus_Actual_{portfolio['shortName'].replace(' | ', '/')}"
            normalized_curve = None

            # Extract and align NAV curve
            navs = portfolio.get("monthlyNavs", [])

            if navs:
                # Convert list of [date, nav] into a Series
                dates = [pd.to_datetime(row[0]) for row in navs]
                values = [float(row[1]) for row in navs]
                s = pd.Series(values, index=dates)

                # Filter to backtest window
                mask = (s.index >= pd.to_datetime(START_DATE)) & (
                    s.index <= pd.to_datetime(END_DATE)
                )
                s = s[mask]

                if not s.empty:
                    # Normalize to starting equity
                    s = (s / s.iloc[0]) * INITIAL_EQUITY
                    s.name = name
                    normalized_curve = s
                    curves[name] = s

            # Extract official metrics
            perf = portfolio.get("performanceMetrics", {})
            if perf or normalized_curve is not None:
                ann_return = perf.get("annualisedReturn") if perf else None
                max_dd = perf.get("maxDrawDown", {}).get("drawDown") if perf else None

                sharpe_str = "N/A"
                if normalized_curve is not None and len(normalized_curve) > 1:
                    monthly_returns = normalized_curve.pct_change().dropna()
                    if not monthly_returns.empty:
                        monthly_rf = (1 + RISK_FREE_RATE) ** (1 / 12) - 1
                        monthly_excess = monthly_returns - monthly_rf
                        vol = monthly_excess.std()
                        if vol > 0:
                            monthly_sharpe = (monthly_excess.mean() / vol) * np.sqrt(12)
                            sharpe_str = f"{monthly_sharpe:.3f}"
                        else:
                            sharpe_str = "0.000"

                        # Fallbacks if official metrics are absent.
                        if ann_return is None:
                            total_return = (
                                normalized_curve.iloc[-1] / normalized_curve.iloc[0]
                            ) - 1
                            n_years = len(normalized_curve) / 12
                            ann_return = (
                                (1 + total_return) ** (1 / n_years) - 1
                                if n_years > 0
                                else 0.0
                            )

                        if max_dd is None:
                            rolling_max = normalized_curve.cummax()
                            drawdowns = (normalized_curve - rolling_max) / rolling_max
                            max_dd = drawdowns.min()

                ann_return = 0.0 if ann_return is None else ann_return
                max_dd = 0.0 if max_dd is None else max_dd
                calmar = ann_return / abs(max_dd) if max_dd != 0 else np.nan

                metrics_list.append(
                    {
                        "Strategy": name,
                        "Ann. Return": f"{ann_return:.2%}",
                        "Sharpe Ratio": sharpe_str,
                        "Max Drawdown": f"{max_dd:.2%}",
                        "Calmar Ratio": f"{calmar:.3f}" if pd.notna(calmar) else "N/A",
                    }
                )
    except Exception as e:
        print(f"Error loading historical Endowus data: {e}")


def plot_equity_curves(
    curves_df: pd.DataFrame, output_path: str = "results/endowus_comparison_plot.png"
):
    """Plots the actual Endowus funds alongside our simulated portfolios."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Define styles to distinguish actual Endowus funds vs our simulations
    for column in curves_df.columns:
        if "Endowus_Actual" in column:
            # Render the actual funds as dashed lines with lower alpha so they form a "background" spectrum
            ax.plot(
                curves_df.index,
                curves_df[column],
                label=column.replace("Endowus_Actual_", "Actual "),
                ls="--",
                lw=1.5,
                alpha=0.7,
            )
        elif column == "Endowus-60/40":
            # Our proxy benchmark
            ax.plot(
                curves_df.index,
                curves_df[column],
                label="Proxy 60/40 Benchmark",
                color="black",
                lw=2.5,
                ls="-",
            )
        elif "LLM-BL" in column:
            # Our AI portfolios
            ax.plot(curves_df.index, curves_df[column], label=column, lw=2.0, ls="-")

    ax.set_title(
        "AI Black-Litterman Portfolios vs Actual Endowus Flagship Funds (2020-2024)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("Portfolio Value (Base $10,000)")
    ax.set_xlabel("Date")

    # Formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    # Place legend outside to not obscure the curves
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")


def main():
    print("=" * 60)
    print("  Endowus Black-Litterman Portfolio Engine")
    print("=" * 60)

    global ENDOWUS_WEIGHTS
    ENDOWUS_WEIGHTS = load_endowus_6040_weights()

    tickers = list(ENDOWUS_WEIGHTS.keys())
    print(f"Backtesting universe: {tickers}")

    daily_prices, daily_open = download_prices(tickers, START_DATE, END_DATE)
    print(f"Downloaded price data with shape {daily_prices.shape}")
    print(f"Price data columns: {daily_prices}")
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
        ("LLM-BL-D", signals_D),
    ]

    for name, sigs in strategies:
        print(f"Running {name}...")
        if name.startswith("LLM") and sigs.empty:
            print(f"  Skipping {name} - no signal data.")
            continue

        curve, wlog = run_endowus_backtest(
            name,
            sigs,
            weekly_prices,
            weekly_exec_prices,
            daily_prices,
            transaction_cost=TRANSACTION_COST,
        )
        curves[name] = curve
        weights_logs[name] = wlog
        print(f"  Final equity: ${curve.iloc[-1]:,.2f}")

    # Inject actual historical Endowus curves.
    # Metrics are recomputed after alignment/rebasing for apples-to-apples comparison.
    load_endowus_historical(curves, [])

    aligned_curves, common_start = align_and_rebase_curves(curves)
    if not aligned_curves:
        print("No valid curves available after alignment.")
        return

    curves = aligned_curves
    if common_start is not None:
        print(
            f"Aligned and rebased all curves from common start: {common_start.date()}"
        )

    metrics_list = [compute_metrics(c, n) for n, c in curves.items()]
    print("\n[Equity Curves]")
    for name, curve in curves.items():
        print(f"{name}: Final equity ${curve.iloc[-1]:,.2f}")
    print("\n[Metrics Summary]")
    print("=" * 70)
    summary = pd.DataFrame([m for m in metrics_list if m])  # Filter out any empties
    if not summary.empty:
        print(summary.to_string(index=False))
    print("=" * 70)

    os.makedirs("results", exist_ok=True)
    summary.to_csv("results/endowus_bl_performance_summary.csv", index=False)

    # Save curves, handle forward filling for the monthly Endowus data to align with weekly dates
    curves_df = pd.DataFrame(curves).ffill()
    curves_df.to_csv("results/endowus_bl_equity_curves.csv")

    # Generate the comparison plot
    plot_equity_curves(curves_df)

    print("Backtest complete. Results saved to results/ folder.")


if __name__ == "__main__":
    main()
