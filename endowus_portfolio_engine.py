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
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yfinance as yf
import portfolio_common as common_portfolio
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
RISK_FREE_TICKER = "^IRX"
RISK_FREE_SERIES = pd.Series(dtype=float)
WEEKS_PER_YEAR = 52
WEIGHT_SWING_ALERT_THRESHOLD = 0.80
EFFICIENT_FRONTIER_L2_GAMMA = 0.5
OPTIMIZER_METHOD = "max-sharpe"
OPTIMIZER_RISK_AVERSION = 2.5
OPTIMIZER_CHOICES = ("max-sharpe", "max-quadratic-utility", "min-volatility")


LLM_DIAGNOSTICS_LOG: list[dict] = []


def configure_optimizer(method: str = "max-sharpe", risk_aversion: float = 2.5):
    global OPTIMIZER_METHOD, OPTIMIZER_RISK_AVERSION
    OPTIMIZER_METHOD, OPTIMIZER_RISK_AVERSION = common_portfolio.set_optimizer_config(
        method, risk_aversion, OPTIMIZER_CHOICES
    )


def configure_risk_free_rate(rate: float):
    global RISK_FREE_RATE
    RISK_FREE_RATE = common_portfolio.set_risk_free_rate(rate)


def configure_risk_free_series(series: pd.Series | None):
    global RISK_FREE_SERIES
    if series is None:
        RISK_FREE_SERIES = pd.Series(dtype=float)
        return
    clean = pd.Series(series.values, index=pd.to_datetime(series.index)).sort_index()
    RISK_FREE_SERIES = clean.dropna()


def load_us_3m_tbill_rate(start: str, end: str) -> float:
    return common_portfolio.load_us_3m_tbill_rate(
        start=start,
        end=end,
        ticker=RISK_FREE_TICKER,
        fallback_rate=RISK_FREE_RATE,
    )


def load_us_3m_tbill_series(start: str, end: str) -> pd.Series:
    return common_portfolio.load_us_3m_tbill_series(
        start=start,
        end=end,
        ticker=RISK_FREE_TICKER,
    )


def optimize_efficient_frontier(
    expected_rets, cov_matrix, context: str = ""
) -> tuple[EfficientFrontier, str]:
    return common_portfolio.optimize_efficient_frontier(
        expected_rets=expected_rets,
        cov_matrix=cov_matrix,
        optimizer_method=OPTIMIZER_METHOD,
        risk_aversion=OPTIMIZER_RISK_AVERSION,
        risk_free_rate=RISK_FREE_RATE,
        l2_gamma=EFFICIENT_FRONTIER_L2_GAMMA,
        context=context,
    )


def is_invalid_view(value) -> bool:
    """Return True when a view value is missing or non-finite (None/NaN/inf)."""
    return common_portfolio.is_invalid_view(value)


def resolve_view_value(views: dict, ticker: str, fallback: float) -> float:
    return common_portfolio.resolve_view_value(views, ticker, fallback)


def reset_diagnostics_log():
    LLM_DIAGNOSTICS_LOG.clear()


def _append_diagnostics_event(event: dict):
    LLM_DIAGNOSTICS_LOG.append(event)


def get_diagnostics_log_df() -> pd.DataFrame:
    if not LLM_DIAGNOSTICS_LOG:
        return pd.DataFrame()
    return pd.DataFrame(LLM_DIAGNOSTICS_LOG)


def save_diagnostics_log(
    output_dir: str = "results/diagnostics",
    file_name: str = "endowus_engine_llm_diagnostics.csv",
):
    os.makedirs(output_dir, exist_ok=True)
    df = get_diagnostics_log_df()
    out_path = os.path.join(output_dir, file_name)
    if df.empty:
        pd.DataFrame(columns=["EventType"]).to_csv(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    print(f"Saved LLM diagnostics to {out_path}")


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
TARGET_EQUITY_ALLOCATION = 0.60

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
    return common_portfolio.load_endowus_6040_weights(
        raw_json_path=raw_json_path,
        id_to_yahoo_csv_path=id_to_yahoo_csv_path,
    )


def load_bl_signal_csvs(data_dir: str) -> pd.DataFrame:
    return common_portfolio.load_bl_signal_csvs(data_dir)


def to_weekly_monday(df: pd.DataFrame) -> pd.DataFrame:
    return common_portfolio.to_weekly_monday(df)


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
    return common_portfolio.get_execution_prices(daily_open, weekly_dates)


def get_latest_price_on_or_before(
    data: pd.DataFrame, ticker: str, as_of: pd.Timestamp
) -> float | None:
    """Return latest non-null price at or before as_of for ticker."""
    return common_portfolio.get_latest_price_on_or_before(data, ticker, as_of)


def compute_bl_weights(
    active_set: list[str],
    price_history: pd.DataFrame,
    views: dict,
    confidences: dict,
    diagnostics_context: dict | None = None,
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
                t: resolve_view_value(views, t, float(market_prior.get(t, 0.05)))
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

        context = (
            f"{diagnostics_context.get('strategy') if diagnostics_context else 'Unknown'}, "
            f"{diagnostics_context.get('date') if diagnostics_context else 'Unknown'}"
        )
        ef, optimizer_mode = optimize_efficient_frontier(
            bl_returns, bl_cov, context=context
        )

        cleaned = ef.clean_weights()
        cleaned_dict = {str(k): float(v) for k, v in cleaned.items()}
        raw_cleaned = {str(t): float(cleaned_dict.get(str(t), 0.0)) for t in active_set}

        if diagnostics_context is not None:
            _append_diagnostics_event(
                {
                    "EventType": "optimizer_result",
                    "Strategy": diagnostics_context.get("strategy"),
                    "Date": diagnostics_context.get("date"),
                    "OptimizerMode": optimizer_mode,
                    "ActiveSetSize": len(active_set),
                    "ActiveSet": json.dumps(active_set),
                    "Views": json.dumps(views, default=float),
                    "Confidences": json.dumps(confidences, default=float),
                    "RawCleanedWeights": json.dumps(raw_cleaned, default=float),
                }
            )

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
    prev_active_set: set[str] = set()

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
        active_set_now = set(active_set)
        if active_set_now != prev_active_set:
            added = sorted(active_set_now - prev_active_set)
            removed = sorted(prev_active_set - active_set_now)
            if prev_active_set:
                _append_diagnostics_event(
                    {
                        "EventType": "active_set_change",
                        "Strategy": strategy,
                        "Date": pd.to_datetime(date),
                        "PrevActiveSetSize": len(prev_active_set),
                        "ActiveSetSize": len(active_set_now),
                        "Added": json.dumps(added),
                        "Removed": json.dumps(removed),
                        "ActiveSet": json.dumps(sorted(active_set_now)),
                    }
                )
                if added or removed:
                    print(
                        f"[INFO] Active set changed for {strategy} on {pd.to_datetime(date).date()}"
                        f" | +{added} -{removed}"
                    )
            prev_active_set = active_set_now

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
            configure_risk_free_rate(
                common_portfolio.get_risk_free_rate_for_date(
                    pd.to_datetime(date), RISK_FREE_SERIES, RISK_FREE_RATE
                )
            )
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

                weights = compute_bl_weights(
                    active_set,
                    history,
                    views,
                    confidences,
                    diagnostics_context={"strategy": strategy, "date": date},
                )

                _append_diagnostics_event(
                    {
                        "EventType": "llm_views",
                        "Strategy": strategy,
                        "Date": pd.to_datetime(date),
                        "ActiveSetSize": len(active_set),
                        "ActiveSet": json.dumps(active_set),
                        "Views": json.dumps(views, default=float),
                        "Confidences": json.dumps(confidences, default=float),
                    }
                )

                if last_target_weights:
                    union_assets = set(last_target_weights.keys()) | set(weights.keys())
                    max_abs_change = max(
                        abs(
                            float(weights.get(asset, 0.0))
                            - float(last_target_weights.get(asset, 0.0))
                        )
                        for asset in union_assets
                    )
                    if max_abs_change >= WEIGHT_SWING_ALERT_THRESHOLD:
                        print(
                            f"[ALERT] Large weight swing for {strategy} on {pd.to_datetime(date).date()}"
                            f" | max abs change={max_abs_change:.2%}"
                        )
                        _append_diagnostics_event(
                            {
                                "EventType": "weight_swing_alert",
                                "Strategy": strategy,
                                "Date": pd.to_datetime(date),
                                "Threshold": WEIGHT_SWING_ALERT_THRESHOLD,
                                "MaxAbsWeightChange": float(max_abs_change),
                                "Views": json.dumps(views, default=float),
                                "Confidences": json.dumps(confidences, default=float),
                                "PrevWeights": json.dumps(
                                    last_target_weights, default=float
                                ),
                                "NewWeights": json.dumps(weights, default=float),
                            }
                        )

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

    period_rf = common_portfolio.get_period_risk_free_returns(
        periodic_returns.index,
        periods_per_year,
        RISK_FREE_SERIES,
        RISK_FREE_RATE,
    )
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
                        monthly_rf = common_portfolio.get_period_risk_free_returns(
                            monthly_returns.index,
                            12,
                            RISK_FREE_SERIES,
                            RISK_FREE_RATE,
                        )
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

    # Place legend below the plot so portfolio names are fully visible
    handles, labels = ax.get_legend_handles_labels()
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
    print(f"Plot saved to {output_path}")


def _run_single_strategy_endowus(payload: dict) -> dict:
    name = payload["name"]
    sigs = payload["signals"]

    if name.startswith("LLM") and sigs.empty:
        return {"name": name, "skipped": True}

    curve, wlog = run_endowus_backtest(
        name,
        sigs,
        payload["weekly_prices"],
        payload["weekly_exec_prices"],
        payload["daily_prices"],
        transaction_cost=TRANSACTION_COST,
    )
    return {"name": name, "curve": curve, "weights": wlog, "skipped": False}


def _run_single_strategy_endowus_worker(payload: dict) -> dict:
    configure_optimizer(payload["optimizer_method"], payload["risk_aversion"])
    configure_risk_free_series(payload.get("risk_free_series"))
    configure_risk_free_rate(payload["risk_free_rate"])
    return _run_single_strategy_endowus(payload)


def main():
    print("=" * 60)

    parser = argparse.ArgumentParser(description="Run Endowus BL portfolio backtests")
    parser.add_argument(
        "--optimizer",
        choices=list(OPTIMIZER_CHOICES),
        default=OPTIMIZER_METHOD,
        help="Optimizer to solve EfficientFrontier",
    )
    parser.add_argument(
        "--risk-aversion",
        type=float,
        default=OPTIMIZER_RISK_AVERSION,
        help="Risk aversion for max-quadratic-utility",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Number of worker processes to run strategies in parallel",
    )
    args = parser.parse_args()
    configure_optimizer(args.optimizer, args.risk_aversion)
    configure_risk_free_series(load_us_3m_tbill_series(START_DATE, END_DATE))
    configure_risk_free_rate(
        common_portfolio.get_risk_free_rate_for_date(
            pd.to_datetime(END_DATE), RISK_FREE_SERIES, RISK_FREE_RATE
        )
    )
    print(
        f"Optimizer: {OPTIMIZER_METHOD}"
        f" (risk_aversion={OPTIMIZER_RISK_AVERSION}, l2_gamma={EFFICIENT_FRONTIER_L2_GAMMA})"
    )
    print(f"Risk-free rate: {RISK_FREE_RATE:.4%} ({RISK_FREE_TICKER})")
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

    strategy_payloads = [
        {
            "name": name,
            "signals": sigs,
            "weekly_prices": weekly_prices,
            "weekly_exec_prices": weekly_exec_prices,
            "daily_prices": daily_prices,
            "optimizer_method": OPTIMIZER_METHOD,
            "risk_aversion": OPTIMIZER_RISK_AVERSION,
            "risk_free_rate": RISK_FREE_RATE,
            "risk_free_series": RISK_FREE_SERIES,
        }
        for name, sigs in strategies
    ]

    if args.parallel_workers > 1:
        print(f"Running strategies in parallel with {args.parallel_workers} workers...")
        results_map: dict[str, dict] = {}
        with ProcessPoolExecutor(max_workers=args.parallel_workers) as executor:
            future_map = {
                executor.submit(_run_single_strategy_endowus_worker, payload): payload[
                    "name"
                ]
                for payload in strategy_payloads
            }
            for future in as_completed(future_map):
                name = future_map[future]
                try:
                    result = future.result()
                    results_map[name] = result
                    if result.get("skipped"):
                        print(f"  Skipping {name} - no signal data.")
                    else:
                        print(
                            f"Completed {name}: Final equity ${result['curve'].iloc[-1]:,.2f}"
                        )
                except Exception as ex:
                    print(f"[ERROR] Strategy failed for {name}: {ex}")

        for name, _ in strategies:
            result = results_map.get(name)
            if not result or result.get("skipped"):
                continue
            curves[name] = result["curve"]
            weights_logs[name] = result["weights"]
    else:
        for payload in strategy_payloads:
            name = payload["name"]
            print(f"Running {name}...")
            result = _run_single_strategy_endowus(payload)
            if result.get("skipped"):
                print(f"  Skipping {name} - no signal data.")
                continue
            curves[name] = result["curve"]
            weights_logs[name] = result["weights"]
            print(f"  Final equity: ${result['curve'].iloc[-1]:,.2f}")

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
