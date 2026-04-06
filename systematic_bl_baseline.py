import glob
import json
import os
import re
import warnings
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import portfolio_common as common_portfolio
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
import statsmodels.api as sm
from pypfopt import (
    BlackLittermanModel,
    risk_models,
    EfficientFrontier,
)

warnings.filterwarnings("ignore")

DATA_DIR = "data"
START_DATE = "2019-09-01"
END_DATE = "2024-12-31"
INITIAL_EQUITY = 10_000.0
TRANSACTION_COST = 0.001
RISK_FREE_RATE = 0.04
RISK_FREE_TICKER = "^IRX"
RISK_FREE_SERIES = pd.Series(dtype=float)
WEEKS_PER_YEAR = 52
WEIGHT_SWING_ALERT_THRESHOLD = 0.80
EFFICIENT_FRONTIER_L2_GAMMA = 0.2
OPTIMIZER_METHOD = "max-sharpe"
OPTIMIZER_RISK_AVERSION = 2
OPTIMIZER_CHOICES = ("max-sharpe", "max-quadratic-utility", "min-volatility")

# Endowus Flagship Target Weights
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
    "0P0001CC3M": 0.04,  # iShares G lobal Aggregate 1-5 Year Bond Index Fund (IE) SGD-Hedged
    "PEBIX": 0.04,  # iShares Emerging Markets Government Bond Index Fund (IE)
    "EIMI.L": 0.033,  # Amundi Core MSCI Emerging Markets Fund
    "0P0001DWI0.SI": 0.02,  # PIMCO GIS Emerging Markets Bond Fund SGD-Hedged
}

EQUITY_ASSETS = [
    "0P0001AF7U.SI",
    "SPY",
    "^990100-USD-STRD",
    "DE000SLA4YD9.SG",
    "0P0001AF7Z.SI",
    "0P0001EF2T.SI",
    "EIMI.L",
]

FIXED_INCOME_ASSETS = [
    "0P0000KYEE.SI",
    "AGGG.L",
    "IE0002461055.IR",
    "0P0001EQUE.SI",
    "0P0001CC3M",
    "PEBIX",
    "0P0001DWI0.SI",
]

# Adjustable class-allocation target for optimization outputs.
TARGET_EQUITY_ALLOCATION = 0.60
TACTICAL_EQUITY_MIN = 0.4
TACTICAL_EQUITY_MAX = 0.8
TACTICAL_TAU_MIN = 0.05
TACTICAL_TAU_MAX = 0.05


def configure_optimizer(method: str = "max-sharpe", risk_aversion: float = 2.5):
    global OPTIMIZER_METHOD, OPTIMIZER_RISK_AVERSION
    OPTIMIZER_METHOD, OPTIMIZER_RISK_AVERSION = common_portfolio.set_optimizer_config(
        method, risk_aversion, OPTIMIZER_CHOICES
    )


def configure_risk_free_rate(rate: float):
    global RISK_FREE_RATE
    RISK_FREE_RATE = common_portfolio.set_risk_free_rate(rate)


def configure_l2_gamma(gamma: float):
    global EFFICIENT_FRONTIER_L2_GAMMA
    if not np.isfinite(gamma):
        raise ValueError(f"l2 gamma must be finite, got: {gamma}")
    EFFICIENT_FRONTIER_L2_GAMMA = float(gamma)


def configure_risk_free_series(series: pd.Series | None):
    global RISK_FREE_SERIES
    if series is None:
        RISK_FREE_SERIES = pd.Series(dtype=float)
        return
    clean = pd.Series(series.values, index=pd.to_datetime(series.index)).sort_index()
    RISK_FREE_SERIES = clean.dropna()


def load_us_3m_tbill_rate_from_prices(irx_prices: pd.Series | None) -> float:
    return common_portfolio.extract_us_3m_tbill_rate_from_prices(
        irx_prices,
        ticker=RISK_FREE_TICKER,
        fallback_rate=RISK_FREE_RATE,
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


def load_bl_signal_csvs(data_dir: str) -> pd.DataFrame:
    return common_portfolio.load_bl_signal_csvs(data_dir)


def infer_periods_per_year(index: pd.Index) -> int:
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

            navs = portfolio.get("monthlyNavs", [])
            if navs:
                dates = [pd.to_datetime(row[0]) for row in navs]
                values = [float(row[1]) for row in navs]
                s = pd.Series(values, index=dates)

                mask = (s.index >= pd.to_datetime(START_DATE)) & (
                    s.index <= pd.to_datetime(END_DATE)
                )
                s = s[mask]

                if not s.empty:
                    s = (s / s.iloc[0]) * INITIAL_EQUITY
                    s.name = name
                    normalized_curve = s
                    curves[name] = s

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


def enforce_class_allocation_targets(
    weights: dict[str, float],
    active_set: list[str],
    target_equity_allocation: float = TARGET_EQUITY_ALLOCATION,
) -> dict[str, float]:
    """Rescale long-only weights so class totals match target equity/fixed-income split."""
    if not active_set:
        return {}

    eq_assets = set(EQUITY_ASSETS)
    fi_assets = set(FIXED_INCOME_ASSETS)

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


def enforce_class_allocation_band_targets(
    weights: dict[str, float],
    active_set: list[str],
    equity_min: float = TACTICAL_EQUITY_MIN,
    equity_max: float = TACTICAL_EQUITY_MAX,
) -> dict[str, float]:
    """Rescale long-only weights so equity allocation stays within [equity_min, equity_max]."""
    if not active_set:
        return {}

    eq_assets = set(EQUITY_ASSETS)
    fi_assets = set(FIXED_INCOME_ASSETS)

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

    if not eq_tickers:
        return {
            t: (1.0 / len(fi_tickers) if t in fi_tickers else 0.0) for t in active_set
        }
    if not fi_tickers:
        return {
            t: (1.0 / len(eq_tickers) if t in eq_tickers else 0.0) for t in active_set
        }

    eq_min = min(max(float(equity_min), 0.0), 1.0)
    eq_max = min(max(float(equity_max), eq_min), 1.0)

    eq_weight = sum(base[t] for t in eq_tickers)
    clamped_eq = min(max(eq_weight, eq_min), eq_max)
    target_fi = 1.0 - clamped_eq

    adjusted = {t: 0.0 for t in active_set}

    eq_sum = sum(base[t] for t in eq_tickers)
    fi_sum = sum(base[t] for t in fi_tickers)

    if eq_sum > 0:
        for t in eq_tickers:
            adjusted[t] = (base[t] / eq_sum) * clamped_eq
    else:
        for t in eq_tickers:
            adjusted[t] = clamped_eq / len(eq_tickers)

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


def get_fred_data(series_id: str, start: str, end: str) -> pd.Series:
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_file = os.path.join(DATA_DIR, f"{series_id}_fred.csv")
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df[series_id]

    print(f"Fetching {series_id} from FRED...")
    try:
        df = web.DataReader(series_id, "fred", start, end)
        df.to_csv(cache_file)
        return df[series_id]
    except Exception as e:
        print(f"Error fetching {series_id} from FRED: {e}")
        return pd.Series(dtype=float)


def get_yfinance_data(
    tickers: list[str], start: str, end: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_file_close = os.path.join(
        DATA_DIR, f"yfinance_close_{'_'.join(tickers)[:30]}.csv"
    )
    cache_file_open = os.path.join(
        DATA_DIR, f"yfinance_open_{'_'.join(tickers)[:30]}.csv"
    )

    if os.path.exists(cache_file_close) and os.path.exists(cache_file_open):
        df_close = pd.read_csv(cache_file_close, index_col=0, parse_dates=True)
        df_open = pd.read_csv(cache_file_open, index_col=0, parse_dates=True)
        if all(t in df_close.columns for t in tickers):
            return df_close, df_open

    print(f"Fetching {len(tickers)} tickers from yfinance...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        close_df = raw["Close"]
        open_df = raw["Open"]
    else:
        close_df = raw[["Close"]].rename(columns={"Close": tickers[0]})
        open_df = raw[["Open"]].rename(columns={"Open": tickers[0]})

    close_df.index = pd.to_datetime(close_df.index)
    close_df = close_df.sort_index().dropna(how="all")
    close_df.to_csv(cache_file_close)

    open_df.index = pd.to_datetime(open_df.index)
    open_df = open_df.sort_index().dropna(how="all")
    open_df.to_csv(cache_file_open)

    return close_df, open_df


def to_weekly_monday(df: pd.DataFrame) -> pd.DataFrame:
    return common_portfolio.to_weekly_monday(df)


def to_monthly_end(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("ME").last()


def align_signals_to_month_end(signals_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse signals to one row per ticker per month using the last signal in each month."""
    if signals_df is None or signals_df.empty:
        return pd.DataFrame(columns=["Date", "Ticker", "target_return", "confidence"])

    df = signals_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["MonthEnd"] = df["Date"].dt.to_period("M").dt.to_timestamp("M")

    # Keep latest signal within each month for each ticker.
    df = df.sort_values(["Ticker", "Date"])
    monthly = (
        df.groupby(["Ticker", "MonthEnd"], as_index=False)
        .last()[["MonthEnd", "Ticker", "target_return", "confidence"]]
        .rename(columns={"MonthEnd": "Date"})
    )
    return monthly


def get_execution_prices(
    daily_open: pd.DataFrame, weekly_dates: pd.Index
) -> pd.DataFrame:
    return common_portfolio.get_execution_prices(daily_open, weekly_dates)


def generate_signals(
    date: pd.Timestamp, prices: pd.DataFrame, fred_data: dict
) -> tuple[list, list]:
    eq_assets = [
        "0P0001AF7U.SI",
        "SPY",
        "^990100-USD-STRD",
        "DE000SLA4YD9.SG",
        "0P0001AF7Z.SI",
        "0P0001EF2T.SI",
        "EIMI.L",
    ]
    fi_assets = [
        "0P0000KYEE.SI",
        "AGGG.L",
        "IE0002461055.IR",
        "0P0001EQUE.SI",
        "0P0001CC3M",
        "PEBIX",
        "0P0001DWI0.SI",
    ]

    us_equity_proxy = "SPY"
    em_equity_proxy = "0P0001AF7Z.SI"
    pacific_equity_proxy = "0P0001EF2T.SI"
    em_bond_proxy = "0P0001DWI0.SI"

    short_duration_bond_proxy = "0P0001CC3M"
    broad_agg_bond_proxy_a = "AGGG.L"
    broad_agg_bond_proxy_b = "0P0001EQUE.SI"

    dxy = "DX-Y.NYB"

    P_rows = []
    Q_vals = []

    hist_prices = prices.loc[:date]
    if len(hist_prices) < 252 * 5:
        return [], []

    def get_annualized_return(series):
        return (series.iloc[-1] / series.iloc[0]) ** (252 / len(series)) - 1

    def get_annualized_volatility(series):
        return series.pct_change().std() * np.sqrt(252)

    # --- Signal A: Absolute Trend ---
    for asset in eq_assets:
        asset_prices = hist_prices[asset].dropna()
        if len(asset_prices) < 252 * 5:
            continue

        current_price = asset_prices.iloc[-1]
        sma_10m = asset_prices.tail(210).mean()

        prices_5yr = asset_prices.tail(252 * 5)
        mu_5yr = get_annualized_return(prices_5yr)
        vol_5yr = get_annualized_volatility(prices_5yr)

        q_val = (
            mu_5yr + (0.5 * vol_5yr)
            if current_price > sma_10m
            else mu_5yr - (0.5 * vol_5yr)
        )

        p_row = {a: 0 for a in list(ENDOWUS_WEIGHTS.keys())}
        p_row[asset] = 1.0
        P_rows.append(p_row)
        Q_vals.append(q_val)

    # --- Signal B: Cross-Sectional Momentum ---
    six_mo_returns = {}
    for asset in eq_assets:
        asset_prices = hist_prices[asset].dropna()
        if len(asset_prices) >= 126:
            ret = (asset_prices.iloc[-1] / asset_prices.iloc[-126]) - 1
            six_mo_returns[asset] = ret

    if len(six_mo_returns) >= 4:
        sorted_assets = sorted(six_mo_returns.items(), key=lambda x: x[1], reverse=True)
        top_2 = [sorted_assets[0][0], sorted_assets[1][0]]
        bottom_2 = [sorted_assets[2][0], sorted_assets[3][0]]

        top_ret = (six_mo_returns[top_2[0]] + six_mo_returns[top_2[1]]) / 2
        bottom_ret = (six_mo_returns[bottom_2[0]] + six_mo_returns[bottom_2[1]]) / 2

        top_ann = (1 + top_ret) ** 2 - 1
        bottom_ann = (1 + bottom_ret) ** 2 - 1
        q_val = top_ann - bottom_ann

        p_row = {a: 0 for a in list(ENDOWUS_WEIGHTS.keys())}
        p_row[top_2[0]] = 0.5
        p_row[top_2[1]] = 0.5
        p_row[bottom_2[0]] = -0.5
        p_row[bottom_2[1]] = -0.5

        P_rows.append(p_row)
        Q_vals.append(q_val)

    # --- Signal C: Macro Currency Regime ---
    dxy_prices = hist_prices[dxy].dropna()
    if len(dxy_prices) >= 252 * 5:
        sma_50 = dxy_prices.tail(50).mean()
        sma_200 = dxy_prices.tail(200).mean()

        dxy_5yr = dxy_prices.tail(252 * 5)
        sma50_5yr = dxy_5yr.rolling(50).mean()
        sma200_5yr = dxy_5yr.rolling(200).mean()

        strong_usd_mask = sma50_5yr > sma200_5yr
        weak_usd_mask = sma50_5yr < sma200_5yr

        required_macro_assets = [
            us_equity_proxy,
            em_equity_proxy,
            pacific_equity_proxy,
            em_bond_proxy,
        ]
        if all(t in hist_prices.columns for t in required_macro_assets):
            us_equity_5yr = hist_prices[us_equity_proxy].dropna().tail(252 * 5)

            # Build an equal-weighted EM/Pacific basket in return space.
            em_ret = hist_prices[em_equity_proxy].dropna().pct_change().tail(252 * 5)
            pac_ret = (
                hist_prices[pacific_equity_proxy].dropna().pct_change().tail(252 * 5)
            )
            em_bond_ret = hist_prices[em_bond_proxy].dropna().pct_change().tail(252 * 5)

            em_pac_ret_basket = (em_ret + pac_ret + em_bond_ret).dropna() / 3

            common_idx = us_equity_5yr.index.intersection(
                em_pac_ret_basket.index
            ).intersection(dxy_5yr.index)
            us_common = us_equity_5yr.loc[common_idx]
            em_ret = em_pac_ret_basket.loc[common_idx]
            strong_mask = strong_usd_mask.loc[common_idx]
            weak_mask = weak_usd_mask.loc[common_idx]

            us_ret = us_common.pct_change()

            p_row = {a: 0 for a in list(ENDOWUS_WEIGHTS.keys())}

            if sma_50 > sma_200:
                if strong_mask.sum() > 20:
                    us_cond_ret = us_ret[strong_mask].mean() * 252
                    em_cond_ret = em_ret[strong_mask].mean() * 252
                    q_val = us_cond_ret - em_cond_ret

                    p_row[us_equity_proxy] = 1.0
                    p_row[em_equity_proxy] = -1 / 3
                    p_row[pacific_equity_proxy] = -1 / 3
                    p_row[em_bond_proxy] = -1 / 3

                    P_rows.append(p_row)
                    Q_vals.append(q_val)
            else:
                if weak_mask.sum() > 20:
                    em_cond_ret = em_ret[weak_mask].mean() * 252
                    us_cond_ret = us_ret[weak_mask].mean() * 252
                    q_val = em_cond_ret - us_cond_ret

                    p_row[em_equity_proxy] = 1 / 3
                    p_row[pacific_equity_proxy] = 1 / 3
                    p_row[em_bond_proxy] = 1 / 3
                    p_row[us_equity_proxy] = -1.0

                    P_rows.append(p_row)
                    Q_vals.append(q_val)

    # --- Signal D: Yield Curve and Inflation Expectations ---
    t10yie = fred_data["T10YIE"].loc[:date].dropna()
    t10y2y = fred_data["T10Y2Y"].loc[:date].dropna()

    if len(t10yie) >= 63 and len(t10y2y) > 0:
        current_spread = t10y2y.iloc[-1]

        current_inf = t10yie.iloc[-1]
        past_inf = t10yie.iloc[-63]
        inf_roc = current_inf - past_inf

        p_row = {a: 0 for a in list(ENDOWUS_WEIGHTS.keys())}

        if inf_roc > 0 and current_spread < 0:
            p_row[short_duration_bond_proxy] = 1.0
            p_row[broad_agg_bond_proxy_a] = -0.5
            p_row[broad_agg_bond_proxy_b] = -0.5
            q_val = abs(current_spread) / 100.0

            P_rows.append(p_row)
            Q_vals.append(q_val)
        elif inf_roc < 0 and current_spread > 0:
            p_row[broad_agg_bond_proxy_a] = 0.5
            p_row[broad_agg_bond_proxy_b] = 0.5
            p_row[short_duration_bond_proxy] = -1.0
            q_val = abs(current_spread) / 100.0

            P_rows.append(p_row)
            Q_vals.append(q_val)

    return P_rows, Q_vals


def compute_systematic_bl_weights(
    active_set: list[str], price_history: pd.DataFrame, P_rows: list, Q_vals: list
) -> dict[str, float]:

    sub = price_history[active_set].ffill().bfill().dropna(axis=0, how="all")

    target_weights_dict = {t: ENDOWUS_WEIGHTS.get(t, 0.0) for t in active_set}
    total_w = sum(target_weights_dict.values())

    if total_w > 0:
        target_weights = np.array(
            [target_weights_dict[t] / total_w for t in active_set]
        )
        fallback_weights = {t: target_weights_dict[t] / total_w for t in active_set}
    else:
        fallback_weights = {t: 1.0 / len(active_set) for t in active_set}
        target_weights = np.array(list(fallback_weights.values()))

    fallback_weights = enforce_class_allocation_targets(fallback_weights, active_set)

    if sub.shape[0] < 10 or len(active_set) < 2:
        return fallback_weights

    try:
        S = risk_models.CovarianceShrinkage(sub, frequency=252).ledoit_wolf()
        delta = 2.5
        market_prior = delta * S.dot(target_weights)

        if not P_rows:
            return fallback_weights

        P_filtered = []
        Q_filtered = []
        for p, q in zip(P_rows, Q_vals):
            valid = True
            for asset, weight in p.items():
                if weight != 0 and asset not in active_set:
                    valid = False
                    break

            if valid:
                p_arr = np.array([p.get(t, 0.0) for t in active_set])
                P_filtered.append(p_arr)
                Q_filtered.append(q)

        if not P_filtered:
            return fallback_weights

        P_matrix = np.array(P_filtered)
        Q_vector = np.array(Q_filtered)

        tau = 0.05
        cov_values = np.asarray(S)
        omega = np.diag(np.diag(tau * P_matrix @ cov_values @ P_matrix.T))

        bl = BlackLittermanModel(
            S, pi=market_prior, P=P_matrix, Q=Q_vector, omega=omega
        )
        bl_returns = bl.bl_returns()
        bl_cov = bl.bl_cov()

        ef, _ = optimize_efficient_frontier(bl_returns, bl_cov, context="Systematic-BL")

        cleaned = ef.clean_weights()
        weights = {str(t): float(w) for t, w in cleaned.items() if w > 1e-6}
        if not weights:
            return fallback_weights

        return enforce_class_allocation_targets(weights, active_set)

    except Exception as e:
        print(f"[DEBUG] BL Math failed for {active_set}: {e}")
        return fallback_weights


def compute_systematic_bl_weights_tactical(
    active_set: list[str],
    price_history: pd.DataFrame,
    P_rows: list,
    Q_vals: list,
    tau_min: float = TACTICAL_TAU_MIN,
    tau_max: float = TACTICAL_TAU_MAX,
    equity_min: float = TACTICAL_EQUITY_MIN,
    equity_max: float = TACTICAL_EQUITY_MAX,
) -> dict[str, float]:
    """Compute BL weights with dynamic tau and equity band constraints for tactical monthly allocation."""
    sub = price_history[active_set].ffill().bfill().dropna(axis=0, how="all")

    target_weights_dict = {t: ENDOWUS_WEIGHTS.get(t, 0.0) for t in active_set}
    total_w = sum(target_weights_dict.values())

    if total_w > 0:
        target_weights = np.array(
            [target_weights_dict[t] / total_w for t in active_set]
        )
        fallback_weights = {t: target_weights_dict[t] / total_w for t in active_set}
    else:
        fallback_weights = {t: 1.0 / len(active_set) for t in active_set}
        target_weights = np.array(list(fallback_weights.values()))

    fallback_weights = enforce_class_allocation_band_targets(
        fallback_weights, active_set, equity_min, equity_max
    )

    if sub.shape[0] < 10 or len(active_set) < 2:
        return fallback_weights

    try:
        S = risk_models.CovarianceShrinkage(sub, frequency=252).ledoit_wolf()
        delta = 2.5
        market_prior = delta * S.dot(target_weights)

        if not P_rows:
            return fallback_weights

        P_filtered = []
        Q_filtered = []
        for p, q in zip(P_rows, Q_vals):
            valid = True
            for asset, weight in p.items():
                if weight != 0 and asset not in active_set:
                    valid = False
                    break

            if valid:
                p_arr = np.array([p.get(t, 0.0) for t in active_set])
                P_filtered.append(p_arr)
                Q_filtered.append(q)

        if not P_filtered:
            return fallback_weights

        P_matrix = np.array(P_filtered)
        Q_vector = np.array(Q_filtered)

        tau_floor = max(float(tau_min), 1e-6)
        tau_ceiling = max(float(tau_max), tau_floor)

        abs_q = np.abs(Q_vector)
        q_scale = np.nanpercentile(abs_q, 75) if abs_q.size > 0 else 0.0
        if not np.isfinite(q_scale) or q_scale <= 1e-9:
            confidence = 0.5
        else:
            confidence = float(np.clip(np.nanmean(abs_q) / q_scale, 0.0, 1.0))

        tau = tau_floor + confidence * (tau_ceiling - tau_floor)
        cov_values = np.asarray(S)
        omega = np.diag(np.diag(tau * P_matrix @ cov_values @ P_matrix.T))

        bl = BlackLittermanModel(
            S, pi=market_prior, P=P_matrix, Q=Q_vector, omega=omega
        )
        bl_returns = bl.bl_returns()
        bl_cov = bl.bl_cov()

        ef, _ = optimize_efficient_frontier(
            bl_returns, bl_cov, context="Systematic-BL-Tactical-Monthly"
        )

        cleaned = ef.clean_weights()
        weights = {str(t): float(w) for t, w in cleaned.items() if w > 1e-6}
        if not weights:
            return fallback_weights

        return enforce_class_allocation_band_targets(
            weights, active_set, equity_min, equity_max
        )

    except Exception as e:
        print(f"[DEBUG] Tactical BL Math failed for {active_set}: {e}")
        return fallback_weights


def compute_markowitz_weights(
    active_set: list[str],
    price_history: pd.DataFrame,
    target_equity_allocation: float = TARGET_EQUITY_ALLOCATION,
) -> dict[str, float]:
    """Compute Markowitz (mean-variance) optimized long-only weights for the active set.

    Uses historical daily returns from `price_history` (expects columns for tickers).
    Falls back to equal-weights when there isn't enough data or optimization fails.
    """
    if not active_set:
        return {}

    sub = price_history[active_set].ffill().bfill().dropna(axis=0, how="all")
    if sub.shape[0] < 10 or len(active_set) < 2:
        return {t: 1.0 / len(active_set) for t in active_set}

    try:
        returns = sub.pct_change().dropna()
        mu = returns.mean() * 252
        cov = returns.cov() * 252

        ef, _ = optimize_efficient_frontier(mu, cov, context="Markowitz")

        cleaned = ef.clean_weights()
        weights = {str(t): float(cleaned.get(t, 0.0)) for t in active_set}
        if not weights:
            return {t: 1.0 / len(active_set) for t in active_set}

        return enforce_class_allocation_targets(
            weights, active_set, target_equity_allocation
        )

    except Exception as e:
        print(f"[DEBUG] Markowitz optimization failed for {active_set}: {e}")
        return {t: 1.0 / len(active_set) for t in active_set}


def get_latest_price_on_or_before(
    data: pd.DataFrame, ticker: str, as_of: pd.Timestamp
) -> float | None:
    """Return latest non-null price at or before `as_of` for `ticker`."""
    return common_portfolio.get_latest_price_on_or_before(data, ticker, as_of)


def compute_llm_bl_weights_banded(
    active_set: list[str],
    price_history: pd.DataFrame,
    views: dict,
    confidences: dict,
    tau_min: float = TACTICAL_TAU_MIN,
    tau_max: float = TACTICAL_TAU_MAX,
    equity_min: float = TACTICAL_EQUITY_MIN,
    equity_max: float = TACTICAL_EQUITY_MAX,
    diagnostics_context: dict | None = None,
) -> dict[str, float]:
    """Compute LLM BL weights using Endowus-style views with equity band constraints."""
    sub = price_history[active_set].ffill().bfill().dropna(axis=0, how="all")

    target_weights_dict = {t: ENDOWUS_WEIGHTS.get(t, 0.0) for t in active_set}
    total_w = sum(target_weights_dict.values())
    if total_w > 0:
        target_weights = np.array(
            [target_weights_dict[t] / total_w for t in active_set]
        )
        fallback_weights = {t: target_weights_dict[t] / total_w for t in active_set}
    else:
        fallback_weights = {t: 1.0 / len(active_set) for t in active_set}
        target_weights = np.array(list(fallback_weights.values()))

    fallback_weights = enforce_class_allocation_band_targets(
        fallback_weights, active_set, equity_min, equity_max
    )

    if sub.shape[0] < 10 or len(active_set) < 2:
        return fallback_weights

    try:
        S = risk_models.CovarianceShrinkage(sub, frequency=252).ledoit_wolf()
        cov_df = (
            S
            if isinstance(S, pd.DataFrame)
            else pd.DataFrame(np.asarray(S), index=active_set, columns=active_set)
        )

        delta = 2.5
        market_prior_raw = delta * cov_df.dot(target_weights)
        market_prior = (
            market_prior_raw
            if isinstance(market_prior_raw, pd.Series)
            else pd.Series(np.asarray(market_prior_raw).reshape(-1), index=active_set)
        )

        Q = pd.Series(
            {
                t: resolve_view_value(views, t, float(market_prior.get(t, 0.05)))
                for t in active_set
            }
        )

        omega_diag = []
        for t in active_set:
            conf = float(confidences.get(t, 5.0))
            conf = min(max(conf, 1.0), 10.0)
            scale = max(1.1 - (conf / 10.0), 0.05)
            var_t = float(np.real(np.asarray(cov_df.loc[t, t]).item()))
            omega_diag.append(var_t * scale)
        omega = np.diag(omega_diag)

        if confidences:
            valid_confs = [c for c in confidences.values() if pd.notna(c)]
            avg_conf = np.mean(valid_confs) if valid_confs else 5.0
        else:
            avg_conf = 5.0

        avg_conf = min(max(float(avg_conf), 1.0), 10.0)

        tau = tau_min + ((avg_conf - 1.0) / 9.0) * (tau_max - tau_min)

        bl = BlackLittermanModel(
            cov_df,
            pi=market_prior,
            absolute_views=Q,
            omega=omega,
            tau=tau,
        )
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
        if (
            diagnostics_context is not None
            and diagnostics_context.get("events") is not None
        ):
            diagnostics_context["events"].append(
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

        return enforce_class_allocation_band_targets(
            weights, active_set, equity_min, equity_max
        )

    except Exception as e:
        print(f"[DEBUG] LLM banded BL math failed for {active_set}: {e}")
        return fallback_weights


def compute_llm_bl_weights(
    active_set: list[str],
    price_history: pd.DataFrame,
    views: dict,
    confidences: dict,
    tau_min: float = TACTICAL_TAU_MIN,
    tau_max: float = TACTICAL_TAU_MAX,
    target_equity_allocation: float = TARGET_EQUITY_ALLOCATION,
    diagnostics_context: dict | None = None,
) -> dict[str, float]:
    """Compute LLM BL weights pinned to an exact equity allocation target."""
    target_eq = min(max(float(target_equity_allocation), 0.0), 1.0)
    return compute_llm_bl_weights_banded(
        active_set=active_set,
        price_history=price_history,
        views=views,
        confidences=confidences,
        tau_min=tau_min,
        tau_max=tau_max,
        equity_min=target_eq,
        equity_max=target_eq,
        diagnostics_context=diagnostics_context,
    )


def run_llm_banded_backtest(
    strategy: str,
    signals_df: pd.DataFrame,
    weekly_prices: pd.DataFrame,
    weekly_exec_prices: pd.DataFrame,
    daily_prices: pd.DataFrame,
    rebalance_policy: str = "weekly",
    log_realized_weights: bool = False,
    transaction_cost: float = TRANSACTION_COST,
    diagnostics_rows: list[dict] | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Run LLM BL backtest with equity-band constrained allocations."""
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
                exec_p = val_p
            exec_prices[t] = exec_p

        active_set = list(exec_prices.keys())
        active_set_now = set(active_set)
        if active_set_now != prev_active_set:
            added = sorted(active_set_now - prev_active_set)
            removed = sorted(prev_active_set - active_set_now)
            if prev_active_set and diagnostics_rows is not None:
                diagnostics_rows.append(
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
            if prev_active_set and (added or removed):
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

            if not signals_df.empty:
                day_signals = signals_df[signals_df["Date"] == date]
                for _, row in day_signals.iterrows():
                    t = row["Ticker"]
                    if t in active_set:
                        views[t] = row["target_return"]
                        confidences[t] = row["confidence"]

            if diagnostics_rows is not None:
                diagnostics_rows.append(
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

            lookback_start = date - pd.Timedelta(weeks=52)
            history = daily_prices.loc[lookback_start:date, :]

            weights = compute_llm_bl_weights_banded(
                active_set,
                history,
                views,
                confidences,
                tau_min=TACTICAL_TAU_MIN,
                tau_max=TACTICAL_TAU_MAX,
                equity_min=TACTICAL_EQUITY_MIN,
                equity_max=TACTICAL_EQUITY_MAX,
                diagnostics_context={
                    "strategy": strategy,
                    "date": date,
                    "events": diagnostics_rows,
                },
            )

            if diagnostics_rows is not None and last_target_weights:
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
                    diagnostics_rows.append(
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
            weights_log[date] = {
                t: (port.shares.get(t, 0.0) * val_prices.get(t, 0.0))
                for t in active_set
                if port.shares.get(t, 0.0) > 0 and val_prices.get(t, 0.0) > 0
            }
            total_now = sum(weights_log[date].values()) + port.cash
            if total_now > 0:
                weights_log[date] = {
                    t: v / total_now for t, v in weights_log[date].items()
                }
        elif should_rebalance:
            weights_log[date] = last_target_weights
        elif last_target_weights:
            weights_log[date] = last_target_weights
        else:
            weights_log[date] = {}

    weights_df = pd.DataFrame(weights_log).T.sort_index().fillna(0.0)
    weights_df.index.name = "Date"
    return pd.Series(equity_curve, name=strategy).sort_index(), weights_df


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
            qty = self.shares.get(t, 0.0)
            if qty <= 0:
                self.shares.pop(t, None)
                continue

            px = prices.get(t)
            if px is None or px <= 0:
                # Keep position if no valid execution price is available this week.
                continue

            gross = qty * px
            fee = gross * self.transaction_cost
            net = gross - fee
            self.cash += net
            proceeds += gross
            self.shares.pop(t, None)
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
            px = prices.get(t)
            if px is None or px <= 0:
                continue

            current_val = self.shares.get(t, 0.0) * px
            desired_val = target_dollars.get(t, 0.0)
            if current_val > desired_val + 1e-6:
                excess_shares = (current_val - desired_val) / px
                gross = excess_shares * px
                fee = gross * self.transaction_cost
                self.shares[t] = self.shares.get(t, 0.0) - excess_shares
                self.cash += gross - fee

        for t in target_tickers:
            px = prices.get(t)
            if px is None or px <= 0:
                continue
            current_val = self.shares.get(t, 0.0) * px
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
                self.shares[t] = self.shares.get(t, 0.0) + actual_gross / px
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
    diagnostics_rows: list[dict] | None = None,
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
                exec_p = val_p
            exec_prices[t] = exec_p

        active_set = list(exec_prices.keys())
        active_set_now = set(active_set)
        if active_set_now != prev_active_set:
            added = sorted(active_set_now - prev_active_set)
            removed = sorted(prev_active_set - active_set_now)
            if prev_active_set and diagnostics_rows is not None:
                diagnostics_rows.append(
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
                if not signals_df.empty:
                    day_signals = signals_df[signals_df["Date"] == date]
                    for _, row in day_signals.iterrows():
                        t = row["Ticker"]
                        if t in active_set:
                            views[t] = row["target_return"]
                            confidences[t] = row["confidence"]

                lookback_start = date - pd.Timedelta(weeks=52)
                history = daily_prices.loc[lookback_start:date, :]

                weights = compute_llm_bl_weights(
                    active_set,
                    history,
                    views,
                    confidences,
                    tau_min=TACTICAL_TAU_MIN,
                    tau_max=TACTICAL_TAU_MAX,
                    diagnostics_context={
                        "strategy": strategy,
                        "date": date,
                        "events": diagnostics_rows,
                    },
                )

                if diagnostics_rows is not None:
                    diagnostics_rows.append(
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


def run_systematic_backtest(
    weekly_prices: pd.DataFrame,
    weekly_exec_prices: pd.DataFrame,
    daily_prices: pd.DataFrame,
    fred_data: dict,
) -> tuple[pd.Series, pd.DataFrame]:

    port = Portfolio(INITIAL_EQUITY, TRANSACTION_COST)
    equity_curve = {}
    weights_log = {}

    tickers = list(ENDOWUS_WEIGHTS.keys())
    dates = weekly_prices.index.intersection(weekly_exec_prices.index)

    dates = dates[
        (dates >= pd.to_datetime(START_DATE)) & (dates <= pd.to_datetime(END_DATE))
    ]

    for date in dates:
        configure_risk_free_rate(
            common_portfolio.get_risk_free_rate_for_date(
                pd.to_datetime(date), RISK_FREE_SERIES, RISK_FREE_RATE
            )
        )
        price_row = weekly_prices.loc[date]
        exec_row = weekly_exec_prices.loc[date]

        val_prices = {
            t: float(price_row[t])
            for t in tickers
            if pd.notna(price_row.get(t)) and float(price_row.get(t, 0)) > 0
        }
        exec_prices = {
            t: float(exec_row[t])
            for t in tickers
            if pd.notna(exec_row.get(t)) and float(exec_row.get(t, 0)) > 0
        }

        for t in tickers:
            if t not in exec_prices and t in val_prices:
                exec_prices[t] = val_prices[t]

        active_set = list(exec_prices.keys())

        P_rows, Q_vals = generate_signals(date, daily_prices, fred_data)

        lookback_start = date - pd.Timedelta(weeks=52)
        history = daily_prices.loc[lookback_start:date, :]

        weights = compute_systematic_bl_weights(active_set, history, P_rows, Q_vals)

        te = port.total_equity(val_prices)
        port.buy_target_weights(weights, exec_prices, te)
        equity_curve[date] = port.total_equity(val_prices)
        weights_log[date] = weights

    weights_df = pd.DataFrame(weights_log).T.sort_index().fillna(0.0)
    weights_df.index.name = "Date"
    print(weights_df)
    return pd.Series(equity_curve, name="Systematic-BL").sort_index(), weights_df


def run_markowitz_backtest(
    weekly_prices: pd.DataFrame,
    weekly_exec_prices: pd.DataFrame,
    daily_prices: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """Run a weekly rebalanced Markowitz mean-variance backtest over the Endowus universe.

    This uses `compute_markowitz_weights` at each rebalancing date using the
    last 52 weeks of daily history (or available history).
    """
    port = Portfolio(INITIAL_EQUITY, TRANSACTION_COST)
    equity_curve = {}
    weights_log = {}

    tickers = list(ENDOWUS_WEIGHTS.keys())
    dates = weekly_prices.index.intersection(weekly_exec_prices.index)
    dates = dates[
        (dates >= pd.to_datetime(START_DATE)) & (dates <= pd.to_datetime(END_DATE))
    ]

    for date in dates:
        configure_risk_free_rate(
            common_portfolio.get_risk_free_rate_for_date(
                pd.to_datetime(date), RISK_FREE_SERIES, RISK_FREE_RATE
            )
        )
        price_row = weekly_prices.loc[date]
        exec_row = weekly_exec_prices.loc[date]

        val_prices = {
            t: float(price_row[t])
            for t in tickers
            if pd.notna(price_row.get(t)) and float(price_row.get(t, 0)) > 0
        }
        exec_prices = {
            t: float(exec_row[t])
            for t in tickers
            if pd.notna(exec_row.get(t)) and float(exec_row.get(t, 0)) > 0
        }

        for t in tickers:
            if t not in exec_prices and t in val_prices:
                exec_prices[t] = val_prices[t]

        active_set = list(exec_prices.keys())

        lookback_start = date - pd.Timedelta(weeks=52)
        history = daily_prices.loc[lookback_start:date, :]

        weights = compute_markowitz_weights(active_set, history)

        te = port.total_equity(val_prices)
        port.buy_target_weights(weights, exec_prices, te)
        equity_curve[date] = port.total_equity(val_prices)
        weights_log[date] = weights

    weights_df = pd.DataFrame(weights_log).T.sort_index().fillna(0.0)
    weights_df.index.name = "Date"
    return pd.Series(equity_curve, name="Markowitz").sort_index(), weights_df


def run_systematic_tactical_monthly_backtest(
    monthly_prices: pd.DataFrame,
    monthly_exec_prices: pd.DataFrame,
    daily_prices: pd.DataFrame,
    fred_data: dict,
) -> tuple[pd.Series, pd.DataFrame]:
    """Run a separate monthly tactical BL portfolio with dynamic tilting and dynamic tau."""
    port = Portfolio(INITIAL_EQUITY, TRANSACTION_COST)
    equity_curve = {}
    weights_log = {}

    tickers = list(ENDOWUS_WEIGHTS.keys())
    dates = monthly_prices.index.intersection(monthly_exec_prices.index)
    dates = dates[
        (dates >= pd.to_datetime(START_DATE)) & (dates <= pd.to_datetime(END_DATE))
    ]

    for date in dates:
        configure_risk_free_rate(
            common_portfolio.get_risk_free_rate_for_date(
                pd.to_datetime(date), RISK_FREE_SERIES, RISK_FREE_RATE
            )
        )
        price_row = monthly_prices.loc[date]
        exec_row = monthly_exec_prices.loc[date]

        val_prices = {
            t: float(price_row[t])
            for t in tickers
            if pd.notna(price_row.get(t)) and float(price_row.get(t, 0)) > 0
        }
        exec_prices = {
            t: float(exec_row[t])
            for t in tickers
            if pd.notna(exec_row.get(t)) and float(exec_row.get(t, 0)) > 0
        }

        for t in tickers:
            if t not in exec_prices and t in val_prices:
                exec_prices[t] = val_prices[t]

        active_set = list(exec_prices.keys())
        P_rows, Q_vals = generate_signals(date, daily_prices, fred_data)

        lookback_start = date - pd.Timedelta(weeks=52)
        history = daily_prices.loc[lookback_start:date, :]

        weights = compute_systematic_bl_weights_tactical(
            active_set,
            history,
            P_rows,
            Q_vals,
            tau_min=TACTICAL_TAU_MIN,
            tau_max=TACTICAL_TAU_MAX,
            equity_min=TACTICAL_EQUITY_MIN,
            equity_max=TACTICAL_EQUITY_MAX,
        )

        te = port.total_equity(val_prices)
        port.buy_target_weights(weights, exec_prices, te)
        equity_curve[date] = port.total_equity(val_prices)
        weights_log[date] = weights

    weights_df = pd.DataFrame(weights_log).T.sort_index().fillna(0.0)
    weights_df.index.name = "Date"
    return (
        pd.Series(equity_curve, name="Systematic-BL-Tactical-Monthly").sort_index(),
        weights_df,
    )


def plot_combined_curves(
    curves_df: pd.DataFrame,
    output_path: str = "results/endowus_systematic_comparison_plot.png",
):
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


def plot_composition_comparison(
    weights_logs: dict[str, pd.DataFrame],
    output_path: str = "results/endowus_systematic_composition_comparison.png",
):
    """Plot stacked allocation composition over time for each portfolio in one figure."""
    valid_logs = {
        name: w.copy()
        for name, w in weights_logs.items()
        if isinstance(w, pd.DataFrame) and not w.empty
    }
    if not valid_logs:
        print("No weights logs available for composition plotting.")
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
        print("No allocation columns found for composition plotting.")
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

    for i, name in enumerate(strategy_names):
        ax = axes_arr[i]
        df = valid_logs[name].sort_index().copy()
        df = df.reindex(columns=all_assets, fill_value=0.0)
        df = df.clip(lower=0.0)

        row_sums = df.sum(axis=1)
        nonzero = row_sums > 1e-10
        if nonzero.any():
            df.loc[nonzero] = df.loc[nonzero].div(row_sums[nonzero], axis=0)

        x = mdates.date2num(df.index.to_pydatetime())
        y = [df[c].values * 100 for c in all_assets]
        ax.stackplot(x, y, labels=all_assets, colors=colors, alpha=0.9)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
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

    # 1. Get handles from the first subplot
    handles, labels = axes_arr[0].get_legend_handles_labels()

    # 2. Force the legend to the bottom center
    # 'ncol=5' or '6' keeps the legend from being one long vertical strip
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

    # 3. Keep the plot area tall while reserving just enough room for legend rows.
    legend_rows = int(np.ceil(len(all_assets) / legend_cols)) if legend_cols > 0 else 1
    bottom_margin = min(0.08 + 0.03 * legend_rows, 0.18)
    plt.subplots_adjust(bottom=bottom_margin, top=0.90, hspace=0.35)

    # 4. Use bbox_inches='tight' during save to ensure nothing is clipped
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Composition comparison plot saved to {output_path}")


def plot_asset_class_allocation_comparison(
    weights_logs: dict[str, pd.DataFrame],
    output_path: str = "results/endowus_systematic_asset_class_allocation.png",
):
    """Plot equity/fixed-income allocation percentages over time for each portfolio."""
    valid_logs = {
        name: w.copy()
        for name, w in weights_logs.items()
        if isinstance(w, pd.DataFrame) and not w.empty
    }
    if not valid_logs:
        print("No weights logs available for asset-class allocation plotting.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    eq_assets = set(EQUITY_ASSETS)
    fi_assets = set(FIXED_INCOME_ASSETS)
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

        eq_cols = [c for c in df.columns if c in eq_assets]
        fi_cols = [c for c in df.columns if c in fi_assets]

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
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
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
    print(f"Asset-class allocation comparison plot saved to {output_path}")


def plot_individual_ticker_growth(
    daily_close: pd.DataFrame,
    tickers: list[str],
    start: pd.Timestamp | str,
    end: pd.Timestamp | str = END_DATE,
    output_path: str = "results/endowus_individual_tickers.png",
):
    """Plot individual ticker growth rebased to `INITIAL_EQUITY` over the given window.

    Only tickers present in `daily_close` are plotted.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    available = [t for t in tickers if t in daily_close.columns]
    if not available:
        print("No requested tickers found in price data for individual plot.")
        return

    df = daily_close.reindex(columns=available).loc[
        pd.to_datetime(start) : pd.to_datetime(end)
    ]

    df = df.ffill().dropna(how="all")

    if df.empty:
        print(
            "No price data available for the chosen time window for individual tickers."
        )
        return

    # Allocate initial equity equally across tickers so each starts at the same value.
    # Use each ticker's first non-NA price within the window to avoid NaNs when
    # some tickers lack a price on the first date.
    df = df.sort_index()
    first_vals = df.apply(
        lambda s: s.dropna().iloc[0] if s.dropna().shape[0] > 0 else np.nan
    )
    valid_cols = first_vals[first_vals.notna()].index.tolist()
    if not valid_cols:
        print("No tickers with valid prices in the chosen window.")
        return

    alloc = INITIAL_EQUITY / float(len(valid_cols))
    rebased = df[valid_cols].div(first_vals[valid_cols], axis=1).multiply(alloc)

    fig, ax = plt.subplots(figsize=(14, 8))
    for col in rebased.columns:
        ax.plot(rebased.index, rebased[col], label=col, lw=1.6)

    ax.set_title(
        f"Individual Endowus Tickers Growth (Equal-weight start, total ${INITIAL_EQUITY:,.0f})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel(f"Value (per-ticker start = ${alloc:,.2f})")
    ax.set_xlabel("Date")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

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
    print(f"Individual tickers plot saved to {output_path}")


def load_fama_french_data_weekly(start: str, end: str) -> pd.DataFrame | None:
    """Fetch Fama-French 5-factor daily data and compound to weekly (W-MON)."""
    cache_path = "data/ff5_factors_weekly.csv"
    if os.path.exists(cache_path):
        ff_weekly = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if (
            not ff_weekly.empty
            and ff_weekly.index[0] <= pd.to_datetime(start)
            and ff_weekly.index[-1] >= pd.to_datetime(end)
        ):
            return ff_weekly

    try:
        ff_dict = web.DataReader(
            "F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start, end
        )
        ff_daily = ff_dict[0]
        ff_daily = (ff_daily / 100.0) + 1.0
        ff_weekly = ff_daily.resample("W-MON", label="left", closed="left").prod() - 1.0
        ff_weekly = ff_weekly * 100.0

        os.makedirs("data", exist_ok=True)
        ff_weekly.to_csv(cache_path)
        return ff_weekly
    except Exception as e:
        print(f"Error fetching/compounding FF weekly data: {e}")
        return None


def load_fama_french_data_monthly(start: str, end: str) -> pd.DataFrame | None:
    """Fetch Fama-French 5-factor monthly data with PeriodIndex for alignment."""
    cache_path = "data/ff5_factors_monthly.csv"
    if os.path.exists(cache_path):
        ff_monthly = pd.read_csv(cache_path, index_col=0)
        ff_monthly.index = pd.to_datetime(ff_monthly.index.astype(str)).to_period("M")
        return ff_monthly

    try:
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


def compute_full_period_factor_alphas(
    equity: pd.Series,
    ff_data_weekly: pd.DataFrame | None,
    ff_data_monthly: pd.DataFrame | None,
) -> tuple[float, float, float]:
    """Compute annualized CAPM/FF3/FF5 alphas over full period using curve frequency."""
    eq = equity.dropna()
    if len(eq) < 4:
        return np.nan, np.nan, np.nan

    ann_factor = infer_periods_per_year(eq.index)
    use_monthly = ann_factor <= 12

    str_returns = eq.pct_change().dropna()
    if len(str_returns) < 4:
        return np.nan, np.nan, np.nan

    ff_data = ff_data_monthly if use_monthly else ff_data_weekly
    if ff_data is None or ff_data.empty:
        return np.nan, np.nan, np.nan

    ff_aligned = ff_data.copy()
    ret_aligned = str_returns.copy()

    if use_monthly:
        ret_aligned.index = pd.DatetimeIndex(ret_aligned.index).to_period("M")
        if not isinstance(ff_aligned.index, pd.PeriodIndex):
            ff_aligned.index = pd.to_datetime(ff_aligned.index.astype(str)).to_period(
                "M"
            )

    common_idx = ret_aligned.index.intersection(ff_aligned.index)
    if len(common_idx) < 6:
        return np.nan, np.nan, np.nan

    y = ret_aligned.loc[common_idx] * 100.0
    ff = ff_aligned.loc[common_idx]
    if "RF" not in ff.columns or "Mkt-RF" not in ff.columns:
        return np.nan, np.nan, np.nan

    y_ex = y - ff["RF"]

    capm_alpha, ff3_alpha, ff5_alpha = np.nan, np.nan, np.nan

    try:
        X_capm = sm.add_constant(ff[["Mkt-RF"]])
        capm_alpha = sm.OLS(y_ex, X_capm).fit().params.get("const", np.nan) * ann_factor
    except Exception:
        pass

    if all(c in ff.columns for c in ["Mkt-RF", "SMB", "HML"]):
        try:
            X_ff3 = sm.add_constant(ff[["Mkt-RF", "SMB", "HML"]])
            ff3_alpha = (
                sm.OLS(y_ex, X_ff3).fit().params.get("const", np.nan) * ann_factor
            )
        except Exception:
            pass

    if all(c in ff.columns for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]):
        try:
            X_ff5 = sm.add_constant(ff[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]])
            ff5_alpha = (
                sm.OLS(y_ex, X_ff5).fit().params.get("const", np.nan) * ann_factor
            )
        except Exception:
            pass

    return capm_alpha, ff3_alpha, ff5_alpha


def append_factor_alphas_to_summary(
    summary_df: pd.DataFrame,
    curves: dict[str, pd.Series],
    ff_data_weekly: pd.DataFrame | None,
    ff_data_monthly: pd.DataFrame | None,
) -> pd.DataFrame:
    """Append CAPM/FF3/FF5 alpha columns to summary dataframe."""
    if summary_df is None or summary_df.empty:
        return summary_df

    summary = summary_df.copy()
    label_col = "Strategy" if "Strategy" in summary.columns else "Portfolio"
    if label_col not in summary.columns:
        return summary

    capm_vals = []
    ff3_vals = []
    ff5_vals = []

    for _, row in summary.iterrows():
        name = row[label_col]
        curve = curves.get(name)
        if curve is None:
            capm_vals.append("N/A")
            ff3_vals.append("N/A")
            ff5_vals.append("N/A")
            continue

        capm_alpha, ff3_alpha, ff5_alpha = compute_full_period_factor_alphas(
            curve, ff_data_weekly, ff_data_monthly
        )

        capm_vals.append(f"{capm_alpha:.2f}%" if pd.notna(capm_alpha) else "N/A")
        ff3_vals.append(f"{ff3_alpha:.2f}%" if pd.notna(ff3_alpha) else "N/A")
        ff5_vals.append(f"{ff5_alpha:.2f}%" if pd.notna(ff5_alpha) else "N/A")

    summary["CAPM Alpha (Ann)"] = capm_vals
    summary["FF3 Alpha (Ann)"] = ff3_vals
    summary["FF5 Alpha (Ann)"] = ff5_vals
    return summary


def generate_full_period_markdown_report(
    summary_df: pd.DataFrame,
    curves_df: pd.DataFrame,
    output_dir: str = "results",
    combined_plot_path: str = "results/endowus_systematic_comparison_plot.png",
    composition_plot_path: str = "results/endowus_systematic_composition_comparison.png",
    asset_class_plot_path: str = "results/endowus_systematic_asset_class_allocation.png",
    tickers_plot_path: str = "results/endowus_individual_tickers.png",
) -> str:
    """Generate a phase4-style markdown report for the entire test duration."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "systematic_bl_full_period_report.md")

    if curves_df is not None and not curves_df.empty:
        start_date = curves_df.index.min().strftime("%Y-%m-%d")
        end_date = curves_df.index.max().strftime("%Y-%m-%d")
    else:
        start_date = START_DATE
        end_date = END_DATE

    with open(report_path, "w") as f:
        f.write("# Systematic BL Full-Period Backtest Report\n\n")
        f.write(
            "This report summarizes the full backtest-period performance of the Systematic BL suite and related benchmarks.\n\n"
        )
        f.write(f"## Backtest Window ({start_date} to {end_date})\n\n")

        if os.path.exists(combined_plot_path):
            f.write("### Equity Curve Comparison\n\n")
            f.write(
                f"![Equity Curve Comparison]({os.path.basename(combined_plot_path)})\n\n"
            )

        if os.path.exists(composition_plot_path):
            f.write("### Portfolio Composition Comparison\n\n")
            f.write(
                f"![Portfolio Composition Comparison]({os.path.basename(composition_plot_path)})\n\n"
            )

        if os.path.exists(asset_class_plot_path):
            f.write("### Equity vs Fixed Income Allocation\n\n")
            f.write(
                f"![Equity vs Fixed Income Allocation]({os.path.basename(asset_class_plot_path)})\n\n"
            )

        if os.path.exists(tickers_plot_path):
            f.write("### Underlying Ticker Growth\n\n")
            f.write(
                f"![Underlying Ticker Growth]({os.path.basename(tickers_plot_path)})\n\n"
            )

        f.write("### Performance Metrics (Full Period)\n\n")
        if summary_df is not None and not summary_df.empty:
            preferred_cols = [
                "Strategy",
                "Ann. Return",
                "Sharpe Ratio",
                "Max Drawdown",
                "Calmar Ratio",
                "CAPM Alpha (Ann)",
                "FF3 Alpha (Ann)",
                "FF5 Alpha (Ann)",
            ]
            display_cols = [c for c in preferred_cols if c in summary_df.columns]
            metrics_table = summary_df[display_cols] if display_cols else summary_df
            f.write(metrics_table.to_markdown(index=False) + "\n\n")
        else:
            f.write("No performance metrics were generated.\n\n")

        f.write("---\n\n")
        f.write("Generated by `systematic_bl_baseline.py`.\n")

    print(f"Generated markdown report at {report_path}")
    return report_path


def save_raw_weights_logs(
    weights_logs: dict[str, pd.DataFrame], output_dir: str = "results/weights_raw"
):
    """Save raw component weights for each portfolio strategy to individual CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    manifest_rows = []
    for strategy, weights_df in weights_logs.items():
        if not isinstance(weights_df, pd.DataFrame) or weights_df.empty:
            continue

        out = weights_df.copy().sort_index().fillna(0.0)
        out.index.name = "Date"

        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", strategy).strip("_")
        if not safe_name:
            safe_name = "strategy"

        file_name = f"{safe_name}_raw_weights.csv"
        file_path = os.path.join(output_dir, file_name)
        out.to_csv(file_path)

        manifest_rows.append(
            {
                "Strategy": strategy,
                "File": file_name,
                "Rows": int(out.shape[0]),
                "Assets": int(out.shape[1]),
            }
        )

    if manifest_rows:
        manifest_df = pd.DataFrame(manifest_rows).sort_values("Strategy")
        manifest_df.to_csv(
            os.path.join(output_dir, "raw_weights_manifest.csv"), index=False
        )

    print(f"Saved raw portfolio weight logs to {output_dir}")


def save_diagnostics_events(
    diagnostics_rows: list[dict],
    output_dir: str = "results/diagnostics",
    file_name: str = "systematic_llm_banded_diagnostics.csv",
):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, file_name)
    if not diagnostics_rows:
        pd.DataFrame(columns=["EventType"]).to_csv(out_path, index=False)
    else:
        pd.DataFrame(diagnostics_rows).to_csv(out_path, index=False)
    print(f"Saved diagnostics events to {out_path}")


def _run_single_strategy(payload: dict) -> dict:
    name = payload["name"]
    sigs = payload["signals"]
    rebalance_policy = payload["rebalance_policy"]
    diagnostics_rows: list[dict] = []

    if name.startswith("LLM") and sigs is not None and sigs.empty:
        return {"name": name, "skipped": True}

    if name == "Systematic-BL":
        curve, wlog = run_systematic_backtest(
            payload["weekly_prices"],
            payload["weekly_exec_prices"],
            payload["daily_close"],
            payload["fred_data"],
        )
    elif name == "Systematic-BL-Tactical-Monthly":
        curve, wlog = run_systematic_tactical_monthly_backtest(
            payload["monthly_prices"],
            payload["monthly_exec_prices"],
            payload["daily_close"],
            payload["fred_data"],
        )
    elif name == "Markowitz":
        curve, wlog = run_markowitz_backtest(
            payload["weekly_prices"],
            payload["weekly_exec_prices"],
            payload["daily_close"],
        )
    else:
        is_monthly = "-Monthly" in name
        base_prices = (
            payload["monthly_prices"] if is_monthly else payload["weekly_prices"]
        )
        base_exec = (
            payload["monthly_exec_prices"]
            if is_monthly
            else payload["weekly_exec_prices"]
        )

        run_weekly = base_prices[
            [t for t in list(ENDOWUS_WEIGHTS.keys()) if t in base_prices.columns]
        ]
        run_exec = base_exec[
            [t for t in list(ENDOWUS_WEIGHTS.keys()) if t in base_exec.columns]
        ]
        run_daily = payload["daily_close"][
            [
                t
                for t in list(ENDOWUS_WEIGHTS.keys())
                if t in payload["daily_close"].columns
            ]
        ]

        run_weekly = run_weekly[
            (run_weekly.index >= pd.to_datetime(START_DATE))
            & (run_weekly.index <= pd.to_datetime(END_DATE))
        ]
        run_exec = run_exec[
            (run_exec.index >= pd.to_datetime(START_DATE))
            & (run_exec.index <= pd.to_datetime(END_DATE))
        ]

        if name.endswith("-Banded") and name.startswith("LLM-BL"):
            curve, wlog = run_llm_banded_backtest(
                name,
                sigs,
                run_weekly,
                run_exec,
                run_daily,
                rebalance_policy=("weekly" if is_monthly else rebalance_policy),
                log_realized_weights=(rebalance_policy == "buy-and-hold"),
                transaction_cost=TRANSACTION_COST,
                diagnostics_rows=diagnostics_rows,
            )
            return {
                "name": name,
                "curve": curve,
                "weights": wlog,
                "diagnostics": diagnostics_rows,
                "skipped": False,
            }

        curve, wlog = run_endowus_backtest(
            name,
            sigs,
            run_weekly,
            run_exec,
            run_daily,
            rebalance_policy=("weekly" if is_monthly else rebalance_policy),
            log_realized_weights=(rebalance_policy == "buy-and-hold"),
            transaction_cost=TRANSACTION_COST,
            diagnostics_rows=diagnostics_rows,
        )

    return {
        "name": name,
        "curve": curve,
        "weights": wlog,
        "diagnostics": diagnostics_rows,
        "skipped": False,
    }


def _run_single_strategy_worker(payload: dict) -> dict:
    configure_optimizer(payload["optimizer_method"], payload["risk_aversion"])
    configure_l2_gamma(payload["l2_gamma"])
    configure_risk_free_series(payload.get("risk_free_series"))
    configure_risk_free_rate(payload["risk_free_rate"])

    return _run_single_strategy(payload)


def main():
    print("=" * 60)

    parser = argparse.ArgumentParser(
        description="Run Systematic BL portfolio backtests"
    )
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
        "--l2-gamma",
        type=float,
        default=EFFICIENT_FRONTIER_L2_GAMMA,
        help="L2 regularization gamma for EfficientFrontier",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Number of worker processes to run strategies in parallel",
    )
    parser.add_argument(
        "--risk-free-rate-override",
        type=float,
        default=None,
        help=(
            "Override annual risk-free rate for the whole test period (e.g., 0.04). "
            "When set, date-varying ^IRX risk-free series is disabled."
        ),
    )
    args = parser.parse_args()
    configure_optimizer(args.optimizer, args.risk_aversion)
    configure_l2_gamma(args.l2_gamma)
    print(
        f"Optimizer: {OPTIMIZER_METHOD}"
        f" (risk_aversion={OPTIMIZER_RISK_AVERSION}, l2_gamma={EFFICIENT_FRONTIER_L2_GAMMA})"
    )
    print("  Systematic Black-Litterman Portfolio Engine")
    print("=" * 60)

    llm_banded_weekly_diagnostics: list[dict] = []
    llm_banded_monthly_diagnostics: list[dict] = []
    endowus_like_diagnostics: list[dict] = []

    lookback_start = "2014-01-01"
    tickers = list(ENDOWUS_WEIGHTS.keys()) + ["DX-Y.NYB", RISK_FREE_TICKER]

    daily_close, daily_open = get_yfinance_data(tickers, lookback_start, END_DATE)

    configure_risk_free_series(
        common_portfolio.extract_us_3m_tbill_series_from_prices(
            daily_close.get(RISK_FREE_TICKER), ticker=RISK_FREE_TICKER
        )
    )

    if args.risk_free_rate_override is not None:
        configure_risk_free_series(None)
        configure_risk_free_rate(args.risk_free_rate_override)
        print(
            f"Risk-free rate override enabled: {RISK_FREE_RATE:.4%} "
            "(constant across full test period)"
        )
    else:
        configure_risk_free_rate(
            common_portfolio.get_risk_free_rate_for_date(
                pd.to_datetime(END_DATE), RISK_FREE_SERIES, RISK_FREE_RATE
            )
        )
        print(f"Risk-free rate: {RISK_FREE_RATE:.4%} ({RISK_FREE_TICKER})")

    valid_tickers = [t for t in tickers if t in daily_close.columns]
    daily_close = daily_close[valid_tickers]
    daily_open = daily_open[[t for t in valid_tickers if t in daily_open.columns]]

    fred_data = {
        "T10YIE": get_fred_data("T10YIE", lookback_start, END_DATE),
        "T10Y2Y": get_fred_data("T10Y2Y", lookback_start, END_DATE),
    }

    weekly_prices = to_weekly_monday(daily_close)
    common_dates = weekly_prices.index
    weekly_exec_prices = get_execution_prices(daily_open, common_dates)

    monthly_prices = to_monthly_end(daily_close)
    monthly_dates = monthly_prices.index
    monthly_exec_prices = get_execution_prices(daily_open, monthly_dates)

    signals_A = load_bl_signal_csvs("results_endowus_A")
    signals_B = load_bl_signal_csvs("results_endowus_B")
    signals_C = load_bl_signal_csvs("results_endowus_C")
    signals_D = load_bl_signal_csvs("results_endowus_D")

    signals_A_monthly = align_signals_to_month_end(signals_A)
    signals_B_monthly = align_signals_to_month_end(signals_B)
    signals_C_monthly = align_signals_to_month_end(signals_C)
    signals_D_monthly = align_signals_to_month_end(signals_D)

    curves = {}
    weights_logs = {}

    strategies = [
        ("Endowus-60/40", pd.DataFrame(), "weekly"),
        ("Systematic-BL", None, "weekly"),  # None means use the systematic generation
        ("Systematic-BL-Tactical-Monthly", None, "monthly"),
        ("Markowitz", None, "weekly"),
        ("LLM-BL-A", signals_A, "weekly"),
        ("LLM-BL-B", signals_B, "weekly"),
        ("LLM-BL-C", signals_C, "weekly"),
        ("LLM-BL-D", signals_D, "weekly"),
        ("LLM-BL-A-Monthly", signals_A_monthly, "monthly"),
        ("LLM-BL-B-Monthly", signals_B_monthly, "monthly"),
        ("LLM-BL-C-Monthly", signals_C_monthly, "monthly"),
        ("LLM-BL-D-Monthly", signals_D_monthly, "monthly"),
        ("LLM-BL-A-Monthly-Banded", signals_A_monthly, "monthly"),
        ("LLM-BL-B-Monthly-Banded", signals_B_monthly, "monthly"),
        ("LLM-BL-C-Monthly-Banded", signals_C_monthly, "monthly"),
        ("LLM-BL-D-Monthly-Banded", signals_D_monthly, "monthly"),
        ("LLM-BL-A-Banded", signals_A, "weekly"),
        ("LLM-BL-B-Banded", signals_B, "weekly"),
        ("LLM-BL-C-Banded", signals_C, "weekly"),
        ("LLM-BL-D-Banded", signals_D, "weekly"),
    ]

    strategy_payloads = []
    for name, sigs, rebalance_policy in strategies:
        strategy_payloads.append(
            {
                "name": name,
                "signals": sigs,
                "rebalance_policy": rebalance_policy,
                "weekly_prices": weekly_prices,
                "weekly_exec_prices": weekly_exec_prices,
                "monthly_prices": monthly_prices,
                "monthly_exec_prices": monthly_exec_prices,
                "daily_close": daily_close,
                "fred_data": fred_data,
                "optimizer_method": OPTIMIZER_METHOD,
                "risk_aversion": OPTIMIZER_RISK_AVERSION,
                "l2_gamma": EFFICIENT_FRONTIER_L2_GAMMA,
                "risk_free_rate": RISK_FREE_RATE,
                "risk_free_series": RISK_FREE_SERIES,
            }
        )

    if args.parallel_workers > 1:
        print(f"Running strategies in parallel with {args.parallel_workers} workers...")
        results_map: dict[str, dict] = {}
        with ProcessPoolExecutor(max_workers=args.parallel_workers) as executor:
            future_map = {
                executor.submit(_run_single_strategy_worker, payload): payload["name"]
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
                        curve = result["curve"]
                        print(f"Completed {name}: Final equity ${curve.iloc[-1]:,.2f}")
                except Exception as ex:
                    print(f"[ERROR] Strategy failed for {name}: {ex}")

        for name, _, _ in strategies:
            result = results_map.get(name)
            if not result or result.get("skipped"):
                continue
            curves[name] = result["curve"]
            weights_logs[name] = result["weights"]
            if name.endswith("-Banded") and name.startswith("LLM-BL"):
                if "-Monthly" in name:
                    llm_banded_monthly_diagnostics.extend(result.get("diagnostics", []))
                else:
                    llm_banded_weekly_diagnostics.extend(result.get("diagnostics", []))
            else:
                endowus_like_diagnostics.extend(result.get("diagnostics", []))
    else:
        for payload in strategy_payloads:
            name = payload["name"]
            print(f"Running {name}...")
            result = _run_single_strategy(payload)
            if result.get("skipped"):
                print(f"  Skipping {name} - no signal data.")
                continue

            curves[name] = result["curve"]
            weights_logs[name] = result["weights"]
            if name.endswith("-Banded") and name.startswith("LLM-BL"):
                if "-Monthly" in name:
                    llm_banded_monthly_diagnostics.extend(result.get("diagnostics", []))
                else:
                    llm_banded_weekly_diagnostics.extend(result.get("diagnostics", []))
            else:
                endowus_like_diagnostics.extend(result.get("diagnostics", []))
            print(result["curve"])
            print(f"  Final equity: ${result['curve'].iloc[-1]:,.2f}")

    # Inject actual historical Endowus curves.
    # Metrics are computed only after a common alignment/rebase step.
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
    print(curves)
    print("\n[Metrics Summary]")
    print("=" * 70)
    summary = pd.DataFrame([m for m in metrics_list if m])
    if curves:
        min_curve_date = min(s.index.min() for s in curves.values() if len(s) > 0)
        max_curve_date = max(s.index.max() for s in curves.values() if len(s) > 0)

        ff_weekly = load_fama_french_data_weekly(
            min_curve_date.strftime("%Y-%m-%d"), max_curve_date.strftime("%Y-%m-%d")
        )
        ff_monthly = load_fama_french_data_monthly(
            min_curve_date.strftime("%Y-%m-%d"), max_curve_date.strftime("%Y-%m-%d")
        )

        summary = append_factor_alphas_to_summary(
            summary, curves, ff_weekly, ff_monthly
        )

    if not summary.empty:
        print(summary.to_string(index=False))
    print("=" * 70)

    os.makedirs("results", exist_ok=True)
    summary.to_csv("results/endowus_systematic_performance_summary.csv", index=False)

    label_col = None
    if "Strategy" in summary.columns:
        label_col = "Strategy"
    elif "Portfolio" in summary.columns:
        label_col = "Portfolio"

    if label_col is not None:
        tactical_summary = summary[
            summary[label_col] == "Systematic-BL-Tactical-Monthly"
        ]
        if not tactical_summary.empty:
            tactical_summary.to_csv(
                "results/endowus_systematic_tactical_monthly_metrics.csv", index=False
            )

    curves_df = pd.DataFrame(curves).ffill()
    curves_df.to_csv("results/endowus_systematic_equity_curves.csv")

    save_raw_weights_logs(weights_logs, output_dir="results/weights_raw")
    save_diagnostics_events(
        endowus_like_diagnostics,
        output_dir="results/diagnostics",
        file_name="endowus_engine_llm_diagnostics.csv",
    )
    save_diagnostics_events(
        llm_banded_weekly_diagnostics,
        output_dir="results/diagnostics",
        file_name="systematic_llm_banded_weekly_diagnostics.csv",
    )
    save_diagnostics_events(
        llm_banded_monthly_diagnostics,
        output_dir="results/diagnostics",
        file_name="systematic_llm_banded_monthly_diagnostics.csv",
    )
    save_diagnostics_events(
        llm_banded_weekly_diagnostics + llm_banded_monthly_diagnostics,
        output_dir="results/diagnostics",
        file_name="systematic_llm_banded_diagnostics.csv",
    )

    plot_combined_curves(curves_df)
    plot_composition_comparison(weights_logs)
    plot_asset_class_allocation_comparison(weights_logs)

    # Plot individual Endowus tickers over the same common timeframe
    start_date = (
        common_start if common_start is not None else pd.to_datetime(START_DATE)
    )
    plot_individual_ticker_growth(
        daily_close, list(ENDOWUS_WEIGHTS.keys()), start_date, END_DATE
    )

    generate_full_period_markdown_report(
        summary_df=summary,
        curves_df=curves_df,
        output_dir="results",
        combined_plot_path="results/endowus_systematic_comparison_plot.png",
        composition_plot_path="results/endowus_systematic_composition_comparison.png",
        asset_class_plot_path="results/endowus_systematic_asset_class_allocation.png",
        tickers_plot_path="results/endowus_individual_tickers.png",
    )

    print("Backtest complete. Results saved to results/ folder.")


if __name__ == "__main__":
    main()
