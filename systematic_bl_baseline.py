import glob
import json
import os
import warnings

import endowus_portfolio_engine as endowus_engine
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
from pypfopt import BlackLittermanModel, risk_models, EfficientFrontier

# Import logic from endowus_portfolio_engine directly to reuse its load/run logic for LLM/Endowus
from endowus_portfolio_engine import (
    load_bl_signal_csvs,
    compute_metrics,
    run_endowus_backtest,
    load_endowus_historical,
)

warnings.filterwarnings("ignore")

DATA_DIR = "data"
START_DATE = "2019-09-01"
END_DATE = "2024-12-31"
INITIAL_EQUITY = 10_000.0
TRANSACTION_COST = 0.001
RISK_FREE_RATE = 0.04
WEEKS_PER_YEAR = 52

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
    "0P0001CC3M": 0.04,  # iShares Global Aggregate 1-5 Year Bond Index Fund (IE) SGD-Hedged
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
    return df.resample("W-MON", label="left", closed="left").last()


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
        omega = np.diag(np.diag(tau * P_matrix @ S.values @ P_matrix.T))

        bl = BlackLittermanModel(
            S, pi=market_prior, P=P_matrix, Q=Q_vector, omega=omega
        )
        bl_returns = bl.bl_returns()
        bl_cov = bl.bl_cov()

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
    def __init__(self, initial_equity: float):
        self.cash = initial_equity
        self.shares: dict[str, float] = {}

    def total_equity(self, prices: dict[str, float]) -> float:
        mv = sum(self.shares.get(t, 0.0) * prices.get(t, 0.0) for t in self.shares)
        return self.cash + mv

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
            fee = gross * TRANSACTION_COST
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
                fee = gross * TRANSACTION_COST
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
                fee = gross * TRANSACTION_COST
                net_spend = gross + fee
                net_spend = min(net_spend, self.cash)
                actual_gross = net_spend / (1 + TRANSACTION_COST)
                self.shares[t] = self.shares.get(t, 0.0) + actual_gross / px
                self.cash -= net_spend


def run_systematic_backtest(
    weekly_prices: pd.DataFrame,
    weekly_exec_prices: pd.DataFrame,
    daily_prices: pd.DataFrame,
    fred_data: dict,
) -> tuple[pd.Series, pd.DataFrame]:

    port = Portfolio(INITIAL_EQUITY)
    equity_curve = {}
    weights_log = {}

    tickers = list(ENDOWUS_WEIGHTS.keys())
    dates = weekly_prices.index.intersection(weekly_exec_prices.index)

    dates = dates[
        (dates >= pd.to_datetime(START_DATE)) & (dates <= pd.to_datetime(END_DATE))
    ]

    for date in dates:
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

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
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

    handles, labels = axes_arr[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1)

    plt.setp([ax.get_xticklabels() for ax in axes_arr[:n]], rotation=45, ha="right")
    plt.tight_layout(rect=(0, 0, 0.84, 0.96))
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Composition comparison plot saved to {output_path}")


def main():
    print("=" * 60)
    print("  Systematic Black-Litterman Portfolio Engine")
    print("=" * 60)

    # Keep shared backtest helpers on the same Endowus fund universe as this script.
    endowus_engine.ENDOWUS_WEIGHTS = ENDOWUS_WEIGHTS.copy()
    endowus_engine.EQUITY_ASSETS = EQUITY_ASSETS.copy()
    endowus_engine.FIXED_INCOME_ASSETS = FIXED_INCOME_ASSETS.copy()
    endowus_engine.TARGET_EQUITY_ALLOCATION = TARGET_EQUITY_ALLOCATION

    lookback_start = "2014-01-01"
    tickers = list(ENDOWUS_WEIGHTS.keys()) + ["DX-Y.NYB"]

    daily_close, daily_open = get_yfinance_data(tickers, lookback_start, END_DATE)

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

    signals_A = load_bl_signal_csvs("results_endowus_A")
    signals_B = load_bl_signal_csvs("results_endowus_B")
    signals_C = load_bl_signal_csvs("results_endowus_C")
    signals_D = load_bl_signal_csvs("results_endowus_D")

    curves = {}
    weights_logs = {}

    strategies = [
        ("Endowus-60/40", pd.DataFrame(), "weekly"),
        (
            "Endowus-60/40-NoRebalanceIfMissing",
            pd.DataFrame(),
            "full-universe-only",
        ),
        ("Endowus-60/40-BuyHold", pd.DataFrame(), "buy-and-hold"),
        ("Systematic-BL", None, "weekly"),  # None means use the systematic generation
        ("LLM-BL-A", signals_A, "weekly"),
        ("LLM-BL-B", signals_B, "weekly"),
        ("LLM-BL-C", signals_C, "weekly"),
        ("LLM-BL-D", signals_D, "weekly"),
    ]

    for name, sigs, rebalance_policy in strategies:
        print(f"Running {name}...")
        if name.startswith("LLM") and sigs is not None and sigs.empty:
            print(f"  Skipping {name} - no signal data.")
            continue

        if name == "Systematic-BL":
            curve, wlog = run_systematic_backtest(
                weekly_prices, weekly_exec_prices, daily_close, fred_data
            )
        else:
            # We filter out DX-Y.NYB for endowus engine backtest
            # AND we must slice it exactly to the START_DATE to END_DATE window
            run_weekly = weekly_prices[
                [t for t in list(ENDOWUS_WEIGHTS.keys()) if t in weekly_prices.columns]
            ]
            run_exec = weekly_exec_prices[
                [
                    t
                    for t in list(ENDOWUS_WEIGHTS.keys())
                    if t in weekly_exec_prices.columns
                ]
            ]
            run_daily = daily_close[
                [t for t in list(ENDOWUS_WEIGHTS.keys()) if t in daily_close.columns]
            ]

            # Align dates to the backtest window
            run_weekly = run_weekly[
                (run_weekly.index >= pd.to_datetime(START_DATE))
                & (run_weekly.index <= pd.to_datetime(END_DATE))
            ]
            run_exec = run_exec[
                (run_exec.index >= pd.to_datetime(START_DATE))
                & (run_exec.index <= pd.to_datetime(END_DATE))
            ]

            curve, wlog = run_endowus_backtest(
                name,
                sigs,
                run_weekly,
                run_exec,
                run_daily,
                rebalance_policy=rebalance_policy,
                log_realized_weights=(rebalance_policy == "buy-and-hold"),
            )

        curves[name] = curve
        weights_logs[name] = wlog
        print(curve)
        print(f"  Final equity: ${curve.iloc[-1]:,.2f}")

    metrics_list = [compute_metrics(c, n) for n, c in curves.items()]

    # Inject actual historical Endowus numbers
    load_endowus_historical(curves, metrics_list)
    print("\n[Equity Curves]")
    print(curves)
    print("\n[Metrics Summary]")
    print("=" * 70)
    summary = pd.DataFrame([m for m in metrics_list if m])
    if not summary.empty:
        print(summary.to_string(index=False))
    print("=" * 70)

    os.makedirs("results", exist_ok=True)
    summary.to_csv("results/endowus_systematic_performance_summary.csv", index=False)

    curves_df = pd.DataFrame(curves).ffill()
    curves_df.to_csv("results/endowus_systematic_equity_curves.csv")

    plot_combined_curves(curves_df)
    plot_composition_comparison(weights_logs)

    print("Backtest complete. Results saved to results/ folder.")


if __name__ == "__main__":
    main()
