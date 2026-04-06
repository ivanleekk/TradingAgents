import glob
import json
import os
from typing import cast

import numpy as np
import pandas as pd
import yfinance as yf
from pypfopt import EfficientFrontier, objective_functions


def set_optimizer_config(
    method: str,
    risk_aversion: float,
    choices: tuple[str, ...],
) -> tuple[str, float]:
    if method not in choices:
        raise ValueError(f"optimizer method must be one of {choices}, got: {method}")
    return str(method), float(risk_aversion)


def set_risk_free_rate(rate: float) -> float:
    if not np.isfinite(rate):
        raise ValueError(f"risk-free rate must be finite, got: {rate}")
    return float(rate)


def extract_us_3m_tbill_rate_from_prices(
    irx_prices: pd.Series | None,
    ticker: str = "^IRX",
    fallback_rate: float = 0.04,
) -> float:
    if irx_prices is None:
        print(
            f"[WARNING] {ticker} data is unavailable; using fallback "
            f"RISK_FREE_RATE={fallback_rate:.4%}"
        )
        return float(fallback_rate)

    clean = irx_prices.dropna()
    if clean.empty:
        print(
            f"[WARNING] {ticker} has no valid prices; using fallback "
            f"RISK_FREE_RATE={fallback_rate:.4%}"
        )
        return float(fallback_rate)

    latest_value = float(clean.iloc[-1])
    annual_rf = latest_value / 100.0
    if not np.isfinite(annual_rf):
        print(
            f"[WARNING] Invalid {ticker} value ({latest_value}); using fallback "
            f"RISK_FREE_RATE={fallback_rate:.4%}"
        )
        return float(fallback_rate)

    print(
        f"Using risk-free rate from {ticker}: {annual_rf:.4%}"
        f" (as of {clean.index[-1].date()})"
    )
    return annual_rf


def extract_us_3m_tbill_series_from_prices(
    irx_prices: pd.Series | None,
    ticker: str = "^IRX",
) -> pd.Series:
    if irx_prices is None:
        return pd.Series(dtype=float)

    clean = irx_prices.copy()
    clean.index = pd.to_datetime(clean.index)
    clean = clean.sort_index().dropna()
    if clean.empty:
        return pd.Series(dtype=float)

    annual_rf = (
        (clean.astype(float) / 100.0).replace([np.inf, -np.inf], np.nan).dropna()
    )
    annual_rf.name = f"{ticker}_annual_rf"
    return annual_rf


def load_us_3m_tbill_rate(
    start: str,
    end: str,
    ticker: str = "^IRX",
    fallback_rate: float = 0.04,
) -> float:
    try:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
        )
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            raise ValueError(f"empty {ticker} download")

        if isinstance(raw.columns, pd.MultiIndex):
            close_series = raw["Close"][ticker]
        else:
            close_series = raw["Close"]

        return extract_us_3m_tbill_rate_from_prices(
            close_series, ticker=ticker, fallback_rate=fallback_rate
        )
    except Exception as e:
        print(
            f"[WARNING] Failed to load {ticker}; using fallback "
            f"RISK_FREE_RATE={fallback_rate:.4%}: {e}"
        )
        return float(fallback_rate)


def load_us_3m_tbill_series(
    start: str,
    end: str,
    ticker: str = "^IRX",
) -> pd.Series:
    try:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
        )
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            raise ValueError(f"empty {ticker} download")

        if isinstance(raw.columns, pd.MultiIndex):
            close_series = raw["Close"][ticker]
        else:
            close_series = raw["Close"]

        series = extract_us_3m_tbill_series_from_prices(close_series, ticker=ticker)
        if series.empty:
            raise ValueError(f"no valid {ticker} close values")
        return series
    except Exception as e:
        print(f"[WARNING] Failed to load {ticker} time series: {e}")
        return pd.Series(dtype=float)


def get_risk_free_rate_for_date(
    as_of: pd.Timestamp,
    risk_free_series: pd.Series | None,
    fallback_rate: float,
) -> float:
    if risk_free_series is None or risk_free_series.empty:
        return float(fallback_rate)

    idx = pd.to_datetime(risk_free_series.index)
    aligned = pd.Series(risk_free_series.values, index=idx).sort_index()
    hist = aligned.loc[: pd.to_datetime(as_of)].dropna()
    if hist.empty:
        return float(fallback_rate)
    return float(hist.iloc[-1])


def get_period_risk_free_returns(
    index: pd.Index,
    periods_per_year: int,
    risk_free_series: pd.Series | None,
    fallback_rate: float,
) -> pd.Series:
    dt_index = pd.DatetimeIndex(index)
    if periods_per_year <= 0:
        periods_per_year = 1

    if risk_free_series is None or risk_free_series.empty:
        period_rf = (1 + float(fallback_rate)) ** (1 / periods_per_year) - 1
        return pd.Series(period_rf, index=dt_index)

    annual_rf = pd.Series(
        risk_free_series.values, index=pd.to_datetime(risk_free_series.index)
    ).sort_index()
    aligned_annual = annual_rf.reindex(dt_index, method="ffill")
    if aligned_annual.isna().all():
        period_rf = (1 + float(fallback_rate)) ** (1 / periods_per_year) - 1
        return pd.Series(period_rf, index=dt_index)

    aligned_annual = aligned_annual.fillna(float(fallback_rate))
    period_values = np.power(1 + aligned_annual.astype(float), 1 / periods_per_year) - 1
    return pd.Series(period_values, index=dt_index)


def _build_efficient_frontier(expected_rets, cov_matrix, l2_gamma: float):
    ef = EfficientFrontier(expected_rets, cov_matrix)
    ef.add_objective(objective_functions.L2_reg, gamma=float(l2_gamma))
    return ef


def optimize_efficient_frontier(
    expected_rets,
    cov_matrix,
    optimizer_method: str,
    risk_aversion: float,
    risk_free_rate: float,
    l2_gamma: float,
    context: str = "",
):
    ef = _build_efficient_frontier(expected_rets, cov_matrix, l2_gamma)

    if optimizer_method == "min-volatility":
        ef.min_volatility()
        return ef, "min_volatility"

    if optimizer_method == "max-quadratic-utility":
        try:
            ef.max_quadratic_utility(risk_aversion=int(round(risk_aversion)))
            return ef, "max_quadratic_utility"
        except Exception as ex:
            print(
                f"[WARNING] Max Quadratic Utility failed, switching to Min Volatility"
                f" ({context}): {ex}"
            )
            ef = _build_efficient_frontier(expected_rets, cov_matrix, l2_gamma)
            ef.min_volatility()
            return ef, "min_volatility_fallback_from_max_quadratic_utility"

    try:
        ef.max_sharpe(risk_free_rate=risk_free_rate)
        return ef, "max_sharpe"
    except Exception as ex:
        print(f"[WARNING] Max Sharpe failed, switching to Quadratic Utility: {ex}")
        ef = _build_efficient_frontier(expected_rets, cov_matrix, l2_gamma)

        # Fallback to Quadratic Utility instead of min_volatility
        # Note: You will need to ensure 'risk_aversion' is passed down to this block
        try:
            ef.max_quadratic_utility(risk_aversion=risk_aversion)
            return ef, "max_quadratic_utility_fallback"
        except:
            # Double failsafe if utility also crashes
            ef = _build_efficient_frontier(expected_rets, cov_matrix, l2_gamma)
            ef.min_volatility()
            return ef, "min_volatility_double_fallback"


def is_invalid_view(value) -> bool:
    if value is None:
        return True
    try:
        val = float(value)
    except (TypeError, ValueError):
        return True
    return not np.isfinite(val)


def resolve_view_value(views: dict, ticker: str, fallback: float) -> float:
    raw = views.get(ticker)
    if is_invalid_view(raw):
        return float(fallback)
    return float(cast(float | int | str, raw))


def to_weekly_monday(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("W-MON", label="left", closed="left").last()


def get_execution_prices(
    daily_open: pd.DataFrame, weekly_dates: pd.Index
) -> pd.DataFrame:
    trading_days = daily_open.index
    rows = {}
    for monday in pd.DatetimeIndex(weekly_dates):
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
    if ticker not in data.columns:
        return None

    hist = data.loc[:as_of, ticker].dropna()
    if hist.empty:
        return None
    return float(hist.iloc[-1])


DEFAULT_ENDOWUS_WEIGHTS = {
    "0P0001AF7U.SI": 0.195,
    "0P0000KYEE.SI": 0.1,
    "SPY": 0.09,
    "^990100-USD-STRD": 0.09,
    "DE000SLA4YD9.SG": 0.084,
    "AGGG.L": 0.08,
    "IE0002461055.IR": 0.07,
    "0P0001AF7Z.SI": 0.06,
    "0P0001EQUE.SI": 0.05,
    "0P0001EF2T.SI": 0.048,
    "0P0001CC3M": 0.04,
    "PEBIX": 0.04,
    "EIMI.L": 0.033,
    "0P0001DWI0.SI": 0.02,
}


def load_endowus_6040_weights(
    raw_json_path: str = "data/endowus_raw.json",
    id_to_yahoo_csv_path: str = "data/endowus_yfinance/instrument_id_to_yahoo.csv",
) -> dict[str, float]:
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

            for _, row in df.iterrows():
                decision_str = str(row["decision"]).strip()

                if decision_str.startswith("```json"):
                    decision_str = decision_str[7:]
                if decision_str.startswith("```"):
                    decision_str = decision_str[3:]
                if decision_str.endswith("```"):
                    decision_str = decision_str[:-3]
                decision_str = decision_str.strip()

                try:
                    data = json.loads(decision_str)

                    ret_str = (
                        str(data.get("Target_Return_30d", "")).replace("%", "").strip()
                    )
                    if not ret_str or ret_str.lower() in ["null", "nan", "none"]:
                        target_returns.append(None)
                    else:
                        ret = float(ret_str) / 100.0
                        annual_ret = (1 + ret) ** (365 / 30) - 1
                        target_returns.append(annual_ret)

                    conf = data.get("Confidence_Score")
                    if (
                        conf is None
                        or str(conf).strip() == ""
                        or str(conf).lower() in ["null", "nan", "none"]
                    ):
                        confidences.append(None)
                    else:
                        confidences.append(float(conf))

                except Exception:
                    target_returns.append(None)
                    confidences.append(None)

            df["target_return"] = target_returns
            df["confidence"] = confidences
            df = df.sort_values("Date")
            df["target_return"] = df["target_return"].ffill()
            df["confidence"] = df["confidence"].ffill().fillna(5.0)

            frames.append(df[["Date", "Ticker", "target_return", "confidence"]])

        except Exception as e:
            print(f"Error parsing {fp}: {e}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
