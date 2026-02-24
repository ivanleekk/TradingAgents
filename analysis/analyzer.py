"""
Core trading strategy analyzer module.

This module provides the main TradingAnalyzer class that orchestrates
the analysis of trading decisions from CSV files.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from typing import Optional, Dict, Any
import re
from datetime import datetime


class TradingAnalyzer:
    """
    Main analyzer class for trading strategy performance.

    This class reads trading decision files, fetches price data,
    simulates portfolio performance, and prepares data for metrics calculation.
    """

    def __init__(
        self,
        csv_path: str,
        initial_capital: float = 10000.0,
        risk_free_rate: float = 0.03,
    ):
        """
        Initialize the trading analyzer.

        Args:
            csv_path: Path to the CSV file with trading decisions
            initial_capital: Starting portfolio value (default: $10,000)
            risk_free_rate: Annual risk-free rate for Sharpe ratio (default: 3%)
        """
        self.csv_path = Path(csv_path)
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate

        # Parse filename to extract metadata
        self.ticker, self.start_date, self.end_date = self._parse_filename()

        # Load data
        self.decisions_df = pd.read_csv(csv_path)
        self.decisions_df["test_date"] = pd.to_datetime(self.decisions_df["test_date"])
        self.decisions_df = self.decisions_df.sort_values("test_date")

        # Fetch price data
        self.price_df = self._fetch_price_data()

        # Simulate portfolio
        self.portfolio_df = self._simulate_portfolio()

    def _parse_filename(self) -> tuple[str, str, str]:
        """
        Parse the CSV filename to extract ticker and date range.

        Returns:
            Tuple of (ticker, start_date, end_date)
        """
        filename = self.csv_path.stem

        # Pattern: {ticker}_decisions_{start_date}_{end_date}
        pattern = r"(.+?)_decisions_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})"
        match = re.match(pattern, filename)

        if not match:
            raise ValueError(f"Filename does not match expected pattern: {filename}")

        ticker, start_date, end_date = match.groups()
        return ticker, start_date, end_date

    def _fetch_price_data(self) -> pd.DataFrame:
        """
        Fetch historical price data for the ticker.

        Returns:
            DataFrame with date index and OHLCV data
        """
        # Add buffer days to ensure we have data before first decision
        start_buffer = pd.to_datetime(self.start_date) - pd.Timedelta(days=30)
        end_buffer = pd.to_datetime(self.end_date) + pd.Timedelta(days=30)

        print(
            f"Fetching price data for {self.ticker} from {start_buffer.date()} to {end_buffer.date()}..."
        )

        ticker_obj = yf.Ticker(self.ticker)
        df = ticker_obj.history(start=start_buffer, end=end_buffer)

        if df.empty:
            raise ValueError(f"No price data found for {self.ticker}")

        df.index = pd.to_datetime(df.index).tz_localize(None)

        return df

    def _simulate_portfolio(self) -> pd.DataFrame:
        """
        Simulate portfolio performance based on trading decisions.

        Returns:
            DataFrame with portfolio value over time
        """
        # Create a daily DataFrame with forward-filled decisions
        daily_df = pd.DataFrame(index=self.price_df.index)
        daily_df["price"] = self.price_df["Close"]

        # Map decisions to daily data
        decisions_dict = dict(
            zip(self.decisions_df["test_date"], self.decisions_df["decision"])
        )

        # Forward fill decisions
        daily_df["decision"] = pd.Series(decisions_dict).reindex(
            daily_df.index, method="ffill"
        )
        daily_df["decision"] = daily_df["decision"].fillna(
            "HOLD"
        )  # Before first decision

        # Calculate position (1 = long, 0 = not invested)
        daily_df["position"] = (daily_df["decision"] == "BUY").astype(int)

        # Calculate returns
        daily_df["price_return"] = daily_df["price"].pct_change()
        daily_df["strategy_return"] = (
            daily_df["position"].shift(1) * daily_df["price_return"]
        )

        # Calculate cumulative returns
        daily_df["strategy_cum_return"] = (1 + daily_df["strategy_return"]).cumprod()
        daily_df["portfolio_value"] = (
            self.initial_capital * daily_df["strategy_cum_return"]
        )

        # Calculate buy & hold benchmark
        first_price = daily_df["price"].iloc[0]
        daily_df["bnh_value"] = self.initial_capital * (daily_df["price"] / first_price)

        # Calculate drawdown
        daily_df["cum_max"] = daily_df["portfolio_value"].cummax()
        daily_df["drawdown"] = (
            (daily_df["portfolio_value"] - daily_df["cum_max"])
            / daily_df["cum_max"]
            * 100
        )

        # Drop NaN rows from start
        daily_df = daily_df.dropna(subset=["strategy_return"])

        return daily_df

    def get_trade_log(self) -> pd.DataFrame:
        """
        Generate a trade log with entry/exit points and P&L.

        Returns:
            DataFrame with trade details
        """
        trades = []
        position = 0
        entry_date = None
        entry_price = None

        for idx, row in self.portfolio_df.iterrows():
            decision = row["decision"]
            price = row["price"]

            if decision == "BUY" and position == 0:
                # Entry
                position = 1
                entry_date = idx
                entry_price = price

            elif decision == "SELL" and position == 1:
                # Exit
                exit_date = idx
                exit_price = price
                pnl = (exit_price - entry_price) / entry_price * 100

                trades.append(
                    {
                        "entry_date": entry_date,
                        "entry_price": entry_price,
                        "exit_date": exit_date,
                        "exit_price": exit_price,
                        "pnl_pct": pnl,
                        "win": pnl > 0,
                    }
                )

                position = 0
                entry_date = None
                entry_price = None

        # If still in position at end, close it
        if position == 1:
            exit_date = self.portfolio_df.index[-1]
            exit_price = self.portfolio_df["price"].iloc[-1]
            pnl = (exit_price - entry_price) / entry_price * 100

            trades.append(
                {
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "exit_date": exit_date,
                    "exit_price": exit_price,
                    "pnl_pct": pnl,
                    "win": pnl > 0,
                }
            )

        return pd.DataFrame(trades)

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get basic summary statistics.

        Returns:
            Dictionary with summary statistics
        """
        pf = self.portfolio_df

        total_return = (pf["portfolio_value"].iloc[-1] / self.initial_capital - 1) * 100
        bnh_return = (pf["bnh_value"].iloc[-1] / self.initial_capital - 1) * 100

        return {
            "ticker": self.ticker,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "final_value": pf["portfolio_value"].iloc[-1],
            "total_return_pct": total_return,
            "bnh_return_pct": bnh_return,
            "outperformance_pct": total_return - bnh_return,
            "total_days": len(pf),
            "trading_days": len(pf),
        }
