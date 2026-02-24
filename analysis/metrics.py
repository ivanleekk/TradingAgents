"""
Performance metrics calculation module.

This module provides the PerformanceMetrics class for calculating
all key trading strategy performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


class PerformanceMetrics:
    """
    Calculate comprehensive performance metrics for trading strategies.

    Metrics include:
    - Annualized Return
    - Annualized Sharpe Ratio
    - Maximum Drawdown
    - Win Rate
    - Calmar Ratio
    """

    def __init__(self, analyzer):
        """
        Initialize metrics calculator.

        Args:
            analyzer: TradingAnalyzer instance with portfolio data
        """
        self.analyzer = analyzer
        self.portfolio_df = analyzer.portfolio_df
        self.risk_free_rate = analyzer.risk_free_rate

    def calculate_all_metrics(self) -> Dict[str, float]:
        """
        Calculate all performance metrics.

        Returns:
            Dictionary with all metrics
        """
        metrics = {
            "annualized_return": self.annualized_return(),
            "annualized_sharpe_ratio": self.annualized_sharpe_ratio(),
            "maximum_drawdown": self.maximum_drawdown(),
            "win_rate": self.win_rate(),
            "calmar_ratio": self.calmar_ratio(),
            "total_return": self.total_return(),
            "volatility": self.annualized_volatility(),
            "sortino_ratio": self.sortino_ratio(),
            "profit_factor": self.profit_factor(),
            "total_trades": self.total_trades(),
            "winning_trades": self.winning_trades(),
            "losing_trades": self.losing_trades(),
            "avg_win": self.avg_win(),
            "avg_loss": self.avg_loss(),
            "largest_win": self.largest_win(),
            "largest_loss": self.largest_loss(),
            "avg_trade_duration": self.avg_trade_duration(),
        }

        return metrics

    def annualized_return(self) -> float:
        """
        Calculate annualized return (CAGR).

        Returns:
            Annualized return as percentage
        """
        start_value = self.analyzer.initial_capital
        end_value = self.portfolio_df["portfolio_value"].iloc[-1]

        # Calculate number of years
        start_date = self.portfolio_df.index[0]
        end_date = self.portfolio_df.index[-1]
        years = (end_date - start_date).days / 365.25

        if years <= 0:
            return 0.0

        # CAGR formula: (End Value / Start Value)^(1/years) - 1
        cagr = (end_value / start_value) ** (1 / years) - 1

        return cagr * 100  # Return as percentage

    def annualized_sharpe_ratio(self) -> float:
        """
        Calculate annualized Sharpe ratio.

        The Sharpe ratio measures risk-adjusted returns by comparing
        excess returns over the risk-free rate to return volatility.

        Returns:
            Annualized Sharpe ratio
        """
        returns = self.portfolio_df["strategy_return"].dropna()

        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        # Calculate excess returns
        # Convert annual risk-free rate to daily
        daily_rf = (1 + self.risk_free_rate) ** (1 / 252) - 1
        excess_returns = returns - daily_rf

        # Calculate Sharpe ratio
        sharpe = excess_returns.mean() / returns.std()

        # Annualize (assuming 252 trading days per year)
        annualized_sharpe = sharpe * np.sqrt(252)

        return annualized_sharpe

    def maximum_drawdown(self) -> float:
        """
        Calculate maximum drawdown.

        Maximum drawdown is the largest peak-to-trough decline
        in portfolio value during the evaluation period.

        Returns:
            Maximum drawdown as percentage (negative value)
        """
        drawdown = self.portfolio_df["drawdown"].min()
        return drawdown

    def win_rate(self) -> float:
        """
        Calculate win rate.

        Win rate is the percentage of trades that generate positive returns.

        Returns:
            Win rate as percentage
        """
        trade_log = self.analyzer.get_trade_log()

        if len(trade_log) == 0:
            return 0.0

        winning_trades = trade_log["win"].sum()
        total_trades = len(trade_log)

        return (winning_trades / total_trades) * 100

    def calmar_ratio(self) -> float:
        """
        Calculate Calmar ratio.

        The Calmar ratio is the ratio of annualized return to maximum drawdown,
        providing a risk-adjusted metric focused on downside protection.

        Returns:
            Calmar ratio
        """
        ann_return = self.annualized_return()
        max_dd = abs(self.maximum_drawdown())

        if max_dd == 0:
            return 0.0

        return ann_return / max_dd

    def total_return(self) -> float:
        """
        Calculate total return over the period.

        Returns:
            Total return as percentage
        """
        start_value = self.analyzer.initial_capital
        end_value = self.portfolio_df["portfolio_value"].iloc[-1]

        return ((end_value - start_value) / start_value) * 100

    def annualized_volatility(self) -> float:
        """
        Calculate annualized volatility (standard deviation of returns).

        Returns:
            Annualized volatility as percentage
        """
        returns = self.portfolio_df["strategy_return"].dropna()

        if len(returns) == 0:
            return 0.0

        daily_vol = returns.std()
        annualized_vol = daily_vol * np.sqrt(252)

        return annualized_vol * 100

    def sortino_ratio(self) -> float:
        """
        Calculate Sortino ratio (similar to Sharpe but only considers downside volatility).

        Returns:
            Annualized Sortino ratio
        """
        returns = self.portfolio_df["strategy_return"].dropna()

        if len(returns) == 0:
            return 0.0

        # Convert annual risk-free rate to daily
        daily_rf = (1 + self.risk_free_rate) ** (1 / 252) - 1
        excess_returns = returns - daily_rf

        # Calculate downside deviation (only negative returns)
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        downside_std = downside_returns.std()
        sortino = excess_returns.mean() / downside_std

        # Annualize
        annualized_sortino = sortino * np.sqrt(252)

        return annualized_sortino

    def profit_factor(self) -> float:
        """
        Calculate profit factor (ratio of gross profits to gross losses).

        Returns:
            Profit factor
        """
        trade_log = self.analyzer.get_trade_log()

        if len(trade_log) == 0:
            return 0.0

        winning_pnl = trade_log[trade_log["win"]]["pnl_pct"].sum()
        losing_pnl = abs(trade_log[~trade_log["win"]]["pnl_pct"].sum())

        if losing_pnl == 0:
            return np.inf if winning_pnl > 0 else 0.0

        return winning_pnl / losing_pnl

    def total_trades(self) -> int:
        """Get total number of trades."""
        return len(self.analyzer.get_trade_log())

    def winning_trades(self) -> int:
        """Get number of winning trades."""
        trade_log = self.analyzer.get_trade_log()
        return trade_log["win"].sum()

    def losing_trades(self) -> int:
        """Get number of losing trades."""
        trade_log = self.analyzer.get_trade_log()
        return (~trade_log["win"]).sum()

    def avg_win(self) -> float:
        """Get average winning trade percentage."""
        trade_log = self.analyzer.get_trade_log()
        winning_trades = trade_log[trade_log["win"]]

        if len(winning_trades) == 0:
            return 0.0

        return winning_trades["pnl_pct"].mean()

    def avg_loss(self) -> float:
        """Get average losing trade percentage."""
        trade_log = self.analyzer.get_trade_log()
        losing_trades = trade_log[~trade_log["win"]]

        if len(losing_trades) == 0:
            return 0.0

        return losing_trades["pnl_pct"].mean()

    def largest_win(self) -> float:
        """Get largest winning trade percentage."""
        trade_log = self.analyzer.get_trade_log()

        if len(trade_log) == 0:
            return 0.0

        return trade_log["pnl_pct"].max()

    def largest_loss(self) -> float:
        """Get largest losing trade percentage."""
        trade_log = self.analyzer.get_trade_log()

        if len(trade_log) == 0:
            return 0.0

        return trade_log["pnl_pct"].min()

    def avg_trade_duration(self) -> float:
        """Get average trade duration in days."""
        trade_log = self.analyzer.get_trade_log()

        if len(trade_log) == 0:
            return 0.0

        durations = (trade_log["exit_date"] - trade_log["entry_date"]).dt.days
        return durations.mean()

    def get_metrics_summary(self) -> str:
        """
        Get a formatted string summary of key metrics.

        Returns:
            Formatted metrics summary
        """
        metrics = self.calculate_all_metrics()

        summary = f"""
=== PERFORMANCE METRICS ===

Key Risk-Adjusted Metrics:
  Annualized Return:        {metrics['annualized_return']:>10.2f}%
  Annualized Sharpe Ratio:  {metrics['annualized_sharpe_ratio']:>10.2f}
  Maximum Drawdown:         {metrics['maximum_drawdown']:>10.2f}%
  Win Rate:                 {metrics['win_rate']:>10.2f}%
  Calmar Ratio:             {metrics['calmar_ratio']:>10.2f}

Additional Metrics:
  Total Return:             {metrics['total_return']:>10.2f}%
  Annualized Volatility:    {metrics['volatility']:>10.2f}%
  Sortino Ratio:            {metrics['sortino_ratio']:>10.2f}
  Profit Factor:            {metrics['profit_factor']:>10.2f}

Trade Statistics:
  Total Trades:             {metrics['total_trades']:>10.0f}
  Winning Trades:           {metrics['winning_trades']:>10.0f}
  Losing Trades:            {metrics['losing_trades']:>10.0f}
  Average Win:              {metrics['avg_win']:>10.2f}%
  Average Loss:             {metrics['avg_loss']:>10.2f}%
  Largest Win:              {metrics['largest_win']:>10.2f}%
  Largest Loss:             {metrics['largest_loss']:>10.2f}%
  Avg Trade Duration:       {metrics['avg_trade_duration']:>10.1f} days
"""

        return summary
