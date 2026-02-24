"""
Portfolio analysis module for combining multiple trading strategies.

This module allows you to combine multiple trading strategies into a single
portfolio with equal or custom weights, and analyze the combined performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from analysis import TradingAnalyzer, PerformanceMetrics


class PortfolioAnalyzer:
    """
    Combine multiple trading strategies into a weighted portfolio.

    This class loads multiple strategy CSV files, combines them with specified
    weights, and calculates portfolio-level performance metrics.
    """

    def __init__(
        self,
        strategy_files: Dict[str, str],
        weights: Optional[Dict[str, float]] = None,
        initial_capital: float = 10000.0,
        risk_free_rate: float = 0.03,
        rebalance: bool = False,
    ):
        """
        Initialize portfolio analyzer.

        Args:
            strategy_files: Dict mapping ticker to CSV file path
                           e.g., {'AAPL': 'path/to/aapl.csv', 'MSFT': 'path/to/msft.csv'}
            weights: Dict mapping ticker to weight (must sum to 1.0)
                    If None, uses equal weights
            initial_capital: Total starting capital for portfolio
            risk_free_rate: Annual risk-free rate for Sharpe ratio
            rebalance: Whether to rebalance to target weights daily (default: False)
        """
        self.strategy_files = strategy_files
        self.tickers = list(strategy_files.keys())
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.rebalance = rebalance

        # Set weights
        if weights is None:
            # Equal weights
            n = len(self.tickers)
            self.weights = {ticker: 1.0 / n for ticker in self.tickers}
        else:
            # Validate weights
            if set(weights.keys()) != set(self.tickers):
                raise ValueError("Weights must be specified for all tickers")

            weight_sum = sum(weights.values())
            if not np.isclose(weight_sum, 1.0):
                raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")

            self.weights = weights

        # Load individual strategies
        print(f"Loading {len(self.tickers)} strategies...")
        self.strategies = {}
        for ticker, csv_file in strategy_files.items():
            try:
                analyzer = TradingAnalyzer(
                    csv_file, initial_capital * self.weights[ticker], risk_free_rate
                )
                self.strategies[ticker] = analyzer
                print(f"  ✓ {ticker}: {len(analyzer.portfolio_df)} days")
            except Exception as e:
                print(f"  ✗ {ticker}: Error - {e}")
                raise

        # Calculate portfolio performance
        print("Calculating portfolio performance...")
        self.portfolio_df = self._calculate_portfolio_performance()
        print(f"  ✓ Portfolio: {len(self.portfolio_df)} days")

    def _calculate_portfolio_performance(self) -> pd.DataFrame:
        """
        Calculate combined portfolio performance.

        Returns:
            DataFrame with portfolio performance
        """
        # Find common date range across all strategies
        start_dates = [s.portfolio_df.index[0] for s in self.strategies.values()]
        end_dates = [s.portfolio_df.index[-1] for s in self.strategies.values()]

        common_start = max(start_dates)
        common_end = min(end_dates)

        # Validate date range
        if common_start >= common_end:
            # Print detailed info about each strategy's date range
            print("\n  ERROR: No overlapping date range found!")
            print("\n  Strategy date ranges:")
            for ticker, strategy in self.strategies.items():
                start = strategy.portfolio_df.index[0].date()
                end = strategy.portfolio_df.index[-1].date()
                print(f"    {ticker:10s}: {start} to {end}")

            raise ValueError(
                f"No overlapping date range across all strategies. "
                f"Latest start date ({common_start.date()}) is after "
                f"earliest end date ({common_end.date()}). "
                f"Please ensure all strategies have overlapping time periods."
            )

        print(f"  Common period: {common_start.date()} to {common_end.date()}")

        # Align all strategies to common dates
        aligned_strategies = {}
        for ticker, strategy in self.strategies.items():
            df = strategy.portfolio_df.loc[common_start:common_end].copy()

            # Validate that we have data
            if len(df) == 0:
                raise ValueError(
                    f"Strategy {ticker} has no data in the common period "
                    f"({common_start.date()} to {common_end.date()})"
                )

            aligned_strategies[ticker] = df

        # Get common index
        common_index = aligned_strategies[self.tickers[0]].index

        # Create portfolio DataFrame
        portfolio_df = pd.DataFrame(index=common_index)

        # Calculate weighted portfolio value
        if self.rebalance:
            # Daily rebalancing to target weights
            portfolio_df["portfolio_value"] = 0

            for ticker, weight in self.weights.items():
                strategy_returns = aligned_strategies[ticker]["strategy_return"]
                # Each strategy maintains its weight
                weighted_value = (
                    self.initial_capital * weight * (1 + strategy_returns).cumprod()
                )
                portfolio_df["portfolio_value"] += weighted_value
        else:
            # Buy and hold weights (drift with performance)
            portfolio_df["portfolio_value"] = 0

            for ticker, weight in self.weights.items():
                weighted_value = (
                    aligned_strategies[ticker]["portfolio_value"]
                    / aligned_strategies[ticker]["portfolio_value"].iloc[0]
                    * (self.initial_capital * weight)
                )
                portfolio_df["portfolio_value"] += weighted_value

        # Calculate portfolio returns
        portfolio_df["portfolio_return"] = portfolio_df["portfolio_value"].pct_change(
            fill_method=None
        )
        # Alias for metrics compatibility (PerformanceMetrics expects 'strategy_return')
        portfolio_df["strategy_return"] = portfolio_df["portfolio_return"]

        # Calculate drawdown
        portfolio_df["cum_max"] = portfolio_df["portfolio_value"].cummax()
        portfolio_df["drawdown"] = (
            (portfolio_df["portfolio_value"] - portfolio_df["cum_max"])
            / portfolio_df["cum_max"]
            * 100
        )

        # Add individual strategy values for reference
        for ticker in self.tickers:
            portfolio_df[f"{ticker}_value"] = aligned_strategies[ticker][
                "portfolio_value"
            ]
            portfolio_df[f"{ticker}_return"] = aligned_strategies[ticker][
                "strategy_return"
            ]

        # Calculate buy & hold benchmark (equal weight across all tickers)
        portfolio_df["bnh_value"] = 0
        for ticker in self.tickers:
            bnh_weight = 1.0 / len(self.tickers)
            weighted_bnh = (
                aligned_strategies[ticker]["bnh_value"]
                / aligned_strategies[ticker]["bnh_value"].iloc[0]
                * (self.initial_capital * bnh_weight)
            )
            portfolio_df["bnh_value"] += weighted_bnh

        return portfolio_df

    def get_portfolio_metrics(self) -> Dict:
        """
        Calculate portfolio-level metrics.

        Returns:
            Dictionary with portfolio metrics
        """

        # Create a temporary analyzer-like object for metrics calculation
        class PortfolioMetricsCalculator:
            def __init__(self, portfolio_df, initial_capital, risk_free_rate):
                self.portfolio_df = portfolio_df
                self.initial_capital = initial_capital
                self.risk_free_rate = risk_free_rate

            def get_trade_log(self):
                """Return empty trade log for portfolio (trades are per-strategy)."""
                return pd.DataFrame(
                    columns=[
                        "entry_date",
                        "entry_price",
                        "exit_date",
                        "exit_price",
                        "pnl_pct",
                        "win",
                    ]
                )

        calc = PortfolioMetricsCalculator(
            self.portfolio_df, self.initial_capital, self.risk_free_rate
        )
        metrics = PerformanceMetrics(calc)

        portfolio_metrics = metrics.calculate_all_metrics()

        # Add portfolio-specific info
        portfolio_metrics["num_strategies"] = len(self.tickers)
        portfolio_metrics["strategy_names"] = ", ".join(self.tickers)
        portfolio_metrics["rebalancing"] = "Daily" if self.rebalance else "Buy & Hold"

        return portfolio_metrics

    def get_individual_metrics(self) -> pd.DataFrame:
        """
        Get metrics for each individual strategy.

        Returns:
            DataFrame with metrics for each strategy
        """
        individual_results = []

        for ticker, strategy in self.strategies.items():
            metrics = PerformanceMetrics(strategy)
            all_metrics = metrics.calculate_all_metrics()

            individual_results.append(
                {
                    "Ticker": ticker,
                    "Weight": self.weights[ticker] * 100,
                    "Annual Return (%)": all_metrics["annualized_return"],
                    "Sharpe Ratio": all_metrics["annualized_sharpe_ratio"],
                    "Max Drawdown (%)": all_metrics["maximum_drawdown"],
                    "Win Rate (%)": all_metrics["win_rate"],
                    "Calmar Ratio": all_metrics["calmar_ratio"],
                    "Volatility (%)": all_metrics["volatility"],
                    "Total Trades": all_metrics["total_trades"],
                }
            )

        return pd.DataFrame(individual_results)

    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Calculate correlation matrix of strategy returns.

        Returns:
            Correlation matrix DataFrame
        """
        returns_df = pd.DataFrame()

        for ticker in self.tickers:
            returns_df[ticker] = self.portfolio_df[f"{ticker}_return"]

        return returns_df.corr()

    def get_contribution_analysis(self) -> pd.DataFrame:
        """
        Analyze contribution of each strategy to portfolio performance.

        Returns:
            DataFrame with contribution analysis
        """
        contributions = []

        total_return = (
            self.portfolio_df["portfolio_value"].iloc[-1] / self.initial_capital - 1
        ) * 100

        for ticker in self.tickers:
            strategy_val_start = self.portfolio_df[f"{ticker}_value"].iloc[0]
            strategy_val_end = self.portfolio_df[f"{ticker}_value"].iloc[-1]

            strategy_return = (strategy_val_end / strategy_val_start - 1) * 100
            weight = self.weights[ticker]
            contribution = strategy_return * weight

            contributions.append(
                {
                    "Ticker": ticker,
                    "Weight (%)": weight * 100,
                    "Strategy Return (%)": strategy_return,
                    "Contribution to Portfolio (%)": contribution,
                    "Contribution Share (%)": (
                        (contribution / total_return * 100) if total_return != 0 else 0
                    ),
                }
            )

        return pd.DataFrame(contributions)

    def get_summary(self) -> str:
        """
        Get formatted summary of portfolio analysis.

        Returns:
            Formatted string summary
        """
        portfolio_metrics = self.get_portfolio_metrics()
        individual_df = self.get_individual_metrics()
        contribution_df = self.get_contribution_analysis()

        summary = f"""
{'='*80}
PORTFOLIO ANALYSIS SUMMARY
{'='*80}

Portfolio Composition:
  Number of Strategies:      {len(self.tickers)}
  Strategies:                {', '.join(self.tickers)}
  Rebalancing:               {portfolio_metrics['rebalancing']}
  Initial Capital:           ${self.initial_capital:,.2f}

{'='*80}
PORTFOLIO PERFORMANCE
{'='*80}

Risk-Adjusted Returns:
  Annualized Return:         {portfolio_metrics['annualized_return']:>12.2f}%
  Annualized Sharpe Ratio:   {portfolio_metrics['annualized_sharpe_ratio']:>12.2f}
  Annualized Sortino Ratio:  {portfolio_metrics['sortino_ratio']:>12.2f}
  
Risk Metrics:
  Maximum Drawdown:          {portfolio_metrics['maximum_drawdown']:>12.2f}%
  Annualized Volatility:     {portfolio_metrics['volatility']:>12.2f}%
  Calmar Ratio:              {portfolio_metrics['calmar_ratio']:>12.2f}

Return Metrics:
  Total Return:              {portfolio_metrics['total_return']:>12.2f}%
  Total Trades (all):        {portfolio_metrics['total_trades']:>12.0f}
  Win Rate (all):            {portfolio_metrics['win_rate']:>12.2f}%
  Profit Factor (all):       {portfolio_metrics['profit_factor']:>12.2f}

{'='*80}
INDIVIDUAL STRATEGY PERFORMANCE
{'='*80}

{individual_df.to_string(index=False)}

{'='*80}
STRATEGY CONTRIBUTION ANALYSIS
{'='*80}

{contribution_df.to_string(index=False)}

{'='*80}
CORRELATION MATRIX
{'='*80}

{self.get_correlation_matrix().to_string()}

{'='*80}
"""

        return summary

    def save_results(self, output_dir: str):
        """
        Save portfolio analysis results to files.

        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save portfolio metrics
        portfolio_metrics = self.get_portfolio_metrics()
        metrics_file = output_path / "portfolio_metrics.json"

        import json

        with open(metrics_file, "w") as f:
            json.dump(portfolio_metrics, f, indent=2, default=str)
        print(f"  Saved: {metrics_file.name}")

        # Save individual strategy metrics
        individual_df = self.get_individual_metrics()
        individual_file = output_path / "individual_strategies.csv"
        individual_df.to_csv(individual_file, index=False)
        print(f"  Saved: {individual_file.name}")

        # Save contribution analysis
        contribution_df = self.get_contribution_analysis()
        contribution_file = output_path / "contribution_analysis.csv"
        contribution_df.to_csv(contribution_file, index=False)
        print(f"  Saved: {contribution_file.name}")

        # Save correlation matrix
        corr_matrix = self.get_correlation_matrix()
        corr_file = output_path / "correlation_matrix.csv"
        corr_matrix.to_csv(corr_file)
        print(f"  Saved: {corr_file.name}")

        # Save portfolio history
        history_file = output_path / "portfolio_history.csv"
        self.portfolio_df.to_csv(history_file)
        print(f"  Saved: {history_file.name}")

        # Save summary report
        summary_file = output_path / "portfolio_summary.txt"
        with open(summary_file, "w") as f:
            f.write(self.get_summary())
        print(f"  Saved: {summary_file.name}")

        print(f"\n✓ All results saved to {output_dir}")
