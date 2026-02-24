"""
Visualization module for trading strategy performance.

This module provides the PerformanceVisualizer class for creating
charts and graphs for strategy reports.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List


class PerformanceVisualizer:
    """
    Create visualizations for trading strategy performance.

    Generates:
    - Equity curve
    - Drawdown chart
    - Monthly returns heatmap
    - Trade distribution
    - Returns distribution
    """

    def __init__(self, analyzer, metrics):
        """
        Initialize visualizer.

        Args:
            analyzer: TradingAnalyzer instance
            metrics: PerformanceMetrics instance
        """
        self.analyzer = analyzer
        self.metrics = metrics
        self.portfolio_df = analyzer.portfolio_df

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 6)
        plt.rcParams["font.size"] = 10

    def plot_equity_curve(self, save_path: Optional[str] = None, show: bool = True):
        """
        Plot equity curve comparing strategy to buy & hold.

        Args:
            save_path: Path to save figure (optional)
            show: Whether to display the plot
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot strategy and benchmark
        ax.plot(
            self.portfolio_df.index,
            self.portfolio_df["portfolio_value"],
            label="Strategy",
            linewidth=2,
            color="#2E86AB",
        )
        ax.plot(
            self.portfolio_df.index,
            self.portfolio_df["bnh_value"],
            label="Buy & Hold",
            linewidth=2,
            color="#A23B72",
            alpha=0.7,
        )

        # Formatting
        ax.set_title(
            f"{self.analyzer.ticker} - Strategy Equity Curve",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Date", fontsize=12, fontweight="bold")
        ax.set_ylabel("Portfolio Value ($)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=11, loc="upper left")
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45, ha="right")

        # Add metrics text box
        metrics_text = f"""
        Annualized Return: {self.metrics.annualized_return():.2f}%
        Sharpe Ratio: {self.metrics.annualized_sharpe_ratio():.2f}
        Max Drawdown: {self.metrics.maximum_drawdown():.2f}%
        Win Rate: {self.metrics.win_rate():.2f}%
        """

        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        ax.text(
            0.02,
            0.98,
            metrics_text.strip(),
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved equity curve to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_drawdown(self, save_path: Optional[str] = None, show: bool = True):
        """
        Plot drawdown over time.

        Args:
            save_path: Path to save figure (optional)
            show: Whether to display the plot
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot drawdown
        ax.fill_between(
            self.portfolio_df.index,
            self.portfolio_df["drawdown"],
            0,
            color="#C1403D",
            alpha=0.6,
        )
        ax.plot(
            self.portfolio_df.index,
            self.portfolio_df["drawdown"],
            color="#8B0000",
            linewidth=1.5,
        )

        # Formatting
        ax.set_title(
            f"{self.analyzer.ticker} - Drawdown Over Time",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Date", fontsize=12, fontweight="bold")
        ax.set_ylabel("Drawdown (%)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45, ha="right")

        # Add max drawdown line
        max_dd = self.metrics.maximum_drawdown()
        ax.axhline(
            y=max_dd,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Max Drawdown: {max_dd:.2f}%",
        )
        ax.legend(fontsize=11)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved drawdown chart to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_monthly_returns(self, save_path: Optional[str] = None, show: bool = True):
        """
        Plot monthly returns heatmap.

        Args:
            save_path: Path to save figure (optional)
            show: Whether to display the plot
        """
        # Calculate monthly returns
        monthly_df = self.portfolio_df.copy()
        monthly_df["month"] = monthly_df.index.to_period("M")

        # Get first and last value of each month
        monthly_returns = monthly_df.groupby("month").agg(
            {"portfolio_value": ["first", "last"]}
        )
        monthly_returns.columns = ["first", "last"]
        monthly_returns["return_pct"] = (
            (monthly_returns["last"] - monthly_returns["first"])
            / monthly_returns["first"]
            * 100
        )

        # Convert to year/month format
        monthly_returns.index = monthly_returns.index.to_timestamp()
        monthly_returns["year"] = monthly_returns.index.year
        monthly_returns["month"] = monthly_returns.index.month

        # Pivot for heatmap
        heatmap_data = monthly_returns.pivot(
            index="year", columns="month", values="return_pct"
        )

        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 8))

        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            center=0,
            cbar_kws={"label": "Return (%)"},
            linewidths=0.5,
            ax=ax,
        )

        # Formatting
        ax.set_title(
            f"{self.analyzer.ticker} - Monthly Returns Heatmap (%)",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Month", fontsize=12, fontweight="bold")
        ax.set_ylabel("Year", fontsize=12, fontweight="bold")

        # Set month labels
        month_labels = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        ax.set_xticklabels(month_labels)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved monthly returns heatmap to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_trade_distribution(
        self, save_path: Optional[str] = None, show: bool = True
    ):
        """
        Plot distribution of trade returns.

        Args:
            save_path: Path to save figure (optional)
            show: Whether to display the plot
        """
        trade_log = self.analyzer.get_trade_log()

        if len(trade_log) == 0:
            print("No trades to plot")
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Histogram
        ax1.hist(
            trade_log["pnl_pct"], bins=30, color="#2E86AB", alpha=0.7, edgecolor="black"
        )
        ax1.axvline(x=0, color="red", linestyle="--", linewidth=2)
        ax1.axvline(
            x=trade_log["pnl_pct"].mean(),
            color="green",
            linestyle="--",
            linewidth=2,
            label="Mean",
        )
        ax1.set_title("Distribution of Trade Returns", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Return (%)", fontsize=12)
        ax1.set_ylabel("Frequency", fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Win/Loss breakdown
        wins = trade_log[trade_log["win"]]["pnl_pct"]
        losses = trade_log[~trade_log["win"]]["pnl_pct"]

        ax2.boxplot(
            [wins, losses],
            labels=["Wins", "Losses"],
            patch_artist=True,
            boxprops=dict(facecolor="#2E86AB", alpha=0.7),
            medianprops=dict(color="red", linewidth=2),
        )
        ax2.set_title("Win/Loss Distribution", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Return (%)", fontsize=12)
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.axhline(y=0, color="red", linestyle="--", linewidth=1)

        plt.suptitle(
            f"{self.analyzer.ticker} - Trade Analysis",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved trade distribution to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_returns_distribution(
        self, save_path: Optional[str] = None, show: bool = True
    ):
        """
        Plot distribution of daily returns.

        Args:
            save_path: Path to save figure (optional)
            show: Whether to display the plot
        """
        returns = self.portfolio_df["strategy_return"].dropna() * 100  # Convert to %

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Histogram
        ax1.hist(returns, bins=50, color="#2E86AB", alpha=0.7, edgecolor="black")
        ax1.axvline(
            x=returns.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {returns.mean():.3f}%",
        )
        ax1.set_title("Daily Returns Distribution", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Return (%)", fontsize=12)
        ax1.set_ylabel("Frequency", fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Q-Q plot
        from scipy import stats

        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title("Q-Q Plot (Normality Check)", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        plt.suptitle(
            f"{self.analyzer.ticker} - Returns Analysis",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved returns distribution to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_rolling_metrics(
        self, window: int = 252, save_path: Optional[str] = None, show: bool = True
    ):
        """
        Plot rolling Sharpe ratio and volatility.

        Args:
            window: Rolling window in days (default: 252 = 1 year)
            save_path: Path to save figure (optional)
            show: Whether to display the plot
        """
        returns = self.portfolio_df["strategy_return"]

        # Calculate rolling metrics
        rolling_sharpe = (
            returns.rolling(window).mean()
            / returns.rolling(window).std()
            * np.sqrt(252)
        )
        rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Rolling Sharpe
        ax1.plot(self.portfolio_df.index, rolling_sharpe, color="#2E86AB", linewidth=2)
        ax1.axhline(
            y=1.0,
            color="orange",
            linestyle="--",
            label="Sharpe = 1.0 (Good)",
            alpha=0.7,
        )
        ax1.axhline(
            y=2.0,
            color="green",
            linestyle="--",
            label="Sharpe = 2.0 (Excellent)",
            alpha=0.7,
        )
        ax1.set_title(
            f"Rolling Sharpe Ratio ({window}-day window)",
            fontsize=14,
            fontweight="bold",
        )
        ax1.set_ylabel("Sharpe Ratio", fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Rolling Volatility
        ax2.plot(self.portfolio_df.index, rolling_vol, color="#C1403D", linewidth=2)
        ax2.set_title(
            f"Rolling Volatility ({window}-day window)", fontsize=14, fontweight="bold"
        )
        ax2.set_xlabel("Date", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Annualized Volatility (%)", fontsize=12)
        ax2.grid(True, alpha=0.3)

        # Format x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

        plt.xticks(rotation=45, ha="right")
        plt.suptitle(
            f"{self.analyzer.ticker} - Rolling Risk Metrics",
            fontsize=16,
            fontweight="bold",
            y=1.00,
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved rolling metrics to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def create_all_charts(self, output_dir: str, show: bool = False):
        """
        Create all charts and save to directory.

        Args:
            output_dir: Directory to save charts
            show: Whether to display plots

        Returns:
            List of saved file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        ticker = self.analyzer.ticker
        saved_files = []

        # Generate all charts
        charts = [
            (
                "equity_curve",
                lambda: self.plot_equity_curve(
                    save_path=output_path / f"{ticker}_equity_curve.png", show=show
                ),
            ),
            (
                "drawdown",
                lambda: self.plot_drawdown(
                    save_path=output_path / f"{ticker}_drawdown.png", show=show
                ),
            ),
            (
                "monthly_returns",
                lambda: self.plot_monthly_returns(
                    save_path=output_path / f"{ticker}_monthly_returns.png", show=show
                ),
            ),
            (
                "trade_distribution",
                lambda: self.plot_trade_distribution(
                    save_path=output_path / f"{ticker}_trade_distribution.png",
                    show=show,
                ),
            ),
            (
                "returns_distribution",
                lambda: self.plot_returns_distribution(
                    save_path=output_path / f"{ticker}_returns_distribution.png",
                    show=show,
                ),
            ),
            (
                "rolling_metrics",
                lambda: self.plot_rolling_metrics(
                    save_path=output_path / f"{ticker}_rolling_metrics.png", show=show
                ),
            ),
        ]

        for chart_name, chart_func in charts:
            try:
                chart_func()
                saved_files.append(output_path / f"{ticker}_{chart_name}.png")
            except Exception as e:
                print(f"Error creating {chart_name}: {e}")

        print(f"\nSaved {len(saved_files)} charts to {output_dir}")
        return saved_files
