"""
Visualization module for portfolio analysis.

Extends the PerformanceVisualizer to create portfolio-specific charts.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


class PortfolioVisualizer:
    """
    Create visualizations for portfolio analysis.
    """

    def __init__(self, portfolio_analyzer):
        """
        Initialize portfolio visualizer.

        Args:
            portfolio_analyzer: PortfolioAnalyzer instance
        """
        self.analyzer = portfolio_analyzer
        self.portfolio_df = portfolio_analyzer.portfolio_df
        self.tickers = portfolio_analyzer.tickers
        self.weights = portfolio_analyzer.weights

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (14, 8)
        plt.rcParams["font.size"] = 10

    def plot_portfolio_equity_curve(
        self, save_path: Optional[str] = None, show: bool = True
    ):
        """
        Plot portfolio equity curve with individual strategies.

        Args:
            save_path: Path to save figure
            show: Whether to display the plot
        """
        fig, ax = plt.subplots(figsize=(16, 8))

        # Plot portfolio
        ax.plot(
            self.portfolio_df.index,
            self.portfolio_df["portfolio_value"],
            label="Portfolio",
            linewidth=3,
            color="#2E86AB",
            zorder=10,
        )

        # Plot benchmark
        ax.plot(
            self.portfolio_df.index,
            self.portfolio_df["bnh_value"],
            label="Buy & Hold (Equal Weight)",
            linewidth=2,
            color="#A23B72",
            alpha=0.7,
            linestyle="--",
            zorder=5,
        )

        # Plot individual strategies
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.tickers)))
        for i, ticker in enumerate(self.tickers):
            ax.plot(
                self.portfolio_df.index,
                self.portfolio_df[f"{ticker}_value"],
                label=f"{ticker} ({self.weights[ticker]*100:.1f}%)",
                linewidth=1.5,
                alpha=0.6,
                color=colors[i],
            )

        # Formatting
        ax.set_title(
            "Portfolio Equity Curve - All Strategies",
            fontsize=18,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Date", fontsize=14, fontweight="bold")
        ax.set_ylabel("Portfolio Value ($)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11, loc="upper left", framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved portfolio equity curve to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_portfolio_weights_evolution(
        self, save_path: Optional[str] = None, show: bool = True
    ):
        """
        Plot how portfolio weights evolve over time (if not rebalancing).

        Args:
            save_path: Path to save figure
            show: Whether to display the plot
        """
        fig, ax = plt.subplots(figsize=(16, 8))

        # Calculate actual weights over time
        weights_over_time = pd.DataFrame(index=self.portfolio_df.index)

        for ticker in self.tickers:
            weights_over_time[ticker] = (
                self.portfolio_df[f"{ticker}_value"]
                / self.portfolio_df["portfolio_value"]
                * 100
            )

        # Create stacked area chart
        ax.stackplot(
            weights_over_time.index,
            [weights_over_time[ticker] for ticker in self.tickers],
            labels=self.tickers,
            alpha=0.7,
        )

        # Add target weight lines
        cumulative = 0
        for ticker in self.tickers:
            target_weight = self.weights[ticker] * 100
            cumulative += target_weight
            ax.axhline(
                y=cumulative, color="red", linestyle="--", alpha=0.5, linewidth=1
            )

        # Formatting
        ax.set_title(
            "Portfolio Weight Evolution Over Time",
            fontsize=18,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Date", fontsize=14, fontweight="bold")
        ax.set_ylabel("Weight (%)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11, loc="upper left", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved weights evolution to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_correlation_matrix(
        self, save_path: Optional[str] = None, show: bool = True
    ):
        """
        Plot correlation matrix heatmap.

        Args:
            save_path: Path to save figure
            show: Whether to display the plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        corr_matrix = self.analyzer.get_correlation_matrix()

        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=1,
            ax=ax,
            cbar_kws={"label": "Correlation"},
        )

        ax.set_title(
            "Strategy Return Correlation Matrix", fontsize=16, fontweight="bold", pad=20
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved correlation matrix to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_contribution_analysis(
        self, save_path: Optional[str] = None, show: bool = True
    ):
        """
        Plot contribution of each strategy to portfolio performance.

        Args:
            save_path: Path to save figure
            show: Whether to display the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        contribution_df = self.analyzer.get_contribution_analysis()

        # Bar chart of contributions
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.tickers)))
        bars = ax1.bar(
            contribution_df["Ticker"],
            contribution_df["Contribution to Portfolio (%)"],
            color=colors,
            alpha=0.7,
            edgecolor="black",
        )

        ax1.set_title(
            "Strategy Contribution to Portfolio Return", fontsize=14, fontweight="bold"
        )
        ax1.set_xlabel("Strategy", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Contribution (%)", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3, axis="y")
        ax1.axhline(y=0, color="red", linestyle="--", linewidth=1)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}%",
                ha="center",
                va="bottom" if height > 0 else "top",
                fontsize=10,
                fontweight="bold",
            )

        # Pie chart of weights vs contribution share
        x = np.arange(len(contribution_df))
        width = 0.35

        ax2.bar(
            x - width / 2,
            contribution_df["Weight (%)"],
            width,
            label="Target Weight",
            color="#2E86AB",
            alpha=0.7,
        )
        ax2.bar(
            x + width / 2,
            contribution_df["Contribution Share (%)"],
            width,
            label="Actual Contribution",
            color="#A23B72",
            alpha=0.7,
        )

        ax2.set_title(
            "Target Weight vs Actual Contribution", fontsize=14, fontweight="bold"
        )
        ax2.set_xlabel("Strategy", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Percentage (%)", fontsize=12, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(contribution_df["Ticker"])
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, axis="y")

        plt.suptitle(
            "Portfolio Contribution Analysis", fontsize=18, fontweight="bold", y=1.02
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved contribution analysis to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_drawdown_comparison(
        self, save_path: Optional[str] = None, show: bool = True
    ):
        """
        Compare drawdowns of portfolio vs individual strategies.

        Args:
            save_path: Path to save figure
            show: Whether to display the plot
        """
        fig, ax = plt.subplots(figsize=(16, 8))

        # Plot portfolio drawdown
        ax.fill_between(
            self.portfolio_df.index,
            self.portfolio_df["drawdown"],
            0,
            color="#2E86AB",
            alpha=0.6,
            label="Portfolio",
        )
        ax.plot(
            self.portfolio_df.index,
            self.portfolio_df["drawdown"],
            color="#2E86AB",
            linewidth=2,
        )

        # Plot individual strategy drawdowns
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.tickers)))
        for i, ticker in enumerate(self.tickers):
            strategy = self.analyzer.strategies[ticker]
            # Align to common dates
            aligned_dd = strategy.portfolio_df.loc[self.portfolio_df.index, "drawdown"]
            ax.plot(
                self.portfolio_df.index,
                aligned_dd,
                label=ticker,
                linewidth=1.5,
                alpha=0.7,
                color=colors[i],
                linestyle="--",
            )

        # Formatting
        ax.set_title(
            "Drawdown Comparison - Portfolio vs Individual Strategies",
            fontsize=18,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Date", fontsize=14, fontweight="bold")
        ax.set_ylabel("Drawdown (%)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11, loc="lower left", framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved drawdown comparison to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_strategy_performance_comparison(
        self, save_path: Optional[str] = None, show: bool = True
    ):
        """
        Create a comprehensive comparison chart of all strategies.

        Args:
            save_path: Path to save figure
            show: Whether to display the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

        individual_df = self.analyzer.get_individual_metrics()

        # 1. Returns comparison
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.tickers) + 1))

        portfolio_metrics = self.analyzer.get_portfolio_metrics()
        all_returns = list(individual_df["Annual Return (%)"]) + [
            portfolio_metrics["annualized_return"]
        ]
        all_labels = list(individual_df["Ticker"]) + ["Portfolio"]

        bars1 = ax1.bar(
            all_labels, all_returns, color=colors, alpha=0.7, edgecolor="black"
        )
        ax1.set_title("Annualized Returns Comparison", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Return (%)", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3, axis="y")
        ax1.axhline(y=0, color="red", linestyle="--", linewidth=1)

        # 2. Sharpe ratio comparison
        all_sharpe = list(individual_df["Sharpe Ratio"]) + [
            portfolio_metrics["annualized_sharpe_ratio"]
        ]
        bars2 = ax2.bar(
            all_labels, all_sharpe, color=colors, alpha=0.7, edgecolor="black"
        )
        ax2.set_title("Sharpe Ratio Comparison", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Sharpe Ratio", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.axhline(
            y=1.0, color="orange", linestyle="--", linewidth=1, label="Good (>1.0)"
        )
        ax2.axhline(
            y=2.0, color="green", linestyle="--", linewidth=1, label="Excellent (>2.0)"
        )
        ax2.legend(fontsize=9)

        # 3. Max drawdown comparison
        all_dd = list(individual_df["Max Drawdown (%)"]) + [
            portfolio_metrics["maximum_drawdown"]
        ]
        bars3 = ax3.bar(all_labels, all_dd, color=colors, alpha=0.7, edgecolor="black")
        ax3.set_title("Maximum Drawdown Comparison", fontsize=14, fontweight="bold")
        ax3.set_ylabel("Max Drawdown (%)", fontsize=12, fontweight="bold")
        ax3.grid(True, alpha=0.3, axis="y")

        # 4. Risk-return scatter
        ax4.scatter(
            individual_df["Volatility (%)"],
            individual_df["Annual Return (%)"],
            s=200,
            alpha=0.7,
            c=colors[:-1],
            edgecolors="black",
            linewidth=2,
        )
        ax4.scatter(
            [portfolio_metrics["volatility"]],
            [portfolio_metrics["annualized_return"]],
            s=400,
            marker="*",
            c=[colors[-1]],
            edgecolors="black",
            linewidth=2,
            label="Portfolio",
            zorder=10,
        )

        for idx, row in individual_df.iterrows():
            ax4.annotate(
                row["Ticker"],
                (row["Volatility (%)"], row["Annual Return (%)"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
            )

        ax4.annotate(
            "Portfolio",
            (portfolio_metrics["volatility"], portfolio_metrics["annualized_return"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
        )

        ax4.set_title("Risk-Return Profile", fontsize=14, fontweight="bold")
        ax4.set_xlabel("Volatility (%)", fontsize=12, fontweight="bold")
        ax4.set_ylabel("Return (%)", fontsize=12, fontweight="bold")
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=11)

        plt.suptitle(
            "Strategy Performance Comparison", fontsize=18, fontweight="bold", y=0.995
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved performance comparison to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def create_all_charts(self, output_dir: str, show: bool = False):
        """
        Create all portfolio charts.

        Args:
            output_dir: Directory to save charts
            show: Whether to display plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("Generating portfolio visualizations...")

        charts = [
            (
                "equity_curve",
                lambda: self.plot_portfolio_equity_curve(
                    save_path=output_path / "portfolio_equity_curve.png", show=show
                ),
            ),
            (
                "weights_evolution",
                lambda: self.plot_portfolio_weights_evolution(
                    save_path=output_path / "portfolio_weights_evolution.png", show=show
                ),
            ),
            (
                "correlation_matrix",
                lambda: self.plot_correlation_matrix(
                    save_path=output_path / "portfolio_correlation_matrix.png",
                    show=show,
                ),
            ),
            (
                "contribution_analysis",
                lambda: self.plot_contribution_analysis(
                    save_path=output_path / "portfolio_contribution_analysis.png",
                    show=show,
                ),
            ),
            (
                "drawdown_comparison",
                lambda: self.plot_drawdown_comparison(
                    save_path=output_path / "portfolio_drawdown_comparison.png",
                    show=show,
                ),
            ),
            (
                "performance_comparison",
                lambda: self.plot_strategy_performance_comparison(
                    save_path=output_path / "portfolio_performance_comparison.png",
                    show=show,
                ),
            ),
        ]

        saved_files = []
        for chart_name, chart_func in charts:
            try:
                chart_func()
                saved_files.append(output_path / f"portfolio_{chart_name}.png")
            except Exception as e:
                print(f"Error creating {chart_name}: {e}")

        print(f"\n✓ Saved {len(saved_files)} charts to {output_dir}")
        return saved_files
