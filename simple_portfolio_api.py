"""
Simple API for portfolio analysis.

Provides easy-to-use functions for combining multiple strategies into portfolios.
"""

from analysis import PortfolioAnalyzer, PortfolioVisualizer
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional


def create_equal_weight_portfolio(
    csv_files: List[str], initial_capital: float = 10000.0, rebalance: bool = False
) -> PortfolioAnalyzer:
    """
    Create a portfolio with equal weights across all strategies.

    Args:
        csv_files: List of CSV file paths with trading decisions
        initial_capital: Total starting capital
        rebalance: Whether to rebalance daily to target weights

    Returns:
        PortfolioAnalyzer instance

    Example:
        >>> portfolio = create_equal_weight_portfolio([
        ...     'results_bulk/AAPL_decisions_*.csv',
        ...     'results_bulk/MSFT_decisions_*.csv'
        ... ])
        >>> print(f"Sharpe: {portfolio.get_portfolio_metrics()['annualized_sharpe_ratio']:.2f}")
    """
    # Extract tickers from filenames
    strategy_files = {}
    for csv_file in csv_files:
        ticker = Path(csv_file).stem.split("_")[0]
        strategy_files[ticker] = csv_file

    return PortfolioAnalyzer(
        strategy_files=strategy_files,
        weights=None,  # Equal weights
        initial_capital=initial_capital,
        rebalance=rebalance,
    )


def create_custom_weight_portfolio(
    strategy_weights: Dict[str, tuple],
    initial_capital: float = 10000.0,
    rebalance: bool = False,
) -> PortfolioAnalyzer:
    """
    Create a portfolio with custom weights.

    Args:
        strategy_weights: Dict mapping ticker to (csv_file, weight)
        initial_capital: Total starting capital
        rebalance: Whether to rebalance daily

    Returns:
        PortfolioAnalyzer instance

    Example:
        >>> weights = {
        ...     'AAPL': ('results_bulk/AAPL_*.csv', 0.6),
        ...     'MSFT': ('results_bulk/MSFT_*.csv', 0.4)
        ... }
        >>> portfolio = create_custom_weight_portfolio(weights)
    """
    strategy_files = {ticker: info[0] for ticker, info in strategy_weights.items()}
    weights = {ticker: info[1] for ticker, info in strategy_weights.items()}

    return PortfolioAnalyzer(
        strategy_files=strategy_files,
        weights=weights,
        initial_capital=initial_capital,
        rebalance=rebalance,
    )


def analyze_all_in_directory(
    directory: str = "results_bulk",
    initial_capital: float = 10000.0,
    rebalance: bool = False,
) -> PortfolioAnalyzer:
    """
    Create equal-weight portfolio from all CSV files in directory.

    Args:
        directory: Directory containing CSV files
        initial_capital: Total starting capital
        rebalance: Whether to rebalance daily

    Returns:
        PortfolioAnalyzer instance

    Example:
        >>> portfolio = analyze_all_in_directory('results_bulk')
        >>> print(portfolio.get_summary())
    """
    csv_files = list(Path(directory).glob("*_decisions_*.csv"))

    if len(csv_files) < 2:
        raise ValueError(
            f"Need at least 2 CSV files in {directory}, found {len(csv_files)}"
        )

    return create_equal_weight_portfolio(
        csv_files=[str(f) for f in csv_files],
        initial_capital=initial_capital,
        rebalance=rebalance,
    )


def get_portfolio_summary(portfolio: PortfolioAnalyzer) -> Dict:
    """
    Get comprehensive portfolio summary.

    Args:
        portfolio: PortfolioAnalyzer instance

    Returns:
        Dict with portfolio metrics, individual metrics, and contribution analysis

    Example:
        >>> summary = get_portfolio_summary(portfolio)
        >>> print(f"Portfolio Return: {summary['portfolio']['annualized_return']:.2f}%")
    """
    return {
        "portfolio": portfolio.get_portfolio_metrics(),
        "individual": portfolio.get_individual_metrics().to_dict("records"),
        "contribution": portfolio.get_contribution_analysis().to_dict("records"),
        "correlation": portfolio.get_correlation_matrix().to_dict(),
    }


def compare_portfolios(portfolios: Dict[str, PortfolioAnalyzer]) -> pd.DataFrame:
    """
    Compare multiple portfolio configurations.

    Args:
        portfolios: Dict mapping name to PortfolioAnalyzer instance

    Returns:
        DataFrame comparing all portfolios

    Example:
        >>> portfolios = {
        ...     'Equal Weight': equal_weight_portfolio,
        ...     'Custom Weight': custom_weight_portfolio
        ... }
        >>> comparison = compare_portfolios(portfolios)
        >>> print(comparison)
    """
    results = []

    for name, portfolio in portfolios.items():
        metrics = portfolio.get_portfolio_metrics()
        results.append(
            {
                "Portfolio": name,
                "Annual Return (%)": metrics["annualized_return"],
                "Sharpe Ratio": metrics["annualized_sharpe_ratio"],
                "Max Drawdown (%)": metrics["maximum_drawdown"],
                "Volatility (%)": metrics["volatility"],
                "Calmar Ratio": metrics["calmar_ratio"],
                "Num Strategies": metrics["num_strategies"],
            }
        )

    return pd.DataFrame(results)


def generate_portfolio_report(
    portfolio: PortfolioAnalyzer, output_dir: str, show_charts: bool = False
) -> Dict[str, str]:
    """
    Generate complete portfolio report with all files and charts.

    Args:
        portfolio: PortfolioAnalyzer instance
        output_dir: Directory to save report
        show_charts: Whether to display charts

    Returns:
        Dict mapping output type to file paths

    Example:
        >>> files = generate_portfolio_report(portfolio, 'reports/my_portfolio')
        >>> print(f"Summary saved to {files['summary']}")
    """
    # Save data files
    portfolio.save_results(output_dir)

    # Generate visualizations
    visualizer = PortfolioVisualizer(portfolio)
    chart_files = visualizer.create_all_charts(output_dir, show=show_charts)

    return {"data": output_dir, "charts": [str(f) for f in chart_files]}


def find_optimal_weights(
    strategy_files: Dict[str, str],
    initial_capital: float = 10000.0,
    weight_increments: float = 0.1,
    metric: str = "sharpe",
) -> Dict:
    """
    Find optimal weights by testing combinations.

    WARNING: Can be slow for many strategies (combinatorial explosion).

    Args:
        strategy_files: Dict mapping ticker to CSV file path
        initial_capital: Starting capital
        weight_increments: Weight step size (e.g., 0.1 = 10% increments)
        metric: Optimization metric ('sharpe', 'calmar', 'return')

    Returns:
        Dict with optimal weights and metrics

    Example:
        >>> optimal = find_optimal_weights(strategy_files, weight_increments=0.2)
        >>> print(f"Best Sharpe: {optimal['sharpe']:.2f}")
        >>> print(f"Optimal weights: {optimal['weights']}")
    """
    import itertools
    import numpy as np

    tickers = list(strategy_files.keys())
    n = len(tickers)

    if n > 5:
        print(
            f"Warning: Testing {n} strategies can be very slow. Consider using fewer strategies."
        )

    # Generate weight combinations
    weight_values = np.arange(0, 1 + weight_increments, weight_increments)

    best_metric_value = float("-inf")
    best_weights = None
    best_metrics = None

    tested = 0
    valid = 0

    print(f"Testing weight combinations (increments of {weight_increments})...")

    for weight_combo in itertools.product(weight_values, repeat=n):
        tested += 1

        if not np.isclose(sum(weight_combo), 1.0):
            continue

        valid += 1
        weights = dict(zip(tickers, weight_combo))

        try:
            portfolio = PortfolioAnalyzer(
                strategy_files=strategy_files,
                weights=weights,
                initial_capital=initial_capital,
                rebalance=False,
            )

            metrics = portfolio.get_portfolio_metrics()

            # Select optimization metric
            if metric == "sharpe":
                metric_value = metrics["annualized_sharpe_ratio"]
            elif metric == "calmar":
                metric_value = metrics["calmar_ratio"]
            elif metric == "return":
                metric_value = metrics["annualized_return"]
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_weights = weights
                best_metrics = metrics

        except Exception as e:
            continue

    print(f"Tested {tested} combinations, {valid} were valid")

    return {"weights": best_weights, "metrics": best_metrics, metric: best_metric_value}


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            """
Simple Portfolio API Examples

1. Equal weight portfolio:
   python simple_portfolio_api.py equal

2. Custom weights:
   python simple_portfolio_api.py custom

3. Find optimal weights:
   python simple_portfolio_api.py optimize
        """
        )
        sys.exit(1)

    command = sys.argv[1]

    if command == "equal":
        # Example: Equal weight portfolio
        csv_files = list(Path("results_bulk").glob("*_decisions_*.csv"))[:3]

        portfolio = create_equal_weight_portfolio(
            csv_files=[str(f) for f in csv_files], initial_capital=10000.0
        )

        print(portfolio.get_summary())

        generate_portfolio_report(portfolio, "reports/portfolio_equal")

    elif command == "custom":
        # Example: Custom weight portfolio
        strategy_weights = {
            "AAPL": ("results_bulk/AAPL_decisions_2020-01-06_2024-12-23.csv", 0.5),
            "MSFT": ("results_bulk/MSFT_decisions_2020-01-01_2024-12-31.csv", 0.5),
        }

        portfolio = create_custom_weight_portfolio(
            strategy_weights=strategy_weights, initial_capital=10000.0
        )

        print(portfolio.get_summary())

        generate_portfolio_report(portfolio, "reports/portfolio_custom")

    elif command == "optimize":
        # Example: Find optimal weights
        strategy_files = {
            "AAPL": "results_bulk/AAPL_decisions_2020-01-06_2024-12-23.csv",
            "MSFT": "results_bulk/MSFT_decisions_2020-01-01_2024-12-31.csv",
            "NVDA": "results_bulk/NVDA_decisions_2020-01-01_2024-12-31.csv",
        }

        optimal = find_optimal_weights(
            strategy_files=strategy_files,
            weight_increments=0.2,  # 20% increments
            metric="sharpe",
        )

        print(f"\nOptimal Sharpe Ratio: {optimal['sharpe']:.2f}")
        print("\nOptimal Weights:")
        for ticker, weight in optimal["weights"].items():
            print(f"  {ticker}: {weight*100:.1f}%")

    else:
        print(f"Unknown command: {command}")
