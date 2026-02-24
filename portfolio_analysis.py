#!/usr/bin/env python3
"""
Portfolio analysis script for combining multiple trading strategies.

This script allows you to combine multiple trading decision files into a
single portfolio with equal or custom weights, and analyze the combined performance.
"""

from analysis.portfolio import PortfolioAnalyzer
from analysis.portfolio_visualizer import PortfolioVisualizer
from pathlib import Path
import json
import sys
from typing import Dict, Optional


def analyze_equal_weight_portfolio(
    csv_files: list,
    output_dir: str = "reports/portfolio_equal_weight",
    initial_capital: float = 10000.0,
    rebalance: bool = False,
    show_charts: bool = False,
):
    """
    Analyze portfolio with equal weights across all strategies.

    Args:
        csv_files: List of CSV file paths
        output_dir: Directory to save results
        initial_capital: Total starting capital
        rebalance: Whether to rebalance daily
        show_charts: Whether to display charts

    Returns:
        PortfolioAnalyzer instance
    """
    # Extract ticker from filename
    strategy_files = {}
    for csv_file in csv_files:
        ticker = Path(csv_file).stem.split("_")[0]
        strategy_files[ticker] = csv_file

    print(f"\n{'='*80}")
    print(f"EQUAL WEIGHT PORTFOLIO ANALYSIS")
    print(f"{'='*80}\n")
    print(f"Strategies: {', '.join(strategy_files.keys())}")
    print(f"Equal weight: {100/len(strategy_files):.2f}% each")
    print(f"Rebalancing: {'Daily' if rebalance else 'Buy & Hold'}")
    print(f"Initial capital: ${initial_capital:,.2f}")
    print()

    # Create portfolio analyzer
    portfolio = PortfolioAnalyzer(
        strategy_files=strategy_files,
        weights=None,  # Equal weights
        initial_capital=initial_capital,
        rebalance=rebalance,
    )

    # Print summary
    print(portfolio.get_summary())

    # Save results
    portfolio.save_results(output_dir)

    # Generate visualizations
    visualizer = PortfolioVisualizer(portfolio)
    visualizer.create_all_charts(output_dir, show=show_charts)

    return portfolio


def analyze_custom_weight_portfolio(
    strategy_weights: Dict[str, tuple],
    output_dir: str = "reports/portfolio_custom_weight",
    initial_capital: float = 10000.0,
    rebalance: bool = False,
    show_charts: bool = False,
):
    """
    Analyze portfolio with custom weights.

    Args:
        strategy_weights: Dict mapping ticker to (csv_file, weight)
                         e.g., {'AAPL': ('path/to/aapl.csv', 0.4), 'MSFT': ('path/to/msft.csv', 0.6)}
        output_dir: Directory to save results
        initial_capital: Total starting capital
        rebalance: Whether to rebalance daily
        show_charts: Whether to display charts

    Returns:
        PortfolioAnalyzer instance
    """
    strategy_files = {ticker: info[0] for ticker, info in strategy_weights.items()}
    weights = {ticker: info[1] for ticker, info in strategy_weights.items()}

    # Validate weights sum to 1
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 0.001:
        raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")

    print(f"\n{'='*80}")
    print(f"CUSTOM WEIGHT PORTFOLIO ANALYSIS")
    print(f"{'='*80}\n")
    print("Strategies and weights:")
    for ticker, weight in weights.items():
        print(f"  {ticker}: {weight*100:.1f}%")
    print(f"\nRebalancing: {'Daily' if rebalance else 'Buy & Hold'}")
    print(f"Initial capital: ${initial_capital:,.2f}")
    print()

    # Create portfolio analyzer
    portfolio = PortfolioAnalyzer(
        strategy_files=strategy_files,
        weights=weights,
        initial_capital=initial_capital,
        rebalance=rebalance,
    )

    # Print summary
    print(portfolio.get_summary())

    # Save results
    portfolio.save_results(output_dir)

    # Generate visualizations
    visualizer = PortfolioVisualizer(portfolio)
    visualizer.create_all_charts(output_dir, show=show_charts)

    return portfolio


def load_weights_from_json(json_file: str) -> Dict[str, tuple]:
    """
    Load strategy weights from JSON file.

    JSON format:
    {
        "AAPL": {
            "csv_file": "results_bulk/AAPL_decisions_2020-01-06_2024-12-23.csv",
            "weight": 0.3
        },
        "MSFT": {
            "csv_file": "results_bulk/MSFT_decisions_2020-01-01_2024-12-31.csv",
            "weight": 0.7
        }
    }

    Args:
        json_file: Path to JSON configuration file

    Returns:
        Dict mapping ticker to (csv_file, weight)
    """
    with open(json_file, "r") as f:
        config = json.load(f)

    strategy_weights = {}
    for ticker, info in config.items():
        strategy_weights[ticker] = (info["csv_file"], info["weight"])

    return strategy_weights


def main():
    """Main function with CLI interface."""

    if len(sys.argv) < 2:
        print(
            """
Portfolio Analysis Tool - Combine Multiple Trading Strategies

USAGE:
  
  1. Equal Weight Portfolio:
     python portfolio_analysis.py equal <csv_file1> <csv_file2> [csv_file3 ...]
     
     Example:
     python portfolio_analysis.py equal results_bulk/AAPL_*.csv results_bulk/MSFT_*.csv
  
  2. Custom Weight Portfolio from JSON:
     python portfolio_analysis.py custom <config.json>
     
     Example:
     python portfolio_analysis.py custom portfolio_config.json
  
  3. All Stocks Equal Weight:
     python portfolio_analysis.py all
     (Analyzes all CSV files in results_bulk/ with equal weights)

OPTIONS:
  --rebalance    Enable daily rebalancing to target weights
  --capital NUM  Set initial capital (default: 10000)
  --show-charts  Display charts during generation
  --output DIR   Set output directory

EXAMPLES:

  # Equal weight portfolio with 3 stocks
  python portfolio_analysis.py equal \\
    results_bulk/AAPL_decisions_2020-01-06_2024-12-23.csv \\
    results_bulk/MSFT_decisions_2020-01-01_2024-12-31.csv \\
    results_bulk/NVDA_decisions_2020-01-01_2024-12-31.csv
  
  # Custom weight portfolio
  python portfolio_analysis.py custom my_portfolio.json
  
  # All stocks in results_bulk/ with rebalancing
  python portfolio_analysis.py all --rebalance --capital 50000
        """
        )
        sys.exit(1)

    command = sys.argv[1]

    # Parse options
    rebalance = "--rebalance" in sys.argv
    show_charts = "--show-charts" in sys.argv

    initial_capital = 10000.0
    if "--capital" in sys.argv:
        idx = sys.argv.index("--capital")
        initial_capital = float(sys.argv[idx + 1])

    output_dir = None
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        output_dir = sys.argv[idx + 1]

    # Execute command
    if command == "equal":
        # Get CSV files (filter out options)
        csv_files = [
            arg
            for arg in sys.argv[2:]
            if not arg.startswith("--") and arg.endswith(".csv")
        ]

        if len(csv_files) < 2:
            print("Error: Need at least 2 CSV files for portfolio analysis")
            sys.exit(1)

        if output_dir is None:
            output_dir = "reports/portfolio_equal_weight"

        analyze_equal_weight_portfolio(
            csv_files=csv_files,
            output_dir=output_dir,
            initial_capital=initial_capital,
            rebalance=rebalance,
            show_charts=show_charts,
        )

    elif command == "custom":
        json_file = sys.argv[2]

        strategy_weights = load_weights_from_json(json_file)

        if output_dir is None:
            output_dir = "reports/portfolio_custom_weight"

        analyze_custom_weight_portfolio(
            strategy_weights=strategy_weights,
            output_dir=output_dir,
            initial_capital=initial_capital,
            rebalance=rebalance,
            show_charts=show_charts,
        )

    elif command == "all":
        # Analyze all CSV files in results_bulk/
        results_dir = Path("results_merged")

        if not results_dir.exists():
            print(f"Error: Directory {results_dir} not found")
            sys.exit(1)

        csv_files = list(results_dir.glob("*_decisions_*.csv"))

        if len(csv_files) < 2:
            print(f"Error: Found only {len(csv_files)} CSV files in {results_dir}")
            print("Need at least 2 files for portfolio analysis")
            sys.exit(1)

        print(f"Found {len(csv_files)} strategy files in {results_dir}")

        if output_dir is None:
            output_dir = "reports/portfolio_all_stocks"

        analyze_equal_weight_portfolio(
            csv_files=[str(f) for f in csv_files],
            output_dir=output_dir,
            initial_capital=initial_capital,
            rebalance=rebalance,
            show_charts=show_charts,
        )

    else:
        print(f"Unknown command: {command}")
        print("Use: equal, custom, or all")
        sys.exit(1)


if __name__ == "__main__":
    main()
