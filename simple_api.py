"""
Simple API for quick analysis.

This module provides simplified functions for common use cases.
No need to instantiate classes - just call functions directly.
"""

from analysis import (
    TradingAnalyzer,
    PerformanceMetrics,
    PerformanceVisualizer,
    ReportGenerator,
)
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional


def analyze_strategy(
    csv_path: str, initial_capital: float = 10000.0, risk_free_rate: float = 0.03
) -> Dict:
    """
    Quick analysis of a single trading strategy.

    Args:
        csv_path: Path to CSV file with trading decisions
        initial_capital: Starting capital (default: $10,000)
        risk_free_rate: Annual risk-free rate (default: 3%)

    Returns:
        Dictionary with all metrics

    Example:
        >>> results = analyze_strategy('results_bulk/AAPL_decisions_*.csv')
        >>> print(f"Sharpe: {results['annualized_sharpe_ratio']:.2f}")
    """
    analyzer = TradingAnalyzer(csv_path, initial_capital, risk_free_rate)
    metrics = PerformanceMetrics(analyzer)

    results = metrics.calculate_all_metrics()
    results["ticker"] = analyzer.ticker
    results["start_date"] = analyzer.start_date
    results["end_date"] = analyzer.end_date

    return results


def generate_report(
    csv_path: str,
    output_dir: str,
    initial_capital: float = 10000.0,
    show_charts: bool = False,
) -> Dict[str, str]:
    """
    Generate complete report with all files and charts.

    Args:
        csv_path: Path to CSV file with trading decisions
        output_dir: Directory to save report files
        initial_capital: Starting capital (default: $10,000)
        show_charts: Whether to display charts (default: False)

    Returns:
        Dictionary mapping output type to file paths

    Example:
        >>> files = generate_report('results_bulk/AAPL_decisions_*.csv',
        ...                         output_dir='reports/AAPL')
        >>> print(f"Report saved to {files['summary_report']}")
    """
    analyzer = TradingAnalyzer(csv_path, initial_capital)
    metrics = PerformanceMetrics(analyzer)
    visualizer = PerformanceVisualizer(analyzer, metrics)
    report = ReportGenerator(analyzer, metrics, visualizer)

    return report.generate_full_report(output_dir, show_charts)


def compare_strategies(
    csv_files: List[str], initial_capital: float = 10000.0
) -> pd.DataFrame:
    """
    Compare multiple trading strategies.

    Args:
        csv_files: List of CSV file paths
        initial_capital: Starting capital for all strategies

    Returns:
        DataFrame with comparison metrics

    Example:
        >>> files = ['AAPL_decisions_*.csv', 'MSFT_decisions_*.csv']
        >>> df = compare_strategies(files)
        >>> print(df.sort_values('annualized_sharpe_ratio', ascending=False))
    """
    results = []

    for csv_file in csv_files:
        try:
            metrics = analyze_strategy(csv_file, initial_capital)
            results.append(
                {
                    "Ticker": metrics["ticker"],
                    "Annual Return (%)": metrics["annualized_return"],
                    "Sharpe Ratio": metrics["annualized_sharpe_ratio"],
                    "Max Drawdown (%)": metrics["maximum_drawdown"],
                    "Win Rate (%)": metrics["win_rate"],
                    "Calmar Ratio": metrics["calmar_ratio"],
                    "Total Trades": metrics["total_trades"],
                    "Profit Factor": metrics["profit_factor"],
                    "Start Date": metrics["start_date"],
                    "End Date": metrics["end_date"],
                }
            )
        except Exception as e:
            print(f"Error analyzing {csv_file}: {e}")
            continue

    df = pd.DataFrame(results)
    return df.sort_values("Sharpe Ratio", ascending=False)


def get_key_metrics(
    csv_path: str, initial_capital: float = 10000.0
) -> Dict[str, float]:
    """
    Get only the 5 key metrics (for quick overview).

    Args:
        csv_path: Path to CSV file with trading decisions
        initial_capital: Starting capital

    Returns:
        Dictionary with 5 key metrics

    Example:
        >>> metrics = get_key_metrics('results_bulk/AAPL_decisions_*.csv')
        >>> print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
    """
    analyzer = TradingAnalyzer(csv_path, initial_capital)
    metrics = PerformanceMetrics(analyzer)

    return {
        "annualized_return": metrics.annualized_return(),
        "sharpe_ratio": metrics.annualized_sharpe_ratio(),
        "max_drawdown": metrics.maximum_drawdown(),
        "win_rate": metrics.win_rate(),
        "calmar_ratio": metrics.calmar_ratio(),
    }


def print_summary(csv_path: str, initial_capital: float = 10000.0) -> None:
    """
    Print formatted summary to console.

    Args:
        csv_path: Path to CSV file with trading decisions
        initial_capital: Starting capital

    Example:
        >>> print_summary('results_bulk/AAPL_decisions_*.csv')
    """
    analyzer = TradingAnalyzer(csv_path, initial_capital)
    metrics = PerformanceMetrics(analyzer)

    print(f"\n{'='*60}")
    print(f"Strategy: {analyzer.ticker}")
    print(f"{'='*60}")
    print(metrics.get_metrics_summary())


def batch_analyze(
    directory: str = "results_bulk",
    output_dir: str = "reports",
    initial_capital: float = 10000.0,
    generate_reports: bool = True,
) -> pd.DataFrame:
    """
    Analyze all CSV files in a directory.

    Args:
        directory: Directory containing CSV files
        output_dir: Base directory for reports
        initial_capital: Starting capital
        generate_reports: Whether to generate full reports for each

    Returns:
        DataFrame with comparison of all strategies

    Example:
        >>> df = batch_analyze('results_bulk', generate_reports=True)
        >>> df.to_csv('all_strategies_comparison.csv', index=False)
    """
    csv_files = list(Path(directory).glob("*_decisions_*.csv"))

    print(f"Found {len(csv_files)} strategy files")

    if generate_reports:
        print("Generating full reports...")
        for csv_file in csv_files:
            try:
                ticker = csv_file.stem.split("_")[0]
                report_dir = f"{output_dir}/{ticker}"
                generate_report(
                    str(csv_file), report_dir, initial_capital, show_charts=False
                )
            except Exception as e:
                print(f"Error generating report for {csv_file.name}: {e}")

    print("\nComparing all strategies...")
    comparison_df = compare_strategies([str(f) for f in csv_files], initial_capital)

    # Save comparison
    comparison_file = f"{output_dir}/all_strategies_comparison.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\nComparison saved to {comparison_file}")

    return comparison_df


def quick_chart(
    csv_path: str, chart_type: str = "equity", save_path: Optional[str] = None
) -> None:
    """
    Generate a single chart quickly.

    Args:
        csv_path: Path to CSV file with trading decisions
        chart_type: Type of chart ('equity', 'drawdown', 'monthly', 'trades', 'returns', 'rolling')
        save_path: Path to save chart (optional)

    Example:
        >>> quick_chart('results_bulk/AAPL_decisions_*.csv', 'equity', 'aapl_equity.png')
    """
    analyzer = TradingAnalyzer(csv_path)
    metrics = PerformanceMetrics(analyzer)
    viz = PerformanceVisualizer(analyzer, metrics)

    chart_map = {
        "equity": viz.plot_equity_curve,
        "drawdown": viz.plot_drawdown,
        "monthly": viz.plot_monthly_returns,
        "trades": viz.plot_trade_distribution,
        "returns": viz.plot_returns_distribution,
        "rolling": viz.plot_rolling_metrics,
    }

    if chart_type not in chart_map:
        raise ValueError(
            f"Unknown chart type: {chart_type}. Choose from: {list(chart_map.keys())}"
        )

    chart_func = chart_map[chart_type]
    chart_func(save_path=save_path, show=True)


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage examples:")
        print("  python simple_api.py analyze results_bulk/AAPL_decisions_*.csv")
        print(
            "  python simple_api.py report results_bulk/AAPL_decisions_*.csv reports/AAPL"
        )
        print("  python simple_api.py compare results_bulk/*.csv")
        print("  python simple_api.py batch results_bulk")
        sys.exit(1)

    command = sys.argv[1]

    if command == "analyze":
        metrics = analyze_strategy(sys.argv[2])
        print(f"\nTicker: {metrics['ticker']}")
        print(f"Sharpe Ratio: {metrics['annualized_sharpe_ratio']:.2f}")
        print(f"Annual Return: {metrics['annualized_return']:.2f}%")
        print(f"Max Drawdown: {metrics['maximum_drawdown']:.2f}%")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")

    elif command == "report":
        files = generate_report(sys.argv[2], sys.argv[3])
        print(f"\nReport generated: {files['summary_report']}")

    elif command == "compare":
        df = compare_strategies(sys.argv[2:])
        print(df)

    elif command == "batch":
        df = batch_analyze(sys.argv[2] if len(sys.argv) > 2 else "results_bulk")
        print("\nTop 5 strategies by Sharpe Ratio:")
        print(df.head())

    else:
        print(f"Unknown command: {command}")
