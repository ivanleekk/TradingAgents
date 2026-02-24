#!/usr/bin/env python3
"""
Example usage of the trading analysis library.

This script demonstrates how to use the analysis library to:
1. Load trading decisions from CSV
2. Calculate performance metrics
3. Generate visualizations
4. Create comprehensive reports
"""

from analysis import (
    TradingAnalyzer,
    PerformanceMetrics,
    PerformanceVisualizer,
    ReportGenerator,
)
from pathlib import Path


def analyze_single_file(
    csv_path: str,
    output_dir: str = None,
    initial_capital: float = 10000.0,
    show_charts: bool = False,
):
    """
    Analyze a single trading decisions file.

    Args:
        csv_path: Path to CSV file with trading decisions
        output_dir: Directory to save outputs (default: creates one based on ticker)
        initial_capital: Starting capital (default: $10,000)
        show_charts: Whether to display charts interactively
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING: {csv_path}")
    print(f"{'='*80}\n")

    # Initialize analyzer
    analyzer = TradingAnalyzer(
        csv_path=csv_path,
        initial_capital=initial_capital,
        risk_free_rate=0.03,  # 3% annual risk-free rate
    )

    # Calculate metrics
    metrics = PerformanceMetrics(analyzer)

    # Create visualizer
    visualizer = PerformanceVisualizer(analyzer, metrics)

    # Print summary to console
    print(metrics.get_metrics_summary())

    # Generate full report if output directory specified
    if output_dir:
        report_gen = ReportGenerator(analyzer, metrics, visualizer)
        saved_files = report_gen.generate_full_report(
            output_dir=output_dir, show_charts=show_charts
        )
        return analyzer, metrics, visualizer, saved_files
    else:
        return analyzer, metrics, visualizer, None


def analyze_multiple_files(
    csv_files: list, output_base_dir: str = "reports", initial_capital: float = 10000.0
):
    """
    Analyze multiple trading decisions files.

    Args:
        csv_files: List of CSV file paths
        output_base_dir: Base directory for all reports
        initial_capital: Starting capital for each analysis
    """
    results = []

    for csv_file in csv_files:
        try:
            ticker = Path(csv_file).stem.split("_")[0]
            output_dir = f"{output_base_dir}/{ticker}"

            analyzer, metrics, visualizer, saved_files = analyze_single_file(
                csv_path=csv_file,
                output_dir=output_dir,
                initial_capital=initial_capital,
                show_charts=False,
            )

            results.append(
                {
                    "ticker": ticker,
                    "csv_file": csv_file,
                    "output_dir": output_dir,
                    "metrics": metrics.calculate_all_metrics(),
                }
            )

        except Exception as e:
            print(f"Error analyzing {csv_file}: {e}")
            continue

    # Create comparison report
    if results:
        create_comparison_report(results, output_base_dir)

    return results


def create_comparison_report(results: list, output_dir: str):
    """
    Create a comparison report across multiple strategies.

    Args:
        results: List of analysis results
        output_dir: Directory to save comparison report
    """
    import pandas as pd

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract key metrics for comparison
    comparison_data = []
    for result in results:
        metrics = result["metrics"]
        comparison_data.append(
            {
                "Ticker": result["ticker"],
                "Annual Return (%)": metrics["annualized_return"],
                "Sharpe Ratio": metrics["annualized_sharpe_ratio"],
                "Max Drawdown (%)": metrics["maximum_drawdown"],
                "Win Rate (%)": metrics["win_rate"],
                "Calmar Ratio": metrics["calmar_ratio"],
                "Total Trades": metrics["total_trades"],
                "Profit Factor": metrics["profit_factor"],
            }
        )

    df = pd.DataFrame(comparison_data)

    # Save to CSV
    csv_path = output_path / "strategy_comparison.csv"
    df.to_csv(csv_path, index=False)

    # Create markdown report
    md_path = output_path / "strategy_comparison.md"
    with open(md_path, "w") as f:
        f.write("# Strategy Comparison Report\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Rankings\n\n")

        # Add rankings
        f.write("### By Sharpe Ratio\n")
        f.write(
            df.nlargest(10, "Sharpe Ratio")[["Ticker", "Sharpe Ratio"]].to_markdown(
                index=False
            )
        )

        f.write("\n\n### By Annualized Return\n")
        f.write(
            df.nlargest(10, "Annual Return (%)")[
                ["Ticker", "Annual Return (%)"]
            ].to_markdown(index=False)
        )

        f.write("\n\n### By Calmar Ratio\n")
        f.write(
            df.nlargest(10, "Calmar Ratio")[["Ticker", "Calmar Ratio"]].to_markdown(
                index=False
            )
        )

    print(f"\n✓ Comparison report saved to {output_dir}")
    print(f"  - {csv_path.name}")
    print(f"  - {md_path.name}")


def main():
    """Main example usage."""
    import sys

    # Example 1: Analyze a single file
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        analyze_single_file(
            csv_path=csv_file,
            output_dir=f"reports/{Path(csv_file).stem}",
            initial_capital=10000.0,
            show_charts=False,
        )
    else:
        # Example 2: Analyze multiple files
        results_dir = Path("results_merged")
        if results_dir.exists():
            csv_files = list(results_dir.glob("*_decisions_*.csv"))

            if csv_files:
                print(f"Found {len(csv_files)} decision files")

                analyze_multiple_files(
                    csv_files=csv_files,  # Analyze first 3 as example
                    output_base_dir="reports",
                    initial_capital=10000.0,
                )
            else:
                print("No decision CSV files found in results_bulk/")
        else:
            print("results_bulk/ directory not found")
            print("\nUsage:")
            print("  python example_usage.py <path_to_csv_file>")
            print("  or")
            print("  python example_usage.py  # to analyze files in results_bulk/")


if __name__ == "__main__":
    main()
