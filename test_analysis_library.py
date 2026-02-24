"""
Quick test script for the trading analysis library.

Run this to verify the library is working correctly.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_basic_functionality():
    """Test basic library functionality."""
    try:
        from analysis import (
            TradingAnalyzer,
            PerformanceMetrics,
            PerformanceVisualizer,
            ReportGenerator,
        )

        print("✓ Successfully imported all modules")

        # Find a sample CSV file
        results_dir = Path("results_bulk")
        csv_files = list(results_dir.glob("*_decisions_*.csv"))

        if not csv_files:
            print("✗ No CSV files found in results_bulk/")
            return False

        test_file = csv_files[0]
        print(f"✓ Found test file: {test_file.name}")

        # Test analyzer
        print("\nTesting TradingAnalyzer...")
        analyzer = TradingAnalyzer(
            csv_path=str(test_file), initial_capital=10000.0, risk_free_rate=0.03
        )
        print(
            f"✓ Loaded {analyzer.ticker} with {len(analyzer.portfolio_df)} days of data"
        )

        # Test metrics
        print("\nTesting PerformanceMetrics...")
        metrics = PerformanceMetrics(analyzer)
        all_metrics = metrics.calculate_all_metrics()
        print(f"✓ Calculated {len(all_metrics)} metrics")
        print(f"  - Annualized Return: {all_metrics['annualized_return']:.2f}%")
        print(f"  - Sharpe Ratio: {all_metrics['annualized_sharpe_ratio']:.2f}")
        print(f"  - Max Drawdown: {all_metrics['maximum_drawdown']:.2f}%")
        print(f"  - Win Rate: {all_metrics['win_rate']:.2f}%")

        # Test visualizer
        print("\nTesting PerformanceVisualizer...")
        visualizer = PerformanceVisualizer(analyzer, metrics)
        print("✓ Visualizer initialized")

        # Test report generator
        print("\nTesting ReportGenerator...")
        report = ReportGenerator(analyzer, metrics, visualizer)
        print("✓ Report generator initialized")

        # Test summary
        print("\n" + "=" * 60)
        print("METRICS SUMMARY:")
        print("=" * 60)
        print(metrics.get_metrics_summary())

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe library is ready to use. Try:")
        print(f"  python3 example_analysis.py {test_file}")
        print("\nOr see analysis/README.md for full documentation.")

        return True

    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("\nMake sure all dependencies are installed:")
        print("  pip install pandas numpy matplotlib seaborn yfinance scipy")
        return False
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
