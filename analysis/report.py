"""
Report generation module for trading strategy analysis.

This module provides the ReportGenerator class for creating comprehensive
reports with metrics and visualizations saved to files.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


class ReportGenerator:
    """
    Generate comprehensive trading strategy reports.

    Creates:
    - JSON file with all metrics
    - CSV file with trade log
    - CSV file with portfolio history
    - Text file with summary
    - All visualizations
    """

    def __init__(self, analyzer, metrics, visualizer):
        """
        Initialize report generator.

        Args:
            analyzer: TradingAnalyzer instance
            metrics: PerformanceMetrics instance
            visualizer: PerformanceVisualizer instance
        """
        self.analyzer = analyzer
        self.metrics = metrics
        self.visualizer = visualizer

    def generate_full_report(
        self, output_dir: str, show_charts: bool = False
    ) -> Dict[str, str]:
        """
        Generate complete report with all outputs.

        Args:
            output_dir: Directory to save all report files
            show_charts: Whether to display charts during generation

        Returns:
            Dictionary mapping output type to file path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        ticker = self.analyzer.ticker
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"\n{'='*60}")
        print(f"Generating report for {ticker}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}\n")

        saved_files = {}

        # 1. Save metrics as JSON
        print("Saving metrics...")
        metrics_file = output_path / f"{ticker}_metrics.json"
        self._save_metrics_json(metrics_file)
        saved_files["metrics_json"] = str(metrics_file)

        # 2. Save metrics as CSV
        metrics_csv_file = output_path / f"{ticker}_metrics.csv"
        self._save_metrics_csv(metrics_csv_file)
        saved_files["metrics_csv"] = str(metrics_csv_file)

        # 3. Save trade log
        print("Saving trade log...")
        trade_log_file = output_path / f"{ticker}_trade_log.csv"
        self._save_trade_log(trade_log_file)
        saved_files["trade_log"] = str(trade_log_file)

        # 4. Save portfolio history
        print("Saving portfolio history...")
        portfolio_file = output_path / f"{ticker}_portfolio_history.csv"
        self._save_portfolio_history(portfolio_file)
        saved_files["portfolio_history"] = str(portfolio_file)

        # 5. Save summary report
        print("Generating text summary...")
        summary_file = output_path / f"{ticker}_summary_report.txt"
        self._save_summary_report(summary_file)
        saved_files["summary_report"] = str(summary_file)

        # 6. Generate all charts
        print("Generating visualizations...")
        chart_files = self.visualizer.create_all_charts(output_dir, show=show_charts)
        saved_files["charts"] = [str(f) for f in chart_files]

        # 7. Create index/readme
        print("Creating report index...")
        index_file = output_path / f"{ticker}_REPORT_INDEX.md"
        self._create_report_index(index_file, saved_files)
        saved_files["index"] = str(index_file)

        print(f"\n{'='*60}")
        print(f"✓ Report generation complete!")
        print(f"✓ All files saved to: {output_dir}")
        print(f"✓ See {index_file.name} for report overview")
        print(f"{'='*60}\n")

        return saved_files

    def _save_metrics_json(self, file_path: Path):
        """Save all metrics as JSON."""
        metrics = self.metrics.calculate_all_metrics()
        summary = self.analyzer.get_summary_stats()

        output = {
            "metadata": {
                "ticker": summary["ticker"],
                "start_date": summary["start_date"],
                "end_date": summary["end_date"],
                "generated_at": datetime.now().isoformat(),
            },
            "summary": summary,
            "metrics": metrics,
        }

        with open(file_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        print(f"  Saved: {file_path.name}")

    def _save_metrics_csv(self, file_path: Path):
        """Save metrics as CSV for easy importing."""
        metrics = self.metrics.calculate_all_metrics()

        df = pd.DataFrame(
            [{"metric": key, "value": value} for key, value in metrics.items()]
        )

        df.to_csv(file_path, index=False)
        print(f"  Saved: {file_path.name}")

    def _save_trade_log(self, file_path: Path):
        """Save detailed trade log."""
        trade_log = self.analyzer.get_trade_log()

        if len(trade_log) > 0:
            # Add calculated columns
            trade_log["duration_days"] = (
                trade_log["exit_date"] - trade_log["entry_date"]
            ).dt.days

            trade_log.to_csv(file_path, index=False)
            print(f"  Saved: {file_path.name} ({len(trade_log)} trades)")
        else:
            print(f"  No trades to save")

    def _save_portfolio_history(self, file_path: Path):
        """Save complete portfolio history."""
        history = self.analyzer.portfolio_df[
            [
                "price",
                "decision",
                "position",
                "portfolio_value",
                "bnh_value",
                "drawdown",
            ]
        ].copy()

        history.to_csv(file_path)
        print(f"  Saved: {file_path.name} ({len(history)} days)")

    def _save_summary_report(self, file_path: Path):
        """Save formatted text summary."""
        summary = self.analyzer.get_summary_stats()
        metrics = self.metrics.calculate_all_metrics()
        trade_log = self.analyzer.get_trade_log()

        report = f"""
{'='*80}
TRADING STRATEGY PERFORMANCE REPORT
{'='*80}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}
STRATEGY INFORMATION
{'='*80}

Ticker Symbol:           {summary['ticker']}
Analysis Period:         {summary['start_date']} to {summary['end_date']}
Total Trading Days:      {summary['trading_days']:,}
Initial Capital:         ${summary['initial_capital']:,.2f}
Final Portfolio Value:   ${summary['final_value']:,.2f}

{'='*80}
KEY PERFORMANCE METRICS
{'='*80}

Risk-Adjusted Returns:
  Annualized Return:           {metrics['annualized_return']:>12.2f}%
  Annualized Sharpe Ratio:     {metrics['annualized_sharpe_ratio']:>12.2f}
  Annualized Sortino Ratio:    {metrics['sortino_ratio']:>12.2f}
  
Risk Metrics:
  Maximum Drawdown:            {metrics['maximum_drawdown']:>12.2f}%
  Annualized Volatility:       {metrics['volatility']:>12.2f}%
  Calmar Ratio:                {metrics['calmar_ratio']:>12.2f}

Return Metrics:
  Total Return:                {metrics['total_return']:>12.2f}%
  Buy & Hold Return:           {summary['bnh_return_pct']:>12.2f}%
  Outperformance:              {summary['outperformance_pct']:>12.2f}%

{'='*80}
TRADE STATISTICS
{'='*80}

Trade Summary:
  Total Trades:                {metrics['total_trades']:>12.0f}
  Winning Trades:              {metrics['winning_trades']:>12.0f}
  Losing Trades:               {metrics['losing_trades']:>12.0f}
  Win Rate:                    {metrics['win_rate']:>12.2f}%
  Profit Factor:               {metrics['profit_factor']:>12.2f}

Trade Performance:
  Average Win:                 {metrics['avg_win']:>12.2f}%
  Average Loss:                {metrics['avg_loss']:>12.2f}%
  Largest Win:                 {metrics['largest_win']:>12.2f}%
  Largest Loss:                {metrics['largest_loss']:>12.2f}%
  Avg Trade Duration:          {metrics['avg_trade_duration']:>12.1f} days

{'='*80}
INTERPRETATION GUIDE
{'='*80}

Sharpe Ratio:
  > 1.0 = Good risk-adjusted returns
  > 2.0 = Excellent risk-adjusted returns
  > 3.0 = Exceptional risk-adjusted returns
  Current: {metrics['annualized_sharpe_ratio']:.2f} - {'Exceptional' if metrics['annualized_sharpe_ratio'] > 3.0 else 'Excellent' if metrics['annualized_sharpe_ratio'] > 2.0 else 'Good' if metrics['annualized_sharpe_ratio'] > 1.0 else 'Below Average'}

Maximum Drawdown:
  The worst peak-to-trough decline experienced by the strategy.
  Lower is better. Most institutional investors have strict limits.
  Current: {metrics['maximum_drawdown']:.2f}%

Win Rate:
  Percentage of profitable trades. Consider alongside profit factor.
  High win rate doesn't guarantee profitability if losses are large.
  Current: {metrics['win_rate']:.2f}%

Calmar Ratio:
  Return per unit of downside risk (annual return / max drawdown).
  Higher is better. Good for risk-averse investors.
  Current: {metrics['calmar_ratio']:.2f}

{'='*80}
END OF REPORT
{'='*80}
"""

        with open(file_path, "w") as f:
            f.write(report)

        print(f"  Saved: {file_path.name}")

    def _create_report_index(self, file_path: Path, saved_files: Dict[str, Any]):
        """Create markdown index of all report files."""
        ticker = self.analyzer.ticker

        index = f"""# Trading Strategy Report: {ticker}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Report Files

### 📊 Metrics and Data

- **Metrics (JSON):** `{Path(saved_files['metrics_json']).name}`
  - All metrics in JSON format for programmatic access

- **Metrics (CSV):** `{Path(saved_files['metrics_csv']).name}`
  - Metrics in CSV format for spreadsheet analysis

- **Trade Log:** `{Path(saved_files['trade_log']).name}`
  - Detailed log of all trades with entry/exit points and P&L

- **Portfolio History:** `{Path(saved_files['portfolio_history']).name}`
  - Daily portfolio values and positions

- **Summary Report:** `{Path(saved_files['summary_report']).name}`
  - Human-readable text summary with all key metrics

### 📈 Visualizations

"""

        if "charts" in saved_files:
            for chart_file in saved_files["charts"]:
                chart_name = (
                    Path(chart_file)
                    .stem.replace(f"{ticker}_", "")
                    .replace("_", " ")
                    .title()
                )
                index += f"- **{chart_name}:** `{Path(chart_file).name}`\n"

        index += f"""
## Quick Start

1. **Read the summary:** Open `{Path(saved_files['summary_report']).name}` for a complete overview
2. **View charts:** Check the PNG files for visual analysis
3. **Analyze trades:** Open `{Path(saved_files['trade_log']).name}` in a spreadsheet
4. **Import metrics:** Use `{Path(saved_files['metrics_json']).name}` for programmatic analysis

## Key Metrics Summary

"""

        metrics = self.metrics.calculate_all_metrics()

        index += f"""
| Metric | Value |
|--------|-------|
| Annualized Return | {metrics['annualized_return']:.2f}% |
| Sharpe Ratio | {metrics['annualized_sharpe_ratio']:.2f} |
| Maximum Drawdown | {metrics['maximum_drawdown']:.2f}% |
| Win Rate | {metrics['win_rate']:.2f}% |
| Calmar Ratio | {metrics['calmar_ratio']:.2f} |
| Total Trades | {metrics['total_trades']:.0f} |

---

*Generated by TradingAgents Analysis Library*
"""

        with open(file_path, "w") as f:
            f.write(index)

        print(f"  Saved: {file_path.name}")

    def print_summary(self):
        """Print summary to console."""
        print(self.metrics.get_metrics_summary())
