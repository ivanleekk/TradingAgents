# Trading Strategy Analysis Library

A comprehensive Python library for analyzing trading strategy performance, calculating risk-adjusted metrics, and generating professional reports with visualizations.

## Features

### 📊 Performance Metrics

Calculate all key trading performance metrics:

- **Annualized Return (%)**: Compound annual growth rate (CAGR)
- **Annualized Sharpe Ratio**: Risk-adjusted returns (>1.0 good, >2.0 excellent, >3.0 exceptional)
- **Maximum Drawdown (%)**: Largest peak-to-trough decline
- **Win Rate (%)**: Percentage of profitable trades
- **Calmar Ratio**: Return-to-drawdown ratio for downside protection
- **Additional metrics**: Sortino Ratio, Profit Factor, Volatility, and more

### 📈 Visualizations

Generate professional charts for reports:

- Equity curve (strategy vs buy & hold)
- Drawdown over time
- Monthly returns heatmap
- Trade distribution analysis
- Returns distribution with Q-Q plot
- Rolling Sharpe ratio and volatility

### 📄 Reports

Automatically generate comprehensive reports:

- JSON metrics for programmatic access
- CSV files for spreadsheet analysis
- Detailed trade logs with P&L
- Daily portfolio history
- Human-readable text summaries
- Markdown index with all files

## Installation

### Dependencies

```bash
pip install pandas numpy matplotlib seaborn yfinance scipy
```

Or install all project dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

```python
from analysis import TradingAnalyzer, PerformanceMetrics, PerformanceVisualizer, ReportGenerator

# Analyze a single trading decisions file
analyzer = TradingAnalyzer(
    csv_path='results_bulk/AAPL_decisions_2020-01-06_2024-12-23.csv',
    initial_capital=10000.0,
    risk_free_rate=0.03
)

# Calculate metrics
metrics = PerformanceMetrics(analyzer)
print(metrics.get_metrics_summary())

# Create visualizations
visualizer = PerformanceVisualizer(analyzer, metrics)
visualizer.plot_equity_curve(save_path='equity_curve.png')

# Generate full report
report = ReportGenerator(analyzer, metrics, visualizer)
report.generate_full_report(output_dir='reports/AAPL')
```

### Using the Example Script

Analyze a single file:

```bash
python example_analysis.py results_bulk/AAPL_decisions_2020-01-06_2024-12-23.csv
```

Analyze multiple files:

```bash
python example_analysis.py
```

### Custom Analysis

```python
from analysis import TradingAnalyzer, PerformanceMetrics

# Load and analyze
analyzer = TradingAnalyzer('path/to/decisions.csv', initial_capital=50000)
metrics = PerformanceMetrics(analyzer)

# Get specific metrics
sharpe = metrics.annualized_sharpe_ratio()
max_dd = metrics.maximum_drawdown()
win_rate = metrics.win_rate()

# Get all metrics
all_metrics = metrics.calculate_all_metrics()

# Get trade log
trades = analyzer.get_trade_log()
print(f"Total trades: {len(trades)}")
print(f"Winning trades: {trades['win'].sum()}")
```

## Input Data Format

The library expects CSV files with trading decisions in this format:

```csv
test_date,decision
2020-01-06,BUY
2020-01-13,BUY
2020-01-20,SELL
2020-01-27,BUY
2020-02-03,HOLD
...
```

### Requirements

- **Columns**: `test_date` (date), `decision` (BUY/SELL/HOLD)
- **Filename pattern**: `{TICKER}_decisions_{START_DATE}_{END_DATE}.csv`
- **Example**: `AAPL_decisions_2020-01-06_2024-12-23.csv`

## API Reference

### TradingAnalyzer

Main class for loading and analyzing trading decisions.

```python
analyzer = TradingAnalyzer(
    csv_path: str,              # Path to decisions CSV
    initial_capital: float,     # Starting capital (default: 10000)
    risk_free_rate: float       # Annual risk-free rate (default: 0.03)
)

# Properties
analyzer.ticker              # Extracted ticker symbol
analyzer.portfolio_df        # Daily portfolio values
analyzer.price_df            # Historical price data

# Methods
analyzer.get_trade_log()           # DataFrame of all trades
analyzer.get_summary_stats()       # Basic summary statistics
```

### PerformanceMetrics

Calculate all performance metrics.

```python
metrics = PerformanceMetrics(analyzer)

# Key metrics
metrics.annualized_return()           # CAGR %
metrics.annualized_sharpe_ratio()     # Sharpe ratio
metrics.maximum_drawdown()            # Max DD %
metrics.win_rate()                    # Win rate %
metrics.calmar_ratio()                # Calmar ratio

# Additional metrics
metrics.annualized_volatility()       # Volatility %
metrics.sortino_ratio()               # Sortino ratio
metrics.profit_factor()               # Profit factor
metrics.total_trades()                # Number of trades

# All at once
all_metrics = metrics.calculate_all_metrics()

# Formatted summary
print(metrics.get_metrics_summary())
```

### PerformanceVisualizer

Generate charts and visualizations.

```python
visualizer = PerformanceVisualizer(analyzer, metrics)

# Individual charts
visualizer.plot_equity_curve(save_path='equity.png', show=True)
visualizer.plot_drawdown(save_path='drawdown.png')
visualizer.plot_monthly_returns(save_path='monthly.png')
visualizer.plot_trade_distribution(save_path='trades.png')
visualizer.plot_returns_distribution(save_path='returns.png')
visualizer.plot_rolling_metrics(window=252, save_path='rolling.png')

# Generate all charts
visualizer.create_all_charts(output_dir='charts/', show=False)
```

### ReportGenerator

Create comprehensive reports with all outputs.

```python
report = ReportGenerator(analyzer, metrics, visualizer)

# Generate complete report
saved_files = report.generate_full_report(
    output_dir='reports/AAPL',
    show_charts=False
)

# Print summary to console
report.print_summary()
```

## Output Files

A complete report includes:

### Data Files

- `{ticker}_metrics.json` - All metrics in JSON format
- `{ticker}_metrics.csv` - Metrics in CSV format
- `{ticker}_trade_log.csv` - Detailed trade log
- `{ticker}_portfolio_history.csv` - Daily portfolio values
- `{ticker}_summary_report.txt` - Human-readable summary

### Visualizations

- `{ticker}_equity_curve.png` - Strategy vs buy & hold
- `{ticker}_drawdown.png` - Drawdown over time
- `{ticker}_monthly_returns.png` - Monthly returns heatmap
- `{ticker}_trade_distribution.png` - Trade analysis
- `{ticker}_returns_distribution.png` - Returns distribution
- `{ticker}_rolling_metrics.png` - Rolling Sharpe & volatility

### Index

- `{ticker}_REPORT_INDEX.md` - Markdown overview of all files

## Interpretation Guide

### Sharpe Ratio

- **> 1.0**: Good risk-adjusted returns
- **> 2.0**: Excellent risk-adjusted returns
- **> 3.0**: Exceptional risk-adjusted returns

### Maximum Drawdown

- Measures worst-case decline from peak
- Lower is better (less negative)
- Most institutional investors have strict limits (e.g., -20%)

### Win Rate

- High win rate doesn't guarantee profitability
- Must consider average win vs average loss
- 40% win rate can be profitable if wins are large

### Calmar Ratio

- Return per unit of downside risk
- Higher is better
- Particularly relevant for risk-averse investors

## Examples

### Batch Analysis

Analyze all stocks in `results_bulk/`:

```python
from pathlib import Path
from analysis import TradingAnalyzer, PerformanceMetrics, ReportGenerator

results_dir = Path('results_bulk')
csv_files = list(results_dir.glob('*_decisions_*.csv'))

for csv_file in csv_files:
    try:
        analyzer = TradingAnalyzer(str(csv_file))
        metrics = PerformanceMetrics(analyzer)

        ticker = analyzer.ticker
        print(f"\n{ticker}:")
        print(f"  Sharpe: {metrics.annualized_sharpe_ratio():.2f}")
        print(f"  Return: {metrics.annualized_return():.2f}%")
        print(f"  Max DD: {metrics.maximum_drawdown():.2f}%")

    except Exception as e:
        print(f"Error with {csv_file}: {e}")
```

### Custom Metrics Extraction

```python
import pandas as pd
from analysis import TradingAnalyzer, PerformanceMetrics

# Analyze multiple strategies
results = []

for csv_file in csv_files:
    analyzer = TradingAnalyzer(str(csv_file))
    metrics = PerformanceMetrics(analyzer)

    all_metrics = metrics.calculate_all_metrics()
    all_metrics['ticker'] = analyzer.ticker
    results.append(all_metrics)

# Create comparison DataFrame
df = pd.DataFrame(results)
df = df.sort_values('annualized_sharpe_ratio', ascending=False)

# Export top performers
df.to_csv('top_strategies.csv', index=False)
print(df[['ticker', 'annualized_return', 'annualized_sharpe_ratio', 'maximum_drawdown']])
```

## Project Structure

```
TradingAgents/
├── analysis/
│   ├── __init__.py          # Package initialization
│   ├── analyzer.py          # Core TradingAnalyzer class
│   ├── metrics.py           # PerformanceMetrics class
│   ├── visualizer.py        # PerformanceVisualizer class
│   └── report.py            # ReportGenerator class
├── example_analysis.py      # Example usage script
├── results_bulk/            # Input CSV files
└── reports/                 # Generated reports (output)
```

## Contributing

Feel free to extend the library with:

- Additional metrics (e.g., Value at Risk, Conditional VaR)
- More visualizations (e.g., scatter plots, correlation matrices)
- Different report formats (e.g., HTML, PDF)
- Performance optimizations

## License

This library is part of the TradingAgents project. See LICENSE file for details.

## Acknowledgments

Metrics calculations follow industry-standard methodologies:

- Sharpe Ratio: Nobel Prize-winning framework by William F. Sharpe
- Drawdown analysis: Standard risk management practices
- Performance attribution: CFA Institute guidelines
