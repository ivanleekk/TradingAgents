# Trading Strategy Analysis Library - Summary

## Overview

A comprehensive Python library for analyzing trading strategy performance from CSV decision files. The library calculates key risk-adjusted metrics, generates professional visualizations, and creates detailed reports suitable for research papers and institutional presentations.

## What's Included

### Core Modules (`analysis/` directory)

1. **analyzer.py** - `TradingAnalyzer` class
    - Parses trading decision CSV files
    - Fetches historical price data via yfinance
    - Simulates portfolio performance
    - Generates trade logs

2. **metrics.py** - `PerformanceMetrics` class
    - Annualized Return (CAGR)
    - Annualized Sharpe Ratio
    - Maximum Drawdown
    - Win Rate
    - Calmar Ratio
    - 12+ additional metrics

3. **visualizer.py** - `PerformanceVisualizer` class
    - Equity curve (strategy vs buy & hold)
    - Drawdown chart
    - Monthly returns heatmap
    - Trade distribution analysis
    - Returns distribution with Q-Q plot
    - Rolling Sharpe ratio and volatility

4. **report.py** - `ReportGenerator` class
    - Generates JSON metrics files
    - Exports CSV data files
    - Creates formatted text reports
    - Produces markdown index
    - Saves all visualizations

### Usage Examples

1. **example_analysis.py** - Command-line tool
    - Analyze single file: `python3 example_analysis.py results_bulk/AAPL_decisions_*.csv`
    - Batch analyze: `python3 example_analysis.py` (processes results_bulk/)
    - Creates comparison reports for multiple strategies

2. **analysis_example.ipynb** - Jupyter notebook
    - Interactive analysis workflow
    - Step-by-step demonstrations
    - Visualization examples
    - Comparison techniques

3. **test_analysis_library.py** - Test script
    - Verifies installation
    - Tests all modules
    - Displays sample output

### Documentation

1. **analysis/README.md** - Complete documentation
    - Full API reference
    - Detailed examples
    - Interpretation guides
    - Advanced usage patterns

2. **ANALYSIS_QUICKSTART.md** - Quick reference
    - Installation instructions
    - Common tasks
    - Troubleshooting
    - Metrics explained

3. **analysis/requirements.txt** - Dependencies
    - pandas, numpy, matplotlib, seaborn
    - yfinance, scipy

## Key Features

### 📊 Metrics Calculated

**Risk-Adjusted Returns:**

- Annualized Return (CAGR)
- Sharpe Ratio (>1.0 good, >2.0 excellent, >3.0 exceptional)
- Sortino Ratio (downside-focused)
- Calmar Ratio (return/max drawdown)

**Risk Metrics:**

- Maximum Drawdown (worst peak-to-trough decline)
- Annualized Volatility
- Value at Risk (via drawdown)

**Trade Statistics:**

- Win Rate (% of profitable trades)
- Profit Factor (gross profits/gross losses)
- Average Win/Loss
- Largest Win/Loss
- Trade Duration

### 📈 Visualizations Generated

1. **Equity Curve** - Strategy performance vs buy & hold benchmark
2. **Drawdown Chart** - Risk exposure over time
3. **Monthly Returns Heatmap** - Seasonal performance patterns
4. **Trade Distribution** - Win/loss analysis with box plots
5. **Returns Distribution** - Statistical properties with Q-Q plot
6. **Rolling Metrics** - Time-varying Sharpe ratio and volatility

All charts are publication-quality (300 DPI) and customizable.

### 📄 Report Outputs

For each strategy, generates:

**Data Files:**

- `{ticker}_metrics.json` - All metrics (programmatic access)
- `{ticker}_metrics.csv` - Metrics (spreadsheet-friendly)
- `{ticker}_trade_log.csv` - Complete trade history with P&L
- `{ticker}_portfolio_history.csv` - Daily portfolio values
- `{ticker}_summary_report.txt` - Human-readable summary

**Visualizations:**

- 6 high-quality PNG charts (see above)

**Index:**

- `{ticker}_REPORT_INDEX.md` - Overview with quick stats

## Quick Start

### Installation

```bash
pip install pandas numpy matplotlib seaborn yfinance scipy
```

### Basic Usage

```python
from analysis import TradingAnalyzer, PerformanceMetrics, PerformanceVisualizer, ReportGenerator

# Analyze a trading strategy
analyzer = TradingAnalyzer('results_bulk/AAPL_decisions_2020-01-06_2024-12-23.csv')
metrics = PerformanceMetrics(analyzer)
visualizer = PerformanceVisualizer(analyzer, metrics)

# Print summary
print(metrics.get_metrics_summary())

# Generate full report
report = ReportGenerator(analyzer, metrics, visualizer)
report.generate_full_report(output_dir='reports/AAPL')
```

### Command Line

```bash
# Analyze single file
python3 example_analysis.py results_bulk/AAPL_decisions_2020-01-06_2024-12-23.csv

# Analyze all files
python3 example_analysis.py
```

## Input Format

CSV files with trading decisions:

```csv
test_date,decision
2020-01-06,BUY
2020-01-13,BUY
2020-01-20,SELL
2020-01-27,BUY
```

**Filename pattern:** `{TICKER}_decisions_{START_DATE}_{END_DATE}.csv`

**Supported decisions:** BUY, SELL, HOLD

## Use Cases

### 1. Research Papers

- Publication-quality charts
- Industry-standard metrics
- Statistical analysis tools
- Comprehensive documentation

### 2. Performance Reports

- Professional formatting
- Multiple output formats
- Interpretation guides
- Executive summaries

### 3. Strategy Comparison

- Batch processing
- Unified metrics
- Ranking tables
- Visual comparisons

### 4. Risk Management

- Drawdown analysis
- Rolling metrics
- Worst-case scenarios
- Volatility tracking

## File Structure

```
TradingAgents/
├── analysis/                           # Core library
│   ├── __init__.py                    # Package init
│   ├── analyzer.py                    # TradingAnalyzer class
│   ├── metrics.py                     # PerformanceMetrics class
│   ├── visualizer.py                  # PerformanceVisualizer class
│   ├── report.py                      # ReportGenerator class
│   ├── README.md                      # Full documentation
│   └── requirements.txt               # Dependencies
├── example_analysis.py                # CLI tool
├── analysis_example.ipynb             # Jupyter notebook
├── test_analysis_library.py           # Test script
├── ANALYSIS_QUICKSTART.md             # Quick reference
└── results_bulk/                      # Input CSV files
    ├── AAPL_decisions_*.csv
    ├── MSFT_decisions_*.csv
    └── ...
```

## Example Output

### Console Output

```
=== PERFORMANCE METRICS ===

Key Risk-Adjusted Metrics:
  Annualized Return:        25.34%
  Annualized Sharpe Ratio:   2.15
  Maximum Drawdown:        -18.45%
  Win Rate:                 62.50%
  Calmar Ratio:              1.37

Trade Statistics:
  Total Trades:                 24
  Winning Trades:               15
  Losing Trades:                 9
  Average Win:               8.32%
  Average Loss:             -4.21%
```

### Generated Files

```
reports/AAPL/
├── AAPL_metrics.json
├── AAPL_metrics.csv
├── AAPL_trade_log.csv
├── AAPL_portfolio_history.csv
├── AAPL_summary_report.txt
├── AAPL_equity_curve.png
├── AAPL_drawdown.png
├── AAPL_monthly_returns.png
├── AAPL_trade_distribution.png
├── AAPL_returns_distribution.png
├── AAPL_rolling_metrics.png
└── AAPL_REPORT_INDEX.md
```

## Testing

```bash
# Run test suite
python3 test_analysis_library.py

# Expected output: All tests pass with sample metrics displayed
```

## Next Steps

1. **Install dependencies:** `pip install -r analysis/requirements.txt`
2. **Run test:** `python3 test_analysis_library.py`
3. **Try example:** `python3 example_analysis.py results_bulk/AAPL_decisions_*.csv`
4. **Read docs:** See `analysis/README.md` for complete API reference
5. **Customize:** Extend classes for custom metrics or visualizations

## Metrics Interpretation

### Sharpe Ratio

- **> 1.0:** Good (compensated for risk)
- **> 2.0:** Excellent (strong risk-adjusted returns)
- **> 3.0:** Exceptional (elite performance)

### Maximum Drawdown

- Institutional limit typically -20% to -30%
- Shows worst investor experience
- Critical for capital preservation

### Win Rate

- High win rate ≠ profitability
- 40% can be profitable if wins >> losses
- Consider with profit factor

### Calmar Ratio

- Return per unit of drawdown risk
- > 1.0 is solid
- Important for risk-averse investors

## Support

- **Full documentation:** `analysis/README.md`
- **Quick reference:** `ANALYSIS_QUICKSTART.md`
- **Interactive examples:** `analysis_example.ipynb`
- **Test suite:** `test_analysis_library.py`

## License

Part of the TradingAgents project. See LICENSE file.
