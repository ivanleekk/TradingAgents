# Quick Start Guide - Trading Analysis Library

## Installation

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn yfinance scipy

# Or use the requirements file
pip install -r analysis/requirements.txt
```

## Basic Usage

### 1. Analyze a Single File

```python
from analysis import TradingAnalyzer, PerformanceMetrics, PerformanceVisualizer, ReportGenerator

# Load trading decisions
analyzer = TradingAnalyzer('results_bulk/AAPL_decisions_2020-01-06_2024-12-23.csv')

# Calculate metrics
metrics = PerformanceMetrics(analyzer)

# Print summary
print(metrics.get_metrics_summary())

# Create charts
visualizer = PerformanceVisualizer(analyzer, metrics)
visualizer.plot_equity_curve(save_path='aapl_equity.png')

# Generate full report
report = ReportGenerator(analyzer, metrics, visualizer)
report.generate_full_report(output_dir='reports/AAPL')
```

### 2. Using the Command Line

```bash
# Analyze one file
python3 example_analysis.py results_bulk/AAPL_decisions_2020-01-06_2024-12-23.csv

# Analyze all files in results_bulk/
python3 example_analysis.py
```

### 3. Get Specific Metrics

```python
from analysis import TradingAnalyzer, PerformanceMetrics

analyzer = TradingAnalyzer('your_file.csv')
metrics = PerformanceMetrics(analyzer)

# Individual metrics
sharpe = metrics.annualized_sharpe_ratio()
max_dd = metrics.maximum_drawdown()
win_rate = metrics.win_rate()
calmar = metrics.calmar_ratio()

print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_dd:.2f}%")
print(f"Win Rate: {win_rate:.2f}%")
```

## Key Metrics Explained

### Annualized Return

- **What**: Compound annual growth rate (CAGR) of your portfolio
- **Good values**: Depends on asset class, but >10% is solid for equities
- **Formula**: `(End Value / Start Value)^(1/years) - 1`

### Sharpe Ratio

- **What**: Risk-adjusted returns (excess return per unit of volatility)
- **Good values**:
    - \> 1.0 = Good
    - \> 2.0 = Excellent
    - \> 3.0 = Exceptional
- **Interpretation**: Higher is better - you want more return per unit of risk

### Maximum Drawdown

- **What**: Worst peak-to-trough decline in portfolio value
- **Good values**: Smaller (less negative) is better
- **Typical limits**: Many funds have -20% to -30% limits
- **Use**: Shows worst-case scenario investor would experience

### Win Rate

- **What**: Percentage of trades that are profitable
- **Good values**: >50% is generally good, but not required for profitability
- **Note**: Can be profitable with 40% win rate if winners are larger than losers

### Calmar Ratio

- **What**: Annualized return divided by maximum drawdown
- **Good values**: Higher is better, >1.0 is solid
- **Use**: Particularly relevant for risk-averse investors

## Output Files

After running `generate_full_report()`, you'll get:

```
reports/AAPL/
├── AAPL_metrics.json              # All metrics (programmatic access)
├── AAPL_metrics.csv               # Metrics (spreadsheet)
├── AAPL_trade_log.csv             # All trades with P&L
├── AAPL_portfolio_history.csv     # Daily values
├── AAPL_summary_report.txt        # Human-readable summary
├── AAPL_equity_curve.png          # Chart: Strategy vs Buy & Hold
├── AAPL_drawdown.png              # Chart: Drawdown over time
├── AAPL_monthly_returns.png       # Chart: Monthly returns heatmap
├── AAPL_trade_distribution.png    # Chart: Win/loss analysis
├── AAPL_returns_distribution.png  # Chart: Returns distribution
├── AAPL_rolling_metrics.png       # Chart: Rolling Sharpe & vol
└── AAPL_REPORT_INDEX.md           # Overview of all files
```

## Common Tasks

### Compare Multiple Strategies

```python
from pathlib import Path
from analysis import TradingAnalyzer, PerformanceMetrics
import pandas as pd

results = []
for csv_file in Path('results_bulk').glob('*.csv'):
    analyzer = TradingAnalyzer(str(csv_file))
    metrics = PerformanceMetrics(analyzer)

    results.append({
        'Ticker': analyzer.ticker,
        'Sharpe': metrics.annualized_sharpe_ratio(),
        'Return': metrics.annualized_return(),
        'Max DD': metrics.maximum_drawdown(),
        'Win Rate': metrics.win_rate()
    })

df = pd.DataFrame(results).sort_values('Sharpe', ascending=False)
print(df)
df.to_csv('strategy_comparison.csv', index=False)
```

### Export Metrics to JSON

```python
import json
from analysis import TradingAnalyzer, PerformanceMetrics

analyzer = TradingAnalyzer('your_file.csv')
metrics = PerformanceMetrics(analyzer)

all_metrics = metrics.calculate_all_metrics()

with open('metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=2)
```

### Create Custom Charts

```python
from analysis import TradingAnalyzer, PerformanceMetrics, PerformanceVisualizer

analyzer = TradingAnalyzer('your_file.csv')
metrics = PerformanceMetrics(analyzer)
viz = PerformanceVisualizer(analyzer, metrics)

# Individual charts
viz.plot_equity_curve(save_path='equity.png', show=True)
viz.plot_drawdown(save_path='drawdown.png', show=False)
viz.plot_monthly_returns(save_path='monthly.png')

# All charts at once
viz.create_all_charts(output_dir='my_charts/', show=False)
```

## Testing the Library

```bash
# Quick test to verify everything works
python3 test_analysis_library.py
```

## Troubleshooting

### Missing Dependencies

```bash
pip install pandas numpy matplotlib seaborn yfinance scipy
```

### yfinance Data Issues

- Check your internet connection
- Verify ticker symbol is correct
- Some international tickers may need exchange suffix (e.g., `0700.HK`)

### No Trades Found

- Make sure CSV has BUY and SELL decisions (not just BUY or just HOLD)
- Check date range covers actual trading activity

## Advanced Usage

See [analysis/README.md](analysis/README.md) for:

- Complete API reference
- Advanced examples
- Custom metric calculations
- Batch processing
- Report customization

## Questions?

- Check [analysis/README.md](analysis/README.md) for full documentation
- Run `test_analysis_library.py` to verify installation
- See `example_analysis.py` for usage examples
