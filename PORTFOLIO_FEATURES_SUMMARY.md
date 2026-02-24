# Portfolio Analysis - Summary

## What's New

Added comprehensive portfolio analysis capabilities to combine multiple trading strategies with equal or custom weights and analyze diversification benefits.

## New Files Created

### Core Portfolio Modules

- `analysis/portfolio.py` - PortfolioAnalyzer class for combining strategies
- `analysis/portfolio_visualizer.py` - Portfolio-specific visualizations
- `analysis/__init__.py` - Updated to include portfolio classes

### Usage Scripts

- `portfolio_analysis.py` - Command-line tool for portfolio analysis
- `simple_portfolio_api.py` - Simple function-based API
- `portfolio_config_example.json` - Example configuration file

### Documentation

- `PORTFOLIO_README.md` - Complete portfolio analysis guide
- `PORTFOLIO_ANALYSIS_GUIDE.md` - Quick reference guide

## Quick Start

### Command Line

```bash
# Equal weight portfolio
python3 portfolio_analysis.py equal \
  results_bulk/AAPL_decisions_*.csv \
  results_bulk/MSFT_decisions_*.csv \
  results_bulk/NVDA_decisions_*.csv

# All stocks with rebalancing
python3 portfolio_analysis.py all --rebalance --capital 50000

# Custom weights from JSON
python3 portfolio_analysis.py custom portfolio_config_example.json
```

### Python API (Simple)

```python
from simple_portfolio_api import create_equal_weight_portfolio, generate_portfolio_report

# Create equal-weight portfolio
csv_files = [
    'results_bulk/AAPL_decisions_*.csv',
    'results_bulk/MSFT_decisions_*.csv',
    'results_bulk/NVDA_decisions_*.csv'
]

portfolio = create_equal_weight_portfolio(csv_files, initial_capital=10000)

# Get metrics
metrics = portfolio.get_portfolio_metrics()
print(f"Portfolio Sharpe: {metrics['annualized_sharpe_ratio']:.2f}")

# Generate full report
generate_portfolio_report(portfolio, output_dir='reports/my_portfolio')
```

### Python API (Full Control)

```python
from analysis import PortfolioAnalyzer, PortfolioVisualizer

# Define strategies with custom weights
strategy_files = {
    'AAPL': 'results_bulk/AAPL_decisions_*.csv',
    'MSFT': 'results_bulk/MSFT_decisions_*.csv',
    'NVDA': 'results_bulk/NVDA_decisions_*.csv'
}

weights = {
    'AAPL': 0.4,  # 40%
    'MSFT': 0.3,  # 30%
    'NVDA': 0.3   # 30%
}

# Create portfolio
portfolio = PortfolioAnalyzer(
    strategy_files=strategy_files,
    weights=weights,
    initial_capital=10000.0,
    rebalance=False  # Buy & hold
)

# Analyze
print(portfolio.get_summary())
print(portfolio.get_correlation_matrix())
print(portfolio.get_contribution_analysis())

# Visualize
visualizer = PortfolioVisualizer(portfolio)
visualizer.create_all_charts(output_dir='reports/portfolio_charts')
```

## Key Features

### Portfolio Metrics

- All single-strategy metrics applied to portfolio
- Individual strategy performance breakdown
- Contribution analysis (how much each strategy contributes)
- Correlation matrix (diversification benefits)
- Weight evolution over time

### Rebalancing Options

- **Buy & Hold** (default): Allocate once, let weights drift
- **Daily Rebalancing**: Maintain constant weights (theoretical)

### Visualizations (6 charts)

1. Portfolio equity curve with all strategies
2. Weight evolution over time
3. Correlation matrix heatmap
4. Contribution analysis charts
5. Drawdown comparison
6. 4-panel performance comparison

### Output Files

- `portfolio_metrics.json` - All metrics
- `individual_strategies.csv` - Each strategy's performance
- `contribution_analysis.csv` - Contribution breakdown
- `correlation_matrix.csv` - Strategy correlations
- `portfolio_history.csv` - Daily portfolio values
- `portfolio_summary.txt` - Human-readable report
- 6 PNG charts (300 DPI)

## Usage Patterns

### Equal Weight Portfolio

```bash
python3 portfolio_analysis.py equal results_bulk/*.csv
```

### Custom Weight Portfolio

Create `my_portfolio.json`:

```json
{
    "AAPL": { "csv_file": "results_bulk/AAPL_*.csv", "weight": 0.5 },
    "MSFT": { "csv_file": "results_bulk/MSFT_*.csv", "weight": 0.5 }
}
```

Then run:

```bash
python3 portfolio_analysis.py custom my_portfolio.json
```

### Find Optimal Weights

```python
from simple_portfolio_api import find_optimal_weights

optimal = find_optimal_weights(
    strategy_files=strategy_files,
    weight_increments=0.1,
    metric='sharpe'
)

print(f"Optimal Sharpe: {optimal['sharpe']:.2f}")
print(f"Optimal Weights: {optimal['weights']}")
```

### Compare Portfolios

```python
from simple_portfolio_api import compare_portfolios

portfolios = {
    'Equal Weight': equal_weight_portfolio,
    'Custom Weight': custom_weight_portfolio,
    'Optimized': optimized_portfolio
}

comparison = compare_portfolios(portfolios)
print(comparison.sort_values('Sharpe Ratio', ascending=False))
```

## Integration with Existing Library

The portfolio analysis seamlessly integrates with existing single-strategy analysis:

```python
from analysis import TradingAnalyzer, PerformanceMetrics, PortfolioAnalyzer

# Analyze individual strategies first
aapl = TradingAnalyzer('results_bulk/AAPL_*.csv')
msft = TradingAnalyzer('results_bulk/MSFT_*.csv')

aapl_metrics = PerformanceMetrics(aapl)
msft_metrics = PerformanceMetrics(msft)

print(f"AAPL Sharpe: {aapl_metrics.annualized_sharpe_ratio():.2f}")
print(f"MSFT Sharpe: {msft_metrics.annualized_sharpe_ratio():.2f}")

# Then combine into portfolio
portfolio = PortfolioAnalyzer({
    'AAPL': 'results_bulk/AAPL_*.csv',
    'MSFT': 'results_bulk/MSFT_*.csv'
})

portfolio_metrics = portfolio.get_portfolio_metrics()
print(f"Portfolio Sharpe: {portfolio_metrics['annualized_sharpe_ratio']:.2f}")
```

## Benefits

1. **Diversification** - Reduce risk by combining uncorrelated strategies
2. **Improved Metrics** - Often achieve better Sharpe ratios
3. **Risk Management** - Lower maximum drawdowns
4. **Flexibility** - Test equal weight, custom weight, or optimized portfolios
5. **Comprehensive Analysis** - Understand contribution and correlation

## Documentation

- **Complete Guide**: [PORTFOLIO_README.md](PORTFOLIO_README.md)
- **Quick Reference**: [PORTFOLIO_ANALYSIS_GUIDE.md](PORTFOLIO_ANALYSIS_GUIDE.md)
- **Single Strategy**: [ANALYSIS_QUICKSTART.md](ANALYSIS_QUICKSTART.md)
- **API Reference**: [analysis/README.md](analysis/README.md)

## Examples

See example configuration: `portfolio_config_example.json`

See example usage:

- `portfolio_analysis.py` - Command-line tool
- `simple_portfolio_api.py` - Function-based API

## Summary

You now have a complete suite for:

1. **Single-strategy analysis** - Analyze individual trading strategies
2. **Portfolio analysis** - Combine strategies with custom weights
3. **Optimization** - Find optimal weight allocations
4. **Comparison** - Compare different portfolio configurations
5. **Visualization** - Professional charts for all analyses
6. **Reporting** - Comprehensive reports with all metrics

All with publication-quality outputs suitable for research papers and professional reports!
