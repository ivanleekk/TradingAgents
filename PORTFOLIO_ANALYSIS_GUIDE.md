# Portfolio Analysis - Quick Reference

## Overview

The portfolio analysis module allows you to combine multiple trading strategies into a single portfolio with equal or custom weights, and analyze the combined performance with diversification benefits.

## Quick Start

### Option 1: Equal Weight Portfolio

Combine all your strategies with equal weights:

```bash
python3 portfolio_analysis.py equal \
  results_bulk/AAPL_decisions_*.csv \
  results_bulk/MSFT_decisions_*.csv \
  results_bulk/NVDA_decisions_*.csv
```

### Option 2: Custom Weight Portfolio

Create a JSON configuration file (`my_portfolio.json`):

```json
{
    "AAPL": {
        "csv_file": "results_bulk/AAPL_decisions_2020-01-06_2024-12-23.csv",
        "weight": 0.4
    },
    "MSFT": {
        "csv_file": "results_bulk/MSFT_decisions_2020-01-01_2024-12-31.csv",
        "weight": 0.3
    },
    "NVDA": {
        "csv_file": "results_bulk/NVDA_decisions_2020-01-01_2024-12-31.csv",
        "weight": 0.3
    }
}
```

Then run:

```bash
python3 portfolio_analysis.py custom my_portfolio.json
```

### Option 3: All Stocks in Directory

Analyze all CSV files in `results_bulk/` with equal weights:

```bash
python3 portfolio_analysis.py all
```

## Command Line Options

- `--rebalance` - Enable daily rebalancing to target weights (default: buy & hold)
- `--capital NUM` - Set initial capital (default: $10,000)
- `--show-charts` - Display charts during generation
- `--output DIR` - Set custom output directory

### Examples

```bash
# Equal weight with rebalancing and $50,000 capital
python3 portfolio_analysis.py equal results_bulk/*.csv --rebalance --capital 50000

# Custom weights with charts displayed
python3 portfolio_analysis.py custom my_portfolio.json --show-charts

# All stocks with rebalancing
python3 portfolio_analysis.py all --rebalance --output reports/my_portfolio
```

## Python API

### Equal Weight Portfolio

```python
from analysis import PortfolioAnalyzer, PortfolioVisualizer

# Define strategies
strategy_files = {
    'AAPL': 'results_bulk/AAPL_decisions_2020-01-06_2024-12-23.csv',
    'MSFT': 'results_bulk/MSFT_decisions_2020-01-01_2024-12-31.csv',
    'NVDA': 'results_bulk/NVDA_decisions_2020-01-01_2024-12-31.csv'
}

# Create portfolio with equal weights
portfolio = PortfolioAnalyzer(
    strategy_files=strategy_files,
    weights=None,  # None = equal weights
    initial_capital=10000.0,
    rebalance=False  # Buy & hold
)

# Get metrics
metrics = portfolio.get_portfolio_metrics()
print(f"Portfolio Sharpe: {metrics['annualized_sharpe_ratio']:.2f}")

# Get summary
print(portfolio.get_summary())

# Save results
portfolio.save_results('reports/my_portfolio')
```

### Custom Weight Portfolio

```python
from analysis import PortfolioAnalyzer, PortfolioVisualizer

# Define strategies with custom weights
strategy_files = {
    'AAPL': 'results_bulk/AAPL_decisions_2020-01-06_2024-12-23.csv',
    'MSFT': 'results_bulk/MSFT_decisions_2020-01-01_2024-12-31.csv',
    'NVDA': 'results_bulk/NVDA_decisions_2020-01-01_2024-12-31.csv'
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
    rebalance=False
)

# Get individual strategy metrics
individual_df = portfolio.get_individual_metrics()
print(individual_df)

# Get contribution analysis
contribution_df = portfolio.get_contribution_analysis()
print(contribution_df)

# Get correlation matrix
corr_matrix = portfolio.get_correlation_matrix()
print(corr_matrix)
```

### Generate Visualizations

```python
from analysis import PortfolioAnalyzer, PortfolioVisualizer

# Create portfolio
portfolio = PortfolioAnalyzer(strategy_files, weights)

# Create visualizer
visualizer = PortfolioVisualizer(portfolio)

# Generate individual charts
visualizer.plot_portfolio_equity_curve(save_path='portfolio_equity.png')
visualizer.plot_correlation_matrix(save_path='correlation.png')
visualizer.plot_contribution_analysis(save_path='contribution.png')

# Or generate all charts at once
visualizer.create_all_charts(output_dir='reports/portfolio_charts')
```

## Output Files

Portfolio analysis generates:

### Data Files

- `portfolio_metrics.json` - All portfolio metrics
- `individual_strategies.csv` - Metrics for each strategy
- `contribution_analysis.csv` - How each strategy contributes
- `correlation_matrix.csv` - Strategy return correlations
- `portfolio_history.csv` - Daily portfolio values
- `portfolio_summary.txt` - Human-readable summary

### Visualizations

- `portfolio_equity_curve.png` - Portfolio vs individual strategies
- `portfolio_weights_evolution.png` - How weights drift over time
- `portfolio_correlation_matrix.png` - Correlation heatmap
- `portfolio_contribution_analysis.png` - Contribution charts
- `portfolio_drawdown_comparison.png` - Drawdown comparison
- `portfolio_performance_comparison.png` - 4-panel comparison

## Key Metrics

Portfolio analysis calculates:

### Portfolio-Level Metrics

- Annualized Return
- Sharpe Ratio
- Maximum Drawdown
- Volatility
- Calmar Ratio
- All metrics from single-strategy analysis

### Additional Analysis

- **Individual Strategy Metrics** - Performance of each component
- **Contribution Analysis** - How much each strategy contributes
- **Correlation Matrix** - Diversification benefits
- **Weight Evolution** - How weights drift (if not rebalancing)

## Rebalancing Modes

### Buy & Hold (default: `rebalance=False`)

- Allocate initial capital according to weights
- Let weights drift with performance
- More tax efficient
- Shows actual performance

### Daily Rebalancing (`rebalance=True`)

- Rebalance to target weights every day
- Maintains constant weight allocation
- Higher turnover (theoretical)
- Shows pure strategy combination

## Benefits of Portfolio Analysis

1. **Diversification** - Reduce risk through uncorrelated strategies
2. **Improved Sharpe Ratio** - Often better than individual strategies
3. **Reduced Drawdown** - Lower maximum losses
4. **Risk Management** - Understand correlation and concentration
5. **Optimization** - Find optimal weight combinations

## Common Use Cases

### Compare Individual vs Portfolio

```python
# Analyze each strategy individually
from analysis import TradingAnalyzer, PerformanceMetrics

strategies = ['AAPL', 'MSFT', 'NVDA']
for ticker in strategies:
    analyzer = TradingAnalyzer(f'results_bulk/{ticker}_decisions_*.csv')
    metrics = PerformanceMetrics(analyzer)
    print(f"{ticker} Sharpe: {metrics.annualized_sharpe_ratio():.2f}")

# Then analyze as portfolio
portfolio = PortfolioAnalyzer(strategy_files, weights=None)
portfolio_metrics = portfolio.get_portfolio_metrics()
print(f"Portfolio Sharpe: {portfolio_metrics['annualized_sharpe_ratio']:.2f}")
```

### Find Optimal Weights

```python
import itertools
import numpy as np

# Test different weight combinations
strategies = ['AAPL', 'MSFT', 'NVDA']
best_sharpe = 0
best_weights = None

# Test weights in 10% increments
for weights in itertools.product([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], repeat=3):
    if not np.isclose(sum(weights), 1.0):
        continue

    weight_dict = dict(zip(strategies, weights))

    try:
        portfolio = PortfolioAnalyzer(strategy_files, weight_dict)
        metrics = portfolio.get_portfolio_metrics()
        sharpe = metrics['annualized_sharpe_ratio']

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weights = weight_dict
    except:
        continue

print(f"Best Sharpe: {best_sharpe:.2f}")
print(f"Best weights: {best_weights}")
```

### Analyze Sector Portfolios

```python
# Tech sector portfolio
tech_portfolio = PortfolioAnalyzer({
    'AAPL': 'results_bulk/AAPL_*.csv',
    'MSFT': 'results_bulk/MSFT_*.csv',
    'NVDA': 'results_bulk/NVDA_*.csv'
}, weights=None)

# Consumer sector portfolio
consumer_portfolio = PortfolioAnalyzer({
    'KO': 'results_bulk/KO_*.csv',
    'MCD': 'results_bulk/MCD_*.csv',
    'WMT': 'results_bulk/WMT_*.csv'
}, weights=None)

# Compare sectors
tech_metrics = tech_portfolio.get_portfolio_metrics()
consumer_metrics = consumer_portfolio.get_portfolio_metrics()

print(f"Tech Sharpe: {tech_metrics['annualized_sharpe_ratio']:.2f}")
print(f"Consumer Sharpe: {consumer_metrics['annualized_sharpe_ratio']:.2f}")
```

## Tips

1. **Start with equal weights** - Good baseline for comparison
2. **Check correlations** - High correlation = less diversification benefit
3. **Watch drawdowns** - Portfolio should have lower max drawdown than individuals
4. **Consider rebalancing costs** - Daily rebalancing is theoretical (high turnover)
5. **Use contribution analysis** - Identify which strategies add value
6. **Optimize weights** - Test different combinations to maximize Sharpe ratio

## Troubleshooting

### Different Date Ranges

The portfolio analyzer automatically finds the common date range across all strategies. Strategies with different date ranges will be aligned to the overlapping period.

### Weight Validation

Weights must sum to exactly 1.0. The library will raise an error if weights don't sum correctly.

### Missing Data

If a strategy fails to load, the entire portfolio analysis will fail. Check individual strategy CSV files first.

## Full Documentation

See [analysis/README.md](analysis/README.md) for complete API documentation and [ANALYSIS_QUICKSTART.md](ANALYSIS_QUICKSTART.md) for single-strategy analysis.
