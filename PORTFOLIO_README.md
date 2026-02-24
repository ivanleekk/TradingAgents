# Portfolio Analysis - Complete Guide

## Overview

The portfolio analysis module extends the single-strategy analysis library to combine multiple trading strategies into weighted portfolios. This allows you to:

- **Diversify** across multiple strategies to reduce risk
- **Optimize** portfolio weights to maximize risk-adjusted returns
- **Compare** portfolio performance vs individual strategies
- **Analyze** diversification benefits through correlation analysis
- **Test** different rebalancing strategies (buy & hold vs daily rebalancing)

## Installation

Same as single-strategy analysis:

```bash
pip install pandas numpy matplotlib seaborn yfinance scipy
```

## Quick Examples

### Example 1: Equal Weight Portfolio (3 strategies)

```python
from simple_portfolio_api import create_equal_weight_portfolio, generate_portfolio_report

# Create equal-weight portfolio
csv_files = [
    'results_bulk/AAPL_decisions_2020-01-06_2024-12-23.csv',
    'results_bulk/MSFT_decisions_2020-01-01_2024-12-31.csv',
    'results_bulk/NVDA_decisions_2020-01-01_2024-12-31.csv'
]

portfolio = create_equal_weight_portfolio(csv_files, initial_capital=10000)

# Get portfolio metrics
metrics = portfolio.get_portfolio_metrics()
print(f"Portfolio Sharpe Ratio: {metrics['annualized_sharpe_ratio']:.2f}")
print(f"Portfolio Return: {metrics['annualized_return']:.2f}%")
print(f"Portfolio Max Drawdown: {metrics['maximum_drawdown']:.2f}%")

# Generate full report
generate_portfolio_report(portfolio, output_dir='reports/my_portfolio')
```

### Example 2: Custom Weight Portfolio

```python
from simple_portfolio_api import create_custom_weight_portfolio

# Define custom weights (must sum to 1.0)
strategy_weights = {
    'AAPL': ('results_bulk/AAPL_decisions_*.csv', 0.5),   # 50%
    'MSFT': ('results_bulk/MSFT_decisions_*.csv', 0.3),   # 30%
    'NVDA': ('results_bulk/NVDA_decisions_*.csv', 0.2)    # 20%
}

portfolio = create_custom_weight_portfolio(strategy_weights)

# View contribution analysis
contribution = portfolio.get_contribution_analysis()
print(contribution)
```

### Example 3: Command Line

```bash
# Equal weight portfolio
python3 portfolio_analysis.py equal \
  results_bulk/AAPL_decisions_*.csv \
  results_bulk/MSFT_decisions_*.csv \
  results_bulk/NVDA_decisions_*.csv

# Custom weight portfolio (using JSON config)
python3 portfolio_analysis.py custom portfolio_config_example.json

# All stocks in directory
python3 portfolio_analysis.py all --rebalance --capital 50000
```

## Key Features

### 1. Portfolio Metrics

All metrics from single-strategy analysis, plus:

- **Number of Strategies**: How many strategies in portfolio
- **Weighted Performance**: Combined performance of all strategies
- **Rebalancing Mode**: Buy & hold vs daily rebalancing

### 2. Individual Strategy Analysis

For each strategy in the portfolio:

- Individual performance metrics
- Portfolio weight
- Contribution to overall returns

### 3. Contribution Analysis

Shows how much each strategy contributes:

- Absolute contribution to returns
- Percentage share of total returns
- Comparison to target weights

### 4. Correlation Matrix

Returns correlation between strategies:

- High correlation (>0.7): Less diversification benefit
- Low correlation (<0.3): Good diversification
- Negative correlation (<0): Excellent diversification

### 5. Weight Evolution

Shows how portfolio weights drift over time:

- Relevant for buy & hold portfolios
- Shows which strategies outperform
- Helps identify rebalancing opportunities

## Output Files

Portfolio analysis generates:

```
reports/portfolio_name/
├── portfolio_metrics.json                    # All portfolio metrics
├── individual_strategies.csv                 # Each strategy's metrics
├── contribution_analysis.csv                 # Contribution breakdown
├── correlation_matrix.csv                    # Strategy correlations
├── portfolio_history.csv                     # Daily portfolio values
├── portfolio_summary.txt                     # Human-readable summary
├── portfolio_equity_curve.png                # Portfolio vs strategies chart
├── portfolio_weights_evolution.png           # Weight drift over time
├── portfolio_correlation_matrix.png          # Correlation heatmap
├── portfolio_contribution_analysis.png       # Contribution charts
├── portfolio_drawdown_comparison.png         # Drawdown comparison
└── portfolio_performance_comparison.png      # 4-panel comparison
```

## Rebalancing Strategies

### Buy & Hold (default)

```python
portfolio = PortfolioAnalyzer(
    strategy_files=strategy_files,
    weights=weights,
    rebalance=False  # Buy & hold
)
```

- Allocate capital once at start
- Let weights drift with performance
- Lower turnover (more realistic)
- Tax efficient

### Daily Rebalancing

```python
portfolio = PortfolioAnalyzer(
    strategy_files=strategy_files,
    weights=weights,
    rebalance=True  # Daily rebalancing
)
```

- Rebalance to target weights every day
- Maintains constant allocation
- Higher turnover (theoretical)
- Good for testing pure strategy combination

## Advanced Usage

### Find Optimal Weights

```python
from simple_portfolio_api import find_optimal_weights

strategy_files = {
    'AAPL': 'results_bulk/AAPL_*.csv',
    'MSFT': 'results_bulk/MSFT_*.csv',
    'NVDA': 'results_bulk/NVDA_*.csv'
}

# Find weights that maximize Sharpe ratio
optimal = find_optimal_weights(
    strategy_files=strategy_files,
    weight_increments=0.1,  # Test 0%, 10%, 20%, ..., 100%
    metric='sharpe'  # or 'calmar' or 'return'
)

print(f"Optimal Sharpe: {optimal['sharpe']:.2f}")
print(f"Optimal Weights: {optimal['weights']}")

# Use optimal weights
best_portfolio = PortfolioAnalyzer(
    strategy_files=strategy_files,
    weights=optimal['weights']
)
```

### Compare Portfolio Configurations

```python
from simple_portfolio_api import compare_portfolios

# Create different portfolios
equal_weight = create_equal_weight_portfolio(csv_files)
custom_weight = create_custom_weight_portfolio(strategy_weights)
tech_heavy = create_custom_weight_portfolio({
    'AAPL': ('aapl.csv', 0.4),
    'MSFT': ('msft.csv', 0.4),
    'NVDA': ('nvda.csv', 0.2)
})

# Compare them
portfolios = {
    'Equal Weight': equal_weight,
    'Custom Weight': custom_weight,
    'Tech Heavy': tech_heavy
}

comparison = compare_portfolios(portfolios)
print(comparison.sort_values('Sharpe Ratio', ascending=False))
```

### Analyze Sector Portfolios

```python
# Tech sector
tech_portfolio = PortfolioAnalyzer({
    'AAPL': 'results_bulk/AAPL_*.csv',
    'MSFT': 'results_bulk/MSFT_*.csv',
    'NVDA': 'results_bulk/NVDA_*.csv',
    'META': 'results_bulk/META_*.csv'
})

# Consumer sector
consumer_portfolio = PortfolioAnalyzer({
    'KO': 'results_bulk/KO_*.csv',
    'MCD': 'results_bulk/MCD_*.csv',
    'WMT': 'results_bulk/WMT_*.csv',
    'COST': 'results_bulk/COST_*.csv'
})

# Compare sectors
tech_metrics = tech_portfolio.get_portfolio_metrics()
consumer_metrics = consumer_portfolio.get_portfolio_metrics()

print(f"Tech Sector:")
print(f"  Sharpe: {tech_metrics['annualized_sharpe_ratio']:.2f}")
print(f"  Return: {tech_metrics['annualized_return']:.2f}%")

print(f"\nConsumer Sector:")
print(f"  Sharpe: {consumer_metrics['annualized_sharpe_ratio']:.2f}")
print(f"  Return: {consumer_metrics['annualized_return']:.2f}%")
```

### Risk Parity Portfolio

```python
# Weight by inverse volatility (risk parity approach)
from analysis import TradingAnalyzer, PerformanceMetrics

strategy_files = {
    'AAPL': 'results_bulk/AAPL_*.csv',
    'MSFT': 'results_bulk/MSFT_*.csv',
    'NVDA': 'results_bulk/NVDA_*.csv'
}

# Calculate individual volatilities
vols = {}
for ticker, csv_file in strategy_files.items():
    analyzer = TradingAnalyzer(csv_file)
    metrics = PerformanceMetrics(analyzer)
    vols[ticker] = metrics.annualized_volatility()

# Inverse volatility weights
inv_vols = {ticker: 1/vol for ticker, vol in vols.items()}
total_inv_vol = sum(inv_vols.values())
risk_parity_weights = {ticker: inv_vol/total_inv_vol
                       for ticker, inv_vol in inv_vols.items()}

print("Risk Parity Weights:")
for ticker, weight in risk_parity_weights.items():
    print(f"  {ticker}: {weight*100:.1f}%")

# Create risk parity portfolio
rp_portfolio = PortfolioAnalyzer(
    strategy_files=strategy_files,
    weights=risk_parity_weights
)
```

## JSON Configuration

Create a JSON file for custom portfolios:

```json
{
    "AAPL": {
        "csv_file": "results_bulk/AAPL_decisions_2020-01-06_2024-12-23.csv",
        "weight": 0.3
    },
    "MSFT": {
        "csv_file": "results_bulk/MSFT_decisions_2020-01-01_2024-12-31.csv",
        "weight": 0.3
    },
    "NVDA": {
        "csv_file": "results_bulk/NVDA_decisions_2020-01-01_2024-12-31.csv",
        "weight": 0.2
    },
    "META": {
        "csv_file": "results_bulk/META_decisions_2020-01-06_2024-12-23.csv",
        "weight": 0.2
    }
}
```

Then use:

```bash
python3 portfolio_analysis.py custom my_portfolio.json
```

## Interpretation Guide

### Diversification Benefits

A well-diversified portfolio should show:

- **Higher Sharpe Ratio** than individual strategies
- **Lower Maximum Drawdown** than individual strategies
- **Lower Volatility** while maintaining returns
- **Low correlation** between strategies (<0.5)

### Weight Contribution

If a strategy contributes more/less than its weight:

- **Over-contributing**: Strategy outperforming
- **Under-contributing**: Strategy underperforming
- **Negative contribution**: Strategy losing money

### Rebalancing Decision

Choose buy & hold when:

- Transaction costs are significant
- Tax considerations matter
- Testing actual implementation

Choose daily rebalancing when:

- Testing pure strategy combination
- Transaction costs negligible (theoretical)
- Analyzing optimal constant allocation

## Common Patterns

### Balanced Portfolio (Equal Weight)

```python
# Simple, no bias
portfolio = create_equal_weight_portfolio(csv_files)
```

### Quality-Weighted (Custom Sharpe-based)

```python
# Weight by historical Sharpe ratio
sharpe_weights = calculate_sharpe_weights(csv_files)
portfolio = create_custom_weight_portfolio(sharpe_weights)
```

### Core-Satellite (80/20)

```python
# 80% in core strategies, 20% in satellite
weights = {
    'CORE1': 0.4,
    'CORE2': 0.4,
    'SATELLITE1': 0.1,
    'SATELLITE2': 0.1
}
```

### Minimum Variance (Low Risk)

```python
# Use optimization to find minimum variance portfolio
from scipy.optimize import minimize

# Define optimization function...
# (see advanced optimization examples)
```

## FAQ

**Q: How many strategies should I combine?**  
A: Start with 3-5. More strategies can increase diversification but requires more data and computation.

**Q: What if strategies have different date ranges?**  
A: The library automatically finds the overlapping period. Strategies are aligned to common dates.

**Q: Should weights sum to exactly 1.0?**  
A: Yes, the library validates this and will raise an error if not.

**Q: Which is better: equal weight or optimized weights?**  
A: Start with equal weight as a baseline. Optimize if you want to maximize specific metrics, but beware of overfitting.

**Q: How do I know if diversification is working?**  
A: Check: (1) Portfolio Sharpe > individual Sharpes, (2) Low correlations (<0.5), (3) Lower portfolio drawdown

**Q: Can I combine strategies with different frequencies?**  
A: All strategies should use the same frequency (e.g., weekly decisions). The library assumes consistent timing.

## Performance Tips

1. **Start simple**: Begin with equal-weight portfolios
2. **Check correlations**: High correlation = less benefit from combining
3. **Monitor contributions**: Identify which strategies add value
4. **Test rebalancing**: Compare buy & hold vs rebalancing
5. **Optimize carefully**: Don't overfit to historical data
6. **Diversify**: Combine strategies with different characteristics

## Complete Example

```python
from analysis import PortfolioAnalyzer, PortfolioVisualizer
from simple_portfolio_api import (
    create_equal_weight_portfolio,
    create_custom_weight_portfolio,
    compare_portfolios,
    generate_portfolio_report
)

# 1. Create equal weight portfolio
equal_portfolio = create_equal_weight_portfolio([
    'results_bulk/AAPL_decisions_*.csv',
    'results_bulk/MSFT_decisions_*.csv',
    'results_bulk/NVDA_decisions_*.csv'
])

# 2. Create custom weight portfolio
custom_weights = {
    'AAPL': ('results_bulk/AAPL_*.csv', 0.5),
    'MSFT': ('results_bulk/MSFT_*.csv', 0.3),
    'NVDA': ('results_bulk/NVDA_*.csv', 0.2)
}
custom_portfolio = create_custom_weight_portfolio(custom_weights)

# 3. Compare configurations
comparison = compare_portfolios({
    'Equal Weight': equal_portfolio,
    'Custom Weight': custom_portfolio
})
print(comparison)

# 4. Analyze best portfolio
best = custom_portfolio  # Based on comparison

print(best.get_summary())

# View correlation
print("\nCorrelation Matrix:")
print(best.get_correlation_matrix())

# View contribution
print("\nContribution Analysis:")
print(best.get_contribution_analysis())

# 5. Generate full report
generate_portfolio_report(best, output_dir='reports/final_portfolio')

print("\n✓ Analysis complete! Check reports/final_portfolio/")
```

## Related Documentation

- **Single Strategy Analysis**: [ANALYSIS_QUICKSTART.md](ANALYSIS_QUICKSTART.md)
- **Complete API Reference**: [analysis/README.md](analysis/README.md)
- **Portfolio Quick Reference**: [PORTFOLIO_ANALYSIS_GUIDE.md](PORTFOLIO_ANALYSIS_GUIDE.md)

## Support

For questions or issues:

1. Check the example files: `portfolio_analysis.py`, `simple_portfolio_api.py`
2. Review the configuration example: `portfolio_config_example.json`
3. See the visualization module: `analysis/portfolio_visualizer.py`
