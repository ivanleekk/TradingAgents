# Trading Strategy Analysis Library - Complete Suite

A comprehensive Python library for analyzing trading strategies with support for both **individual strategy analysis** and **portfolio analysis** with customizable weights.

## 🎯 What Can You Do?

### Individual Strategy Analysis

- Calculate 17+ performance metrics (Sharpe, Calmar, Win Rate, etc.)
- Generate 6 professional visualizations
- Create comprehensive reports with all outputs
- Compare multiple strategies

### Portfolio Analysis (New!)

- Combine multiple strategies with equal or custom weights
- Analyze diversification benefits
- Test rebalancing strategies
- Find optimal weight allocations
- Compare portfolio configurations

## 📦 Installation

```bash
pip install pandas numpy matplotlib seaborn yfinance scipy
```

Or:

```bash
pip install -r analysis/requirements.txt
```

## 🚀 Quick Start

### Analyze a Single Strategy

```bash
python3 example_analysis.py results_bulk/AAPL_decisions_2020-01-06_2024-12-23.csv
```

Or in Python:

```python
from analysis import TradingAnalyzer, PerformanceMetrics, ReportGenerator

analyzer = TradingAnalyzer('results_bulk/AAPL_decisions_*.csv')
metrics = PerformanceMetrics(analyzer)

print(metrics.get_metrics_summary())

# Generate full report
from analysis import PerformanceVisualizer
visualizer = PerformanceVisualizer(analyzer, metrics)
report = ReportGenerator(analyzer, metrics, visualizer)
report.generate_full_report(output_dir='reports/AAPL')
```

### Analyze a Portfolio

```bash
# Equal weight portfolio
python3 portfolio_analysis.py equal \
  results_bulk/AAPL_*.csv \
  results_bulk/MSFT_*.csv \
  results_bulk/NVDA_*.csv

# All stocks with rebalancing
python3 portfolio_analysis.py all --rebalance --capital 50000
```

Or in Python:

```python
from simple_portfolio_api import create_equal_weight_portfolio, generate_portfolio_report

portfolio = create_equal_weight_portfolio([
    'results_bulk/AAPL_*.csv',
    'results_bulk/MSFT_*.csv',
    'results_bulk/NVDA_*.csv'
], initial_capital=10000)

print(portfolio.get_summary())
generate_portfolio_report(portfolio, 'reports/my_portfolio')
```

## 📊 Key Metrics Calculated

### Risk-Adjusted Returns

- ✅ **Annualized Return (%)** - CAGR
- ✅ **Sharpe Ratio** - Risk-adjusted returns (>1.0 good, >2.0 excellent, >3.0 exceptional)
- ✅ **Sortino Ratio** - Downside risk-adjusted returns
- ✅ **Calmar Ratio** - Return per unit of max drawdown

### Risk Metrics

- ✅ **Maximum Drawdown (%)** - Worst peak-to-trough decline
- ✅ **Annualized Volatility** - Standard deviation of returns

### Trade Statistics

- ✅ **Win Rate (%)** - Percentage of profitable trades
- ✅ **Profit Factor** - Gross profits / gross losses
- ✅ **Average Win/Loss** - Mean trade P&L
- ✅ **Total Trades** - Number of completed trades

### Portfolio-Specific (New!)

- ✅ **Contribution Analysis** - How much each strategy contributes
- ✅ **Correlation Matrix** - Diversification benefits
- ✅ **Weight Evolution** - How weights drift over time

## 📁 Project Structure

```
TradingAgents/
├── analysis/                                  # Core library
│   ├── analyzer.py                           # Single-strategy analyzer
│   ├── metrics.py                            # Performance metrics
│   ├── visualizer.py                         # Single-strategy charts
│   ├── report.py                             # Report generator
│   ├── portfolio.py                          # Portfolio analyzer (NEW)
│   ├── portfolio_visualizer.py               # Portfolio charts (NEW)
│   └── README.md                             # Full API docs
│
├── example_analysis.py                       # Single-strategy CLI
├── simple_api.py                             # Simple function API
├── portfolio_analysis.py                     # Portfolio CLI (NEW)
├── simple_portfolio_api.py                   # Portfolio function API (NEW)
│
├── analysis_example.ipynb                    # Jupyter notebook
├── test_analysis_library.py                  # Test script
├── portfolio_config_example.json             # Example config (NEW)
│
├── ANALYSIS_QUICKSTART.md                    # Single-strategy quick start
├── PORTFOLIO_ANALYSIS_GUIDE.md               # Portfolio quick start (NEW)
├── PORTFOLIO_README.md                       # Portfolio complete guide (NEW)
└── results_bulk/                             # Your trading decisions
    ├── AAPL_decisions_*.csv
    ├── MSFT_decisions_*.csv
    └── ...
```

## 📈 Visualizations Generated

### Single Strategy (6 charts)

1. Equity curve (strategy vs buy & hold)
2. Drawdown over time
3. Monthly returns heatmap
4. Trade distribution
5. Returns distribution with Q-Q plot
6. Rolling Sharpe ratio and volatility

### Portfolio (6 charts)

1. Portfolio equity curve with all strategies
2. Weight evolution over time
3. Correlation matrix heatmap
4. Contribution analysis
5. Drawdown comparison
6. Performance comparison (4-panel)

All charts are publication-quality (300 DPI).

## 📄 Output Files

### Single Strategy

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

### Portfolio

```
reports/my_portfolio/
├── portfolio_metrics.json
├── individual_strategies.csv
├── contribution_analysis.csv
├── correlation_matrix.csv
├── portfolio_history.csv
├── portfolio_summary.txt
├── portfolio_equity_curve.png
├── portfolio_weights_evolution.png
├── portfolio_correlation_matrix.png
├── portfolio_contribution_analysis.png
├── portfolio_drawdown_comparison.png
└── portfolio_performance_comparison.png
```

## 🎓 Usage Examples

### Example 1: Analyze One Strategy

```python
from simple_api import analyze_strategy, generate_report

# Get metrics
metrics = analyze_strategy('results_bulk/AAPL_decisions_*.csv')
print(f"Sharpe Ratio: {metrics['annualized_sharpe_ratio']:.2f}")

# Generate full report
generate_report('results_bulk/AAPL_decisions_*.csv', 'reports/AAPL')
```

### Example 2: Compare Multiple Strategies

```python
from simple_api import compare_strategies

csv_files = [
    'results_bulk/AAPL_decisions_*.csv',
    'results_bulk/MSFT_decisions_*.csv',
    'results_bulk/NVDA_decisions_*.csv'
]

comparison = compare_strategies(csv_files)
print(comparison.sort_values('Sharpe Ratio', ascending=False))
```

### Example 3: Create Equal Weight Portfolio

```python
from simple_portfolio_api import create_equal_weight_portfolio

portfolio = create_equal_weight_portfolio(csv_files, initial_capital=10000)
print(portfolio.get_summary())
```

### Example 4: Custom Weight Portfolio

```python
from simple_portfolio_api import create_custom_weight_portfolio

strategy_weights = {
    'AAPL': ('results_bulk/AAPL_*.csv', 0.5),
    'MSFT': ('results_bulk/MSFT_*.csv', 0.3),
    'NVDA': ('results_bulk/NVDA_*.csv', 0.2)
}

portfolio = create_custom_weight_portfolio(strategy_weights)
```

### Example 5: Find Optimal Weights

```python
from simple_portfolio_api import find_optimal_weights

strategy_files = {
    'AAPL': 'results_bulk/AAPL_*.csv',
    'MSFT': 'results_bulk/MSFT_*.csv',
    'NVDA': 'results_bulk/NVDA_*.csv'
}

optimal = find_optimal_weights(
    strategy_files=strategy_files,
    weight_increments=0.1,
    metric='sharpe'
)

print(f"Optimal Sharpe: {optimal['sharpe']:.2f}")
print(f"Optimal Weights: {optimal['weights']}")
```

## 📚 Documentation

### Quick Start Guides

- **[ANALYSIS_QUICKSTART.md](ANALYSIS_QUICKSTART.md)** - Single-strategy quick reference
- **[PORTFOLIO_ANALYSIS_GUIDE.md](PORTFOLIO_ANALYSIS_GUIDE.md)** - Portfolio quick reference

### Complete Guides

- **[analysis/README.md](analysis/README.md)** - Complete API reference
- **[PORTFOLIO_README.md](PORTFOLIO_README.md)** - Complete portfolio guide

### Summaries

- **[ANALYSIS_LIBRARY_SUMMARY.md](ANALYSIS_LIBRARY_SUMMARY.md)** - Single-strategy overview
- **[PORTFOLIO_FEATURES_SUMMARY.md](PORTFOLIO_FEATURES_SUMMARY.md)** - Portfolio overview

### Examples

- **[analysis_example.ipynb](analysis_example.ipynb)** - Jupyter notebook tutorial
- **[portfolio_config_example.json](portfolio_config_example.json)** - Portfolio config example

## 🧪 Testing

```bash
# Test the library
python3 test_analysis_library.py

# Should output:
# ✓ Successfully imported all modules
# ✓ Loaded TICKER with X days of data
# ✓ Calculated N metrics
# ✓ ALL TESTS PASSED!
```

## 🎯 Common Workflows

### Workflow 1: Single Strategy Report

```bash
# Analyze and generate full report
python3 example_analysis.py results_bulk/AAPL_decisions_*.csv

# Check output
ls reports/AAPL_*_*/
```

### Workflow 2: Batch Analysis

```bash
# Analyze all strategies
python3 example_analysis.py

# Creates comparison report in reports/
```

### Workflow 3: Portfolio Analysis

```bash
# Create equal weight portfolio
python3 portfolio_analysis.py all --rebalance

# Check output
ls reports/portfolio_all_stocks/
```

### Workflow 4: Custom Portfolio

```bash
# 1. Create config file
cat > my_portfolio.json << EOF
{
  "AAPL": {"csv_file": "results_bulk/AAPL_*.csv", "weight": 0.4},
  "MSFT": {"csv_file": "results_bulk/MSFT_*.csv", "weight": 0.6}
}
EOF

# 2. Analyze
python3 portfolio_analysis.py custom my_portfolio.json

# 3. Check results
cat reports/portfolio_custom_weight/portfolio_summary.txt
```

## 💡 Tips

### For Single Strategy Analysis

1. Start by analyzing individual strategies
2. Check Sharpe ratio (>1.0 good, >2.0 excellent)
3. Look at maximum drawdown (institutional limit ~20-30%)
4. Review trade log for patterns
5. Use monthly returns heatmap to spot seasonality

### For Portfolio Analysis

1. Start with equal weights as baseline
2. Check correlation matrix (low correlation = better diversification)
3. Look for improved Sharpe ratio vs individual strategies
4. Monitor contribution analysis (are all strategies adding value?)
5. Test both buy & hold and rebalancing modes
6. Use optimization carefully (avoid overfitting)

## 🔧 Advanced Features

### Custom Metrics

```python
from analysis import TradingAnalyzer, PerformanceMetrics

analyzer = TradingAnalyzer('your_file.csv')
metrics = PerformanceMetrics(analyzer)

# Get any specific metric
sharpe = metrics.annualized_sharpe_ratio()
calmar = metrics.calmar_ratio()
max_dd = metrics.maximum_drawdown()
```

### Custom Charts

```python
from analysis import PerformanceVisualizer

visualizer = PerformanceVisualizer(analyzer, metrics)

# Individual charts
visualizer.plot_equity_curve(save_path='my_equity.png')
visualizer.plot_drawdown(save_path='my_drawdown.png')

# Or all at once
visualizer.create_all_charts(output_dir='my_charts/')
```

### Portfolio Optimization

```python
from simple_portfolio_api import find_optimal_weights

# Find weights that maximize Sharpe ratio
optimal = find_optimal_weights(
    strategy_files,
    weight_increments=0.1,
    metric='sharpe'
)

# Use optimal weights
from analysis import PortfolioAnalyzer
portfolio = PortfolioAnalyzer(strategy_files, optimal['weights'])
```

## 📊 Input Data Format

CSV files with trading decisions:

```csv
test_date,decision
2020-01-06,BUY
2020-01-13,BUY
2020-01-20,SELL
2020-01-27,BUY
2020-02-03,HOLD
```

**Requirements:**

- Columns: `test_date`, `decision`
- Decisions: `BUY`, `SELL`, or `HOLD`
- Filename: `{TICKER}_decisions_{START_DATE}_{END_DATE}.csv`

## ❓ FAQ

**Q: Can I analyze strategies with different date ranges?**  
A: Yes! Portfolio analysis automatically finds the overlapping period.

**Q: What if I have only BUY decisions (no SELLs)?**  
A: The analyzer treats missing SELLs as HOLD. Metrics will still calculate.

**Q: How do I choose between buy & hold and rebalancing?**  
A: Use buy & hold for realistic results. Use rebalancing to test pure strategy combinations.

**Q: Can I add custom metrics?**  
A: Yes! Extend the `PerformanceMetrics` class or add calculations to your analysis script.

**Q: What's the difference between simple_api.py and the full API?**  
A: `simple_api.py` provides easy functions. Full API gives more control and customization.

## 🎓 Learning Path

1. **Start here**: [ANALYSIS_QUICKSTART.md](ANALYSIS_QUICKSTART.md)
2. **Try it**: `python3 test_analysis_library.py`
3. **Analyze one**: `python3 example_analysis.py results_bulk/AAPL_*.csv`
4. **Portfolio basics**: [PORTFOLIO_ANALYSIS_GUIDE.md](PORTFOLIO_ANALYSIS_GUIDE.md)
5. **Advanced**: [PORTFOLIO_README.md](PORTFOLIO_README.md)
6. **API Reference**: [analysis/README.md](analysis/README.md)

## 📝 License

Part of the TradingAgents project. See LICENSE file.

## 🙏 Acknowledgments

Metrics calculations follow industry standards:

- Sharpe Ratio: William F. Sharpe (Nobel Prize)
- Risk management: CFA Institute guidelines
- Portfolio theory: Harry Markowitz (Modern Portfolio Theory)

---

**Ready to analyze your trading strategies?**

```bash
# Test the library
python3 test_analysis_library.py

# Analyze your first strategy
python3 example_analysis.py results_bulk/YOUR_FILE.csv

# Create your first portfolio
python3 portfolio_analysis.py all
```

**Questions?** Check the documentation files or explore the example scripts!
