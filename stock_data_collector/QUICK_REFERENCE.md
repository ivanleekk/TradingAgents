# Stock Data Collector - Quick Reference

## Installation

```bash
cd stock_data_collector
pip install -r requirements.txt
```

## Basic Usage

```python
from stock_data_collector import StockDataCollector

# Initialize
collector = StockDataCollector()
```

## Common Operations

### Get Price Data
```python
# Get historical prices
prices = collector.get_price_data('AAPL', '2020-01-01', '2024-12-31')

# Get latest price info
info = collector.get_stock_info('AAPL')
```

### Technical Indicators
```python
# Single indicator for one date
rsi = collector.get_indicator('AAPL', 'rsi', '2024-01-15')

# Indicator over a date range
rsi_df = collector.get_indicator_range('AAPL', 'rsi', '2020-01-01', '2024-12-31')

# Multiple indicators at once
indicators = collector.get_multiple_indicators(
    'AAPL',
    ['rsi', 'macd', 'close_50_sma', 'close_200_sma'],
    '2024-01-15'
)

# List all available indicators
all_indicators = collector.list_indicators()
```

### Company Fundamentals
```python
# Company information
company_info = collector.get_company_info('AAPL')

# Financial statements
income = collector.get_financials('AAPL', 'income', quarterly=False)
balance = collector.get_financials('AAPL', 'balance', quarterly=False)
cashflow = collector.get_financials('AAPL', 'cashflow', quarterly=True)
```

### News & Sentiment
```python
# Company news (last 7 days)
news = collector.get_company_news('AAPL', '2024-01-15', look_back_days=7)

# Formatted news summary
news_summary = collector.get_news_summary('AAPL', '2024-01-15', look_back_days=7)

# Insider sentiment
insider = collector.get_insider_sentiment('AAPL', '2024-01-15', look_back_days=30)

# Insider transactions
transactions = collector.get_insider_transactions('AAPL', '2024-01-15', look_back_days=30)
```

### Complete Snapshot
```python
# Get everything at once
snapshot = collector.get_complete_snapshot('AAPL', '2024-01-15', look_back_days=7)

# Access components
print(snapshot['price_data'])
print(snapshot['indicators'])
print(snapshot['news'])
print(snapshot['insider_sentiment'])
```

## Available Indicators

### Momentum
- `rsi` - Relative Strength Index (14-period)
- `rsi_6` - RSI (6-period)
- `macd` - MACD line
- `macds` - MACD signal line
- `macdh` - MACD histogram

### Moving Averages
- `close_10_ema` - 10-day EMA
- `close_20_ema` - 20-day EMA
- `close_50_sma` - 50-day SMA
- `close_200_sma` - 200-day SMA

### Bollinger Bands
- `boll` - Middle band (20-day SMA)
- `boll_ub` - Upper band
- `boll_lb` - Lower band

### Volatility
- `atr` - Average True Range

### Volume
- `vwma` - Volume Weighted Moving Average
- `mfi` - Money Flow Index

### Stochastic
- `kdjk` - KDJ K line
- `kdjd` - KDJ D line
- `kdjj` - KDJ J line

## Configuration

```python
from stock_data_collector import CollectorConfig, StockDataCollector

# Custom configuration
config = CollectorConfig({
    'data_dir': './my_data',
    'cache_dir': './my_cache',
    'finnhub_api_key': 'your_api_key',
})

collector = StockDataCollector(config)
```

## Running Examples

```bash
# Basic usage examples
python stock_data_collector/examples/example_usage.py

# Create backtest datasets
python stock_data_collector/examples/create_backtest_dataset.py
```

## Running Tests

```bash
cd stock_data_collector/tests
python test_collector.py
```

## Common Patterns

### Backtest Dataset Creation
```python
# Get price data
prices = collector.get_price_data(ticker, start_date, end_date)

# Add indicators
rsi = collector.get_indicator_range(ticker, 'rsi', start_date, end_date)
macd = collector.get_indicator_range(ticker, 'macd', start_date, end_date)

# Merge
import pandas as pd
data = prices.merge(rsi, on='Date').merge(macd, on='Date')

# Add features
data['Daily_Return'] = data['Close'].pct_change()
data['Forward_Return'] = data['Close'].shift(-1) / data['Close'] - 1
```

### Multi-Stock Comparison
```python
symbols = ['AAPL', 'MSFT', 'GOOGL']
results = []

for symbol in symbols:
    indicators = collector.get_multiple_indicators(
        symbol,
        ['rsi', 'close_50_sma'],
        '2024-01-15'
    )
    results.append({'Symbol': symbol, **indicators})

df = pd.DataFrame(results)
```

### Daily Signal Generation
```python
def get_trading_signals(ticker, date):
    # Get indicators
    signals = collector.get_multiple_indicators(
        ticker,
        ['rsi', 'macd', 'close_50_sma', 'close_200_sma'],
        date
    )
    
    # Get recent news
    news = collector.get_company_news(ticker, date, look_back_days=3)
    
    # Determine signal
    if signals['rsi'] < 30 and signals['macd'] > 0:
        return 'BUY'
    elif signals['rsi'] > 70 and signals['macd'] < 0:
        return 'SELL'
    else:
        return 'HOLD'
```

## Module Access

### Direct Module Usage
```python
# Access individual collectors directly
from stock_data_collector.yahoo_finance import YahooFinanceCollector
from stock_data_collector.technical_indicators import TechnicalIndicatorCollector

yahoo = YahooFinanceCollector()
indicators = TechnicalIndicatorCollector()

prices = yahoo.get_stock_data('AAPL', '2020-01-01', '2024-12-31')
rsi = indicators.calculate_indicator('AAPL', 'rsi', '2024-01-15')
```

## Troubleshooting

### No data returned
- Check date format: YYYY-MM-DD
- Ensure date is a trading day (not weekend/holiday)
- Verify ticker symbol is correct

### Missing indicators
- Some indicators need sufficient historical data
- Early dates may have NaN values
- Use `dropna()` to remove incomplete rows

### Rate limiting (Google News)
- Built-in retry logic with exponential backoff
- Random delays between requests
- Consider caching results

## Performance Tips

1. **Use caching**: Set `cache_dir` in config
2. **Batch operations**: Use `get_indicator_range()` instead of multiple `get_indicator()` calls
3. **Reuse collector**: Create once, use many times
4. **Filter early**: Request only needed date ranges

## Support

- Check README.md for detailed documentation
- Review examples/ directory for working code
- Run tests/ to verify installation
- See SUMMARY.md for complete feature list
