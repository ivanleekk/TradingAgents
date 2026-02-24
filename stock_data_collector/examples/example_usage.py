"""
Example Usage of Stock Data Collector Library

This script demonstrates how to use the stock data collector library
to gather comprehensive market data for the 2020-2024 period.
"""

from stock_data_collector import StockDataCollector, CollectorConfig
import pandas as pd
from datetime import datetime, timedelta

def example_1_basic_usage():
    """Example 1: Basic usage - get stock price data."""
    print("\n" + "="*70)
    print("Example 1: Basic Stock Price Data")
    print("="*70)
    
    collector = StockDataCollector()
    
    # Get Apple stock data for 2020-2024
    prices = collector.get_price_data('AAPL', '2020-01-01', '2024-12-31')
    
    print(f"\nRetrieved {len(prices)} days of AAPL data")
    print(f"\nFirst few rows:")
    print(prices.head())
    print(f"\nLast few rows:")
    print(prices.tail())
    
    # Calculate some basic statistics
    print(f"\nPrice Statistics (2020-2024):")
    print(f"  Highest Close: ${prices['Close'].max():.2f}")
    print(f"  Lowest Close: ${prices['Close'].min():.2f}")
    print(f"  Average Close: ${prices['Close'].mean():.2f}")
    print(f"  Total Volume: {prices['Volume'].sum():,.0f}")


def example_2_technical_indicators():
    """Example 2: Calculate technical indicators."""
    print("\n" + "="*70)
    print("Example 2: Technical Indicators")
    print("="*70)
    
    collector = StockDataCollector()
    
    # Get indicators for a specific date
    date = '2023-06-15'
    ticker = 'AAPL'
    
    indicators = collector.get_multiple_indicators(
        ticker,
        ['rsi', 'macd', 'close_50_sma', 'close_200_sma', 'boll'],
        date
    )
    
    print(f"\nTechnical Indicators for {ticker} on {date}:")
    for indicator, value in indicators.items():
        print(f"  {indicator}: {value}")
    
    # Get RSI over a period
    print(f"\nCalculating RSI for {ticker} over 2023...")
    rsi_df = collector.get_indicator_range(ticker, 'rsi', '2023-01-01', '2023-12-31')
    
    print(f"Retrieved {len(rsi_df)} days of RSI data")
    print(f"Average RSI in 2023: {rsi_df['rsi'].mean():.2f}")
    print(f"Max RSI in 2023: {rsi_df['rsi'].max():.2f}")
    print(f"Min RSI in 2023: {rsi_df['rsi'].min():.2f}")


def example_3_company_fundamentals():
    """Example 3: Get company information and fundamentals."""
    print("\n" + "="*70)
    print("Example 3: Company Fundamentals")
    print("="*70)
    
    collector = StockDataCollector()
    
    ticker = 'AAPL'
    
    # Get company info
    info = collector.get_company_info(ticker)
    print(f"\nCompany Information:")
    print(info.to_string(index=False))
    
    # Get income statement
    print(f"\nIncome Statement (Annual):")
    income = collector.get_financials(ticker, 'income', quarterly=False)
    if not income.empty:
        print(f"  Available periods: {len(income.columns)}")
        print(f"  Latest period: {income.columns[0]}")
    
    # Get balance sheet
    print(f"\nBalance Sheet (Annual):")
    balance = collector.get_financials(ticker, 'balance', quarterly=False)
    if not balance.empty:
        print(f"  Available periods: {len(balance.columns)}")


def example_4_complete_snapshot():
    """Example 4: Get a complete data snapshot."""
    print("\n" + "="*70)
    print("Example 4: Complete Data Snapshot")
    print("="*70)
    
    collector = StockDataCollector()
    
    ticker = 'TSLA'
    date = '2023-12-15'
    
    print(f"\nGetting complete snapshot for {ticker} on {date}...")
    snapshot = collector.get_complete_snapshot(ticker, date, look_back_days=7)
    
    print(f"\nSnapshot Contents:")
    print(f"  Ticker: {snapshot['ticker']}")
    print(f"  Date: {snapshot['date']}")
    print(f"  Company Info: {len(snapshot['company_info'])} row")
    print(f"  Price Data: {len(snapshot['price_data'])} days")
    print(f"  Indicators: {len(snapshot['indicators'])} calculated")
    print(f"  News Articles: {len(snapshot['news'])}")
    print(f"  Insider Sentiment: {len(snapshot['insider_sentiment'])} entries")
    
    if snapshot['indicators']:
        print(f"\n  Indicator Values:")
        for ind, val in snapshot['indicators'].items():
            print(f"    {ind}: {val}")


def example_5_multi_stock_comparison():
    """Example 5: Compare multiple stocks."""
    print("\n" + "="*70)
    print("Example 5: Multi-Stock Comparison")
    print("="*70)
    
    collector = StockDataCollector()
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    date = '2024-01-15'
    
    print(f"\nComparing stocks on {date}...")
    
    results = []
    for symbol in symbols:
        try:
            # Get indicators
            indicators = collector.get_multiple_indicators(
                symbol,
                ['rsi', 'close_50_sma', 'close_200_sma'],
                date
            )
            
            # Get recent price
            price_data = collector.get_price_data(symbol, date, date)
            current_price = price_data['Close'].iloc[0] if not price_data.empty else None
            
            # Determine trend
            sma50 = indicators.get('close_50_sma')
            sma200 = indicators.get('close_200_sma')
            trend = "Bullish" if (sma50 and sma200 and sma50 > sma200) else "Bearish"
            
            results.append({
                'Symbol': symbol,
                'Price': f"${current_price:.2f}" if current_price else "N/A",
                'RSI': f"{indicators.get('rsi', 'N/A'):.2f}" if indicators.get('rsi') != 'N/A' else 'N/A',
                'SMA50': f"{sma50:.2f}" if sma50 else "N/A",
                'SMA200': f"{sma200:.2f}" if sma200 else "N/A",
                'Trend': trend
            })
        except Exception as e:
            print(f"  Error processing {symbol}: {e}")
    
    df = pd.DataFrame(results)
    print("\nStock Comparison:")
    print(df.to_string(index=False))


def example_6_historical_analysis():
    """Example 6: Historical trend analysis."""
    print("\n" + "="*70)
    print("Example 6: Historical Trend Analysis (2020-2024)")
    print("="*70)
    
    collector = StockDataCollector()
    
    ticker = 'SPY'  # S&P 500 ETF
    
    print(f"\nAnalyzing {ticker} from 2020 to 2024...")
    
    # Get price data
    prices = collector.get_price_data(ticker, '2020-01-01', '2024-12-31')
    
    # Calculate yearly statistics
    prices['Date'] = pd.to_datetime(prices['Date'])
    prices['Year'] = prices['Date'].dt.year
    
    print(f"\nYearly Performance:")
    for year in range(2020, 2025):
        year_data = prices[prices['Year'] == year]
        if not year_data.empty:
            first_close = year_data.iloc[0]['Close']
            last_close = year_data.iloc[-1]['Close']
            yearly_return = ((last_close - first_close) / first_close) * 100
            
            print(f"  {year}: {yearly_return:+.2f}% "
                  f"(${first_close:.2f} → ${last_close:.2f})")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Stock Data Collector Library - Example Usage")
    print("Testing with 2020-2024 Data")
    print("="*70)
    
    try:
        example_1_basic_usage()
        example_2_technical_indicators()
        example_3_company_fundamentals()
        example_4_complete_snapshot()
        example_5_multi_stock_comparison()
        example_6_historical_analysis()
        
        print("\n" + "="*70)
        print("✓ All examples completed successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
