"""
Example: Create Complete Backtesting Dataset

This example shows how to use the stock data collector library to create
a comprehensive dataset suitable for backtesting trading strategies over
the 2020-2024 period.
"""

from stock_data_collector import StockDataCollector, CollectorConfig
import pandas as pd
from datetime import datetime

def create_backtest_dataset(
    ticker: str,
    start_date: str,
    end_date: str,
    output_file: str = None
):
    """
    Create a complete backtesting dataset with price, indicators, and volume data.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_file: Optional CSV file to save the dataset
        
    Returns:
        DataFrame with all data merged
    """
    print(f"Creating backtest dataset for {ticker} from {start_date} to {end_date}")
    
    collector = StockDataCollector()
    
    # Step 1: Get price data
    print("  Fetching price data...")
    prices = collector.get_price_data(ticker, start_date, end_date)
    print(f"    Retrieved {len(prices)} trading days")
    
    # Step 2: Calculate technical indicators
    indicators_to_calculate = [
        'rsi',
        'rsi_6',
        'macd',
        'macds',
        'macdh',
        'close_10_ema',
        'close_20_ema',
        'close_50_sma',
        'close_200_sma',
        'boll',
        'boll_ub',
        'boll_lb',
        'atr',
        'kdjk',
        'kdjd',
    ]
    
    print(f"  Calculating {len(indicators_to_calculate)} technical indicators...")
    indicator_dfs = []
    
    for indicator in indicators_to_calculate:
        try:
            print(f"    Calculating {indicator}...")
            df = collector.get_indicator_range(ticker, indicator, start_date, end_date)
            if not df.empty:
                indicator_dfs.append(df)
        except Exception as e:
            print(f"    Warning: Could not calculate {indicator}: {e}")
    
    # Step 3: Merge all data
    print("  Merging all data...")
    result = prices.copy()
    
    for df in indicator_dfs:
        result = result.merge(df, on='Date', how='left')
    
    # Step 4: Add derived features
    print("  Adding derived features...")
    
    # Price changes
    result['Daily_Return'] = result['Close'].pct_change()
    result['Price_Change'] = result['Close'].diff()
    
    # Volume analysis
    result['Volume_MA_20'] = result['Volume'].rolling(window=20).mean()
    result['Volume_Ratio'] = result['Volume'] / result['Volume_MA_20']
    
    # Volatility
    result['Volatility_20'] = result['Daily_Return'].rolling(window=20).std()
    
    # High-Low range
    result['Daily_Range'] = result['High'] - result['Low']
    result['Range_Pct'] = (result['Daily_Range'] / result['Close']) * 100
    
    # Step 5: Add labels for supervised learning (optional)
    # Forward-looking returns (for prediction targets)
    result['Forward_1d_Return'] = result['Close'].shift(-1) / result['Close'] - 1
    result['Forward_5d_Return'] = result['Close'].shift(-5) / result['Close'] - 1
    result['Forward_10d_Return'] = result['Close'].shift(-10) / result['Close'] - 1
    
    # Binary labels (up/down)
    result['Forward_1d_Up'] = (result['Forward_1d_Return'] > 0).astype(int)
    result['Forward_5d_Up'] = (result['Forward_5d_Return'] > 0).astype(int)
    
    # Step 6: Clean up
    # Remove rows with NaN in critical columns (early rows due to indicators)
    initial_rows = len(result)
    result = result.dropna(subset=['Close', 'rsi', 'macd'])
    print(f"  Removed {initial_rows - len(result)} rows with missing indicator data")
    
    print(f"\nFinal dataset shape: {result.shape}")
    print(f"  Rows: {len(result)}")
    print(f"  Columns: {len(result.columns)}")
    
    # Step 7: Save to file
    if output_file:
        result.to_csv(output_file, index=False)
        print(f"\n✓ Dataset saved to {output_file}")
    
    return result


def analyze_dataset(df: pd.DataFrame):
    """Print summary statistics of the dataset."""
    print("\n" + "="*70)
    print("Dataset Summary")
    print("="*70)
    
    # Date range
    print(f"\nDate Range:")
    print(f"  Start: {df['Date'].min()}")
    print(f"  End: {df['Date'].max()}")
    print(f"  Trading Days: {len(df)}")
    
    # Price statistics
    print(f"\nPrice Statistics:")
    print(f"  Highest Close: ${df['Close'].max():.2f}")
    print(f"  Lowest Close: ${df['Close'].min():.2f}")
    print(f"  Average Close: ${df['Close'].mean():.2f}")
    print(f"  Std Dev: ${df['Close'].std():.2f}")
    
    # Returns
    print(f"\nDaily Returns:")
    print(f"  Mean: {df['Daily_Return'].mean()*100:.3f}%")
    print(f"  Std Dev: {df['Daily_Return'].std()*100:.3f}%")
    print(f"  Max Gain: {df['Daily_Return'].max()*100:.2f}%")
    print(f"  Max Loss: {df['Daily_Return'].min()*100:.2f}%")
    
    # Indicator statistics
    if 'rsi' in df.columns:
        print(f"\nRSI Statistics:")
        print(f"  Average: {df['rsi'].mean():.2f}")
        print(f"  Overbought days (>70): {(df['rsi'] > 70).sum()}")
        print(f"  Oversold days (<30): {(df['rsi'] < 30).sum()}")
    
    # Missing data
    print(f"\nMissing Data:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("  No missing values")
    
    # Available columns
    print(f"\nAvailable Features ({len(df.columns)}):")
    for col in df.columns:
        print(f"  - {col}")


def main():
    """Main function to create example datasets."""
    print("="*70)
    print("Creating Backtesting Datasets for 2020-2024 Period")
    print("="*70)
    
    # Example 1: Apple (AAPL)
    print("\n--- Dataset 1: AAPL ---")
    aapl_data = create_backtest_dataset(
        'AAPL',
        '2020-01-01',
        '2024-12-31',
        'backtest_AAPL_2020_2024.csv'
    )
    analyze_dataset(aapl_data)
    
    # Example 2: S&P 500 (SPY)
    print("\n\n--- Dataset 2: SPY ---")
    spy_data = create_backtest_dataset(
        'SPY',
        '2020-01-01',
        '2024-12-31',
        'backtest_SPY_2020_2024.csv'
    )
    analyze_dataset(spy_data)
    
    # Example 3: Tesla (TSLA)
    print("\n\n--- Dataset 3: TSLA ---")
    tsla_data = create_backtest_dataset(
        'TSLA',
        '2020-01-01',
        '2024-12-31',
        'backtest_TSLA_2020_2024.csv'
    )
    analyze_dataset(tsla_data)
    
    print("\n" + "="*70)
    print("✓ All datasets created successfully!")
    print("="*70)
    print("\nYou can now use these datasets for:")
    print("  - Backtesting trading strategies")
    print("  - Training machine learning models")
    print("  - Technical analysis research")
    print("  - Performance evaluation")


if __name__ == '__main__':
    main()
