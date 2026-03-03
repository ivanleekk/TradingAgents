import yfinance as yf
import pandas as pd
import numpy as np

ETFS = [
    "SPY", "QQQ", "EEM", "TLT", "LQD", "GLD", "USO",
    "PDBC", "UUP", "FXE", "GBTC", "VNQ", "EWS", "FXI",
    "EWU", "EWJ", "EMB", "VNQI"
]

def get_etf_analyst_baseline(ticker):
    """
    Proxy an ETF analyst baseline by getting the top holdings
    and taking a weighted average of their 12-month analyst target returns.
    """
    print(f"Fetching baseline for {ticker}...")
    try:
        etf = yf.Ticker(ticker)
        holdings = etf.info.get('holdings', [])

        if not holdings:
            print(f"No holdings data found for {ticker}")
            # If no holdings data, try to look at historical return as proxy
            hist = etf.history(period="5y")
            if hist.empty:
                return 0.05 # Fallback default

            # Annualized historical return
            annual_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) ** (252 / len(hist)) - 1
            return round(annual_return, 4)

        total_weight = 0
        weighted_return = 0

        # Consider top 10 holdings
        for holding in holdings[:10]:
            sym = holding.get('symbol')
            weight = holding.get('holdingPercent', 0)

            if not sym or weight <= 0:
                continue

            try:
                stock = yf.Ticker(sym)
                info = stock.info
                curr_price = info.get('currentPrice', info.get('regularMarketPrice'))
                target_price = info.get('targetMeanPrice')

                if curr_price and target_price and curr_price > 0:
                    expected_return = (target_price - curr_price) / curr_price
                    weighted_return += expected_return * weight
                    total_weight += weight
            except Exception as e:
                print(f"Error fetching {sym}: {e}")
                continue

        if total_weight > 0:
            final_return = weighted_return / total_weight
            return round(final_return, 4)
        else:
            # Fallback to historical return if holding targets fail
            hist = etf.history(period="5y")
            if not hist.empty:
                annual_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) ** (252 / len(hist)) - 1
                return round(annual_return, 4)
            return 0.05

    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return 0.05 # Fallback default

def download_ff3_factors():
    """Download Fama-French 3-Factor (FF3) daily data from Kenneth French Data Library."""
    print("Downloading Fama-French 3-Factor daily data...")
    import urllib.request
    import zipfile
    import io
    import os

    url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip'
    try:
        r = urllib.request.urlopen(url)
        z = zipfile.ZipFile(io.BytesIO(r.read()))
        df = pd.read_csv(z.open('F-F_Research_Data_Factors_daily.csv'), skiprows=4)
        df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
        df.dropna(inplace=True)

        os.makedirs("data", exist_ok=True)
        df.to_csv('data/ff3_factors.csv', index=False)
        print('Saved FF3 factors to data/ff3_factors.csv')
    except Exception as e:
        print(f"Failed to download FF3 factors: {e}")

def main():
    import os
    os.makedirs("data", exist_ok=True)

    # 1. Download Fama-French data
    download_ff3_factors()

    # 2. Get ETF analyst baselines
    baselines = {}
    for etf in ETFS:
        baseline = get_etf_analyst_baseline(etf)
        baselines[etf] = baseline
        print(f"{etf} Baseline: {baseline:.2%}")

    df = pd.DataFrame(list(baselines.items()), columns=['Ticker', 'Human_Analyst_Baseline_12M_Return'])
    df.to_csv("data/etf_baselines.csv", index=False)
    print("Saved ETF baselines to data/etf_baselines.csv")

if __name__ == "__main__":
    main()
