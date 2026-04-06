import yfinance as yf
import pandas as pd

ticker = "^990100-USD-STRD"
start = "2019-09-01"
end = "2024-12-31"

print(f"--- Testing {ticker} with period='max' ---")
try:
    data_max = yf.download(ticker, period="max", progress=False)
    print(f"Max period empty: {data_max.empty}")
    if not data_max.empty:
        print(f"Start: {data_max.index[0]}")
except Exception as e:
    print(f"Max period error: {e}")

print(f"\n--- Testing {ticker} with start/end dates ---")
try:
    data_range = yf.download(ticker, start=start, end=end, progress=False)
    print(f"Range empty: {data_range.empty}")
    if not data_range.empty:
        print(f"Start: {data_range.index[0]}")
except Exception as e:
    print(f"Range error: {e}")
