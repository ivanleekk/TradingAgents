import yfinance as yf
import pandas as pd
from rich.console import Console
from rich.table import Table

EQUITY_ASSETS = [
    "0P0001AF7U.SI",
    "SPY",
    "^990100-USD-STRD",
    "DE000SLA4YD9.SG",
    "0P0001AF7Z.SI",
    "0P0001EF2T.SI",
    "EIMI.L",
]

FIXED_INCOME_ASSETS = [
    "0P0000KYEE.SI",
    "AGGG.L",
    "IE0002461055.IR",
    "0P0001EQUE.SI",
    "0P0001CC3M",
    "PEBIX",
    "0P0001DWI0.SI",
]

def check_tickers_exactly_like_user(tickers, category_name):
    console = Console()
    table = Table(title=f"Starting Dates for {category_name}")
    table.add_column("Ticker", style="cyan")
    table.add_column("Start Date", style="magenta")
    table.add_column("Status", style="green")

    # Use the EXACT parameters from the user's baseline script
    start = "1990-01-01"
    end = "2024-12-31"

    print(f"Downloading {category_name}...")
    
    # We loop one by one but use the working parameters (string ticker, specific dates)
    for ticker_symbol in tickers:
        try:
            raw = yf.download(
                ticker_symbol, # Pass as string
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
            )
            
            if not raw.empty:
                start_date = raw.index[0].strftime('%Y-%m-%d')
                table.add_row(ticker_symbol, start_date, "Success")
            else:
                table.add_row(ticker_symbol, "N/A", "[red]Empty in this range[/red]")
        except Exception as e:
            table.add_row(ticker_symbol, "Error", f"[red]{str(e)}[/red]")

    console.print(table)

if __name__ == "__main__":
    print("Testing with your specific date range (2019-09-01 to 2024-12-31)...")
    check_tickers_exactly_like_user(EQUITY_ASSETS, "Equity Assets")
    check_tickers_exactly_like_user(FIXED_INCOME_ASSETS, "Fixed Income Assets")
