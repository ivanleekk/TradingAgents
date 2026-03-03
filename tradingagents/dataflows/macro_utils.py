import pandas as pd
import pandas_datareader.data as web
from datetime import datetime, timedelta
import os
from langchain_core.tools import tool

@tool
def get_macro_fundamentals(
    start_date: str,
    end_date: str,
    save_path: str = None
) -> str:
    """
    Retrieves critical macroeconomic fundamental indicators (Interest Rates, Inflation, GDP)
    from the Federal Reserve Economic Data (FRED) database. Used for ETF analysis where
    corporate balance sheets do not exist.

    Args:
        start_date: The start date (YYYY-MM-DD)
        end_date: The end date (YYYY-MM-DD)
    """
    # Fetch 1 year prior to the start date to give the LLM context of the trend
    start = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=365)
    end = datetime.strptime(end_date, "%Y-%m-%d")

    series_dict = {
        'FEDFUNDS': 'Effective Federal Funds Rate (%)',
        'CPIAUCSL': 'Consumer Price Index (Inflation)',
        'GDP': 'Gross Domestic Product (Billions)',
        'UNRATE': 'Unemployment Rate (%)'
    }

    try:
        df = web.DataReader(list(series_dict.keys()), 'fred', start, end)
        df.rename(columns=series_dict, inplace=True)

        # FRED data is often released monthly/quarterly, so we forward fill to the end date
        df = df.ffill().dropna(how='all')

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path)

        # Return as a formatted string for the LLM context
        return df.tail(12).to_markdown()

    except Exception as e:
        return f"Error fetching macroeconomic data from FRED: {str(e)}"

if __name__ == "__main__":
    # Test the wrapped tool directly
    print(get_macro_fundamentals.invoke({"start_date": "2022-01-01", "end_date": "2022-03-01"}))
