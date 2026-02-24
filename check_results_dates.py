#!/usr/bin/env python3
"""
Script to check end dates of all CSV files in results_bulk folder.
Reads actual CSV data to determine the last date in each file.
"""

import os
import re
import csv
from datetime import datetime, timedelta
from collections import defaultdict


def parse_csv_filename(filename):
    """
    Parse CSV filename to extract stock symbol.
    Format: {STOCK}_decisions_{START_DATE}_{END_DATE}.csv
    """
    # Match pattern: STOCK_decisions_YYYY-MM-DD_YYYY-MM-DD.csv
    pattern = r"^(.+?)_decisions_\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}\.csv$"
    match = re.match(pattern, filename)

    if match:
        stock = match.group(1)
        return stock
    return None


def get_last_date_from_csv(filepath):
    """
    Read CSV file and return the last date found in the test_date column.
    """
    try:
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            last_row = None
            for row in reader:
                last_row = row

            if last_row and "test_date" in last_row:
                return last_row["test_date"]
    except Exception as e:
        print(f"⚠️  Error reading {filepath}: {e}")
    return None


def main():
    results_dir = "results_bulk"

    if not os.path.exists(results_dir):
        print(f"❌ Directory '{results_dir}' not found!")
        return

    # Get all CSV files
    csv_files = [f for f in os.listdir(results_dir) if f.endswith(".csv")]

    if not csv_files:
        print(f"❌ No CSV files found in '{results_dir}'")
        return

    print("=" * 80)
    print("RESULTS BULK FOLDER - CSV END DATES (from actual data)")
    print("=" * 80)
    print()

    # Store latest result for each stock
    stock_data = defaultdict(list)

    for filename in sorted(csv_files):
        stock = parse_csv_filename(filename)

        if stock:
            filepath = os.path.join(results_dir, filename)
            last_date_str = get_last_date_from_csv(filepath)

            if last_date_str:
                try:
                    last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
                    stock_data[stock].append(
                        {
                            "filename": filename,
                            "end_date": last_date_str,
                            "end_datetime": last_date,
                        }
                    )
                except ValueError:
                    print(f"⚠️  Could not parse date '{last_date_str}' in {filename}")

    # Print summary for each stock
    target_date = datetime.strptime("2024-12-31", "%Y-%m-%d")
    stocks_needing_update = []

    print(f"{'Stock':<15} {'Latest End Date':<15} {'Status':<30}")
    print("-" * 80)

    for stock in sorted(stock_data.keys()):
        # Get the latest end date for this stock
        latest = max(stock_data[stock], key=lambda x: x["end_datetime"])
        end_date = latest["end_date"]
        end_dt = latest["end_datetime"]

        if end_dt < target_date:
            status = f"⚠️  Needs update (gap: {(target_date - end_dt).days} days)"
            stocks_needing_update.append(
                {"stock": stock, "end_date": end_date, "filename": latest["filename"]}
            )
        else:
            status = "✅ Up to date"

        print(f"{stock:<15} {end_date:<15} {status:<30}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total unique stocks: {len(stock_data)}")
    print(f"Stocks up to date: {len(stock_data) - len(stocks_needing_update)}")
    print(f"Stocks needing update: {len(stocks_needing_update)}")
    print()

    if stocks_needing_update:
        print("=" * 80)
        print("STOCKS NEEDING UPDATE TO 2024-12-31")
        print("=" * 80)
        print()

        for item in stocks_needing_update:
            next_date_obj = datetime.strptime(item["end_date"], "%Y-%m-%d")
            # Add one day to start from the next trading day
            next_start_date = (next_date_obj + timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"{item['stock']:<15} Start from: {next_start_date} to 2024-12-31")


if __name__ == "__main__":
    main()
