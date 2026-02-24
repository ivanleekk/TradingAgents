#!/usr/bin/env python3
"""
Script to merge CSV files for each ticker into a single file covering 2020-01-01 to 2024-12-31.
Handles overlapping date ranges and removes duplicates.
"""

import os
import csv
import re
from datetime import datetime
from collections import defaultdict


def parse_csv_filename(filename):
    """
    Parse CSV filename to extract stock symbol.
    Format: {STOCK}_decisions_{START_DATE}_{END_DATE}.csv
    """
    pattern = r"^(.+?)_decisions_\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}\.csv$"
    match = re.match(pattern, filename)

    if match:
        return match.group(1)
    return None


def read_csv_data(filepath):
    """
    Read CSV file and return list of rows as dictionaries.
    """
    data = []
    try:
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
    except Exception as e:
        print(f"⚠️  Error reading {filepath}: {e}")
    return data


def merge_ticker_data(ticker, files, results_dir):
    """
    Merge all CSV files for a ticker, removing duplicates and sorting by date.
    Normalizes column names to ensure consistency.
    """
    all_data = {}  # Use dict with date as key to automatically handle duplicates

    print(f"  Merging {len(files)} file(s) for {ticker}...")

    for filename in files:
        filepath = os.path.join(results_dir, filename)
        data = read_csv_data(filepath)

        for row in data:
            test_date = row["test_date"]
            # Normalize row to only include test_date and decision
            normalized_row = {
                "test_date": row["test_date"],
                "decision": row["decision"],
            }
            # Keep the data, later date file will overwrite if there are duplicates
            all_data[test_date] = normalized_row

    # Sort by date
    sorted_data = sorted(all_data.values(), key=lambda x: x["test_date"])

    if sorted_data:
        start_date = sorted_data[0]["test_date"]
        end_date = sorted_data[-1]["test_date"]
        print(f"    ✓ {len(sorted_data)} total rows ({start_date} to {end_date})")

    return sorted_data


def write_merged_csv(ticker, data, output_dir):
    """
    Write merged data to a new CSV file.
    """
    if not data:
        print(f"    ⚠️  No data to write for {ticker}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Determine date range from actual data
    start_date = data[0]["test_date"]
    end_date = data[-1]["test_date"]

    output_filename = f"{ticker}_decisions_{start_date}_{end_date}.csv"
    output_path = os.path.join(output_dir, output_filename)

    try:
        with open(output_path, "w", newline="") as f:
            # Get fieldnames from first row
            fieldnames = list(data[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(data)

        print(f"    ✓ Written to {output_filename}")
        return output_path
    except Exception as e:
        print(f"    ✗ Error writing {output_filename}: {e}")
        return None


def main():
    results_dir = "results_bulk"
    output_dir = "results_merged"

    if not os.path.exists(results_dir):
        print(f"❌ Directory '{results_dir}' not found!")
        return

    # Get all CSV files
    csv_files = [f for f in os.listdir(results_dir) if f.endswith(".csv")]

    if not csv_files:
        print(f"❌ No CSV files found in '{results_dir}'")
        return

    print("=" * 80)
    print("MERGING CSV FILES BY TICKER")
    print("=" * 80)
    print()

    # Group files by ticker
    ticker_files = defaultdict(list)

    for filename in csv_files:
        ticker = parse_csv_filename(filename)
        if ticker:
            ticker_files[ticker].append(filename)
        else:
            print(f"⚠️  Could not parse ticker from: {filename}")

    print(f"Found {len(ticker_files)} unique tickers")
    print()

    # Process each ticker
    merged_count = 0

    for ticker in sorted(ticker_files.keys()):
        files = sorted(ticker_files[ticker])

        if len(files) == 1:
            print(f"📄 {ticker}: Only 1 file, copying...")
            # Just copy the single file to merged directory
            data = read_csv_data(os.path.join(results_dir, files[0]))
            if write_merged_csv(ticker, data, output_dir):
                merged_count += 1
        else:
            print(f"🔗 {ticker}: Merging {len(files)} files")
            merged_data = merge_ticker_data(ticker, files, results_dir)
            if write_merged_csv(ticker, merged_data, output_dir):
                merged_count += 1

        print()

    print("=" * 80)
    print("MERGE COMPLETE")
    print("=" * 80)
    print(f"Successfully processed {merged_count}/{len(ticker_files)} tickers")
    print(f"Merged files saved to: {output_dir}/")


if __name__ == "__main__":
    main()
