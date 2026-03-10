#!/usr/bin/env python3
"""
Identify tickers with missing price data from a wide price CSV.

Expected CSV format:
- Index column is date
- Each other column is a ticker

Examples:
  python identify_missing_price_data.py \
    --csv data/yfinance_close_URTH_SPY_EEM_VPL_BNDW_AGG_EMB_.csv

  python identify_missing_price_data.py \
    --csv data/yfinance_close_URTH_SPY_EEM_VPL_BNDW_AGG_EMB_.csv \
    --start 2020-01-01 --end 2024-12-31 --min-gap-length 3
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd


@dataclass
class Gap:
    start: pd.Timestamp
    end: pd.Timestamp
    length: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find tickers with missing prices and report missing-date gaps."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to wide price CSV (date index + ticker columns).",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Optional start date (YYYY-MM-DD) for analysis window.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Optional end date (YYYY-MM-DD) for analysis window.",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Optional ticker whitelist. If omitted, analyze all columns.",
    )
    parser.add_argument(
        "--min-gap-length",
        type=int,
        default=1,
        help="Only display gaps with at least this many consecutive missing rows.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Maximum number of worst tickers to print in detail.",
    )
    return parser.parse_args()


def _find_missing_gaps(missing_mask: pd.Series, min_gap_length: int) -> list[Gap]:
    """Return contiguous missing ranges from a boolean missing-value mask."""
    if missing_mask.empty or not missing_mask.any():
        return []

    gaps: list[Gap] = []
    in_gap = False
    gap_start = None
    gap_len = 0
    prev_date = None

    for dt, is_missing in missing_mask.items():
        if is_missing:
            if not in_gap:
                in_gap = True
                gap_start = dt
                gap_len = 1
            else:
                gap_len += 1
        else:
            if in_gap:
                if gap_len >= min_gap_length:
                    gaps.append(Gap(start=gap_start, end=prev_date, length=gap_len))
                in_gap = False
                gap_start = None
                gap_len = 0
        prev_date = dt

    if in_gap and gap_start is not None and gap_len >= min_gap_length:
        gaps.append(Gap(start=gap_start, end=prev_date, length=gap_len))

    return gaps


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.csv, index_col=0, parse_dates=True).sort_index()
    if df.empty:
        print("No rows in CSV.")
        return

    if args.start:
        df = df[df.index >= pd.to_datetime(args.start)]
    if args.end:
        df = df[df.index <= pd.to_datetime(args.end)]

    if df.empty:
        print("No rows after applying date filters.")
        return

    tickers = list(df.columns)
    if args.tickers:
        missing_cols = [t for t in args.tickers if t not in df.columns]
        if missing_cols:
            print(f"Warning: requested tickers not found in CSV: {missing_cols}")
        tickers = [t for t in args.tickers if t in df.columns]

    if not tickers:
        print("No ticker columns selected for analysis.")
        return

    sub = df[tickers]
    total_rows = len(sub)

    stats: list[dict] = []
    for ticker in tickers:
        s = sub[ticker]
        missing_mask = s.isna()
        missing_count = int(missing_mask.sum())
        missing_pct = (missing_count / total_rows) * 100
        first_valid = s.first_valid_index()
        last_valid = s.last_valid_index()
        gaps = _find_missing_gaps(missing_mask, args.min_gap_length)

        stats.append(
            {
                "ticker": ticker,
                "missing_count": missing_count,
                "missing_pct": missing_pct,
                "first_valid": first_valid,
                "last_valid": last_valid,
                "gap_count": len(gaps),
                "max_gap": max((g.length for g in gaps), default=0),
                "gaps": gaps,
            }
        )

    stats_sorted = sorted(
        stats,
        key=lambda x: (x["missing_count"], x["max_gap"]),
        reverse=True,
    )

    print("=" * 90)
    print("MISSING PRICE DATA SUMMARY")
    print("=" * 90)
    print(f"CSV: {args.csv}")
    print(
        f"Date range in analysis: {sub.index.min().date()} to {sub.index.max().date()}"
    )
    print(f"Rows: {total_rows}")
    print(f"Tickers analyzed: {len(tickers)}")
    print()

    offenders = [x for x in stats_sorted if x["missing_count"] > 0]
    print(f"Tickers with at least one missing value: {len(offenders)} / {len(tickers)}")
    print()

    if not offenders:
        print("No missing prices detected for selected tickers.")
        return

    print("Top tickers by missing observations")
    print("-" * 90)
    print(
        f"{'Ticker':<18} {'Missing':>10} {'Missing %':>10} {'Gaps':>8} {'Max Gap':>8} {'First Valid':>12} {'Last Valid':>12}"
    )
    print("-" * 90)

    for row in offenders[: args.top]:
        first_valid = (
            row["first_valid"].date().isoformat()
            if row["first_valid"] is not None
            else "None"
        )
        last_valid = (
            row["last_valid"].date().isoformat()
            if row["last_valid"] is not None
            else "None"
        )
        print(
            f"{row['ticker']:<18} {row['missing_count']:>10} {row['missing_pct']:>9.2f}% {row['gap_count']:>8} {row['max_gap']:>8} {first_valid:>12} {last_valid:>12}"
        )

    print()
    print("Detailed missing gaps")
    print("-" * 90)

    for row in offenders[: args.top]:
        if not row["gaps"]:
            continue
        print(
            f"\n{row['ticker']} ({row['missing_count']} missing rows, max gap {row['max_gap']}):"
        )
        for gap in row["gaps"]:
            print(
                f"  {gap.start.date().isoformat()} -> {gap.end.date().isoformat()} ({gap.length} rows)"
            )


if __name__ == "__main__":
    main()
