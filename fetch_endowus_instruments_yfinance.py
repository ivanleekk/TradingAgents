#!/usr/bin/env python3
"""Extract Endowus instrument IDs and download mapped Yahoo Finance price data.

Usage examples:
  python fetch_endowus_instruments_yfinance.py
  python fetch_endowus_instruments_yfinance.py --start 2010-01-01 --end 2026-03-01
  python fetch_endowus_instruments_yfinance.py --mapping-file data/endowus_id_to_yf.json

Notes:
- `instrumentId` in Endowus data looks like ISIN. Yahoo Finance downloads by symbol,
  so this script resolves ISIN -> Yahoo symbol first.
- Resolution strategy:
  1) manual mapping override (optional JSON file)
  2) Yahoo search by ISIN
  3) Yahoo search by fund name (fallback)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf


DEFAULT_INPUT = Path("data/endowus_raw.json")
DEFAULT_OUTPUT_DIR = Path("data/endowus_yfinance")
DEFAULT_START = "2007-01-01"
DEFAULT_END = "2026-03-01"


@dataclass(frozen=True)
class InstrumentRecord:
    instrument_id: str
    name: str | None


def extract_instruments(obj: Any) -> list[InstrumentRecord]:
    """Recursively find instrument entries that include `instrumentId`."""
    found: list[InstrumentRecord] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "instrumentId" in node and isinstance(node["instrumentId"], str):
                found.append(
                    InstrumentRecord(
                        instrument_id=node["instrumentId"].strip(),
                        name=(
                            node.get("name")
                            if isinstance(node.get("name"), str)
                            else None
                        ),
                    )
                )
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(obj)

    dedup: dict[str, InstrumentRecord] = {}
    for rec in found:
        if rec.instrument_id not in dedup:
            dedup[rec.instrument_id] = rec
    return sorted(dedup.values(), key=lambda x: x.instrument_id)


def load_manual_mapping(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    if not isinstance(mapping, dict):
        raise ValueError(
            'Mapping file must be a JSON object: {"instrumentId": "YF_SYMBOL"}'
        )
    return {str(k).strip(): str(v).strip() for k, v in mapping.items()}


def _rank_quote(quote: dict[str, Any], query_name: str | None) -> tuple[int, int]:
    """Lower rank is better. Prefer fund-like quote types and better name matches."""
    quote_type = str(quote.get("quoteType", "")).upper()
    type_rank = {
        "ETF": 0,
        "MUTUALFUND": 1,
        "EQUITY": 2,
        "INDEX": 3,
    }.get(quote_type, 5)

    short = str(quote.get("shortname", "")).lower()
    long_name = str(quote.get("longname", "")).lower()
    qn = (query_name or "").lower()
    name_rank = 0 if qn and (qn in short or qn in long_name) else 1

    return (type_rank, name_rank)


def resolve_yahoo_symbol(
    instrument_id: str, instrument_name: str | None
) -> tuple[str | None, str]:
    """Resolve ISIN-like instrument ID into Yahoo symbol via Yahoo Search."""

    def try_query(query: str) -> list[dict[str, Any]]:
        try:
            result = yf.Search(query=query, max_results=10, news_count=0)
            quotes = getattr(result, "quotes", [])
            if isinstance(quotes, list):
                return [q for q in quotes if isinstance(q, dict)]
            return []
        except Exception:
            return []

    quotes = try_query(instrument_id)
    if quotes:
        quotes_sorted = sorted(quotes, key=lambda q: _rank_quote(q, instrument_name))
        symbol = quotes_sorted[0].get("symbol")
        if isinstance(symbol, str) and symbol.strip():
            return symbol.strip(), "search_by_instrument_id"

    if instrument_name:
        name_quotes = try_query(instrument_name)
        if name_quotes:
            name_quotes_sorted = sorted(
                name_quotes, key=lambda q: _rank_quote(q, instrument_name)
            )
            symbol = name_quotes_sorted[0].get("symbol")
            if isinstance(symbol, str) and symbol.strip():
                return symbol.strip(), "search_by_name"

    return None, "unresolved"


def download_symbol_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    data = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if data is None:
        return pd.DataFrame()
    if data.empty:
        return data

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [
            "_".join([str(c) for c in tup if c]).strip("_")
            for tup in data.columns.values
        ]

    data = data.reset_index()
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"]).dt.strftime("%Y-%m-%d")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Endowus instrument IDs and pull Yahoo Finance data."
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT, help="Path to endowus_raw.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory"
    )
    parser.add_argument(
        "--mapping-file",
        type=Path,
        default=None,
        help="Optional JSON mapping of instrumentId to Yahoo symbol",
    )
    parser.add_argument(
        "--start", type=str, default=DEFAULT_START, help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", type=str, default=DEFAULT_END, help="End date (YYYY-MM-DD)"
    )
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    records = extract_instruments(payload)
    manual_map = load_manual_mapping(args.mapping_file)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prices_dir = args.output_dir / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)

    extracted_df = pd.DataFrame(
        [{"instrumentId": r.instrument_id, "name": r.name or ""} for r in records]
    )
    extracted_df.to_csv(args.output_dir / "instrument_ids.csv", index=False)

    mapping_rows: list[dict[str, str]] = []
    for rec in records:
        source = ""
        symbol = None

        if rec.instrument_id in manual_map:
            symbol = manual_map[rec.instrument_id]
            source = "manual_mapping"
        else:
            symbol, source = resolve_yahoo_symbol(rec.instrument_id, rec.name)

        row = {
            "instrumentId": rec.instrument_id,
            "name": rec.name or "",
            "yahooSymbol": symbol or "",
            "mappingSource": source,
            "status": "resolved" if symbol else "unresolved",
        }

        if symbol:
            data = download_symbol_data(symbol, start=args.start, end=args.end)
            if not data.empty:
                out_file = prices_dir / f"{rec.instrument_id}__{symbol}.csv"
                data.to_csv(out_file, index=False)
                row["rowsDownloaded"] = str(len(data))
                row["priceFile"] = str(out_file)
            else:
                row["rowsDownloaded"] = "0"
                row["priceFile"] = ""
                row["status"] = "resolved_but_no_data"
        else:
            row["rowsDownloaded"] = "0"
            row["priceFile"] = ""

        mapping_rows.append(row)
        print(
            f"{rec.instrument_id}: symbol={row['yahooSymbol'] or 'N/A'} "
            f"source={row['mappingSource']} status={row['status']} rows={row['rowsDownloaded']}"
        )

    mapping_df = pd.DataFrame(mapping_rows).sort_values(["status", "instrumentId"])
    mapping_path = args.output_dir / "instrument_id_to_yahoo.csv"
    mapping_df.to_csv(mapping_path, index=False)

    resolved_count = int((mapping_df["status"] == "resolved").sum())
    unresolved_count = int((mapping_df["status"] == "unresolved").sum())
    print("\nDone")
    print(f"Instrument IDs extracted: {len(records)}")
    print(f"Resolved symbols: {resolved_count}")
    print(f"Unresolved symbols: {unresolved_count}")
    print(f"ID list: {args.output_dir / 'instrument_ids.csv'}")
    print(f"Mapping: {mapping_path}")
    print(f"Price files directory: {prices_dir}")


if __name__ == "__main__":
    main()
