import os
import sys
import csv
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
import time
import concurrent.futures
from tenacity import retry, wait_exponential, stop_after_attempt

from dotenv import load_dotenv
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# Ensure output directories exist
os.makedirs("results_endowus_A", exist_ok=True)
os.makedirs("results_endowus_B", exist_ok=True)
os.makedirs("results_endowus_C", exist_ok=True)
os.makedirs("results_endowus_D", exist_ok=True)

# The new Endowus Proxy Universe (8 ETFs representing the Flagship 60/40 Fund)
# ETFS = [
#     "URTH", # Global Developed Equity (28.5%)
#     "SPY",  # US S&P 500 (17.4%)
#     "EEM",  # Emerging Markets Equity (9.3%)
#     "VPL",  # Pacific Basin Small/Mid Cap (4.8%)
#     "BNDW", # Global Aggregate Bond (23.0%)
#     "AGG",  # US Aggregate Bond (7.0%)
#     "EMB",  # Emerging Markets Government Bond (6.0%)
#     "BSV",  # Short-Term Global Bond (4.0%)
# ]

ETFS = [
    "0P0001AF7U.SI",  # Dimensional Global Core Equity Fund
    "0P0000KYEE.SI",  # PIMCO GIS Global Bond Fund SGD-Hedged
    "SPY",  # iShares US Index Fund (IE) S&P 500
    "^990100-USD-STRD",  # iShares Developed World Index Fund (IE)
    "DE000SLA4YD9.SG",  # Amundi Prime USA Fund
    "AGGG.L",  # Amundi Core Global Aggregate Bond SGD-Hedged
    "IE0002461055.IR",  # PIMCO GIS Income Fund SGD-Hedged
    "0P0001AF7Z.SI",  # Dimensional Emerging Markets Large Cap Core Equity Fund
    "0P0001EQUE.SI",  # Dimensional Global Core Fixed Income Fund SGD-Hedged
    "0P0001EF2T.SI",  # Dimensional Pacific Basin Small Companies Fund
    "0P0001CC3M",  # iShares Global Aggregate 1-5 Year Bond Index Fund (IE) SGD-Hedged
    "PEBIX",  # iShares Emerging Markets Government Bond Index Fund (IE)
    "EIMI.L",  # Amundi Core MSCI Emerging Markets Fund
    "0P0001DWI0.SI",  # PIMCO GIS Emerging Markets Bond Fund SGD-Hedged
]

# Event Windows (Core dates, padding will be added programmatically)
EVENTS = {
    "COVID-19 Crash": ("2020-02-01", "2020-05-31"),
    "The Emerging Market Divergence": ("2021-08-01", "2021-12-31"),
    "Inflation Shock": ("2021-11-01", "2022-01-31"),
    "Russia-Ukraine Invasion": ("2022-02-01", "2022-04-30"),
    "The BOJ Yield Curve Surprise": ("2022-12-01", "2023-01-31"),
    "Regional Banking Crisis": ("2023-03-01", "2023-05-31"),
}

# Define the 4 Ablation Variations
VARIATIONS = {
    "A": ["fundamentals", "market"],  # Fundamentals/Macro (Rates, CPI, GDP) + Price
    "B": ["news"],  # News (Geopolitical/Financial headlines)
    "C": ["market"],  # Technicals (SMA, MACD, RSI)
    "D": ["market", "news", "fundamentals", "social"],  # Full Debate setup
}


def generate_weekly_mondays(start_str: str, end_str: str) -> List[str]:
    """Generates a list of all Mondays between padded start and end date (inclusive)."""
    start_date = datetime.strptime(start_str, "%Y-%m-%d") - pd.DateOffset(months=5)
    end_date = datetime.strptime(end_str, "%Y-%m-%d") + pd.DateOffset(months=5)

    # Find first Monday
    days_ahead = 0 - start_date.weekday()
    if days_ahead < 0:
        days_ahead += 7
    first_monday = start_date + timedelta(days=days_ahead)

    mondays = []
    curr = first_monday
    while curr <= end_date:
        mondays.append(curr.strftime("%Y-%m-%d"))
        curr += timedelta(days=7)

    return mondays


def get_trading_dates() -> List[str]:
    """Get all unique Mondays to evaluate across all event windows."""
    dates = set()
    for start, end in EVENTS.values():
        dates.update(generate_weekly_mondays(start, end))
    return sorted(list(dates))


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    reraise=True,
)
def evaluate_ticker_date(
    ticker: str, dt: str, variation_id: str, analysts: List[str], config: Dict
):
    """
    Evaluates a single ticker on a single date by initializing a new TradingAgentsGraph.
    Uses exponential backoff for rate limits/API errors.
    """
    task_start = time.time()
    ta = TradingAgentsGraph(selected_analysts=analysts, config=config)
    state, decision = ta.propagate(company_name=ticker, trade_date=dt)
    elapsed = time.time() - task_start
    return {"ticker": ticker, "test_date": dt, "decision": decision, "elapsed": elapsed}


def run_variation(
    variation_id: str, analysts: List[str], dates: List[str], config: Dict
):
    print(f"\n{'='*50}", flush=True)
    print(
        f"Running Endowus Variation {variation_id} (Analysts: {analysts})", flush=True
    )
    print(f"{'='*50}", flush=True)

    out_dir = f"results_endowus_{variation_id}"

    # 1. Identify which tasks are already done across all tickers to support resume
    completed_tasks_set = set()  # Store (ticker, date) tuples
    for ticker in ETFS:
        csv_file = os.path.join(out_dir, f"{ticker}_decisions.csv")
        if os.path.isfile(csv_file):
            try:
                df = pd.read_csv(csv_file)
                if "test_date" in df.columns:
                    for dt in df["test_date"].astype(str):
                        completed_tasks_set.add((ticker, dt))
            except pd.errors.EmptyDataError:
                pass

    # 2. Build the list of pending tasks
    tasks = []
    for ticker in ETFS:
        for dt in dates:
            if (ticker, dt) not in completed_tasks_set:
                tasks.append((ticker, dt))
            else:
                print(
                    f"[{variation_id}] Skipping {ticker} on {dt} (Already done)",
                    flush=True,
                )

    total_tasks_run = len(tasks)
    if total_tasks_run == 0:
        print(
            f"\nAll tasks for Variation {variation_id} are already completed!",
            flush=True,
        )
        return

    print(
        f"[{variation_id}] {total_tasks_run} pending tasks to evaluate...", flush=True
    )

    start_time_variation = time.time()
    completed_this_run = 0
    task_times = []

    # Prepare CSV files and headers if they don't exist
    for ticker in ETFS:
        csv_file = os.path.join(out_dir, f"{ticker}_decisions.csv")
        if not os.path.isfile(csv_file):
            with open(csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["test_date", "decision"])

    # 3. Multithreaded execution
    max_workers = 10
    print(
        f"[{variation_id}] Starting thread pool with {max_workers} workers...",
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(
                evaluate_ticker_date, t, d, variation_id, analysts, config
            ): (t, d)
            for (t, d) in tasks
        }

        # Process as they complete
        for future in concurrent.futures.as_completed(future_to_task):
            t, d = future_to_task[future]
            try:
                result = future.result()

                # Write to the specific ticker's CSV safely in the main thread
                csv_file = os.path.join(out_dir, f"{result['ticker']}_decisions.csv")
                with open(csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([result["test_date"], result["decision"]])

                elapsed = result["elapsed"]
                task_times.append(elapsed)
                completed_this_run += 1

                avg_time = sum(task_times) / len(task_times)
                tasks_remaining = total_tasks_run - completed_this_run

                # Approximate ETA based on avg time per task and workers
                eta_seconds = (avg_time * tasks_remaining) / max_workers

                print(
                    f"[{variation_id}] [{completed_this_run}/{total_tasks_run}] Evaluated {t} on {d} (Done in {elapsed:.1f}s) -> ETA: {timedelta(seconds=int(eta_seconds))}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[{variation_id}] Error evaluating {t} on {d}: {exc}", flush=True
                )

    total_time = time.time() - start_time_variation
    print(
        f"\nVariation {variation_id} completed {total_tasks_run} tasks in {timedelta(seconds=int(total_time))}",
        flush=True,
    )


def main():
    print("Gathering dates for event windows...", flush=True)
    trading_dates = get_trading_dates()
    print(f"Total evaluation dates: {len(trading_dates)}", flush=True)

    # Load environment variables
    load_dotenv()

    # Use the config format from run_stock_backtest.py
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "openrouter"
    config["backend_url"] = "https://openrouter.ai/api/v1"
    config["embedding_backend_url"] = "https://api.openai.com/v1"
    config["deep_think_llm"] = "z-ai/glm-4-32b"
    config["quick_think_llm"] = "z-ai/glm-4-32b"
    config["max_debate_rounds"] = 1
    config["online_tools"] = True
    config["data_dir"] = "./data_dir"

    if len(sys.argv) > 1:
        # Run specific variation provided as argument
        var_id = sys.argv[1].upper()
        if var_id in VARIATIONS:
            analysts = VARIATIONS[var_id]
            run_variation(var_id, analysts, trading_dates, config)
        else:
            print(
                f"Error: Variation {var_id} not recognized. Must be one of A, B, C, D.",
                flush=True,
            )
            sys.exit(1)
    else:
        # Run all variations sequentially
        for var_id, analysts in VARIATIONS.items():
            run_variation(var_id, analysts, trading_dates, config)

    print("\nExecution completed.", flush=True)


if __name__ == "__main__":
    main()
