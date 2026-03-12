import os
import csv
import pandas as pd
from datetime import datetime, timedelta
import sys
from typing import List, Dict
import time

from dotenv import load_dotenv
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# Ensure output directories exist
os.makedirs("results_LLM_A", exist_ok=True)
os.makedirs("results_LLM_B", exist_ok=True)
os.makedirs("results_LLM_C", exist_ok=True)
os.makedirs("results_LLM_D", exist_ok=True)

# The new ETF universe
ETFS = [
    "SPY", "QQQ", "EEM", "TLT", "LQD", "GLD", "USO",
    "PDBC", "UUP", "FXE", "GBTC", "VNQ", "EWS", "FXI",
    "EWU", "EWJ", "EMB", "VNQI"
]

# Event Windows (from Phase 4, padded programmatically)
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
    "B": ["news"],                    # News (Geopolitical/Financial headlines)
    "C": ["market"],                  # Technicals (SMA, MACD, RSI)
    "D": ["market", "news", "fundamentals", "social"] # Full Debate setup
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

def run_variation(variation_id: str, analysts: List[str], dates: List[str], config: Dict):
    print(f"\n{'='*50}", flush=True)
    print(f"Running Variation {variation_id} (Analysts: {analysts})", flush=True)
    print(f"{'='*50}", flush=True)

    out_dir = f"results_LLM_{variation_id}"

    # Initialize the graph with the specific subset of analysts
    ta = TradingAgentsGraph(selected_analysts=analysts, config=config)

    total_tasks = len(ETFS) * len(dates)
    completed_tasks = 0
    start_time_variation = time.time()
    task_times = []

    for ticker in ETFS:
        csv_file = os.path.join(out_dir, f"{ticker}_decisions.csv")
        file_exists = os.path.isfile(csv_file)

        # Keep track of existing dates to resume if stopped
        existing_dates = set()
        if file_exists:
            try:
                df = pd.read_csv(csv_file)
                if 'test_date' in df.columns:
                    existing_dates = set(df['test_date'].astype(str))
            except pd.errors.EmptyDataError:
                pass

        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['test_date', 'decision'])

            for dt in dates:
                if dt in existing_dates:
                    print(f"[{variation_id}] Skipping {ticker} on {dt} (Already done)", flush=True)
                    completed_tasks += 1
                    continue

                print(f"[{variation_id}] [{completed_tasks+1}/{total_tasks}] Evaluating {ticker} on {dt}...", flush=True)
                task_start = time.time()
                try:
                    state, decision = ta.propagate(company_name=ticker, trade_date=dt)
                    writer.writerow([dt, decision])
                    f.flush()
                except Exception as e:
                    print(f"Error evaluating {ticker} on {dt}: {e}", flush=True)

                task_end = time.time()
                elapsed = task_end - task_start
                task_times.append(elapsed)

                completed_tasks += 1

                avg_time = sum(task_times) / len(task_times)
                tasks_remaining = total_tasks - completed_tasks
                eta_seconds = avg_time * tasks_remaining

                print(f"   -> Done in {elapsed:.1f}s. ETA for Variation {variation_id}: {timedelta(seconds=int(eta_seconds))}", flush=True)

                # Sleep briefly to avoid aggressive rate limits
                time.sleep(2)

    total_time = time.time() - start_time_variation
    print(f"\nVariation {variation_id} completed in {timedelta(seconds=int(total_time))}", flush=True)

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
        # Run specific variation provided as argument (e.g., python run_ablations.py A)
        var_id = sys.argv[1].upper()
        if var_id in VARIATIONS:
            analysts = VARIATIONS[var_id]
            run_variation(var_id, analysts, trading_dates, config)
        else:
            print(f"Error: Variation {var_id} not recognized. Must be one of A, B, C, D.", flush=True)
            sys.exit(1)
    else:
        # Run all variations sequentially
        for var_id, analysts in VARIATIONS.items():
            run_variation(var_id, analysts, trading_dates, config)

    print("\nExecution completed.", flush=True)

if __name__ == "__main__":
    main()
