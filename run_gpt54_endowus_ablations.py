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
from tradingagents.utils.batch_manager import OpenAIBatchManager, CaptureLLM, BatchCaptureException
from pathlib import Path
import json

# Ensure output directories exist
os.makedirs("gpt54_endowus_A", exist_ok=True)
os.makedirs("gpt54_endowus_B", exist_ok=True)
os.makedirs("gpt54_endowus_C", exist_ok=True)
os.makedirs("gpt54_endowus_D", exist_ok=True)

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
    "ALL": ("2020-01-01", "2024-12-31"),
}

# Define the 4 Ablation Variations
VARIATIONS = {
    "A": ["fundamentals", "market"],  # Fundamentals/Macro (Rates, CPI, GDP) + Price
    "B": ["news"],  # News (Geopolitical/Financial headlines)
    "C": ["market"],  # Technicals (SMA, MACD, RSI)
    "D": ["market", "news", "fundamentals"],  # Full Debate setup
}


def generate_weekly_mondays(start_str: str, end_str: str) -> List[str]:
    """Generates a list of all Mondays between padded start and end date (inclusive)."""
    start_date = datetime.strptime(start_str, "%Y-%m-%d") - pd.DateOffset(months=5)
    end_date = datetime.strptime(end_str, "%Y-%m-%d")

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
    variation_id: str, 
    analysts: List[str], 
    dates: List[str], 
    config: Dict, 
    tickers: List[str] = ETFS,
    tasks: List[tuple] = None,
    batch_manager: OpenAIBatchManager = None
):
    print(f"\n{'='*50}", flush=True)
    print(
        f"Running GPT5.4 Endowus Variation {variation_id} (Analysts: {analysts})", flush=True
    )
    print(f"{'='*50}", flush=True)

    out_dir = f"gpt54_endowus_{variation_id}"

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

    # 1. Identify tasks
    if tasks is None:
        tasks = []
        for ticker in tickers:
            csv_file = os.path.join(out_dir, f"{ticker}_decisions.csv")
            existing_dates = set()
            if os.path.isfile(csv_file):
                with open(csv_file, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        existing_dates.add(row["test_date"])

            for d in dates:
                if d not in existing_dates:
                    tasks.append((ticker, d))

    total_tasks = len(tasks)
    if total_tasks == 0:
        print(f"[{variation_id}] All tasks completed!", flush=True)
        return

    print(f"[{variation_id}] {total_tasks} pending tasks. Starting Batch API Wave Execution...", flush=True)

    # 2. Initialize Batch Manager and Graph
    bm = batch_manager if batch_manager else OpenAIBatchManager()
    
    # We create a single graph instance but use different states
    trading_graph = TradingAgentsGraph(selected_analysts=analysts, config=config)
    
    # Replace real LLMs with CaptureLLM
    quick_capture = CaptureLLM(batch_manager=bm, model=config["quick_think_llm"])
    deep_capture = CaptureLLM(batch_manager=bm, model=config["deep_think_llm"])
    
    trading_graph.quick_thinking_llm = quick_capture
    trading_graph.deep_thinking_llm = deep_capture
    # Re-setup graph with capture llm
    trading_graph.graph = trading_graph.graph_setup.setup_graph(analysts)

    # 3. Initialize States
    states = {} # (ticker, date) -> state
    for t, d in tasks:
        states[(t, d)] = trading_graph.propagator.create_initial_state(t, d)

    # 4. Wave Execution
    nodes = trading_graph.get_analyst_nodes() + trading_graph.get_researcher_nodes() + trading_graph.get_manager_nodes()
    
    for node_name in nodes:
        print(f"[{variation_id}] --- Wave: {node_name} ---", flush=True)
        
        # Step A: Run until node and Capture Prompts (Parallelized)
        captured_count = 0
        
        def capture_task(t, d):
            custom_id_base = f"{variation_id}_{t}_{d}_{node_name.replace(' ', '_')}"
            # Important: custom_id must be managed carefully in threads
            # Since quick_capture and deep_capture are global in run_variation,
            # we need to ensure the BatchManager handles ID injection per call or 
            # we use local model instances if needed.
            
            try:
                config_thread = {"configurable": {"thread_id": f"{variation_id}_{t}_{d}"}}
                state_tuple = trading_graph.graph.get_state(config_thread)
                initial_input = None if state_tuple.values else states[(t, d)]
                
                config_step = {"configurable": {"thread_id": f"{variation_id}_{t}_{d}"}, "interrupt_before": [node_name]}
                trading_graph.graph.invoke(initial_input, config=config_step)
                
                config_step["interrupt_after"] = [node_name]
                config_step.pop("interrupt_before")
                trading_graph.graph.invoke(None, config=config_step)
                return "finished"
            except BatchCaptureException:
                return "captured"
            except Exception as e:
                return f"error: {str(e)}"

        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        max_workers = 15 # Adjust based on CPU/Network
        print(f"[{variation_id}] Parallelizing capture for {len(tasks)} tasks on node {node_name}...", flush=True)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(capture_task, t, d): (t, d) for (t, d) in tasks}
            
            # Progress bar for the capture wave
            pbar = tqdm(total=len(tasks), desc=f"[{variation_id}] Wave: {node_name}")
            for future in as_completed(futures):
                res = future.result()
                if res == "captured":
                    captured_count += 1
                elif res.startswith("error"):
                    t_err, d_err = futures[future]
                    print(f"\nError in capture for {t_err} {d_err} node {node_name}: {res}")
                pbar.update(1)
            pbar.close()

        # Step B: Submit Batch if we have requests
        if bm.requests:
            print(f"[{variation_id}] Submitting batch with {len(bm.requests)} requests for {node_name}...", flush=True)
            batch_id = bm.submit_batch(f"batches/batch_{variation_id}_{node_name.replace(' ', '_')}.jsonl")
            
            # Save to pending_batches.json and EXIT
            pending = {}
            if os.path.exists("pending_batches.json"):
                with open("pending_batches.json", "r") as f:
                    pending = json.load(f)
            
            pending[batch_id] = {
                "variation_id": variation_id,
                "node_name": node_name,
                "analysts": analysts,
                "tasks": tasks,
                "timestamp": time.time()
            }
            
            with open("pending_batches.json", "w") as f:
                json.dump(pending, f, indent=4)
                
            print(f"[{variation_id}] Batch {batch_id} submitted for {node_name}. Tracking in pending_batches.json.", flush=True)
            return "pending"
        else:
            print(f"[{variation_id}] No NEW LLM requests for node {node_name}, moving to next node or finalizing.", flush=True)

    # 5. Result Collection Phase
    print(f"[{variation_id}] Finalizing results and logging states...", flush=True)
    for (t, d) in tasks:
        try:
            # Load final state from checkpointer
            config_final = {"configurable": {"thread_id": f"{variation_id}_{t}_{d}"}}
            final_state = trading_graph.graph.get_state(config_final).values
            
            # Save detailed JSON log
            log_dir = Path(f"eval_results/{t}/Variation_{variation_id}_logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            log_keys = [
                "company_of_interest", "trade_date", "market_report", 
                "sentiment_report", "news_report", "fundamentals_report", 
                "investment_debate_state", "trader_investment_plan", 
                "risk_debate_state", "investment_plan", "final_trade_decision"
            ]
            state_log = {k: final_state.get(k, "") for k in log_keys}
            # Rename for user spec
            state_log["trader_investment_decision"] = state_log.pop("trader_investment_plan")
            
            with open(log_dir / f"full_states_log_{d}.json", "w") as f:
                json.dump(state_log, f, indent=4)
                
            # Update summary CSV
            decision_raw = state_log["trader_investment_decision"]
            processed = trading_graph.process_signal(decision_raw)
            
            csv_file = os.path.join(out_dir, f"{t}_decisions.csv")
            file_exists = os.path.isfile(csv_file)
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["test_date", "decision"])
                writer.writerow([d, processed])
        except Exception as e:
            print(f"Error finalizing {t} {d}: {e}")

    print(f"[{variation_id}] Variation completed.", flush=True)
    return "completed"


def check_and_resume_batches(config: Dict):
    """Checks for pending batches and resumes them if completed."""
    if not os.path.exists("pending_batches.json"):
        return

    with open("pending_batches.json", "r") as f:
        pending = json.load(f)

    if not pending:
        return

    bm = OpenAIBatchManager()
    still_pending = {}
    
    # We collect all completed results first to avoid partial resumptions
    completed_any = False
    
    for batch_id, meta in list(pending.items()):
        try:
            status = bm.get_batch_status(batch_id)
            print(f"Checking Batch {batch_id} status: {status}")
            
            if status == "completed":
                print(f"Batch {batch_id} COMPLETED. Downloading results for {meta['variation_id']} at {meta['node_name']}...")
                results = bm._get_results(bm.client.batches.retrieve(batch_id).output_file_id)
                bm.results.update(results)
                completed_any = True
                
                # After downloading results, we can potentially resume the variation
                # But we should do it AFTER checking all batches to maximize progress in one run
                status = run_variation(
                    meta["variation_id"], 
                    meta["analysts"], 
                    [], 
                    config, 
                    tickers=[], 
                    tasks=meta["tasks"],
                    batch_manager=bm
                )
                if status == "pending":
                    # It hit another node and submitted a new batch, which was already added to pending or will be
                    # Since we are iterating on a copy/original list, we need to refresh pending from disk or manage state
                    pass
            elif status in ["failed", "expired", "cancelled"]:
                print(f"WARNING: Batch {batch_id} ended with {status}. Manual intervention required.")
            else:
                still_pending[batch_id] = meta
        except Exception as e:
            print(f"Error checking batch {batch_id}: {e}")
            still_pending[batch_id] = meta

    # Refresh pending from disk in case run_variation added new ones
    if os.path.exists("pending_batches.json"):
        with open("pending_batches.json", "r") as f:
            latest_pending = json.load(f)
            # Remove the ones we processed
            for bid in list(latest_pending.keys()):
                if bid in pending and bid not in still_pending:
                    del latest_pending[bid]
            still_pending = latest_pending

    with open("pending_batches.json", "w") as f:
        json.dump(still_pending, f, indent=4)


def main():
    print("Gathering dates for event windows...", flush=True)
    trading_dates = get_trading_dates()
    print(f"Total evaluation dates: {len(trading_dates)}", flush=True)

    # # SHORT TEST MODE: If you want to verify the batching logic quickly, uncomment these lines:
    # print("DEBUG: Running in SHORT_TEST mode (2 tickers, 2 dates)")
    # trading_dates = trading_dates[:2]
    # # global ETFS
    # ETFS = ["SPY", "0P0001AF7U.SI"]

    # Load environment variables
    load_dotenv()

    # Use the config format from run_stock_backtest.py
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "openai"
    config["backend_url"] = "https://api.openai.com/v1"
    config["embedding_backend_url"] = "https://api.openai.com/v1"
    config["deep_think_llm"] = "gpt-5.4"
    config["quick_think_llm"] = "gpt-5-nano-2025-08-07"
    config["max_debate_rounds"] = 1
    config["online_tools"] = True
    config["data_dir"] = "./data_dir"

    check_and_resume_batches(config)

    if len(sys.argv) > 1:
        # Run specific variation provided as argument
        var_id = sys.argv[1].upper()
        if var_id in VARIATIONS:
            analysts = VARIATIONS[var_id]
            run_variation(var_id, analysts, trading_dates, config, ETFS)
        else:
            print(
                f"Error: Variation {var_id} not recognized. Must be one of A, B, C, D.",
                flush=True,
            )
            sys.exit(1)
    else:
        # Run all variations sequentially
        for var_id, analysts in VARIATIONS.items():
            run_variation(var_id, analysts, trading_dates, config, ETFS)

    # Final check for pending batches
    if os.path.exists("pending_batches.json"):
        with open("pending_batches.json", "r") as f:
            pending = json.load(f)
            if pending:
                print(f"\nThere are {len(pending)} batches still pending at OpenAI. Rerun this script later to resume.", flush=True)
                sys.exit(0)

    print("\nAll tasks and variations completed successfully.", flush=True)


if __name__ == "__main__":
    main()
