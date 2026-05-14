import os
import sys
import json
from dotenv import load_dotenv
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.utils.batch_manager import OpenAIBatchManager, CaptureLLM
from run_gpt54_endowus_ablations import run_variation, VARIATIONS

def main():
    print("Starting DEBUG variation test (Single Ticker, Single Date)...")
    load_dotenv()

    # Configuration for Debug
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "openai"
    config["backend_url"] = "https://api.openai.com/v1"
    config["embedding_backend_url"] = "https://api.openai.com/v1"
    config["deep_think_llm"] = "gpt-5.4"
    config["quick_think_llm"] = "gpt-5-nano-2025-08-07"
    config["max_debate_rounds"] = 1
    config["online_tools"] = True
    config["data_dir"] = "./data_dir"

    # Use a single ticker and a single date for debugging
    debug_tickers = ["SPY"]
    debug_dates = ["2020-01-06"]
    
    # Run only Variation C (Market Only) for simplicity
    var_id = "C"
    analysts = VARIATIONS[var_id]
    
    print(f"DEBUG: Running Variation {var_id} for {debug_tickers} on {debug_dates}")
    
    # Note: run_variation might exit or return 'pending' after submitting a batch
    run_variation(var_id, analysts, debug_dates, config, debug_tickers)

    print("\nDEBUG Test completed. Check batches/ and pending_batches.json.")

if __name__ == "__main__":
    main()
