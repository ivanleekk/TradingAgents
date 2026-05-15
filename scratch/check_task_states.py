import json
import os
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.utils.batch_manager import OpenAIBatchManager

from tradingagents.default_config import DEFAULT_CONFIG

def check_task_states():
    config = DEFAULT_CONFIG.copy()
    config["quick_think_llm"] = "gpt-5-nano-2025-08-07"
    config["deep_think_llm"] = "gpt-5.4"
    config["selected_analysts"] = ["news"]
    
    # Load tasks from the failed batch
    failed_batch_file = "batches/batch_B_News_Analyst_part0.jsonl"
    tasks_to_check = []
    if os.path.exists(failed_batch_file):
        with open(failed_batch_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    parts = data["custom_id"].split("_")
                    variation_id = parts[0]
                    ticker = parts[1]
                    date = parts[2]
                    tasks_to_check.append((variation_id, ticker, date))
                except:
                    continue
    
    # Initialize graph
    graph = TradingAgentsGraph(config=config, selected_analysts=["news"])
    
    print(f"{'Task':<40} | {'Next Node'}")
    print("-" * 60)
    
    for vid, t, d in tasks_to_check:
        config_thread = {"configurable": {"thread_id": f"{vid}_{t}_{d}"}}
        state = graph.graph.get_state(config_thread)
        print(f"{vid}_{t}_{d:<25} | {state.next}")

if __name__ == "__main__":
    check_task_states()
