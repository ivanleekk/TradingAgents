from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from dotenv import load_dotenv
from datetime import date, timedelta
import polars as pl

# Load environment variables
load_dotenv()

# Create a custom config
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openrouter"  # Use a different model
config["backend_url"] = "https://openrouter.ai/api/v1"  # Use a different backend
config["embedding_backend_url"] = "https://api.openai.com/v1"  # Use a different backend
config["deep_think_llm"] = "z-ai/glm-4-32b"  # Use a different model
config["quick_think_llm"] = "z-ai/glm-4-32b"  # Use a different model
config["max_debate_rounds"] = 1  # Increase debate rounds
config["online_tools"] = True  # Increase debate rounds

# Initialize with custom config
ta = TradingAgentsGraph(selected_analysts=["news"], debug=True, config=config)

# forward propagate over date range
start_date = date(2025, 6, 1)
stock = "0P0001AF7U.SI"
dates_to_test = 115
test_range = [
    (start_date + timedelta(days=x)).isoformat()
    for x in range(0, dates_to_test)
    if (start_date + timedelta(days=x)).weekday() < 5
]
df = pl.DataFrame()

try:
    for test_date in test_range:
        print(f"Propagating for date: {test_date}")
        _, decision = ta.propagate(stock, test_date)
        # save decision to dataframe
        df = pl.concat(
            [df, pl.DataFrame({"test_date": [test_date], "decision": [decision]})]
        )
except Exception as e:
    print(f"Error during propagation: {e}")
    df.write_csv(f"results/{stock}_decisions_{test_range[0]}_{test_range[-1]}.csv")


df.write_csv(f"results/{stock}_decisions_{test_range[0]}_{test_range[-1]}.csv")

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
