from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create a custom config
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "llamacpp"  # Use a different model
config["backend_url"] = "http://localhost:8080/v1"  # Use a different backend
config["deep_think_llm"] = "models/gemma-3-4b-it-BF16.gguf"  # Use a different model
config["quick_think_llm"] = "models/gemma-3-4b-it-BF16.gguf"  # Use a different model
# LlamaCpp tuning to avoid decode errors; override as needed
config["llamacpp_n_ctx"] = 131072
config["llamacpp_n_batch"] = 1024
config["llamacpp_n_gpu_layers"] = 80
config["max_debate_rounds"] = 1  # Increase debate rounds
config["online_tools"] = True  # Increase debate rounds

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# forward propagate
_, decision = ta.propagate("AAPL", "2025-08-18")
print(decision)

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
