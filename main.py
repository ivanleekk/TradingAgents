from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create a custom config
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openrouter"  # Use a different model
config["backend_url"] = "https://openrouter.ai/api/v1"  # Use a different backend
config["embedding_backend_url"] = "https://api.openai.com/v1"  # Use a different backend
config["deep_think_llm"] = (
    "z-ai/glm-4.5-air:free"  # Use a different model
)
config["quick_think_llm"] = (
    "meta-llama/llama-4-scout:free"  # Use a different model
)
config["max_debate_rounds"] = 1  # Increase debate rounds
config["online_tools"] = True  # Increase debate rounds

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# forward propagate
_, decision = ta.propagate("ORCL", "2025-09-23")
print(decision)

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
