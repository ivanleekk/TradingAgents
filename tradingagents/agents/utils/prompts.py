# tradingagents/agents/utils/prompts.py

SHARED_COLLABORATION_PROMPT = (
    "You are a helpful AI assistant, collaborating with other assistants. "
    "Use the provided tools to progress towards answering the question. "
    "If you are unable to fully answer, that's OK; another assistant with different tools "
    "will help where you left off. Execute what you can to make progress. "
    "If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable, "
    "prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
)

def get_context_prompt(current_date: str, ticker: str, additional_context: str = "") -> str:
    """Returns a string with dynamic context to be placed at the end of a prompt."""
    context = f"\n\n--- CURRENT CONTEXT ---\nDate: {current_date}\nTicker: {ticker}"
    if additional_context:
        context += f"\n{additional_context}"
    return context
