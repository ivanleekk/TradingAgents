# TradingAgents/graph/signal_processing.py

from langchain_openai import ChatOpenAI


class SignalProcessor:
    """Processes trading signals to extract actionable decisions."""

    def __init__(self, quick_thinking_llm: ChatOpenAI):
        """Initialize with an LLM for processing."""
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(self, full_signal: str) -> str:
        """
        Process a full trading signal to extract the core decision.

        Args:
            full_signal: Complete trading signal text

        Returns:
            Extracted decision (BUY, SELL, or HOLD)
        """
        import json
        import re

        # Try to find JSON block directly in the text
        match = re.search(r"```json(.*?)```", full_signal, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1).strip())
                if "Target_Return_30d" in data and "Confidence_Score" in data:
                    return json.dumps(data)
            except json.JSONDecodeError:
                pass

        # Use LLM as fallback if strict parsing fails
        messages = [
            (
                "system",
                "You are an efficient data extraction assistant designed to extract structured JSON data from text. Extract the 'Target_Return_30d' and 'Confidence_Score' metrics from the text. Respond ONLY with valid JSON. For example: {\"Target_Return_30d\": \"+2.5%\", \"Confidence_Score\": 8}. Do not output any markdown block ticks.",
            ),
            ("human", full_signal),
        ]

        result = self.quick_thinking_llm.invoke(messages).content.strip()
        # Clean up any potential markdown formatting in the fallback response
        if result.startswith("```json"):
            result = result[7:]
        if result.startswith("```"):
            result = result[3:]
        if result.endswith("```"):
            result = result[:-3]
        return result.strip()
