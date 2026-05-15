# TradingAgents/graph/conditional_logic.py

from tradingagents.agents.utils.agent_states import AgentState


class ConditionalLogic:
    """Handles conditional logic for determining graph flow."""

    def __init__(self, max_debate_rounds=1, max_risk_discuss_rounds=1):
        """Initialize with configuration parameters."""
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds

    def should_continue_market(self, state: AgentState):
        """Determine if market analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            if self._is_looping(messages):
                print("WARNING: Loop detected in Market Analyst tool calls. Forcing completion.")
                return "Msg Clear Market"
            return "tools_market"
        return "Msg Clear Market"

    def should_continue_social(self, state: AgentState):
        """Determine if social media analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            if self._is_looping(messages):
                print("WARNING: Loop detected in Social Analyst tool calls. Forcing completion.")
                return "Msg Clear Social"
            return "tools_social"
        return "Msg Clear Social"

    def should_continue_news(self, state: AgentState):
        """Determine if news analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            if self._is_looping(messages):
                print("WARNING: Loop detected in News Analyst tool calls. Forcing completion.")
                return "Msg Clear News"
            return "tools_news"
        return "Msg Clear News"

    def should_continue_fundamentals(self, state: AgentState):
        """Determine if fundamentals analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            if self._is_looping(messages):
                print("WARNING: Loop detected in Fundamentals Analyst tool calls. Forcing completion.")
                return "Msg Clear Fundamentals"
            return "tools_fundamentals"
        return "Msg Clear Fundamentals"

    def should_continue_debate(self, state: AgentState) -> str:
        """Determine if debate should continue."""

        if (
            state["investment_debate_state"]["count"] >= 2 * self.max_debate_rounds
        ):  # 3 rounds of back-and-forth between 2 agents
            return "Research Manager"
        if state["investment_debate_state"]["current_response"].startswith("Bull"):
            return "Bear Researcher"
        return "Bull Researcher"

    def should_continue_risk_analysis(self, state: AgentState) -> str:
        """Determine if risk analysis should continue."""
        if (
            state["risk_debate_state"]["count"] >= 3 * self.max_risk_discuss_rounds
        ):  # 3 rounds of back-and-forth between 3 agents
            return "Risk Judge"
        if state["risk_debate_state"]["latest_speaker"].startswith("Risky"):
            return "Safe Analyst"
        if state["risk_debate_state"]["latest_speaker"].startswith("Safe"):
            return "Neutral Analyst"
        return "Risky Analyst"

    def _is_looping(self, messages, max_repeats=3):
        """Check if the last tool call has been repeated too many times."""
        if not messages or not hasattr(messages[-1], "tool_calls") or not messages[-1].tool_calls:
            return False
            
        last_tool_call = messages[-1].tool_calls[0]
        tool_name = last_tool_call.get("name")
        tool_args = str(last_tool_call.get("args"))
        
        count = 0
        # Iterate backwards through messages
        for msg in reversed(messages):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tc = msg.tool_calls[0]
                if tc.get("name") == tool_name and str(tc.get("args")) == tool_args:
                    count += 1
            if count >= max_repeats:
                return True
        return False
