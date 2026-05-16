import time
import json


def create_research_manager(llm, memory):
    def research_manager_node(state) -> dict:
        history = state["investment_debate_state"].get("history", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        investment_debate_state = state["investment_debate_state"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        from langchain_core.prompts import ChatPromptTemplate
        from tradingagents.agents.utils.prompts import SHARED_COLLABORATION_PROMPT

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", SHARED_COLLABORATION_PROMPT),
            ("system", 
                "As the portfolio manager and debate facilitator, your role is to critically evaluate this round of debate and make a definitive decision: align with the bear analyst, the bull analyst, or choose Hold only if it is strongly justified based on the arguments presented.\n\n"
                "Summarize the key points from both sides concisely, focusing on the most compelling evidence or reasoning. Your recommendation—Buy, Sell, or Hold—must be clear and actionable. Avoid defaulting to Hold simply because both sides have valid points; commit to a stance grounded in the debate's strongest arguments.\n\n"
                "Additionally, develop a detailed investment plan for the trader. This should include:\n"
                "1. Your Recommendation: A decisive stance supported by the most convincing arguments.\n"
                "2. Rationale: An explanation of why these arguments lead to your conclusion.\n"
                "3. Strategic Actions: Concrete steps for implementing the recommendation.\n"
                "Take into account your past mistakes on similar situations. Use these insights to refine your decision-making and ensure you are learning and improving. Present your analysis conversationally, as if speaking naturally, without special formatting."
            ),
            ("user", 
                "Here are your past reflections on mistakes:\n"
                "\"{past_memory_str}\"\n\n"
                "Here is the debate:\n"
                "Debate History:\n"
                "{history}"
            )
        ])

        response = llm.invoke(prompt_template.format_messages(
            past_memory_str=past_memory_str,
            history=history
        ))

        new_investment_debate_state = {
            "judge_decision": response.content,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": response.content,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": response.content,
        }

    return research_manager_node
