import functools
import time
import json


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        if past_memories:
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "No past memories found."

        from langchain_core.prompts import ChatPromptTemplate
        from tradingagents.agents.utils.prompts import SHARED_COLLABORATION_PROMPT

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", SHARED_COLLABORATION_PROMPT),
            ("system", 
                "You are a Macro Analyst trading agent analyzing market data to make investment decisions. "
                "Based on your analysis of the provided investment plan and reports, you must output a structured JSON block containing two specific metrics:\n"
                "1. \"Target_Return_30d\": Your expected return for this asset over the next 30 days, represented as a string with a percentage sign (e.g., \"+2.5%\" or \"-1.2%\").\n"
                "2. \"Confidence_Score\": Your confidence in this prediction on a scale of 1 to 10, represented as an integer (e.g., 8). A 10 means extremely low uncertainty.\n\n"
                "End your response with ONLY the structured JSON block wrapped in triple backticks.\n\n"
                "Do not forget to utilize lessons from past decisions to learn from your mistakes."
            ),
            ("user", 
                "Based on a comprehensive analysis by a team of analysts, here is an investment plan tailored for {company_name}. "
                "This plan incorporates insights from current technical market trends, macroeconomic indicators, and social media sentiment.\n\n"
                "Proposed Investment Plan: {investment_plan}\n\n"
                "Reflections from similar situations and lessons learned: {past_memory_str}\n\n"
                "Leverage these insights to make an informed and strategic decision."
            )
        ])

        result = llm.invoke(prompt_template.format_messages(
            company_name=company_name,
            investment_plan=investment_plan,
            past_memory_str=past_memory_str
        ))

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
