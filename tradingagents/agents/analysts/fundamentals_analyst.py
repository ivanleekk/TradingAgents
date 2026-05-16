from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_fundamentals_analyst(llm, toolkit):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        from tradingagents.dataflows.macro_utils import get_macro_fundamentals

        # Because we are trading ETFs, standard corporate balance sheets do not exist.
        # We replace the SimFin tools with the Macro Fundamentals FRED tool.
        tools = [get_macro_fundamentals]

        system_message = (
            "You are a Macroeconomic Fundamentals Analyst tasked with analyzing the broad market and economic conditions surrounding an ETF. "
            "Because ETFs are baskets of assets and not individual corporations, they do not have corporate balance sheets, cash flows, or insider transactions. "
            "Instead, you must ALWAYS use the `get_macro_fundamentals` tool to fetch the trailing 12-month data for Interest Rates (FEDFUNDS), Inflation (CPI), GDP, and Unemployment Rate. "
            "Please write a comprehensive macroeconomic report analyzing how these broad fundamental indicators might impact the specified ETF. "
            "Provide detailed and finegrained analysis and insights that may help traders make decisions. "
            "Make sure to append a Markdown table at the end of the report to organize key macro points in the report, organized and easy to read."
        )

        from tradingagents.agents.utils.prompts import SHARED_COLLABORATION_PROMPT, get_context_prompt

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    SHARED_COLLABORATION_PROMPT + 
                    "\n\nYou have access to the following tools: {tool_names}.\n\n"
                    "{system_message}" + 
                    "{context}"
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        context = get_context_prompt(current_date, ticker)
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(context=context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
