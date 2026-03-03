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

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. The company we want to look at is {ticker}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

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
