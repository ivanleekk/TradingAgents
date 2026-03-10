from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


ETF_INDUSTRY_MAP = {
    "0P0001AF7U.SI": "Global Equity",
    "0P0000KYEE.SI": "Global Bonds (SGD-Hedged)",
    "SPY": "US Large-Cap Equity (S&P 500)",
    "^990100-USD-STRD": "Developed Markets Equity",
    "DE000SLA4YD9.SG": "US Equity",
    "AGGG.L": "Global Aggregate Bonds (SGD-Hedged)",
    "IE0002461055.IR": "Global Income Bonds (SGD-Hedged)",
    "0P0001AF7Z.SI": "Emerging Markets Large-Cap Equity",
    "0P0001EQUE.SI": "Global Core Fixed Income (SGD-Hedged)",
    "0P0001EF2T.SI": "Pacific Basin Small-Cap Equity",
    "0P0001CC3M": "Global Short-Duration Aggregate Bonds (1-5Y, SGD-Hedged)",
    "PEBIX": "Emerging Markets Government Bonds",
    "EIMI.L": "Emerging Markets Equity",
    "0P0001DWI0.SI": "Emerging Markets Bonds (SGD-Hedged)",
}


def create_news_analyst(llm, toolkit):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        fund_industry = ETF_INDUSTRY_MAP.get(ticker, "Unknown / Diversified Fund")

        if toolkit.config["online_tools"]:
            tools = [
                # toolkit.get_global_news_openai,
                toolkit.get_google_news
            ]
        else:
            tools = [
                toolkit.get_finnhub_news,
                toolkit.get_reddit_news,
                toolkit.get_google_news,
            ]

        system_message = (
            "You are a news researcher tasked with analyzing recent news and trends over the past week. Please write a comprehensive report of the current state of the world that is relevant for trading and macroeconomics. Look at news from EODHD, and finnhub to be comprehensive. Do not simply state the trends are mixed, provide detailed and finegrained analysis and insights that may help traders make decisions."
            + """ Make sure to append a Makrdown table at the end of the report to organize key points in the report, organized and easy to read."""
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
                    "For your reference, the current date is {current_date}. "
                    "We are looking at the fund {ticker}. "
                    "This fund's primary industry/exposure is: {fund_industry}.",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)
        prompt = prompt.partial(fund_industry=fund_industry)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
