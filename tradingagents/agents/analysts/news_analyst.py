from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import tradingagents.dataflows.interface as interface
import time
import json


def create_news_analyst(llm, toolkit):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        if toolkit.config["online_tools"]:
            tools = [
                # toolkit.get_global_news_openai,
                # toolkit.get_google_news,
                # toolkit.get_web_search_news,
                toolkit.get_alpaca_news,
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
                    "For your reference, the current date is {current_date}. We are looking at the company {ticker}",
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

        # Try to parse the model output as a function call dict and execute the matching toolkit function.
        # Fall back to using the raw content if parsing or execution fails.
        try:
            data = None
            # Some model wrappers expose structured tool_calls; prefer parsing content directly
            try:
                data = json.loads(result.content)
            except Exception:
                # not JSON, ignore
                data = None

            if isinstance(data, dict) and "function" in data:
                fname = data.get("function")
                params = data.get("parameters", {}) or {}

                # First try direct attribute on toolkit
                func = None
                if hasattr(toolkit, fname):
                    func = getattr(toolkit, fname)

                # Otherwise search the provided tools list for a matching tool name
                if func is None:
                    for t in tools:
                        tname = getattr(t, "name", None) or getattr(t, "__name__", None)
                        if tname == fname:
                            func = t
                            break

                if callable(func):
                    report_assigned = False
                    try:
                        # Try calling tool directly with kwargs
                        output = func(**params)
                        report = output
                        report_assigned = True
                    except TypeError:
                        # Fallback: try calling underlying interface implementation
                        try:
                            if hasattr(interface, fname):
                                iface = getattr(interface, fname)
                                if fname == "get_alpaca_news":
                                    report = iface(
                                        params.get("ticker"), params.get("curr_date"), 7
                                    )
                                else:
                                    report = iface(*params.values())
                                report_assigned = True
                        except Exception as e2:
                            print(f"Interface fallback failed for {fname}: {e2}")

                        # If fallback hasn't produced a report yet, try common tool invocation methods
                        if not report_assigned:
                            try:
                                if hasattr(func, "invoke"):
                                    try:
                                        report = func.invoke(params)
                                    except TypeError:
                                        report = func.invoke(json.dumps(params))
                                elif hasattr(func, "run"):
                                    try:
                                        report = func.run(params)
                                    except TypeError:
                                        report = func.run(json.dumps(params))
                                else:
                                    print(
                                        f"No suitable invocation method for tool {fname}"
                                    )
                                    report = result.content
                                report_assigned = True
                            except Exception as e3:
                                print(f"Tool execution failed for {fname}: {e3}")
                                report = result.content
                                report_assigned = True
                    except Exception as e:
                        print(f"Tool execution failed for {fname}: {e}")
                        report = result.content
                        report_assigned = True
                else:
                    # couldn't find callable; use content
                    report = result.content
            else:
                report = result.content
        except Exception as e:
            print(f"Error processing tool call: {e}")
            report = getattr(result, "content", "")

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
