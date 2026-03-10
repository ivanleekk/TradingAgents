import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
import random
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_result,
)


def is_rate_limited(response):
    """Check if the response indicates rate limiting (status code 429)"""
    return response.status_code == 429


@retry(
    retry=(retry_if_result(is_rate_limited)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
)
def make_request(url, headers):
    """Make a request with retry logic for rate limiting"""
    # Random delay before each request to avoid detection
    time.sleep(random.uniform(2, 6))
    response = requests.get(url, headers=headers)
    return response


# def getNewsData(query, start_date, end_date):
#     """
#     Scrape Google News search results for a given query and date range.
#     query: str - search query
#     start_date: str - start date in the format yyyy-mm-dd or mm/dd/yyyy
#     end_date: str - end date in the format yyyy-mm-dd or mm/dd/yyyy
#     """
#     if "-" in start_date:
#         start_date = datetime.strptime(start_date, "%Y-%m-%d")
#         start_date = start_date.strftime("%m/%d/%Y")
#     if "-" in end_date:
#         end_date = datetime.strptime(end_date, "%Y-%m-%d")
#         end_date = end_date.strftime("%m/%d/%Y")

#     headers = {
#         "User-Agent": (
#             "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
#             "AppleWebKit/537.36 (KHTML, like Gecko) "
#             "Chrome/101.0.4951.54 Safari/537.36"
#         )
#     }

#     news_results = []
#     page = 0
#     while True:
#         offset = page * 10
#         url = (
#             f"https://www.google.com/search?q={query}"
#             f"&tbs=cdr:1,cd_min:{start_date},cd_max:{end_date}"
#             f"&tbm=nws&start={offset}"
#         )

#         try:
#             response = make_request(url, headers)
#             soup = BeautifulSoup(response.content, "html.parser")
#             results_on_page = soup.select("div.SoaBEf")

#             if not results_on_page:
#                 break  # No more results found

#             for el in results_on_page:
#                 try:
#                     link = el.find("a")["href"]
#                     title = el.select_one("div.MBeuO").get_text()
#                     snippet = el.select_one(".GI74Re").get_text()
#                     date = el.select_one(".LfVVr").get_text()
#                     source = el.select_one(".NUnG9d span").get_text()
#                     news_results.append(
#                         {
#                             "link": link,
#                             "title": title,
#                             "snippet": snippet,
#                             "date": date,
#                             "source": source,
#                         }
#                     )
#                 except Exception as e:
#                     print(f"Error processing result: {e}")
#                     # If one of the fields is not found, skip this result
#                     continue

#             # Update the progress bar with the current count of results scraped

#             # Check for the "Next" link (pagination)
#             next_link = soup.find("a", id="pnnext")
#             if not next_link:
#                 break

#             page += 1

#         except Exception as e:
#             print(f"Failed after multiple retries: {e}")
#             break

#     return news_results

# RSS
import feedparser
import urllib.parse
from datetime import datetime


def getNewsData(query, start_date, end_date):
    """
    Fetch Google News RSS results for a given query and date range.
    query: str - search query
    start_date: str - format yyyy-mm-dd
    end_date: str - format yyyy-mm-dd
    """
    # Ensure dates are in YYYY-MM-DD for the RSS search operators
    try:
        if "/" in start_date:
            start_date = datetime.strptime(start_date, "%m/%d/%Y").strftime("%Y-%m-%d")
        if "/" in end_date:
            end_date = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m-%d")
    except Exception as e:
        print(f"Date formatting error: {e}")
        return []

    # Construct the query with date operators
    # Example: "AAPL after:2023-01-01 before:2023-01-31"
    rss_query = f"{query} after:{start_date} before:{end_date}"
    encoded_query = urllib.parse.quote(rss_query)

    # RSS URL (hl=en-SG and gl=SG for Singapore context, adjust as needed)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-SG&gl=SG&ceid=SG:en"

    news_results = []

    try:
        # Use requests to get the content, which handles SSL more reliably
        response = requests.get(url, timeout=10)

        # Parse the string content instead of the URL
        feed = feedparser.parse(response.content)

        for entry in feed.entries:
            raw_title = entry.title
            source = "Unknown"
            title = raw_title

            if " - " in raw_title:
                parts = raw_title.rsplit(" - ", 1)
                title = parts[0]
                source = parts[1]

            news_results.append(
                {
                    "link": entry.link,
                    "title": title,
                    "snippet": entry.summary if "summary" in entry else "",
                    "date": entry.published,
                    "source": source,
                }
            )

    except Exception as e:
        print(f"Error: {e}")

    return news_results
