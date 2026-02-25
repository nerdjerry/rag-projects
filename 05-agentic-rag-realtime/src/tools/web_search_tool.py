"""
Web Search Tool — Live Internet Search
=======================================
Uses the SerpAPI Google Search wrapper to fetch real-time search results.

Rate limits & cost considerations:
  - SerpAPI free tier: 100 searches / month  (https://serpapi.com/pricing)
  - Each call here counts as ONE search against your quota.
  - The agent should NOT call this for every question — only when the query
    needs up-to-date or external information that isn't in the knowledge base.

Fallback behaviour:
  If no API key is provided, the tool returns a helpful error message instead
  of crashing.  This lets the rest of the agent keep working.
"""

import json
import os

from langchain.tools import Tool


def create_web_search_tool(api_key: str = None):
    """
    Create a LangChain Tool that performs a Google search via SerpAPI.

    Args:
        api_key: Your SerpAPI key.  If None / empty, the tool will return a
                 graceful error when invoked.

    Returns:
        A LangChain Tool the agent can call by name.
    """

    def _web_search(query: str) -> str:
        """Search the web and return top 3 results with title, snippet, URL."""

        # --- Graceful fallback when API key is missing ---
        effective_key = api_key or os.getenv("SERPAPI_API_KEY")
        if not effective_key:
            return (
                "Web search is unavailable because no SerpAPI key is configured. "
                "Please set SERPAPI_API_KEY in your .env file. "
                "Get a free key at https://serpapi.com/"
            )

        try:
            # Import here so the rest of the app works even without serpapi
            from langchain_community.utilities import SerpAPIWrapper

            search = SerpAPIWrapper(serpapi_api_key=effective_key)
            raw_results = search.results(query)

            # SerpAPI returns an "organic_results" list with many fields.
            # We extract only what the agent needs: title, snippet, link.
            organic = raw_results.get("organic_results", [])[:3]

            if not organic:
                return f"No web results found for: {query}"

            formatted = []
            for i, result in enumerate(organic, 1):
                title = result.get("title", "No title")
                snippet = result.get("snippet", "No snippet available")
                link = result.get("link", "")
                formatted.append(
                    f"[{i}] {title}\n    {snippet}\n    URL: {link}"
                )

            return "\n\n".join(formatted)

        except ImportError:
            return (
                "SerpAPI package not installed. "
                "Run: pip install google-search-results"
            )
        except Exception as e:
            return f"Web search error: {str(e)}"

    tool = Tool(
        name="web_search",
        func=_web_search,
        description=(
            "Use this to search the internet for current, real-time information. "
            "Good for recent news, live data, current events, or anything not "
            "covered by the local knowledge base. "
            "Input should be a concise search query string. "
            "NOTE: This uses a rate-limited API — prefer the knowledge base "
            "or Wikipedia for general facts."
        ),
    )

    return tool
