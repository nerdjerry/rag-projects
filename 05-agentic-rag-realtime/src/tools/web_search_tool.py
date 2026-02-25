"""
src/tools/web_search_tool.py

Provides live web search capability via the Tavily API so the agent can answer
questions about current events, recent news, or anything not in the static
knowledge base.

WHEN TO USE WEB SEARCH vs RAG:
    • Current events / breaking news          → web_search  (RAG docs are static)
    • Live prices, exchange rates, scores     → dedicated tool (finance/weather)
    • Recent product releases, news stories  → web_search
    • Internal company policies or strategy  → search_knowledge_base  (private data)
    • Historical or stable reference info    → search_knowledge_base  (faster, no API cost)

RATE LIMITS:
    Tavily free tier allows 1,000 searches/month (~33/day).
    Don't call this tool for every question — reserve it for questions that
    genuinely require real-time or recent information.
"""

from langchain.tools import Tool


def create_web_search_tool(tavily_api_key: str) -> Tool:
    """
    Build a LangChain Tool that queries the Tavily web search API.

    Tavily is purpose-built for LLM agents — it returns clean text snippets
    rather than raw HTML, which keeps the context window efficient.

    Args:
        tavily_api_key: A valid Tavily API key from https://tavily.com.

    Returns:
        A configured LangChain Tool for live web search.
    """

    def _web_search(query: str) -> str:
        """
        Call the Tavily search API and format the top results as plain text.

        The agent provides `query` as whatever it decides to search for.
        We return a structured string so the agent can extract the most
        relevant details in its reasoning step.
        """
        try:
            # tavily-python provides a clean SDK around the REST API.
            from tavily import TavilyClient  # type: ignore

            client = TavilyClient(api_key=tavily_api_key)

            # max_results=3 keeps the context manageable while still giving
            # the agent enough breadth to triangulate information.
            response = client.search(query, max_results=3)

            results = response.get("results", [])
            if not results:
                return "Web search returned no results for that query."

            lines = []
            for r in results:
                title = r.get("title", "No title")
                snippet = r.get("content", r.get("snippet", "No snippet available"))
                url = r.get("url", "")
                lines.append(f"Title: {title}\nSnippet: {snippet}\nURL: {url}\n---")

            return "\n".join(lines)

        except ImportError:
            # Fall back to a direct HTTP call if the SDK isn't installed.
            return _tavily_http_fallback(query, tavily_api_key)

        except Exception as exc:
            # Graceful degradation: the agent will see this message and can
            # note in its response that web search was unavailable.
            return f"Web search unavailable: {exc}"

    def _tavily_http_fallback(query: str, api_key: str) -> str:
        """Direct HTTP fallback when the tavily-python package is missing."""
        import requests  # noqa: PLC0415

        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={"api_key": api_key, "query": query, "max_results": 3},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                return "Web search returned no results."
            lines = []
            for r in results:
                lines.append(
                    f"Title: {r.get('title', '')}\n"
                    f"Snippet: {r.get('content', '')}\n"
                    f"URL: {r.get('url', '')}\n---"
                )
            return "\n".join(lines)
        except Exception as exc:
            return f"Web search unavailable: {exc}"

    return Tool(
        name="web_search",
        func=_web_search,
        description=(
            "Search the live web for current information, news, or recent events. "
            "Use this when the question requires up-to-date information not available "
            "in static documents. "
            "Input: a search query string."
        ),
    )


def create_mock_web_search_tool() -> Tool:
    """
    Fallback tool returned when no Tavily API key is configured.

    The agent will still receive a coherent message explaining why the tool
    is unavailable, rather than raising an exception mid-reasoning.
    """

    def _mock_search(query: str) -> str:  # noqa: ARG001
        return (
            "Web search is not configured. "
            "Please add TAVILY_API_KEY to .env to enable live web search."
        )

    return Tool(
        name="web_search",
        func=_mock_search,
        description=(
            "Search the live web for current information, news, or recent events. "
            "NOTE: Web search is currently disabled (no API key configured). "
            "Input: a search query string."
        ),
    )
