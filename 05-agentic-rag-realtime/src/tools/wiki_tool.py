"""
src/tools/wiki_tool.py

Provides Wikipedia lookups as a lightweight alternative to full web search.
Wikipedia is ideal for factual, encyclopaedic questions where live web crawling
isn't needed but the answer isn't in the internal knowledge base either.

Advantages over web_search:
    • No API key required.
    • Zero cost — Wikipedia has no rate limits for this use case.
    • Results are well-structured and factual (not SEO-optimised content).

Use this before falling back to web_search for definitional or historical queries.
"""

from langchain.tools import Tool


def create_wiki_tool() -> Tool:
    """
    Build a LangChain Tool that searches Wikipedia for encyclopaedic information.

    Uses the `wikipedia` PyPI package which wraps the Wikipedia REST API.

    Returns:
        A configured LangChain Tool for Wikipedia lookups.
    """

    def search_wikipedia(query: str) -> str:
        """
        Search Wikipedia for the query and return a short summary.

        We return only the first 500 characters of the summary to keep the
        context window usage low.  The agent can always call the tool again
        with a more specific query if it needs more detail.

        Args:
            query: Search term or topic provided by the agent.

        Returns:
            A plain-text summary from Wikipedia, or an error message.
        """
        try:
            import wikipedia  # noqa: PLC0415

            query = query.strip()

            # wikipedia.summary() can raise DisambiguationError when the query
            # maps to multiple articles (e.g. "Python").  We handle this by
            # picking the first suggestion automatically.
            try:
                summary = wikipedia.summary(query, sentences=4, auto_suggest=True)
            except wikipedia.exceptions.DisambiguationError as e:
                # Try the first suggested page instead of failing.
                if e.options:
                    summary = wikipedia.summary(e.options[0], sentences=4)
                else:
                    return f"Wikipedia: '{query}' is ambiguous. Please be more specific."
            except wikipedia.exceptions.PageError:
                return f"Wikipedia: No article found for '{query}'. Try a different search term."

            # Truncate to keep context window usage predictable.
            if len(summary) > 800:
                summary = summary[:800] + "…"

            return f"Wikipedia summary for '{query}':\n{summary}"

        except ImportError:
            return (
                "Wikipedia tool is unavailable. "
                "Install the 'wikipedia' package: pip install wikipedia"
            )
        except Exception as exc:
            return f"Wikipedia lookup failed for '{query}'. Error: {exc}"

    return Tool(
        name="search_wikipedia",
        func=search_wikipedia,
        description=(
            "Search Wikipedia for factual, encyclopaedic information about any topic. "
            "Use this for definitions, historical facts, scientific concepts, or "
            "general knowledge questions. No API key required. "
            "Input: a topic or search term string."
        ),
    )
