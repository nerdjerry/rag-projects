"""
Wikipedia Tool — Background Context & General Knowledge
========================================================
Uses the Wikipedia API to fetch summaries of topics.  This is a free,
unlimited API — no key required.

When to use Wikipedia vs. Web Search:
  ┌──────────────────────┬───────────────────────────────────────┐
  │  Use Wikipedia       │  Use Web Search                      │
  ├──────────────────────┼───────────────────────────────────────┤
  │  Historical facts    │  Breaking news / current events       │
  │  Scientific concepts │  Live prices / scores                 │
  │  Biographies         │  Recent product announcements         │
  │  Definitions         │  Opinions / reviews                   │
  │  Background context  │  Anything needing "as of today" data  │
  └──────────────────────┴───────────────────────────────────────┘

  Wikipedia is great for stable, factual information.  Web search is better
  when you need information that changes frequently.
"""

import wikipedia
from langchain.tools import Tool


def create_wiki_tool():
    """
    Create a LangChain Tool that searches Wikipedia for summaries.

    Returns:
        A LangChain Tool the agent can invoke by name.
    """

    def _search_wikipedia(query: str) -> str:
        """
        Search Wikipedia and return a summary of the most relevant article.

        Args:
            query: A topic or question to look up on Wikipedia.
        """
        query = query.strip()
        if not query:
            return "Please provide a topic to search on Wikipedia."

        try:
            # --- Step 1: Search for matching article titles ---
            # wikipedia.search() returns a list of page titles.
            search_results = wikipedia.search(query, results=3)

            if not search_results:
                return f"No Wikipedia articles found for: '{query}'"

            # --- Step 2: Fetch a summary of the top result ---
            # auto_suggest=False prevents Wikipedia from silently redirecting
            # to a different (sometimes wrong) page.
            try:
                summary = wikipedia.summary(
                    search_results[0],
                    sentences=4,
                    auto_suggest=False,
                )
                page = wikipedia.page(search_results[0], auto_suggest=False)
                url = page.url
            except wikipedia.DisambiguationError as e:
                # The search term matches multiple pages — pick the first option.
                summary = wikipedia.summary(
                    e.options[0],
                    sentences=4,
                    auto_suggest=False,
                )
                page = wikipedia.page(e.options[0], auto_suggest=False)
                url = page.url

            # Show related articles so the user can explore further.
            related = ", ".join(search_results[1:]) if len(search_results) > 1 else "None"

            return (
                f"📚 Wikipedia: {page.title}\n\n"
                f"{summary}\n\n"
                f"Source: {url}\n"
                f"Related topics: {related}"
            )

        except wikipedia.PageError:
            return f"No Wikipedia page found for: '{query}'. Try a different search term."
        except Exception as e:
            return f"Error searching Wikipedia: {str(e)}"

    tool = Tool(
        name="wikipedia",
        func=_search_wikipedia,
        description=(
            "Use this to look up background information, historical facts, "
            "scientific concepts, biographies, or definitions on Wikipedia. "
            "Input should be a topic name or concise question. "
            "This is free and unlimited — prefer this over web_search for "
            "general knowledge that doesn't need to be real-time."
        ),
    )

    return tool
