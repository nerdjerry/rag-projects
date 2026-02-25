"""
Search Tool — A LangChain Tool that searches indexed papers for relevant chunks.

WHAT IS A LANGCHAIN TOOL?
A LangChain Tool is a wrapper that gives an LLM-powered agent the ability to
*do things* beyond just generating text. Each tool has:
  - A **name** the agent refers to (e.g. "search_papers")
  - A **description** the agent reads to decide *when* to use the tool
  - A **function** that actually runs when the agent calls the tool

HOW AGENTS USE TOOLS
When you give an agent a list of tools, it reads their descriptions and
decides — at each step of its reasoning loop — whether calling a tool would
help answer the user's question. For example, if the user asks "What methods
did Paper X use?", the agent might decide to call `search_papers` with the
query "Paper X methodology".

The agent never sees the tool's source code — it only sees the name and
description. That's why a clear, specific description is essential.
"""

from langchain_community.vectorstores import FAISS
from langchain_core.tools import Tool


def _build_search_fn(vector_store: FAISS, top_k: int = 5):
    """Return a closure that searches the vector store.

    We use a closure (a function inside a function) so the vector_store
    and top_k values are "baked in" — the agent just passes a query string.
    """

    def search_papers(query: str) -> str:
        """Search indexed research papers and return the most relevant chunks."""
        # similarity_search returns Document objects ranked by relevance
        results = vector_store.similarity_search(query, k=top_k)

        if not results:
            return "No relevant results found for this query."

        # Format results so the agent can read them easily
        formatted = []
        for i, doc in enumerate(results, 1):
            meta = doc.metadata
            formatted.append(
                f"--- Result {i} ---\n"
                f"Paper: {meta.get('paper_title', 'Unknown')}\n"
                f"Section: {meta.get('section', 'Unknown')}\n"
                f"Page: {meta.get('page', '?')}\n"
                f"Content:\n{doc.page_content}\n"
            )
        return "\n".join(formatted)

    return search_papers


def create_search_tool(vector_store: FAISS, top_k: int = 5) -> Tool:
    """Create a LangChain Tool for searching indexed papers.

    Args:
        vector_store: A FAISS vector store built by paper_indexer.
        top_k: Number of results to return per query.

    Returns:
        A LangChain Tool instance the agent can call.
    """
    return Tool(
        name="search_papers",
        # The description is critical — the agent reads this to decide when
        # to use the tool. Be specific about what the tool does and what
        # kind of input it expects.
        description=(
            "Search indexed research papers for information relevant to a "
            "query. Input should be a natural-language search query. Returns "
            "the most relevant text chunks along with paper title, section, "
            "and page number."
        ),
        func=_build_search_fn(vector_store, top_k),
    )
