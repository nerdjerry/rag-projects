"""
src/tools/rag_tool.py

Wraps the FAISS knowledge base search as a LangChain Tool so the agent can
call it alongside web search, finance, and weather tools.

KEY CONCEPT — Tool Description is Everything:
    The agent's LLM reads the `description` field to decide WHEN to call this
    tool.  A vague description like "search documents" leads to the agent using
    the tool for every question.  A specific description that says what the
    knowledge base CONTAINS helps the agent make the right routing decision:
    internal docs → RAG, live prices → finance_tool, current events → web_search.
"""

from typing import List

from langchain.tools import Tool
from langchain_community.vectorstores import FAISS

from src.knowledge_indexer import search_knowledge_base


def create_rag_tool(
    vector_store: FAISS,
    domain_description: str = "internal company documents",
) -> Tool:
    """
    Build a LangChain Tool that searches the FAISS knowledge base.

    Input/Output contract (required by LangChain Tools):
        - Input:  always a plain string — the search query the agent constructs.
        - Output: always a plain string — the formatted retrieval results.
    The agent cannot pass Python objects; everything is serialised to/from text.

    Args:
        vector_store:       Loaded FAISS index returned by knowledge_indexer.
        domain_description: Short phrase describing WHAT is stored in the KB,
                            e.g. "Q3 financial forecasts and product roadmaps".
                            This is injected into the tool description so the
                            LLM knows exactly when to use it.

    Returns:
        A configured LangChain Tool ready to add to the agent's tool list.
    """

    def _search(query: str) -> str:
        """
        Inner function called by LangChain when the agent invokes this tool.
        The agent provides `query` as a plain string.
        """
        chunks: List[str] = search_knowledge_base(query, vector_store, k=3)

        if not chunks:
            return "No relevant information found in knowledge base."

        # Format each retrieved chunk with a numbered label so the LLM can
        # reference specific chunks in its final answer.
        lines = ["Found in knowledge base:"]
        for i, chunk in enumerate(chunks, start=1):
            # Strip excess whitespace from the chunk to keep the context window tidy.
            clean = " ".join(chunk.split())
            lines.append(f"{i}. {clean}")

        return "\n".join(lines)

    # ---------------------------------------------------------------------------
    # The description is intentionally verbose:
    #   • "internal policies, product documentation, or stored knowledge" signals
    #     the agent to use this for anything that would appear in static docs.
    #   • Mentioning the domain_description further narrows the scope.
    #   • Ending with "Input: a search query string" sets the input format
    #     expectation clearly so the agent passes a plain query, not JSON.
    # ---------------------------------------------------------------------------
    description = (
        f"Search {domain_description} for relevant information. "
        "Use this for questions about internal policies, product documentation, "
        "or stored knowledge. "
        "Input: a search query string."
    )

    return Tool(
        name="search_knowledge_base",
        func=_search,
        description=description,
    )
