"""
src/tools/search_tool.py
------------------------
LangChain Tool that lets the agent do semantic search over indexed papers.

WHAT IS A LANGCHAIN TOOL?
--------------------------
A Tool is a Python function wrapped in a thin object that carries three things:
  1. name        – a short identifier (e.g., "search_papers")
  2. description – a plain-English explanation of WHEN and HOW to use the tool
  3. func        – the actual callable that receives a string and returns a string

The agent never inspects the function's source code.  It only reads the
name + description to decide whether to call the tool.

HOW THE AGENT DECIDES WHEN TO USE THIS TOOL
--------------------------------------------
During each reasoning step the ReAct agent compares its current sub-goal
(e.g., "I need to find papers about attention") to every tool's description.
If "search_papers" says "Search across all indexed research papers …" that
is an obvious match.  Poor descriptions cause the agent to either skip a
useful tool or call the wrong one.

THE INPUT/OUTPUT CONTRACT
--------------------------
  Input  – a plain string (the search query).
  Output – a plain string that the agent reads as an observation.

LangChain enforces this contract: whatever your func returns is converted to
str and injected into the agent's prompt as the "Observation:" line.

WHY TOOL DESCRIPTIONS MUST BE PRECISE
---------------------------------------
The agent is stateless — it has no memory of tool internals.  If the
description says "search papers" without clarifying the expected input format,
the agent might pass a JSON object or a question instead of a keyword query,
producing poor results.  Explicit examples in the description (like "Input: a
search query string") dramatically improve reliability.
"""

from langchain.tools import Tool

from src.paper_indexer import search_papers


def create_search_tool(vector_store) -> Tool:
    """Build and return a LangChain Tool that searches the FAISS index.

    Parameters
    ----------
    vector_store : FAISS
        The populated FAISS vector store built by paper_indexer.index_papers().

    Returns
    -------
    Tool
        Ready-to-use LangChain Tool instance.
    """

    def _search(query: str) -> str:
        """Internal function called by the agent with a plain query string."""
        docs = search_papers(query, vector_store, k=3)

        if not docs:
            return "No relevant passages found for that query."

        parts = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "Unknown paper")
            # page is 0-indexed in PyPDFLoader; add 1 for human readability
            page = doc.metadata.get("page", 0) + 1
            snippet = doc.page_content.strip()[:400]  # keep response concise
            parts.append(
                f"[Result {i}]\n"
                f"  Paper : {source}\n"
                f"  Page  : {page}\n"
                f"  Text  : {snippet}…"
            )

        return "\n\n".join(parts)

    return Tool(
        name="search_papers",
        description=(
            "Search across all indexed research papers. "
            "Use this to find relevant information, methodologies, or findings. "
            "Input: a search query string."
        ),
        func=_search,
    )
