"""
RAG Tool — Knowledge Base Retrieval
====================================
This wraps the FAISS vector store as a LangChain Tool so the agent can
query your local documents alongside real-time data sources.

IMPORTANT: The tool *description* is critical!
  The agent's LLM reads each tool's description to decide which tool to
  invoke for a given user query.  A vague description → the agent won't
  know when to use the tool.  A precise description → the agent picks it
  at the right moment.

Example flow:
  User: "What does our company policy say about remote work?"
  Agent thinks: "This is about internal documents → use knowledge_base tool"
  Agent calls: knowledge_base("remote work policy")
  Tool returns: relevant chunks from your indexed PDFs / text files
"""

from langchain.tools import Tool


def create_rag_tool(vector_store):
    """
    Create a LangChain Tool backed by a FAISS vector store.

    Args:
        vector_store: A FAISS (or compatible) vector store with indexed docs.

    Returns:
        A LangChain Tool that the agent can invoke by name.
    """
    # Build a retriever that returns the top-k most similar chunks.
    # k=3 is a good default — enough context without overwhelming the LLM.
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    def _search_knowledge_base(query: str) -> str:
        """Search the local knowledge base and return relevant passages."""
        docs = retriever.invoke(query)

        if not docs:
            return "No relevant information found in the knowledge base."

        # Format results so the agent (and the user) can see the source.
        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            results.append(
                f"[Result {i}] (source: {source})\n{doc.page_content}"
            )

        return "\n\n".join(results)

    # --- The description tells the agent WHEN to use this tool ---
    # Be specific: mention what kind of information lives here.
    tool = Tool(
        name="knowledge_base",
        func=_search_knowledge_base,
        description=(
            "Use this to answer questions about your knowledge base documents. "
            "This searches through locally indexed PDFs and text files such as "
            "company policies, research papers, or reference material. "
            "Input should be a natural-language question or search query."
        ),
    )

    return tool
