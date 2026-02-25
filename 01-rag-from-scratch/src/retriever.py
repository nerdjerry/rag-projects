"""
retriever.py — Find the most relevant chunks for a user's question.

HOW RETRIEVAL WORKS
  1. The user's question is converted into a vector (embedding).
  2. FAISS compares that vector against every chunk vector in the index.
  3. The top-k chunks with the highest similarity scores are returned.

WHAT DOES 'k' MEAN?
  'k' is the number of chunks to retrieve.  A higher k gives the LLM
  more context but also more noise.  A lower k is more focused but
  risks missing relevant information.  k=3 is a sensible default —
  feel free to experiment.
"""


def retrieve_relevant_chunks(vector_store, query: str, k: int = 3) -> list:
    """
    Search the vector store for the chunks most similar to *query*.

    Args:
        vector_store: A FAISS vector store (from create/load_vector_store).
        query:        The user's natural-language question.
        k:            How many top results to return (default 3).

    Returns:
        A list of Document objects ranked by similarity (most similar first).
    """
    results = vector_store.similarity_search(query, k=k)

    # Print retrieved chunks so learners can inspect what gets passed
    # to the LLM in the generation step.
    print(f"\n🔍 Retrieved {len(results)} chunk(s) for: \"{query}\"\n")
    for i, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        print(f"--- Chunk {i} (source: {source}) ---")
        print(doc.page_content[:300])
        if len(doc.page_content) > 300:
            print("...")
        print()

    return results


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from document_loader import load_documents
    from chunker import chunk_documents
    from embedder import get_embeddings
    from vector_store import create_vector_store

    docs = load_documents("data")
    chunks = chunk_documents(docs)
    emb = get_embeddings()
    vs = create_vector_store(chunks, emb)

    retrieve_relevant_chunks(vs, "What is this document about?")
