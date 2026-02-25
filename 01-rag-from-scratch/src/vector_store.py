"""
vector_store.py — Store and search embeddings with FAISS.

WHAT IS FAISS?
  FAISS (Facebook AI Similarity Search) is a library for *fast*
  nearest-neighbour lookups in high-dimensional vector spaces.
  Think of it as a specialised search engine: instead of matching
  keywords, it finds vectors (and therefore text chunks) that are
  closest in meaning to a query vector.

PERSISTENCE
  Building the index from scratch every time you run the app is slow.
  We save the index to disk (the "faiss_index/" folder) so that on
  subsequent runs we can load it instantly.  If you add new documents,
  simply delete the folder and re-run to rebuild.
"""

import os
from langchain_community.vectorstores import FAISS


def create_vector_store(
    chunks: list,
    embeddings,
    index_path: str = "faiss_index",
) -> FAISS:
    """
    Create a FAISS vector store from document chunks, then save it.

    Args:
        chunks:     List of Document objects (output of chunk_documents()).
        embeddings: The embedding model (output of get_embeddings()).
        index_path: Directory where the index will be saved.

    Returns:
        A FAISS vector store ready for similarity search.
    """
    print(f"Creating FAISS index from {len(chunks)} chunk(s) ...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Persist to disk so we don't have to rebuild next time.
    vector_store.save_local(index_path)
    print(f"FAISS index saved to '{index_path}/'.")
    return vector_store


def load_vector_store(
    embeddings,
    index_path: str = "faiss_index",
) -> FAISS:
    """
    Load a previously saved FAISS index from disk.

    Args:
        embeddings: The same embedding model used when the index was created.
        index_path: Directory where the index was saved.

    Returns:
        A FAISS vector store ready for similarity search.

    Raises:
        FileNotFoundError: If the index directory does not exist.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"No FAISS index found at '{index_path}/'. "
            "Run the pipeline first to build one."
        )

    print(f"Loading FAISS index from '{index_path}/' ...")
    vector_store = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print("FAISS index loaded.")
    return vector_store


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from document_loader import load_documents
    from chunker import chunk_documents
    from embedder import get_embeddings

    docs = load_documents("data")
    chunks = chunk_documents(docs)
    emb = get_embeddings()

    vs = create_vector_store(chunks, emb)
    vs2 = load_vector_store(emb)
    print("Vector store round-trip OK.")
