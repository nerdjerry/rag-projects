"""
text_indexer.py
---------------
Builds and queries a FAISS vector index for plain-text chunks.

Same embedding approach as Project 1 — this is the text modality index.

We reuse the all-MiniLM-L6-v2 sentence-transformer model because:
  * It is fast and runs fully locally (no API calls, no cost).
  * Its 384-dimensional embeddings strike a good balance between quality
    and memory / speed.
  * It has proven strong retrieval performance on diverse Q&A benchmarks.

The only difference from Project 1 is that this index is *one of three*
indexes in the multimodal pipeline.  The query router decides whether to
hit this index, the image index, the table index, or all three.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document


# Shared embedding model — instantiated once to avoid repeated model loading.
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return a HuggingFaceEmbeddings instance for all-MiniLM-L6-v2."""
    return HuggingFaceEmbeddings(model_name=_EMBED_MODEL_NAME)


def index_text_chunks(
    text_blocks: list[str],
    index_path: str = "text_faiss_index",
) -> FAISS:
    """
    Embed a list of text strings and persist them as a FAISS index.

    Parameters
    ----------
    text_blocks : Raw text strings (one per page, paragraph, or chunk).
    index_path  : Directory path where the FAISS index files are saved.

    Returns
    -------
    A LangChain FAISS vector store ready for similarity search.
    """
    if not text_blocks:
        raise ValueError("text_blocks is empty — nothing to index.")

    # Wrap each string in a LangChain Document so we can store metadata.
    # We record the chunk number so retrieved results can be traced back.
    docs = [
        Document(page_content=block, metadata={"chunk_id": i, "modality": "text"})
        for i, block in enumerate(text_blocks)
    ]

    embeddings = _get_embeddings()

    # FAISS.from_documents embeds all docs in a single batch and builds
    # the index in memory, then we persist it to disk.
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(index_path)

    print(f"[text_indexer] Indexed {len(docs)} text chunks → '{index_path}'")
    return vector_store


def load_text_index(index_path: str) -> FAISS:
    """
    Load a previously saved FAISS text index from disk.

    Parameters
    ----------
    index_path : Directory path that was passed to index_text_chunks().

    Returns
    -------
    A LangChain FAISS vector store.
    """
    embeddings = _get_embeddings()
    vector_store = FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )
    print(f"[text_indexer] Loaded text index from '{index_path}'")
    return vector_store


def search_text(query: str, vector_store: FAISS, k: int = 3) -> list:
    """
    Retrieve the top-k most relevant text chunks for a query.

    Parameters
    ----------
    query        : Natural language question or search string.
    vector_store : A loaded or freshly-built FAISS text index.
    k            : Number of results to return.

    Returns
    -------
    List of (Document, score) tuples ordered by descending similarity.
    Lower L2 distance = higher similarity in FAISS.
    """
    results = vector_store.similarity_search_with_score(query, k=k)
    return results
