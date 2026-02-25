"""
table_indexer.py
----------------
Builds and queries a FAISS vector index over natural-language table
descriptions produced by table_processor.py.

The descriptions are embedded with the same all-MiniLM-L6-v2 model used for
text and image captions, giving us a single, consistent semantic space across
all three modalities.  The metadata carries the table_id and csv_path so
callers can retrieve the exact CSV data when needed.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document


_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=_EMBED_MODEL_NAME)


def index_table_descriptions(
    table_data: list[dict],
    index_path: str = "table_faiss_index",
) -> FAISS:
    """
    Embed table descriptions and save a FAISS index to disk.

    Parameters
    ----------
    table_data  : List of dicts with keys "table_id", "csv_path",
                  "description" (as returned by table_processor.process_all_tables()).
    index_path  : Directory where FAISS index files are written.

    Returns
    -------
    A LangChain FAISS vector store whose documents are descriptions with
    table_id and csv_path stored in metadata.
    """
    if not table_data:
        raise ValueError("table_data is empty — nothing to index.")

    docs = [
        Document(
            page_content=item["description"],
            metadata={
                "table_id": item["table_id"],
                "csv_path": item["csv_path"],
                "page": item.get("page", 0),
                "modality": "table",
            },
        )
        for item in table_data
    ]

    embeddings = _get_embeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(index_path)

    print(f"[table_indexer] Indexed {len(docs)} table descriptions → '{index_path}'")
    return vector_store


def load_table_index(index_path: str) -> FAISS:
    """
    Load a previously saved FAISS table-description index from disk.

    Parameters
    ----------
    index_path : Directory path passed to index_table_descriptions().

    Returns
    -------
    A LangChain FAISS vector store.
    """
    embeddings = _get_embeddings()
    vector_store = FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )
    print(f"[table_indexer] Loaded table index from '{index_path}'")
    return vector_store


def search_tables(
    query: str,
    vector_store: FAISS,
    k: int = 3,
) -> list[dict]:
    """
    Retrieve the top-k table descriptions most relevant to a query.

    Parameters
    ----------
    query        : Natural language question or search string.
    vector_store : A loaded or freshly-built FAISS table index.
    k            : Number of results to return.

    Returns
    -------
    List of dicts:
      {
        "description" : str   — natural-language summary of the table
        "table_id"    : str   — unique table identifier
        "csv_path"    : str   — path to the raw CSV file
        "page"        : int   — source page in the original document
        "score"       : float — FAISS L2 distance (lower = more similar)
      }
    """
    raw_results = vector_store.similarity_search_with_score(query, k=k)

    return [
        {
            "description": doc.page_content,
            "table_id": doc.metadata.get("table_id", ""),
            "csv_path": doc.metadata.get("csv_path", ""),
            "page": doc.metadata.get("page", 0),
            "score": float(score),
        }
        for doc, score in raw_results
    ]
