"""
table_indexer.py — Embed table descriptions and store them in a FAISS index.

Each document in this index contains:
  • page_content = the natural-language description of the table
  • metadata     = {"source_table": "/path/to/table.csv"}

When a table description matches a user query, we can load the raw CSV
from metadata to show exact numbers.
"""

import os
import logging

from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def index_table_descriptions(
    table_data: list[dict],
    embeddings,
    index_path: str = "data/indexes/table_index",
) -> FAISS:
    """
    Build a FAISS vector store from table descriptions.

    Parameters
    ----------
    table_data : list[dict]
        Each dict has {"path": str, "description": str} — output of
        table_processor.process_tables().
    embeddings : Embeddings
        LangChain embedding model.
    index_path : str
        Where to save the persisted index.

    Returns
    -------
    FAISS vector store, or None if there are no tables.
    """
    if not table_data:
        logger.warning("No table descriptions to index.")
        return None

    documents = [
        Document(
            page_content=item["description"],
            metadata={"source_table": item["path"]},
        )
        for item in table_data
    ]

    vectorstore = FAISS.from_documents(documents, embeddings)

    os.makedirs(os.path.dirname(index_path) if os.path.dirname(index_path) else ".", exist_ok=True)
    vectorstore.save_local(index_path)
    logger.info("Table-description index saved → %s  (%d entries)", index_path, len(documents))

    return vectorstore


def load_table_index(embeddings, index_path: str = "data/indexes/table_index") -> FAISS:
    """
    Reload a previously-saved table-description FAISS index.

    Parameters
    ----------
    embeddings : Embeddings
        Must match the model used at build time.
    index_path : str
        Directory containing the FAISS index files.

    Returns
    -------
    FAISS vector store, or None if the path does not exist.
    """
    if not os.path.isdir(index_path):
        logger.warning("Table index not found at %s", index_path)
        return None

    vectorstore = FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )
    logger.info("Loaded table index from %s", index_path)
    return vectorstore
