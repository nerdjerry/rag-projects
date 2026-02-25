"""
text_indexer.py — Chunk, embed, and index text passages with FAISS.

This is the same pattern used in Project 1 (basic RAG).  We split the raw
text into overlapping chunks, embed each chunk with a sentence-transformer
model, and store the vectors in a FAISS index for fast similarity search.
"""

import os
import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def index_text_chunks(
    chunks: list[str],
    embeddings,
    index_path: str = "data/indexes/text_index",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> FAISS:
    """
    Split raw text passages into smaller chunks, embed them, and persist a
    FAISS vector store to disk.

    Parameters
    ----------
    chunks : list[str]
        Raw text blocks (e.g. one per PDF page) from the parser.
    embeddings : Embeddings
        A LangChain-compatible embedding model (e.g. HuggingFaceEmbeddings).
    index_path : str
        Where to save the FAISS index on disk.
    chunk_size : int
        Target size (in characters) for each text chunk.
    chunk_overlap : int
        Overlap between consecutive chunks so we don't cut sentences in half.

    Returns
    -------
    FAISS vector store ready for similarity search.
    """
    if not chunks:
        logger.warning("No text chunks to index.")
        return None

    # --- Step 1: split text into smaller, overlapping pieces ----------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    documents = splitter.create_documents(chunks)
    logger.info("Split %d raw blocks → %d chunks (size=%d, overlap=%d)",
                len(chunks), len(documents), chunk_size, chunk_overlap)

    # --- Step 2: embed and build the FAISS index ----------------------------
    vectorstore = FAISS.from_documents(documents, embeddings)

    # --- Step 3: persist to disk so we can reload later ---------------------
    os.makedirs(os.path.dirname(index_path) if os.path.dirname(index_path) else ".", exist_ok=True)
    vectorstore.save_local(index_path)
    logger.info("Text index saved → %s", index_path)

    return vectorstore


def load_text_index(embeddings, index_path: str = "data/indexes/text_index") -> FAISS:
    """
    Reload a previously-saved FAISS text index from disk.

    Parameters
    ----------
    embeddings : Embeddings
        Must be the **same** embedding model used when the index was built.
    index_path : str
        Directory that contains the FAISS index files.

    Returns
    -------
    FAISS vector store, or None if the index does not exist.
    """
    if not os.path.isdir(index_path):
        logger.warning("Text index not found at %s", index_path)
        return None

    vectorstore = FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )
    logger.info("Loaded text index from %s", index_path)
    return vectorstore
