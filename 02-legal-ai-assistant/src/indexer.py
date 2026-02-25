"""
indexer.py — Embed document chunks and store them in a FAISS vector index.

This module reuses the same core RAG pattern from Project 1 (rag-from-scratch):
    1. Split documents into overlapping chunks.
    2. Embed each chunk using a sentence-transformer model.
    3. Store embeddings in a FAISS index for fast similarity search.

FAISS (Facebook AI Similarity Search) is used because:
    - It runs locally with no API costs.
    - It's fast enough for single-document analysis.
    - It stores vectors on disk so we can reload without re-embedding.
"""

import os
from typing import List, Optional

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ---------------------------------------------------------------------------
# Default configuration — can be overridden via environment variables.
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _get_text_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Create a text splitter tuned for legal documents.

    We use RecursiveCharacterTextSplitter because it tries to split on natural
    boundaries (paragraphs, then sentences, then words) rather than cutting
    mid-sentence — important when a single sentence might define a legal term.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Legal docs use numbered lists and paragraph breaks heavily.
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        length_function=len,
    )


def _get_embeddings(model_name: Optional[str] = None) -> HuggingFaceEmbeddings:
    """Load the sentence-transformer embedding model.

    We default to 'all-MiniLM-L6-v2' — a small, fast model that produces
    384-dimensional embeddings. It strikes a good balance between quality
    and speed for a learning project.
    """
    model_name = model_name or DEFAULT_EMBEDDING_MODEL
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def index_document(
    documents: List[Document],
    index_path: str = "faiss_index",
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> FAISS:
    """Split documents into chunks, embed them, and store in a FAISS index.

    Args:
        documents:     List of Document objects from the parser.
        index_path:    Directory where the FAISS index will be saved.
        chunk_size:    Max characters per chunk (default: 1000).
        chunk_overlap: Overlap between consecutive chunks (default: 200).

    Returns:
        A FAISS vector store ready for similarity search.
    """
    # Read config from environment if not explicitly provided.
    chunk_size = chunk_size or int(os.environ.get("CHUNK_SIZE", DEFAULT_CHUNK_SIZE))
    chunk_overlap = chunk_overlap or int(os.environ.get("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP))

    print(f"  Splitting into chunks (size={chunk_size}, overlap={chunk_overlap})...")
    splitter = _get_text_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(documents)
    print(f"  Created {len(chunks)} chunks from {len(documents)} document sections.")

    print("  Loading embedding model (first run downloads ~90 MB)...")
    embeddings = _get_embeddings()

    print("  Building FAISS index...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Persist to disk so we don't have to re-embed next time.
    vector_store.save_local(index_path)
    print(f"  Index saved to '{index_path}/'.")

    return vector_store


def load_index(index_path: str = "faiss_index") -> FAISS:
    """Load a previously saved FAISS index from disk.

    Args:
        index_path: Directory containing the saved FAISS index.

    Returns:
        A FAISS vector store ready for similarity search.

    Raises:
        FileNotFoundError: If the index directory does not exist.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"No FAISS index found at '{index_path}'. "
            "Run index_document() first to create one."
        )

    print(f"  Loading existing FAISS index from '{index_path}'...")
    embeddings = _get_embeddings()
    vector_store = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print("  Index loaded successfully.")
    return vector_store


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Create a few dummy documents to verify the pipeline works.
    test_docs = [
        Document(
            page_content="This Agreement is entered into between Acme Corp and "
                         "Widget Inc as of January 1, 2024.",
            metadata={"source": "test.pdf", "page": 1},
        ),
        Document(
            page_content="The term of this Agreement shall be twelve (12) months "
                         "from the Effective Date, unless terminated earlier.",
            metadata={"source": "test.pdf", "page": 2},
        ),
    ]

    vs = index_document(test_docs, index_path="/tmp/test_faiss_index")
    results = vs.similarity_search("How long is the contract?", k=1)
    print(f"\nSearch result: {results[0].page_content[:100]}...")
