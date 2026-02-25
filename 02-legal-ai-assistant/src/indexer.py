"""
indexer.py — Vector-Store Indexing for Legal Documents

This module is intentionally similar to the RAG indexer from Project 1
(01-rag-from-scratch). The same pattern — chunk → embed → store in FAISS —
works equally well for legal documents. The only difference is that legal
chunks benefit from slightly larger sizes because legal sentences are long
and context-dependent (a 512-token chunk mid-clause may miss the crucial
subject from the sentence above).

Reuse note:
    If you already built a FAISS index in Project 1, the load_index() and
    get_retriever() helpers here are identical. The legal domain just requires
    a different source document and potentially different chunk sizes.
"""

import os

# HuggingFaceEmbeddings runs locally — no API key needed for embedding.
# We default to "all-MiniLM-L6-v2" which is fast and good for semantic search.
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# RecursiveCharacterTextSplitter tries to split on paragraphs, then sentences,
# then words — preserving as much semantic context as possible per chunk.
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document


# ---------------------------------------------------------------------------
# Chunking configuration
# ---------------------------------------------------------------------------

# Legal sentences are verbose; 1200-char chunks with 200-char overlap keeps
# clauses intact while still providing sufficient retrieval granularity.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# Embedding model — same as Project 1, works well for legal text
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return a HuggingFaceEmbeddings instance (downloaded on first call)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _chunk_text(full_text: str) -> list[Document]:
    """
    Split raw contract text into overlapping chunks suitable for embedding.

    The RecursiveCharacterTextSplitter cascades through separators:
      ["\n\n", "\n", " ", ""] — so it prefers to break at paragraph boundaries,
      then line breaks, then spaces. This keeps sentences from being split in
      the middle of a legal obligation where the subject is at the start and
      the verb is at the end.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    # Wrap in a single Document so the splitter returns Document objects
    docs = splitter.create_documents([full_text])
    return docs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def index_document(file_path: str, index_path: str = "legal_faiss_index") -> FAISS:
    """
    Parse, chunk, embed, and persist a contract document to a FAISS index.

    Parameters
    ----------
    file_path  : str — path to the PDF/DOCX contract
    index_path : str — directory where the FAISS index will be saved

    Returns
    -------
    FAISS vector store ready for similarity search.
    """
    # --- Step 1: Load raw text ---
    # Use PyPDFLoader for PDFs; for DOCX we read via document_parser then wrap
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        raw_docs = loader.load()
        full_text = "\n".join(d.page_content for d in raw_docs)
    else:
        # For non-PDF files, fall back to document_parser's full_text
        from src.document_parser import extract_full_text
        full_text = extract_full_text(file_path)

    # --- Step 2: Chunk ---
    chunks = _chunk_text(full_text)
    print(f"[Indexer] Created {len(chunks)} chunks from '{os.path.basename(file_path)}'")

    # --- Step 3: Embed & build FAISS index ---
    embeddings = _get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    # --- Step 4: Persist to disk ---
    os.makedirs(index_path, exist_ok=True)
    vector_store.save_local(index_path)
    print(f"[Indexer] Index saved to '{index_path}'")

    return vector_store


def load_index(index_path: str = "legal_faiss_index") -> FAISS:
    """
    Load a previously saved FAISS index from disk.

    Parameters
    ----------
    index_path : str — directory containing the saved FAISS index files

    Returns
    -------
    FAISS vector store ready for similarity search.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"No FAISS index found at '{index_path}'. "
            "Run index_document() first to create the index."
        )
    embeddings = _get_embeddings()
    vector_store = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True,  # required by newer LangChain versions
    )
    print(f"[Indexer] Loaded index from '{index_path}'")
    return vector_store


def get_retriever(vector_store: FAISS, k: int = 4):
    """
    Wrap a FAISS vector store as a LangChain retriever.

    Parameters
    ----------
    vector_store : FAISS — the in-memory or loaded vector store
    k            : int  — number of chunks to retrieve per query (default 4)

    Returns
    -------
    A LangChain BaseRetriever that can be plugged into any chain.

    Note: k=4 is a good balance for legal Q&A — enough context to answer most
    clause-level questions without exceeding typical context-window limits.
    """
    return vector_store.as_retriever(search_kwargs={"k": k})
