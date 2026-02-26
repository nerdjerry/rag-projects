"""
src/paper_indexer.py
--------------------
Embeds research papers and stores them in a FAISS vector index.

HOW METADATA TAGGING WORKS IN FAISS (via LangChain)
-----------------------------------------------------
LangChain's FAISS wrapper stores a Python dict alongside each embedded chunk.
When you call `FAISS.from_documents(docs)`, each Document's `.metadata` dict
is persisted verbatim next to its vector.  At search time, every returned
Document carries its original metadata, so you can read `doc.metadata["source"]`
to know which paper a chunk came from.

METADATA FILTERING: "SEARCH ONLY WITHIN PAPER X"
--------------------------------------------------
FAISS itself does not support SQL-style WHERE clauses — it returns the k nearest
vectors globally.  We implement per-paper filtering post-hoc: run the query
across the full index, then discard results whose `doc.metadata["source"]` does
not match the requested paper title.  This is simple and correct for small
collections (< a few thousand chunks).  For larger collections a dedicated
vector DB (Pinecone, Weaviate, Qdrant) with native metadata filters is better.

WHY INDEX ALL PAPERS TOGETHER?
-------------------------------
A single shared index enables cross-paper queries like "which papers discuss
attention mechanisms?".  If each paper had its own index you would have to
query N indexes and merge results manually.  The trade-off is that per-paper
filtering requires a post-search step, but that cost is negligible at the
scale of a typical research collection (3-50 papers).
"""

import os
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
# "all-MiniLM-L6-v2" is a fast, lightweight model (80 MB) that works well for
# semantic similarity on academic text and runs entirely locally (no API key).
_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunk parameters: 1 000 chars with 200-char overlap.
# Research paragraphs average ~500-800 chars, so a 1 000-char window usually
# captures a complete idea.  Overlap prevents a sentence at a chunk boundary
# from being split across two embeddings.
_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 200


def index_papers(
    papers_dir: str,
    index_path: str = "papers_faiss_index",
) -> FAISS:
    """Load all PDFs in *papers_dir*, chunk them, embed them, and build a FAISS index.

    Each chunk's metadata contains:
        source    – paper title derived from filename (used for filtering)
        file_path – absolute path to the source PDF
        chunk_id  – sequential integer within that paper

    Parameters
    ----------
    papers_dir : str
        Directory containing *.pdf files.
    index_path : str
        Directory where the FAISS index will be saved to disk.

    Returns
    -------
    FAISS
        The populated vector store.
    """
    papers_path = Path(papers_dir)
    pdf_files = sorted(papers_path.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{papers_dir}'.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],  # prefer paragraph → line → word splits
    )

    all_docs = []
    for pdf in pdf_files:
        print(f"[paper_indexer] Loading: {pdf.name}")
        loader = PyPDFLoader(str(pdf))
        pages = loader.load()

        # Derive a human-readable source label from the filename
        paper_title = pdf.stem.replace("_", " ").replace("-", " ")

        chunks = splitter.split_documents(pages)
        for i, chunk in enumerate(chunks):
            # Enrich metadata — LangChain's PyPDFLoader already adds 'source'
            # and 'page'; we add our own fields on top.
            chunk.metadata["source"] = paper_title
            chunk.metadata["file_path"] = str(pdf)
            chunk.metadata["chunk_id"] = i

        all_docs.extend(chunks)
        print(f"[paper_indexer]   → {len(chunks)} chunk(s)")

    print(f"[paper_indexer] Embedding {len(all_docs)} total chunks…")
    embeddings = HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(all_docs, embeddings)

    # Persist to disk so we can reload without re-embedding
    vector_store.save_local(index_path)
    print(f"[paper_indexer] Index saved to '{index_path}'.")

    return vector_store


def load_index(index_path: str = "papers_faiss_index") -> FAISS:
    """Load a previously saved FAISS index from disk.

    Parameters
    ----------
    index_path : str
        Directory where the index was saved by :func:`index_papers`.

    Returns
    -------
    FAISS
    """
    embeddings = HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL)
    vector_store = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True,  # required by LangChain ≥ 0.1
    )
    print(f"[paper_indexer] Loaded index from '{index_path}'.")
    return vector_store


def search_papers(
    query: str,
    vector_store: FAISS,
    k: int = 5,
    paper_filter: str = None,
) -> list:
    """Semantic search over the FAISS index.

    Parameters
    ----------
    query : str
        Natural-language search query.
    vector_store : FAISS
        The populated vector store.
    k : int
        Number of results to return (before optional filtering).
    paper_filter : str or None
        If provided, only return chunks whose metadata["source"] contains
        this string (case-insensitive).  This implements per-paper search.

    Returns
    -------
    list[Document]
        Matching document chunks, each with .page_content and .metadata.
    """
    # Retrieve more candidates when filtering so we still get k results after
    # dropping non-matching papers
    fetch_k = k * 4 if paper_filter else k
    results = vector_store.similarity_search(query, k=fetch_k)

    if paper_filter:
        filter_lower = paper_filter.lower()
        results = [
            doc for doc in results
            if filter_lower in doc.metadata.get("source", "").lower()
        ]

    return results[:k]
