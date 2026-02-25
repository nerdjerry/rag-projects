"""
src/knowledge_indexer.py

This is the same RAG indexing pattern from Project 1, reused here as one of the
agent's tools. The key difference: in Project 1 this was the ONLY retrieval path;
here it is just one tool the agent may or may not call depending on the question.
"""

import os
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------------
# Embedding model — same lightweight model used in Project 1.
# Using a local sentence-transformers model avoids OpenAI embedding API calls
# and keeps costs at zero for the indexing step.
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFace embedding model instance."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def index_knowledge_base(
    kb_dir: str,
    index_path: str = "kb_faiss_index",
) -> FAISS:
    """
    Load every .pdf and .txt file from kb_dir, chunk the text, build a FAISS
    vector index, and persist it to disk.

    If the index already exists on disk it is loaded directly — re-indexing is
    skipped so the agent starts up fast after the first run.

    Args:
        kb_dir:      Directory containing source documents (.pdf / .txt).
        index_path:  Path where the FAISS index folder will be saved.

    Returns:
        A ready-to-query FAISS vector store.
    """
    # If the index was already built, load it and return immediately.
    if os.path.exists(index_path):
        print(f"[Indexer] Loading existing FAISS index from '{index_path}'")
        return load_index(index_path)

    print(f"[Indexer] Building new FAISS index from '{kb_dir}'")

    # --- Step 1: Load documents using two loaders — one for PDFs, one for TXTs ---
    documents = []

    pdf_loader = DirectoryLoader(
        kb_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        silent_errors=True,
    )
    txt_loader = DirectoryLoader(
        kb_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        silent_errors=True,
    )

    for loader in (pdf_loader, txt_loader):
        try:
            docs = loader.load()
            documents.extend(docs)
            print(f"[Indexer]   Loaded {len(docs)} pages/docs via {loader.__class__.__name__}")
        except Exception as exc:
            print(f"[Indexer]   Warning: loader {loader.__class__.__name__} failed — {exc}")

    if not documents:
        print(f"[Indexer] No documents found in '{kb_dir}'. Index will be empty.")
        # Create a minimal placeholder doc so FAISS doesn't crash on an empty list.
        from langchain.schema import Document
        documents = [
            Document(
                page_content="Knowledge base is empty. Add .pdf or .txt files to the knowledge_base/ folder.",
                metadata={"source": "placeholder"},
            )
        ]

    # --- Step 2: Split documents into overlapping chunks ---
    # chunk_size=500 tokens keeps chunks small enough for a single context window slot.
    # chunk_overlap=50 ensures sentences cut at a boundary don't lose context.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(documents)
    print(f"[Indexer] Split into {len(chunks)} chunks")

    # --- Step 3: Embed and index ---
    embeddings = _get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    # --- Step 4: Persist to disk ---
    vector_store.save_local(index_path)
    print(f"[Indexer] FAISS index saved to '{index_path}'")

    return vector_store


def load_index(index_path: str) -> FAISS:
    """
    Load a previously saved FAISS index from disk.

    Args:
        index_path: Path to the directory created by FAISS.save_local().

    Returns:
        Loaded FAISS vector store ready for similarity search.
    """
    embeddings = _get_embeddings()
    vector_store = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True,  # required since LangChain 0.1.x
    )
    print(f"[Indexer] Loaded FAISS index from '{index_path}'")
    return vector_store


def search_knowledge_base(
    query: str,
    vector_store: FAISS,
    k: int = 3,
) -> List[str]:
    """
    Perform a similarity search and return the top-k text chunks.

    The agent calls this via rag_tool.py. Returning plain strings (not Documents)
    keeps the tool output easy to format and display.

    Args:
        query:        The natural-language search query.
        vector_store: A loaded FAISS index.
        k:            Number of top results to return.

    Returns:
        List of text strings — the retrieved chunks, most relevant first.
    """
    docs = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]
