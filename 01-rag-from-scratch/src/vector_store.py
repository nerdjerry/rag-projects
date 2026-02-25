# src/vector_store.py
#
# STEP 4 OF THE RAG PIPELINE: STORING VECTORS IN FAISS
#
# WHAT IS FAISS?
# ---------------
# FAISS (Facebook AI Similarity Search) is an open-source library developed by
# Meta (Facebook) Research. It is specifically designed for one task:
#
#   Given a query vector, quickly find the most similar vectors in a large collection.
#
# This is called "Approximate Nearest Neighbor" (ANN) search. Doing this naively
# (comparing the query against every stored vector one by one) would be too slow
# at scale. FAISS builds an *index* — a special data structure that lets it find
# the top-k similar vectors in milliseconds, even across millions of documents.
#
# HOW FAISS WORKS (CONCEPTUALLY):
# ---------------------------------
# 1. During indexing:  FAISS takes all your chunk vectors and organizes them into
#    a spatial data structure (e.g., an inverted file index or HNSW graph).
# 2. During search:    Given a query vector, FAISS navigates the data structure to
#    find the nearest neighbors without checking every single vector.
#
# For our use case (hundreds to thousands of chunks), FAISS is near-instant.
# It really shines at millions of vectors, but it's a great habit to use from day one.
#
# WHY SAVE THE INDEX TO DISK?
# -----------------------------
# Embedding documents takes time (each chunk must be processed by the neural network).
# If we re-ran embedding every time we started the app, we'd waste seconds/minutes
# on every run even when the documents haven't changed.
#
# By saving the FAISS index to disk, we only embed once. On subsequent runs, we
# load the pre-built index from disk in milliseconds.
#
# The saved index consists of two files:
#   faiss_index/index.faiss  → the actual vector index (binary)
#   faiss_index/index.pkl    → metadata mapping (which chunk belongs to which vector)

import os
from langchain_community.vectorstores import FAISS


def create_vector_store(chunks: list, embedding_model) -> FAISS:
    """
    Embed all chunks and build a FAISS vector store from them.

    This is the "indexing" phase — it calls the embedding model once per chunk
    (or in batches) and stores all resulting vectors in a FAISS index.

    Args:
        chunks (list):          List of LangChain Document objects (from chunker.py).
        embedding_model:        A loaded HuggingFaceEmbeddings model (from embedder.py).

    Returns:
        FAISS: An in-memory FAISS vector store ready for similarity search.
    """

    print(f"\n🗄️  Building FAISS vector store from {len(chunks)} chunks...")
    print(f"   (Embedding each chunk — this may take a moment on first run)")

    # FAISS.from_documents() does two things in one call:
    #   1. Calls embedding_model.embed_documents() on all chunks
    #   2. Builds the FAISS index from the resulting vectors
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model,
    )

    print(f"✅ Vector store created with {len(chunks)} vectors")
    return vector_store


def save_vector_store(vector_store: FAISS, path: str = "faiss_index") -> None:
    """
    Persist the FAISS index to disk so we don't have to re-embed next time.

    Saves two files:
        {path}/index.faiss  — the binary vector index
        {path}/index.pkl    — the document metadata mapping

    Args:
        vector_store (FAISS): The in-memory FAISS vector store to save.
        path (str):           Directory path where index files will be written.
                              Default: "faiss_index"
    """

    # Create the directory if it doesn't exist yet
    os.makedirs(path, exist_ok=True)

    # LangChain's FAISS wrapper handles the actual serialization
    vector_store.save_local(path)

    print(f"💾 Vector store saved to '{path}/'")
    print(f"   Files: {path}/index.faiss, {path}/index.pkl")


def load_vector_store(path: str, embedding_model) -> FAISS:
    """
    Load a previously saved FAISS index from disk.

    Args:
        path (str):        Directory path where index files are stored.
        embedding_model:   The SAME embedding model used when the index was created.
                           IMPORTANT: If you use a different model, the vectors won't
                           match and search results will be nonsense.

    Returns:
        FAISS: The loaded vector store, ready for similarity search.
    """

    print(f"\n📂 Loading existing FAISS index from '{path}/'...")

    # allow_dangerous_deserialization=True is required because FAISS uses pickle
    # under the hood. This is safe as long as you trust the source of the index file
    # (which you do, since you created it yourself).
    vector_store = FAISS.load_local(
        folder_path=path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )

    print(f"✅ Vector store loaded from disk")
    return vector_store


def get_or_create_vector_store(
    chunks: list,
    embedding_model,
    path: str = "faiss_index",
) -> FAISS:
    """
    Convenience function: load index from disk if it exists, otherwise build it.

    This is the function you'll call in main.py. It implements a simple cache:
      - If '{path}/index.faiss' exists → load it (fast, skips re-embedding)
      - Otherwise → embed all chunks and build a new index, then save it

    Args:
        chunks (list):        List of LangChain Document chunks.
        embedding_model:      Loaded embedding model.
        path (str):           Path to save/load the FAISS index. Default: "faiss_index"

    Returns:
        FAISS: Ready-to-use vector store.
    """

    # Check if a saved index already exists on disk
    index_file = os.path.join(path, "index.faiss")

    if os.path.exists(index_file):
        print(f"\n♻️  Found existing FAISS index at '{path}/' — loading from disk")
        print(f"   (Skipping re-embedding. Delete '{path}/' to force rebuild.)")
        return load_vector_store(path, embedding_model)
    else:
        print(f"\n🆕 No existing index found at '{path}/' — building from scratch")
        vector_store = create_vector_store(chunks, embedding_model)
        save_vector_store(vector_store, path)
        return vector_store
