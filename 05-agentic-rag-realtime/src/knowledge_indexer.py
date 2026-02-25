"""
Knowledge Base Indexer
=====================
This module handles indexing local documents (PDFs, text files) into a FAISS
vector store. FAISS (Facebook AI Similarity Search) enables fast similarity
search over document embeddings — this is the "static" knowledge base that
the agent can query alongside real-time tools.

How it works:
  1. Load documents from a directory (supports PDF and .txt files)
  2. Split documents into smaller chunks for better retrieval
  3. Create embeddings for each chunk using a sentence-transformer model
  4. Store the embeddings in a FAISS index on disk for fast lookup
"""

import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


def index_knowledge_base(data_dir: str, embeddings, index_path: str = "faiss_index"):
    """
    Index all documents in `data_dir` and save the FAISS index to disk.

    Args:
        data_dir:    Path to the folder containing your documents (.pdf, .txt).
        embeddings:  A LangChain embeddings object (e.g., HuggingFaceEmbeddings).
        index_path:  Where to save the FAISS index on disk.

    Returns:
        A FAISS vector store ready for similarity search.
    """
    documents = []

    # --- Load PDF files ---
    # PyPDFLoader reads each page as a separate Document object.
    pdf_loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    try:
        pdf_docs = pdf_loader.load()
        documents.extend(pdf_docs)
        print(f"  Loaded {len(pdf_docs)} pages from PDF files.")
    except Exception as e:
        print(f"  No PDF files found or error loading PDFs: {e}")

    # --- Load plain text files ---
    txt_loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
    )
    try:
        txt_docs = txt_loader.load()
        documents.extend(txt_docs)
        print(f"  Loaded {len(txt_docs)} text files.")
    except Exception as e:
        print(f"  No text files found or error loading text files: {e}")

    if not documents:
        print("  ⚠  No documents found. Creating an empty vector store.")
        # Create a minimal vector store with a placeholder so the agent still works.
        vector_store = FAISS.from_texts(
            ["No knowledge base documents have been indexed yet."],
            embeddings,
        )
        vector_store.save_local(index_path)
        return vector_store

    # --- Split documents into chunks ---
    # Smaller chunks (500 tokens) with overlap improve retrieval accuracy.
    # The overlap ensures we don't lose context at chunk boundaries.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"  Split into {len(chunks)} chunks.")

    # --- Build and save the FAISS index ---
    print("  Building FAISS index (this may take a moment on first run)...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(index_path)
    print(f"  ✅ Index saved to '{index_path}/'.")

    return vector_store


def load_knowledge_base(embeddings, index_path: str = "faiss_index"):
    """
    Load a previously saved FAISS index from disk.

    Args:
        embeddings:  The same embeddings model used when the index was built.
        index_path:  Path where the FAISS index was saved.

    Returns:
        A FAISS vector store, or None if the index doesn't exist.
    """
    if not os.path.exists(index_path):
        print(f"  ⚠  No existing index found at '{index_path}'.")
        return None

    print(f"  Loading FAISS index from '{index_path}'...")
    vector_store = FAISS.load_local(
        index_path,
        embeddings,
        # allow_dangerous_deserialization is required by newer FAISS versions
        # because loading pickled data can execute arbitrary code. Only use
        # this with indexes YOU created.
        allow_dangerous_deserialization=True,
    )
    print("  ✅ Index loaded successfully.")
    return vector_store
