"""
document_loader.py — Load documents from the data/ folder.

A LangChain "Document" object is a simple container with two fields:
  - page_content (str): the actual text extracted from the file
  - metadata (dict): information about where the text came from
    (e.g., file path, page number for PDFs)

This module reads .pdf, .txt, and .docx files and converts each one
into a list of Document objects so the rest of the pipeline can
process them uniformly regardless of the original file format.
"""

import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)


# Map file extensions to their corresponding LangChain loader class.
# Each loader knows how to read a specific file format and return
# a list of Document objects.
LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
}


def load_documents(data_dir: str = "data") -> list:
    """
    Walk through *data_dir*, find every supported file, and return
    a flat list of LangChain Document objects.

    Args:
        data_dir: Path to the folder that contains your source files.
                  Sub-folders (e.g., data/sample_docs/) are included
                  automatically.

    Returns:
        A list of Document objects — one per page (PDF) or one per
        file (TXT / DOCX).
    """
    documents = []

    # os.walk lets us recurse into sub-directories like data/sample_docs/
    for root, _dirs, files in os.walk(data_dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            ext = os.path.splitext(filename)[1].lower()

            if ext not in LOADER_MAP:
                # Skip unsupported file types silently.
                continue

            loader_cls = LOADER_MAP[ext]
            try:
                loader = loader_cls(filepath)
                docs = loader.load()  # returns a list of Document objects
                documents.extend(docs)
                print(f"  ✓ Loaded {filename} ({len(docs)} document(s))")
            except Exception as exc:
                print(f"  ✗ Failed to load {filename}: {exc}")

    print(f"\nTotal documents loaded: {len(documents)}")
    return documents


# ---------------------------------------------------------------------------
# Quick self-test: run this file directly to see if loading works.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    docs = load_documents("data")
    for doc in docs[:3]:
        print(f"\n--- {doc.metadata} ---")
        print(doc.page_content[:200], "...")
