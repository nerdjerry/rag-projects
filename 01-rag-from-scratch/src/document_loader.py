# src/document_loader.py
#
# STEP 1 OF THE RAG PIPELINE: LOADING DOCUMENTS
#
# Before we can answer questions about your documents, we first need to READ them.
# This module handles loading different file types (.pdf, .txt, .docx) into a
# common format that LangChain can work with.
#
# What is a LangChain "Document" object?
# ----------------------------------------
# LangChain uses a Document object to represent a piece of text. It has two fields:
#
#   document.page_content  → the actual text string (e.g., "The capital of France is Paris...")
#   document.metadata      → a dict with info about where the text came from
#                            e.g., {"source": "data/sample_docs/report.pdf", "page": 2}
#
# Why use Documents instead of plain strings?
#   Because we want to keep track of WHERE each piece of text came from.
#   When the LLM answers a question, we can tell the user "this answer came from page 3 of report.pdf"
#   — that's only possible if we preserve the metadata through the pipeline.

import os
from pathlib import Path

# LangChain community loaders for different file types.
# These loaders know how to read each format and return a list of Document objects.
from langchain_community.document_loaders import (
    PyPDFLoader,      # Reads PDF files — returns one Document per page
    TextLoader,       # Reads plain .txt files — returns one Document per file
    Docx2txtLoader,   # Reads .docx (Word) files — returns one Document per file (needs the docx2txt package)
)


def load_documents(data_dir: str) -> list:
    """
    Load all supported documents from a directory.

    Walks through the given directory, finds all .pdf, .txt, and .docx files,
    loads each one using the appropriate loader, and returns a flat list of
    LangChain Document objects.

    Args:
        data_dir (str): Path to the folder containing your documents.
                        e.g., "data/sample_docs"

    Returns:
        list: A list of LangChain Document objects. Each document has:
              - page_content: the text extracted from the file
              - metadata: dict containing at minimum {"source": <file_path>}

    Example:
        documents = load_documents("data/sample_docs")
        print(documents[0].page_content)   # prints raw text
        print(documents[0].metadata)       # prints {"source": "data/sample_docs/report.pdf", "page": 0}
    """

    # Convert to a Path object for easier cross-platform file handling
    data_path = Path(data_dir)

    # Make sure the directory actually exists before trying to read it
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data directory '{data_dir}' does not exist. "
            f"Please create it and add some .pdf, .txt, or .docx files."
        )

    all_documents = []  # We'll collect all Document objects here

    # Map each file extension to its corresponding LangChain loader class.
    # This makes it easy to add new file types later — just add an entry here.
    loader_map = {
        ".pdf":  PyPDFLoader,
        ".txt":  TextLoader,
        ".docx": Docx2txtLoader,
    }

    # Walk through every file in the directory (and subdirectories)
    for file_path in sorted(data_path.rglob("*")):

        # Skip directories — we only want files
        if not file_path.is_file():
            continue

        # Skip hidden files (e.g., .gitkeep, .DS_Store)
        if file_path.name.startswith("."):
            continue

        file_ext = file_path.suffix.lower()  # e.g., ".pdf", ".txt", ".docx"

        # Check if we have a loader for this file type
        if file_ext not in loader_map:
            # Unsupported file type — skip it with a warning
            print(f"  ⚠️  Skipping unsupported file type: {file_path.name} ({file_ext})")
            continue

        print(f"  📄 Loading: {file_path.name}")

        try:
            # Instantiate the appropriate loader with the file path
            loader_class = loader_map[file_ext]
            loader = loader_class(str(file_path))

            # .load() returns a list of Document objects.
            # For PDFs, each page becomes its own Document.
            # For TXT/DOCX, the whole file is usually one Document.
            documents = loader.load()

            print(f"      → Loaded {len(documents)} document chunk(s)")
            all_documents.extend(documents)

        except Exception as e:
            # Don't crash the whole pipeline if one file fails to load.
            # Print the error and continue with the remaining files.
            print(f"  ❌ Failed to load {file_path.name}: {e}")
            continue

    if len(all_documents) == 0:
        print(
            f"\n⚠️  No documents were loaded from '{data_dir}'.\n"
            f"   Add some .pdf, .txt, or .docx files and try again."
        )
    else:
        print(f"\n✅ Total documents loaded: {len(all_documents)}")

    return all_documents
