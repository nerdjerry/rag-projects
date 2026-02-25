"""
Paper Indexer — Chunks research papers and stores them in a FAISS vector index.

HOW METADATA FILTERING WORKS
When we store chunks in a vector store, each chunk gets a metadata dict
attached to it (e.g. {"paper_title": "...", "section": "Introduction"}).
Later, when the agent searches for information, it can filter by metadata —
for example, "only search within Paper X" or "only look at methodology
sections". This is much faster and more precise than scanning every chunk
in the store.

Think of metadata like tags on a blog post: you can search all posts, or
narrow your search to posts with a specific tag.
"""

import os
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_pages(file_path: str) -> list[tuple[str, int]]:
    """Return a list of (page_text, page_number) tuples from a PDF."""
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append((text, i + 1))
    return pages


def _guess_section(text: str) -> str:
    """Heuristic to label a chunk with its likely section name.

    This is intentionally simple — a production system might use the LLM
    itself to classify sections. For learning purposes, keyword matching
    is transparent and easy to debug.
    """
    lower = text[:300].lower()
    if "abstract" in lower:
        return "abstract"
    if "introduction" in lower:
        return "introduction"
    if "method" in lower:
        return "methodology"
    if "result" in lower:
        return "results"
    if "discussion" in lower:
        return "discussion"
    if "conclusion" in lower:
        return "conclusion"
    if "reference" in lower:
        return "references"
    return "body"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def index_papers(
    papers_dir: str,
    embeddings,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> FAISS:
    """Read all PDFs in *papers_dir*, chunk them, and build a FAISS index.

    Each chunk is annotated with metadata so downstream tools can filter
    by paper title, page number, or section.

    Args:
        papers_dir:    Directory containing PDF files.
        embeddings:    A LangChain Embeddings instance (e.g. HuggingFace).
        chunk_size:    Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks to preserve context.

    Returns:
        A FAISS vector store ready for similarity search.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # These separators work well for academic text — paragraphs first,
        # then sentences, then words.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_documents: list[Document] = []

    # Walk through every PDF in the directory
    pdf_files = sorted(
        f
        for f in os.listdir(papers_dir)
        if f.lower().endswith(".pdf")
    )

    if not pdf_files:
        print(f"⚠️  No PDF files found in {papers_dir}")

    for filename in pdf_files:
        file_path = os.path.join(papers_dir, filename)
        paper_title = os.path.splitext(filename)[0]
        print(f"📄 Indexing: {filename}")

        pages = _extract_pages(file_path)
        for page_text, page_num in pages:
            # Split the page into smaller chunks
            chunks = splitter.split_text(page_text)
            for chunk in chunks:
                # Attach metadata to every chunk — this is the key part!
                # The metadata travels with the chunk into the vector store
                # and can be used for filtering at query time.
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "paper_title": paper_title,
                        "source_file": filename,
                        "page": page_num,
                        "section": _guess_section(chunk),
                    },
                )
                all_documents.append(doc)

    if not all_documents:
        # Create a placeholder so the vector store is still usable
        all_documents.append(
            Document(
                page_content="No papers indexed yet.",
                metadata={"paper_title": "placeholder", "source_file": "none"},
            )
        )

    # Build the FAISS index from all collected documents
    vector_store = FAISS.from_documents(all_documents, embeddings)
    print(f"✅ Indexed {len(all_documents)} chunks from {len(pdf_files)} paper(s)")
    return vector_store


def save_paper_index(vector_store: FAISS, index_path: str) -> None:
    """Persist the FAISS index to disk so it can be loaded later."""
    vector_store.save_local(index_path)
    print(f"💾 Index saved to {index_path}")


def load_paper_index(index_path: str, embeddings) -> Optional[FAISS]:
    """Load a previously saved FAISS index from disk.

    Args:
        index_path: Directory where the index was saved.
        embeddings: The same embeddings model used when the index was created
                    (must match — different models produce different vectors).

    Returns:
        A FAISS vector store, or None if the index doesn't exist.
    """
    if not os.path.exists(index_path):
        print(f"⚠️  No index found at {index_path}")
        return None

    vector_store = FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )
    print(f"📂 Loaded index from {index_path}")
    return vector_store
