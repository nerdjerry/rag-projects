"""
chunker.py — Split documents into smaller, overlapping chunks.

WHY DO WE CHUNK?
  LLMs have a limited context window (the amount of text they can read
  at once).  Instead of feeding entire documents, we break them into
  bite-sized pieces so we can later retrieve only the most relevant ones.

WHY OVERLAP?
  Imagine a sentence that starts at the end of one chunk and finishes at
  the beginning of the next.  Without overlap the meaning would be split
  across two chunks and neither chunk would capture the full idea.
  A small overlap (e.g., 50 characters) ensures important context at
  chunk boundaries is preserved in both neighbouring chunks.

CHUNK SIZE — TOO LARGE vs. TOO SMALL:
  • Too large  → Each chunk covers many topics, making retrieval less
                  precise.  The LLM also receives more irrelevant text.
  • Too small  → Individual chunks may lack enough context for the LLM
                  to give a good answer, and retrieval may return
                  fragments instead of coherent passages.
  A size of ~500 characters is a reasonable starting point for most
  documents.  Experiment with your own data to find the sweet spot.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: list,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list:
    """
    Split a list of LangChain Document objects into smaller chunks.

    Args:
        documents:     Output of load_documents() — a list of Documents.
        chunk_size:    Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive
                       chunks (see "WHY OVERLAP?" above).

    Returns:
        A list of Document objects, each representing one chunk.  The
        original metadata (source file, page number, etc.) is carried
        over so you can always trace a chunk back to its origin.
    """
    # RecursiveCharacterTextSplitter tries a hierarchy of separators
    # ("\n\n", "\n", " ", "") so it splits at natural boundaries first.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} document(s) into {len(chunks)} chunk(s) "
          f"(chunk_size={chunk_size}, overlap={chunk_overlap})")
    return chunks


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from document_loader import load_documents

    docs = load_documents("data")
    chunks = chunk_documents(docs)
    if chunks:
        print(f"\nExample chunk (first 200 chars):\n{chunks[0].page_content[:200]}")
