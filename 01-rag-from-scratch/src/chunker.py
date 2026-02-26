# src/chunker.py
#
# STEP 2 OF THE RAG PIPELINE: CHUNKING DOCUMENTS
#
# WHY DO WE SPLIT DOCUMENTS INTO CHUNKS?
# ----------------------------------------
# Large Language Models (LLMs) have a "context window" — a hard limit on how much
# text they can receive in a single prompt. For example, GPT-3.5-turbo has a ~16k
# token limit (roughly 12,000 words). If your document is a 200-page PDF, you
# CANNOT send the whole thing to the LLM at once.
#
# Even if you could, it's wasteful: most of the document is irrelevant to any
# given question. We only want to send the 2-3 paragraphs that are actually useful.
#
# The solution: split documents into small "chunks", embed each chunk as a vector,
# store them in a vector database, and at query time retrieve ONLY the most relevant
# chunks to include in the LLM prompt.
#
# WHAT IS chunk_overlap AND WHY DOES IT MATTER?
# -----------------------------------------------
# Imagine a document with this text:
#   "...the policy expires on December 31st. Renewal must be submitted 30 days..."
#
# Without overlap, if the split happens between "December 31st." and "Renewal",
# one chunk ends with an incomplete thought and the other starts mid-context.
# With overlap (e.g., 50 characters), the second chunk will start a bit before
# "Renewal", capturing "...policy expires on December 31st. Renewal..." — giving
# the LLM enough context to understand the sentence properly.
#
# HOW DOES RecursiveCharacterTextSplitter WORK?
# -----------------------------------------------
# It tries to split text in a smart order of preference:
#   1. Split on paragraph breaks ("\n\n") first — preserves paragraph structure
#   2. If still too long, split on newlines ("\n") — preserves line structure
#   3. If still too long, split on spaces (" ") — preserves word boundaries
#   4. Last resort: split on individual characters — avoids going over limit
#
# This is smarter than a naive "split every N characters" approach because it
# tries to keep semantically coherent units together.
#
# TUNING chunk_size:
# -------------------
#   Too LARGE (e.g., 2000):  Fewer chunks, retrieval is less precise.
#                             The LLM may receive a lot of irrelevant text.
#
#   Too SMALL (e.g., 50):    Many chunks, each missing context.
#                             A sentence like "See section above" becomes meaningless alone.
#
#   Sweet spot: 300–800 characters (roughly 1-2 short paragraphs).
#   Default here is 500 with 50-character overlap — a good starting point.

from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: list,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list:
    """
    Split a list of LangChain Documents into smaller chunks.

    Each chunk is itself a LangChain Document object, inheriting the metadata
    of the original document (so we still know which file each chunk came from).

    Args:
        documents (list):     List of LangChain Document objects (from document_loader).
        chunk_size (int):     Maximum number of characters per chunk. Default: 500.
        chunk_overlap (int):  Number of characters to overlap between consecutive chunks.
                              Helps preserve context at boundaries. Default: 50.

    Returns:
        list: A (usually much longer) list of smaller LangChain Document objects.

    Example:
        chunks = chunk_documents(documents, chunk_size=500, chunk_overlap=50)
        print(f"Created {len(chunks)} chunks")
        print(chunks[0].page_content)   # first chunk's text
        print(chunks[0].metadata)       # same metadata as parent document
    """

    print(f"\n📐 Chunking {len(documents)} document(s)...")
    print(f"   Settings: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    # Create the splitter with our chosen settings.
    # separators: the list of strings it will try to split on, in order of preference.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Try these separators in order — paragraph breaks → newlines → spaces → characters
        separators=["\n\n", "\n", " ", ""],
        # length_function: how to measure "size". len() counts characters.
        # You could swap this for a token counter if you want chunk_size in tokens.
        length_function=len,
    )

    # split_documents() handles the full list at once and preserves metadata.
    # It returns a new list of Document objects — one per chunk.
    chunks = splitter.split_documents(documents)

    print(f"✅ Created {len(chunks)} chunks from {len(documents)} document(s)")
    print(
        f"   Average chunk size: "
        f"~{sum(len(c.page_content) for c in chunks) // max(len(chunks), 1)} characters"
    )

    return chunks
