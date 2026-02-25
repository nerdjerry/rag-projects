#!/usr/bin/env python3
"""
main.py — Multimodal RAG Pipeline (Project 4)

End-to-end workflow:
  1. Parse PDF  → separate text, images, tables
  2. Process    → caption images, describe tables
  3. Index      → build three FAISS indexes (text, image, table)
  4. Query      → route → retrieve → generate answer

Usage:
  python main.py                         # interactive mode
  python main.py --pdf path/to/doc.pdf   # parse + index a new document
"""

import os
import sys
import argparse
import logging

from dotenv import load_dotenv

# ── Local imports ──────────────────────────────────────────────────────────
from src.multimodal_parser import parse_multimodal_document
from src.text_indexer import index_text_chunks, load_text_index
from src.image_processor import process_images
from src.image_indexer import index_image_captions, load_image_index
from src.table_processor import process_tables
from src.table_indexer import index_table_descriptions, load_table_index
from src.query_router import route_query
from src.multi_retriever import retrieve_multimodal, format_context
from src.generator import get_llm, create_multimodal_qa_chain, generate_answer

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = "data"
EXTRACTED_DIR = os.path.join(DATA_DIR, "extracted")
INDEX_DIR = os.path.join(DATA_DIR, "indexes")
TEXT_INDEX = os.path.join(INDEX_DIR, "text_index")
IMAGE_INDEX = os.path.join(INDEX_DIR, "image_index")
TABLE_INDEX = os.path.join(INDEX_DIR, "table_index")


# ── Helpers ────────────────────────────────────────────────────────────────

def _get_embeddings():
    """Return the shared embedding model used for all indexes."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean from an environment variable."""
    return os.getenv(name, str(default)).lower() in ("true", "1", "yes")


def _env_int(name: str, default: int = 0) -> int:
    """Read an integer from an environment variable."""
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


# ── Ingest pipeline ───────────────────────────────────────────────────────

def ingest_document(pdf_path: str):
    """
    Full ingest: parse → process → index.

    This runs once per new document.  After ingestion the three FAISS
    indexes are saved to disk and can be reloaded for querying.
    """
    logger.info("=" * 60)
    logger.info("INGEST: %s", pdf_path)
    logger.info("=" * 60)

    # 1. Parse the PDF into modalities
    parsed = parse_multimodal_document(pdf_path, output_dir=EXTRACTED_DIR)
    text_chunks = parsed["text_chunks"]
    image_paths = parsed["image_paths"]
    table_paths = parsed["table_paths"]

    # 2. Set up shared resources
    embeddings = _get_embeddings()
    use_ollama = _env_bool("USE_OLLAMA")
    model_name = os.getenv("OLLAMA_MODEL", "llama3") if use_ollama else "gpt-4o"
    llm = get_llm(use_ollama=use_ollama, model=model_name)

    # 3. Index text chunks
    chunk_size = _env_int("CHUNK_SIZE", 500)
    chunk_overlap = _env_int("CHUNK_OVERLAP", 50)
    index_text_chunks(
        text_chunks, embeddings, TEXT_INDEX,
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
    )

    # 4. Caption images, then index captions
    if image_paths:
        vision_model = os.getenv("VISION_MODEL", "gpt-4o")
        image_data = process_images(
            image_paths, use_ollama=use_ollama, vision_model=vision_model,
        )
        index_image_captions(image_data, embeddings, IMAGE_INDEX)
    else:
        logger.info("No images found — skipping image indexing.")

    # 5. Describe tables, then index descriptions
    if table_paths:
        table_data = process_tables(table_paths, llm=llm)
        index_table_descriptions(table_data, embeddings, TABLE_INDEX)
    else:
        logger.info("No tables found — skipping table indexing.")

    logger.info("Ingestion complete ✓")


# ── Query pipeline ────────────────────────────────────────────────────────

def query_pipeline(query: str, llm, embeddings, top_k: int = 3) -> str:
    """
    Route → retrieve → generate for a single user query.
    """
    # Load whichever indexes exist
    indexes = {
        "TEXT": load_text_index(embeddings, TEXT_INDEX),
        "IMAGE": load_image_index(embeddings, IMAGE_INDEX),
        "TABLE": load_table_index(embeddings, TABLE_INDEX),
    }

    # Check that at least one index exists
    if not any(indexes.values()):
        return (
            "No indexes found.  Please ingest a document first:\n"
            "  python main.py --pdf data/sample_docs/your_doc.pdf"
        )

    # Route the query to the relevant indexes
    router_result = route_query(query, llm)

    # Retrieve top-k from relevant indexes
    chunks = retrieve_multimodal(query, indexes, router_result, k=top_k)

    if not chunks:
        return "No relevant information found in the indexed documents."

    # Format and generate
    context = format_context(chunks)
    chain = create_multimodal_qa_chain(llm)
    answer = generate_answer(chain, query, context)

    return answer


# ── Interactive loop ──────────────────────────────────────────────────────

def interactive_loop():
    """Run an interactive Q&A session in the terminal."""
    embeddings = _get_embeddings()
    use_ollama = _env_bool("USE_OLLAMA")
    model_name = os.getenv("OLLAMA_MODEL", "llama3") if use_ollama else "gpt-4o"
    llm = get_llm(use_ollama=use_ollama, model=model_name)
    top_k = _env_int("TOP_K", 3)

    print("\n" + "=" * 60)
    print("  Multimodal RAG — Interactive Q&A")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        answer = query_pipeline(query, llm, embeddings, top_k=top_k)
        print(f"\nAssistant: {answer}\n")


# ── CLI entry point ───────────────────────────────────────────────────────

def main():
    load_dotenv()  # Load .env file (API keys, config)

    parser = argparse.ArgumentParser(description="Multimodal RAG Pipeline")
    parser.add_argument("--pdf", type=str, help="Path to a PDF to ingest.")
    parser.add_argument(
        "--query", type=str,
        help="Single query to answer (non-interactive mode).",
    )
    args = parser.parse_args()

    # If a PDF is provided, run the ingest pipeline
    if args.pdf:
        ingest_document(args.pdf)
        # If no query follows, exit after ingestion
        if not args.query:
            print("\nDocument ingested.  Run without --pdf to start querying.")
            return

    # Single-query mode
    if args.query:
        embeddings = _get_embeddings()
        use_ollama = _env_bool("USE_OLLAMA")
        model_name = os.getenv("OLLAMA_MODEL", "llama3") if use_ollama else "gpt-4o"
        llm = get_llm(use_ollama=use_ollama, model=model_name)
        top_k = _env_int("TOP_K", 3)

        answer = query_pipeline(args.query, llm, embeddings, top_k=top_k)
        print(f"\n{answer}")
        return

    # Default: interactive mode
    interactive_loop()


if __name__ == "__main__":
    main()
