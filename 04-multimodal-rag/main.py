"""
main.py — Multimodal RAG Pipeline
-----------------------------------
Orchestrates the full multimodal retrieval-augmented generation pipeline:
  1. Parse PDF → extract text, images, tables
  2. Index each modality in its own FAISS vector store
  3. Route a user query to the relevant index(es)
  4. Retrieve top-k results
  5. Generate a grounded answer

Usage examples
--------------
# Full pipeline (index + query)
python main.py --file data/sample_docs/annual_report.pdf --query "What was Q4 revenue?"

# Skip image captioning to save GPT-4V cost during development
python main.py --file data/sample_docs/annual_report.pdf --query "Summarise the findings" --skip-images

# Interactive mode — ask multiple questions after indexing once
python main.py --file data/sample_docs/annual_report.pdf --interactive
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI

from src.multimodal_parser import parse_document
from src.text_indexer import index_text_chunks
from src.image_processor import process_all_images
from src.image_indexer import index_image_captions
from src.table_processor import process_all_tables
from src.table_indexer import index_table_descriptions
from src.query_router import classify_query
from src.multi_retriever import retrieve_all, merge_and_rank_results
from src.generator import generate_answer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multimodal RAG: answer questions over text, images, and tables in a PDF."
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to the PDF document to process.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Question to answer.  Required unless --interactive is set.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="OpenAI model for text generation (default: env OPENAI_MODEL or gpt-4).",
    )
    parser.add_argument(
        "--vision-model",
        default=None,
        dest="vision_model",
        help="OpenAI vision model for image captioning (default: gpt-4-vision-preview).",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        dest="skip_images",
        help="Skip GPT-4V image captioning (saves cost/time during development).",
    )
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        dest="skip_tables",
        help="Skip LLM-based table description generation.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="After indexing, enter an interactive Q&A loop.",
    )
    return parser


def answer_query(
    query: str,
    llm,
    text_index,
    image_index,
    table_index,
) -> str:
    """Route → retrieve → generate for a single query."""
    print(f"\n[main] Query: {query}")

    query_types = classify_query(query, llm)
    print(f"[main] Router selected modalities: {[qt.value for qt in query_types]}")

    raw_results = retrieve_all(
        query=query,
        query_types=query_types,
        text_index=text_index,
        image_index=image_index,
        table_index=table_index,
        k=3,
    )

    ranked_results = merge_and_rank_results(raw_results)
    print(f"[main] Retrieved {len(ranked_results)} result(s) after merge/de-dup.")

    answer = generate_answer(query, ranked_results, llm)
    return answer


def main() -> None:
    load_dotenv()

    parser = build_arg_parser()
    args = parser.parse_args()

    # ── Validate arguments ────────────────────────────────────────────────────
    if not args.interactive and args.query is None:
        parser.error("--query is required unless --interactive is set.")

    if not os.path.isfile(args.file):
        print(f"[main] ERROR: File not found: {args.file}")
        sys.exit(1)

    # ── Resolve model names ───────────────────────────────────────────────────
    text_model = args.model or os.getenv("OPENAI_MODEL", "gpt-4")
    vision_model = args.vision_model or os.getenv("VISION_MODEL", "gpt-4-vision-preview")
    images_dir = os.getenv("IMAGES_OUTPUT_DIR", "data/extracted/images")
    tables_dir = os.getenv("TABLES_OUTPUT_DIR", "data/extracted/tables")

    # ── Initialise clients ────────────────────────────────────────────────────
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("[main] ERROR: OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")
        sys.exit(1)

    llm = ChatOpenAI(model=text_model, openai_api_key=openai_api_key)
    openai_client = OpenAI(api_key=openai_api_key)

    # ── Step 1: Parse document ────────────────────────────────────────────────
    print(f"\n[main] Parsing document: {args.file}")
    doc = parse_document(args.file, images_dir=images_dir, tables_dir=tables_dir)

    print(
        f"[main] Found {len(doc.text_blocks)} text blocks, "
        f"{len(doc.image_paths)} images, "
        f"{len(doc.tables)} tables."
    )

    # ── Step 2: Index text ────────────────────────────────────────────────────
    text_index = None
    if doc.text_blocks:
        print(f"\n[main] Indexing {len(doc.text_blocks)} text blocks …")
        text_index = index_text_chunks(doc.text_blocks, index_path="text_faiss_index")
    else:
        print("[main] No text blocks found — skipping text index.")

    # ── Step 3: Caption and index images ──────────────────────────────────────
    image_index = None
    if not args.skip_images and doc.image_paths:
        print(f"\n[main] Captioning {len(doc.image_paths)} image(s) with {vision_model} …")
        print("       ⚠️  GPT-4V calls cost more than text models.")
        print("       Use --skip-images during development to avoid these charges.")
        image_data = process_all_images(doc.image_paths, openai_client, vision_model)
        print(f"\n[main] Indexing {len(image_data)} image caption(s) …")
        image_index = index_image_captions(image_data, index_path="image_faiss_index")
    elif args.skip_images:
        print("\n[main] --skip-images set: skipping image captioning and indexing.")
    else:
        print("\n[main] No images found in document.")

    # ── Step 4: Process and index tables ─────────────────────────────────────
    table_index = None
    if not args.skip_tables and doc.tables:
        print(f"\n[main] Processing {len(doc.tables)} table(s) …")
        table_data = process_all_tables(doc.tables, llm, tables_dir=tables_dir)
        print(f"[main] Indexing {len(table_data)} table description(s) …")
        table_index = index_table_descriptions(table_data, index_path="table_faiss_index")
    elif args.skip_tables:
        print("\n[main] --skip-tables set: skipping table processing and indexing.")
    else:
        print("\n[main] No tables found in document.")

    # ── Step 5: Answer query / interactive loop ───────────────────────────────
    print("\n" + "─" * 60)

    if args.interactive:
        print("[main] Interactive mode. Type 'quit' or 'exit' to stop.\n")
        while True:
            try:
                query = input("Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[main] Exiting.")
                break
            if query.lower() in ("quit", "exit", "q"):
                print("[main] Exiting.")
                break
            if not query:
                continue
            answer = answer_query(query, llm, text_index, image_index, table_index)
            print(f"\nAnswer:\n{answer}\n")
            print("─" * 60)
    else:
        answer = answer_query(args.query, llm, text_index, image_index, table_index)
        print(f"\nAnswer:\n{answer}\n")


if __name__ == "__main__":
    main()
