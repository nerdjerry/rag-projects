"""
main.py
-------
Entry point for the 03-research-agent pipeline.

Usage examples
--------------
# Parse all PDFs and start an interactive research Q&A session
python main.py --papers-dir data/papers --interactive

# Ask the agent a single question and exit
python main.py --papers-dir data/papers --query "What methodologies are used across these papers?"

# Generate a full gap analysis report
python main.py --papers-dir data/papers --topic "transformer models" --report

# Combine: generate a report and also run an interactive session
python main.py --papers-dir data/papers --topic "NLP" --report --interactive

# Save the report to a specific file
python main.py --papers-dir data/papers --topic "BERT fine-tuning" --report --output reports/bert.md
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables from .env before any other imports that might
# need OPENAI_API_KEY (e.g., langchain_openai)
# ---------------------------------------------------------------------------
load_dotenv()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="AI Research Agent — analyse a collection of research PDFs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--papers-dir",
        default=os.getenv("PAPERS_DIR", "data/papers"),
        metavar="DIR",
        help="Directory containing *.pdf files  (default: data/papers)",
    )
    parser.add_argument(
        "--topic",
        default="Research Analysis",
        metavar="TOPIC",
        help="Research topic label used in the report title  (default: 'Research Analysis')",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4"),
        metavar="MODEL",
        help="OpenAI model name  (default: gpt-4)",
    )
    parser.add_argument(
        "--query",
        default=None,
        metavar="QUESTION",
        help="Ask the agent a single question and exit.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Run gap analysis and generate a Markdown report.",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Output file path for the Markdown report (only used with --report).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive Q&A session with the agent.",
    )
    return parser


def _check_api_key() -> None:
    """Exit early with a clear error if the OpenAI key is missing."""
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "[main] ERROR: OPENAI_API_KEY environment variable is not set.\n"
            "       Copy .env.example to .env and add your key.",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _check_api_key()

    # ------------------------------------------------------------------
    # Lazy imports so startup is fast when there are argument errors
    # ------------------------------------------------------------------
    from langchain_openai import ChatOpenAI

    from src.agent import create_research_agent, run_agent
    from src.gap_analyzer import analyze_gaps, format_gap_analysis
    from src.paper_indexer import index_papers
    from src.paper_parser import parse_all_papers
    from src.report_generator import generate_report

    # ------------------------------------------------------------------
    # Step 1: Validate papers directory
    # ------------------------------------------------------------------
    papers_dir = Path(args.papers_dir)
    if not papers_dir.exists():
        print(f"[main] ERROR: Papers directory '{papers_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    pdf_count = len(list(papers_dir.glob("*.pdf")))
    if pdf_count == 0:
        print(
            f"[main] ERROR: No PDF files found in '{papers_dir}'.\n"
            "       Add research papers as .pdf files and try again.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[main] Found {pdf_count} PDF file(s) in '{papers_dir}'.")

    # ------------------------------------------------------------------
    # Step 2: Initialise LLM
    # ------------------------------------------------------------------
    print(f"[main] Using model: {args.model}")
    llm = ChatOpenAI(
        model=args.model,
        temperature=0,        # deterministic output for research tasks
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )

    # ------------------------------------------------------------------
    # Step 3: Parse all papers with LLM
    # ------------------------------------------------------------------
    print("\n[main] === Step 1/3: Parsing papers ===")
    paper_metadata = parse_all_papers(args.papers_dir, llm)

    if not paper_metadata:
        print("[main] ERROR: No papers were successfully parsed.", file=sys.stderr)
        sys.exit(1)

    print(f"[main] Parsed {len(paper_metadata)} paper(s).")

    # ------------------------------------------------------------------
    # Step 4: Index papers in FAISS
    # ------------------------------------------------------------------
    print("\n[main] === Step 2/3: Indexing papers in FAISS ===")
    vector_store = index_papers(args.papers_dir)

    # ------------------------------------------------------------------
    # Step 5: Create the research agent
    # ------------------------------------------------------------------
    print("\n[main] === Step 3/3: Building research agent ===")
    agent = create_research_agent(vector_store, paper_metadata, llm)
    print("[main] Agent ready.\n")

    # ------------------------------------------------------------------
    # Step 6a: Generate report (--report)
    # ------------------------------------------------------------------
    if args.report:
        print("[main] Running gap analysis…")
        gaps = analyze_gaps(paper_metadata, llm)

        print(format_gap_analysis(gaps))

        report = generate_report(
            paper_metadata_list=paper_metadata,
            gap_analysis=gaps,
            topic=args.topic,
            output_path=args.output,
        )
        print(f"[main] Report generated ({len(report)} characters).")

    # ------------------------------------------------------------------
    # Step 6b: Single query (--query)
    # ------------------------------------------------------------------
    if args.query:
        run_agent(args.query, agent)

    # ------------------------------------------------------------------
    # Step 6c: Interactive session (--interactive)
    # ------------------------------------------------------------------
    if args.interactive:
        print("\n[main] Entering interactive mode. Type 'exit' or 'quit' to stop.\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[main] Exiting.")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q"}:
                print("[main] Goodbye!")
                break

            run_agent(user_input, agent)

    # If no action flag was given, print help
    if not args.report and not args.query and not args.interactive:
        parser.print_help()
        print(
            "\n[main] No action specified. Use --query, --report, or --interactive."
        )


if __name__ == "__main__":
    main()
