"""
main.py — Legal AI Assistant Entry Point

Full analysis pipeline for legal contracts:
  1. Parse document  → structured text + sections
  2. Index for RAG   → FAISS vector store
  3. Summarize       → executive summary (parties, type, obligations)
  4. Extract clauses → indemnification, IP, termination, etc.
  5. Analyze risks   → HIGH/MEDIUM/LOW risk flags
  6. Detect conflicts→ internal contradictions
  7. Q&A             → answer specific questions or enter interactive mode

Usage examples:
  python main.py --file data/sample_contracts/service_agreement.pdf
  python main.py --file contract.pdf --model gpt-3.5-turbo --skip-conflicts
  python main.py --file contract.pdf --question "What are my termination rights?"
  python main.py --file contract.pdf --interactive
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box

# ---------------------------------------------------------------------------
# Load environment variables from .env file
# ---------------------------------------------------------------------------
load_dotenv()

console = Console()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Legal AI Assistant — contract analysis powered by LLMs + RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --file contract.pdf
  python main.py --file contract.pdf --model gpt-3.5-turbo --skip-risks
  python main.py --file contract.pdf --question "Who owns the IP?"
  python main.py --file contract.pdf --interactive
        """,
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to the contract file (PDF or DOCX). Required unless --interactive.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("OPENAI_MODEL", "gpt-4"),
        help="OpenAI model to use (default: gpt-4).",
    )
    parser.add_argument(
        "--skip-risks",
        action="store_true",
        help="Skip the risk analysis step.",
    )
    parser.add_argument(
        "--skip-conflicts",
        action="store_true",
        help="Skip the conflict detection step.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Ask a single question about the contract and exit.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive Q&A loop after analysis.",
    )
    return parser


# ---------------------------------------------------------------------------
# Rich display helpers
# ---------------------------------------------------------------------------

def print_disclaimer() -> None:
    """Print the mandatory legal disclaimer prominently."""
    console.print(
        Panel(
            "⚠️  [bold yellow]DISCLAIMER[/bold yellow]\n\n"
            "This tool is for [bold]educational purposes only[/bold]. "
            "It does [bold red]NOT[/bold red] constitute legal advice.\n"
            "Always consult a qualified attorney before making any legal or business decisions.",
            title="[bold red]LEGAL NOTICE[/bold red]",
            border_style="red",
            padding=(1, 4),
        )
    )


def print_section(title: str) -> None:
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]", style="cyan"))


def print_summary(summary: dict) -> None:
    from src.summarizer import format_summary_output
    console.print(
        Panel(
            format_summary_output(summary),
            title="[bold green]Executive Summary[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )


def print_clauses(clauses: list) -> None:
    if not clauses:
        console.print("[dim]No clauses extracted.[/dim]")
        return

    table = Table(
        title="Extracted Clauses",
        box=box.ROUNDED,
        show_lines=True,
        style="blue",
    )
    table.add_column("Type", style="bold cyan", no_wrap=True, min_width=20)
    table.add_column("Section", style="dim", min_width=10)
    table.add_column("Plain English", style="white")

    for clause in clauses:
        table.add_row(
            clause.get("clause_type", "").replace("_", " ").title(),
            clause.get("section_reference", "Unknown"),
            clause.get("plain_english", ""),
        )

    console.print(table)


def print_risks(risks: list) -> None:
    from src.risk_analyzer import format_risk_output
    console.print(
        Panel(
            format_risk_output(risks),
            title="[bold red]Risk Analysis[/bold red]",
            border_style="red",
            padding=(1, 2),
        )
    )


def print_conflicts(conflicts: list) -> None:
    from src.conflict_detector import format_conflicts_output
    console.print(
        Panel(
            format_conflicts_output(conflicts),
            title="[bold yellow]Conflict Detection[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        )
    )


# ---------------------------------------------------------------------------
# Interactive Q&A loop
# ---------------------------------------------------------------------------

def run_interactive_qa(qa_chain) -> None:
    """
    Enter a REPL-style loop so the user can ask multiple questions about
    the contract without re-running the full analysis each time.
    """
    console.print(
        Panel(
            "Type your question and press [bold]Enter[/bold].\n"
            "Type [bold]'exit'[/bold] or [bold]'quit'[/bold] to stop.\n\n"
            "Sample questions:\n"
            "  • What are my termination rights?\n"
            "  • Who owns the IP I create?\n"
            "  • What is the liability cap?\n"
            "  • How does auto-renewal work?\n"
            "  • What information must I keep confidential?",
            title="[bold cyan]Interactive Q&A Mode[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    from src.qa_chain import ask_question

    while True:
        try:
            question = console.input("\n[bold cyan]Question >[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Exiting Q&A mode.[/dim]")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            console.print("[dim]Exiting Q&A mode.[/dim]")
            break

        with console.status("[bold green]Thinking...[/bold green]"):
            answer = ask_question(question, qa_chain)

        console.print(
            Panel(
                answer,
                title="[bold green]Answer[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Validate API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        console.print(
            "[bold red]ERROR:[/bold red] OPENAI_API_KEY not set. "
            "Copy .env.example to .env and add your key."
        )
        sys.exit(1)

    # Validate file argument
    if not args.file:
        console.print(
            "[bold red]ERROR:[/bold red] --file is required. "
            "Provide a path to a PDF or DOCX contract."
        )
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(args.file):
        console.print(f"[bold red]ERROR:[/bold red] File not found: {args.file}")
        sys.exit(1)

    # ── Startup banner ──────────────────────────────────────────────────────
    console.print(
        Panel(
            "[bold white]Legal AI Assistant[/bold white]\n"
            f"[dim]Contract:[/dim] {os.path.basename(args.file)}\n"
            f"[dim]Model   :[/dim] {args.model}",
            border_style="white",
            padding=(1, 4),
        )
    )
    print_disclaimer()

    # ── Import modules here to keep startup fast for --help ─────────────────
    from langchain_openai import ChatOpenAI
    from src.document_parser import parse_legal_document
    from src.indexer import index_document, get_retriever
    from src.summarizer import generate_summary
    from src.clause_extractor import extract_clauses
    from src.risk_analyzer import analyze_risks
    from src.conflict_detector import detect_conflicts
    from src.qa_chain import build_qa_chain, ask_question

    # Initialise LLM
    llm = ChatOpenAI(
        model=args.model,
        temperature=0,          # deterministic output for legal analysis
        openai_api_key=api_key,
    )

    # ── Step 1: Parse document ───────────────────────────────────────────────
    print_section("Step 1 — Parsing Document")
    with console.status("[bold green]Parsing document...[/bold green]"):
        doc = parse_legal_document(args.file)

    console.print(
        f"  ✅ Parsed [bold]{doc['file_name']}[/bold] — "
        f"{doc['page_count']} page(s), {len(doc['sections'])} section(s) detected"
    )

    # ── Step 2: Index for RAG ────────────────────────────────────────────────
    print_section("Step 2 — Building Vector Index")
    index_path = f"legal_index_{os.path.splitext(doc['file_name'])[0]}"
    with console.status("[bold green]Indexing document...[/bold green]"):
        vector_store = index_document(args.file, index_path=index_path)
    retriever = get_retriever(vector_store, k=4)
    console.print(f"  ✅ Index built at [bold]{index_path}/[/bold]")

    # ── Step 3: Executive Summary ────────────────────────────────────────────
    print_section("Step 3 — Executive Summary")
    with console.status("[bold green]Generating summary...[/bold green]"):
        summary = generate_summary(doc["full_text"], llm)
    print_summary(summary)

    # ── Step 4: Clause Extraction ────────────────────────────────────────────
    print_section("Step 4 — Key Clause Extraction")
    with console.status("[bold green]Extracting clauses...[/bold green]"):
        clauses = extract_clauses(doc["full_text"], llm)
    console.print(f"  ✅ {len(clauses)} clause(s) extracted")
    print_clauses(clauses)

    # ── Step 5: Risk Analysis ────────────────────────────────────────────────
    if not args.skip_risks:
        print_section("Step 5 — Risk Analysis")
        with console.status("[bold green]Analyzing risks...[/bold green]"):
            risks = analyze_risks(clauses, llm)
        console.print(f"  ✅ {len(risks)} risk(s) identified")
        print_risks(risks)
    else:
        console.print("[dim]Risk analysis skipped (--skip-risks).[/dim]")
        risks = []

    # ── Step 6: Conflict Detection ───────────────────────────────────────────
    if not args.skip_conflicts:
        print_section("Step 6 — Conflict Detection")
        with console.status("[bold green]Detecting conflicts...[/bold green]"):
            conflicts = detect_conflicts(clauses, llm)
        console.print(f"  ✅ {len(conflicts)} potential conflict(s) found")
        print_conflicts(conflicts)
    else:
        console.print("[dim]Conflict detection skipped (--skip-conflicts).[/dim]")

    # ── Step 7: Q&A ──────────────────────────────────────────────────────────
    qa_chain = build_qa_chain(retriever, llm)

    if args.question:
        # Single question mode — answer and exit
        print_section("Q&A — Single Question")
        with console.status("[bold green]Thinking...[/bold green]"):
            answer = ask_question(args.question, qa_chain)
        console.print(f"\n[bold cyan]Q:[/bold cyan] {args.question}")
        console.print(
            Panel(
                answer,
                title="[bold green]Answer[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )

    elif args.interactive:
        print_section("Step 7 — Interactive Q&A")
        run_interactive_qa(qa_chain)

    else:
        console.print(
            "\n[dim]Tip: run with [bold]--interactive[/bold] to ask follow-up questions, "
            "or [bold]--question \"...[/bold]\" for a single query.[/dim]"
        )

    console.print(
        Panel(
            "✅  Analysis complete.\n\n"
            "[bold yellow]Reminder:[/bold yellow] Always verify findings with a qualified attorney.",
            border_style="green",
            padding=(1, 2),
        )
    )


if __name__ == "__main__":
    main()
