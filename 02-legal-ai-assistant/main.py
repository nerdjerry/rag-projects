#!/usr/bin/env python3
"""
main.py — Legal AI Assistant: Analyze contracts using RAG and LLMs.

This is the entry point that orchestrates the full analysis pipeline:
    1. Parse the document (PDF / DOCX / TXT)
    2. Build a FAISS vector index for RAG
    3. Generate an executive summary
    4. Extract key clauses with plain-English translations
    5. Run risk analysis on extracted clauses
    6. Detect internal conflicts between clauses
    7. Launch interactive Q&A so you can ask follow-up questions

⚠ DISCLAIMER: This tool is for educational purposes only. It is NOT legal advice.
   Always consult a qualified attorney for legal decisions.

Usage:
    python main.py <path-to-contract>
    python main.py data/sample_contracts/example_nda.pdf
"""

import json
import os
import sys

from dotenv import load_dotenv

# Load environment variables from .env file (if present).
load_dotenv()

from src.document_parser import parse_legal_document
from src.indexer import index_document, load_index
from src.summarizer import generate_summary
from src.clause_extractor import extract_clauses
from src.risk_analyzer import analyze_risks
from src.conflict_detector import detect_conflicts
from src.qa_chain import create_legal_qa_chain, ask_legal_question, format_answer_with_sources


# ---------------------------------------------------------------------------
# Rich is optional — we use it for pretty terminal output if available.
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown
    from rich import print as rprint
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# ---------------------------------------------------------------------------
# LLM setup — supports OpenAI (cloud) and Ollama (local, free)
# ---------------------------------------------------------------------------

def _create_llm():
    """Create the LLM instance based on environment configuration.

    Supports two providers:
        - OpenAI (default): Requires OPENAI_API_KEY in .env
        - Ollama (local):   Set USE_OLLAMA=true in .env (free, runs locally)
    """
    use_ollama = os.environ.get("USE_OLLAMA", "false").lower() == "true"

    if use_ollama:
        from langchain_community.llms import Ollama
        model = os.environ.get("OLLAMA_MODEL", "llama3")
        print(f"Using Ollama with model: {model}")
        print("Make sure Ollama is running: ollama serve")
        return Ollama(model=model)
    else:
        from langchain_openai import ChatOpenAI
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key or api_key == "your-openai-api-key-here":
            print("ERROR: OPENAI_API_KEY not set.")
            print("Either set it in .env or switch to Ollama (USE_OLLAMA=true).")
            sys.exit(1)
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---------------------------------------------------------------------------
# Output formatting helpers
# ---------------------------------------------------------------------------

def _print_header(title: str):
    """Print a section header."""
    if HAS_RICH:
        console.print(f"\n[bold cyan]{'═' * 60}[/bold cyan]")
        console.print(f"[bold cyan]  {title}[/bold cyan]")
        console.print(f"[bold cyan]{'═' * 60}[/bold cyan]")
    else:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")


def _print_summary(summary: dict):
    """Pretty-print the executive summary."""
    _print_header("EXECUTIVE SUMMARY")

    if HAS_RICH:
        # Parties table
        if summary.get("parties"):
            table = Table(title="Parties")
            table.add_column("Name", style="green")
            table.add_column("Role", style="white")
            for party in summary["parties"]:
                table.add_row(
                    party.get("name", "Unknown"),
                    party.get("role", "Unknown"),
                )
            console.print(table)

        # Contract info
        console.print(f"\n[bold]Contract Type:[/bold] {summary.get('type', 'Unknown')}")

        duration = summary.get("duration", {})
        if isinstance(duration, dict):
            console.print(f"[bold]Effective Date:[/bold] {duration.get('effective_date', 'N/A')}")
            console.print(f"[bold]Expiration:[/bold] {duration.get('expiration_date', 'N/A')}")
            console.print(f"[bold]Renewal:[/bold] {duration.get('renewal_terms', 'N/A')}")

        # Key obligations
        obligations = summary.get("key_obligations", [])
        if obligations:
            console.print("\n[bold]Key Obligations:[/bold]")
            for ob in obligations:
                console.print(f"  • [green]{ob.get('party', '?')}[/green]: {ob.get('obligation', '?')}")

        # Summary text
        console.print(Panel(summary.get("summary", "No summary available."), title="Summary"))
    else:
        print(json.dumps(summary, indent=2))


def _print_clauses(clauses: list):
    """Pretty-print extracted clauses."""
    _print_header("EXTRACTED CLAUSES")

    for i, clause in enumerate(clauses, start=1):
        clause_type = clause.get("clause_type", "Unknown")
        original = clause.get("original_text", "N/A")
        plain = clause.get("plain_english", "N/A")

        if HAS_RICH:
            console.print(f"\n[bold yellow]{i}. {clause_type}[/bold yellow]")
            console.print(f"  [dim]Original:[/dim] {original[:200]}{'...' if len(original) > 200 else ''}")
            console.print(f"  [bold green]Plain English:[/bold green] {plain}")
        else:
            print(f"\n{i}. {clause_type}")
            print(f"   Original: {original[:200]}{'...' if len(original) > 200 else ''}")
            print(f"   Plain English: {plain}")


def _print_risks(risks: list):
    """Pretty-print risk assessments."""
    _print_header("RISK ANALYSIS")

    if not risks:
        print("  No risks identified.")
        return

    # Color-code severity levels.
    severity_colors = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "green"}

    for i, risk in enumerate(risks, start=1):
        severity = risk.get("severity", "UNKNOWN").upper()
        category = risk.get("risk_category", "Unknown")
        reason = risk.get("reason", "N/A")
        suggestion = risk.get("suggestion", "N/A")

        if HAS_RICH:
            color = severity_colors.get(severity, "white")
            console.print(f"\n[bold {color}][{severity}] {category}[/bold {color}]")
            console.print(f"  Reason: {reason}")
            console.print(f"  [italic]Suggestion: {suggestion}[/italic]")
        else:
            print(f"\n[{severity}] {category}")
            print(f"  Reason: {reason}")
            print(f"  Suggestion: {suggestion}")


def _print_conflicts(conflicts: list):
    """Pretty-print detected conflicts."""
    _print_header("CONFLICT DETECTION")

    if not conflicts:
        print("  No internal conflicts detected.")
        return

    for i, conflict in enumerate(conflicts, start=1):
        if HAS_RICH:
            console.print(f"\n[bold red]Conflict {i}: {conflict.get('conflict_type', 'Unknown')}[/bold red]")
            console.print(f"  Clause A: {conflict.get('clause_a', 'N/A')}")
            console.print(f"  Clause B: {conflict.get('clause_b', 'N/A')}")
            console.print(f"  Issue: {conflict.get('description', 'N/A')}")
            console.print(f"  Impact: {conflict.get('impact', 'N/A')}")
            console.print(f"  [italic]Fix: {conflict.get('recommendation', 'N/A')}[/italic]")
        else:
            print(f"\nConflict {i}: {conflict.get('conflict_type', 'Unknown')}")
            print(f"  Clause A: {conflict.get('clause_a', 'N/A')}")
            print(f"  Clause B: {conflict.get('clause_b', 'N/A')}")
            print(f"  Issue: {conflict.get('description', 'N/A')}")
            print(f"  Impact: {conflict.get('impact', 'N/A')}")
            print(f"  Fix: {conflict.get('recommendation', 'N/A')}")


# ---------------------------------------------------------------------------
# Interactive Q&A loop
# ---------------------------------------------------------------------------

def _run_interactive_qa(vector_store, llm):
    """Launch an interactive Q&A session about the document."""
    _print_header("INTERACTIVE Q&A")
    print("Ask questions about the contract. Type 'quit' or 'exit' to stop.\n")

    chain = create_legal_qa_chain(vector_store, llm)

    while True:
        try:
            question = input("❓ Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Q&A.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Exiting Q&A. Goodbye!")
            break

        print("  Searching document and generating answer...\n")
        answer, sources = ask_legal_question(chain, question)
        formatted = format_answer_with_sources(answer, sources)

        if HAS_RICH:
            console.print(Panel(formatted, title="Answer", border_style="green"))
        else:
            print(formatted)
        print()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    """Run the full legal document analysis pipeline."""
    # --- Parse command-line arguments ---
    if len(sys.argv) < 2:
        print("Legal AI Assistant — Contract Analysis Tool")
        print()
        print("Usage: python main.py <path-to-contract>")
        print()
        print("Supported formats: .pdf, .docx, .txt")
        print("Example: python main.py data/sample_contracts/nda.pdf")
        sys.exit(1)

    file_path = sys.argv[1]
    print()
    print("⚖  Legal AI Assistant — Contract Analysis Tool")
    print("   ⚠ DISCLAIMER: For learning purposes only. Not legal advice.\n")

    # --- Step 1: Parse the document ---
    _print_header("STEP 1: PARSING DOCUMENT")
    print(f"  File: {file_path}")
    documents = parse_legal_document(file_path)
    print(f"  Parsed {len(documents)} sections.")

    # --- Step 2: Build FAISS index ---
    _print_header("STEP 2: BUILDING SEARCH INDEX")
    index_path = os.path.join(os.path.dirname(file_path) or ".", "faiss_index")
    vector_store = index_document(documents, index_path=index_path)

    # --- Step 3: Create LLM ---
    _print_header("STEP 3: INITIALIZING LLM")
    llm = _create_llm()

    # --- Step 4: Generate summary ---
    _print_header("STEP 4: GENERATING SUMMARY")
    summary = generate_summary(documents, llm)
    _print_summary(summary)

    # --- Step 5: Extract clauses ---
    _print_header("STEP 5: EXTRACTING CLAUSES")
    clauses = extract_clauses(documents, llm)
    _print_clauses(clauses)

    # --- Step 6: Risk analysis ---
    _print_header("STEP 6: RISK ANALYSIS")
    risks = analyze_risks(clauses, llm)
    _print_risks(risks)

    # --- Step 7: Conflict detection ---
    _print_header("STEP 7: CONFLICT DETECTION")
    conflicts = detect_conflicts(clauses, llm)
    _print_conflicts(conflicts)

    # --- Step 8: Interactive Q&A ---
    _run_interactive_qa(vector_store, llm)

    print("\n✅ Analysis complete. Remember: always have a lawyer review the contract!")


if __name__ == "__main__":
    main()
