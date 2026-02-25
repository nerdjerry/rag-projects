"""
main.py — Entry point for the AI Research Agent.

WORKFLOW
========
1. Load environment variables (.env)
2. Initialise the LLM (OpenAI or Ollama)
3. Parse all PDF papers in data/papers/
4. Index paper chunks into a FAISS vector store
5. Create agent tools (search, summarise, compare)
6. Run gap analysis across all papers
7. Generate a Markdown report
8. Enter an interactive query loop so users can ask follow-up questions

USAGE
=====
  python main.py                 # Run the full pipeline
  python main.py --interactive   # Skip analysis, go straight to Q&A
"""

import argparse
import os
import sys

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Local imports — these are the modules we built in src/
# ---------------------------------------------------------------------------
from src.paper_parser import parse_paper
from src.paper_indexer import index_papers, save_paper_index, load_paper_index
from src.tools.search_tool import create_search_tool
from src.tools.summary_tool import create_summary_tool
from src.tools.compare_tool import create_compare_tool
from src.agent import create_research_agent, run_agent
from src.gap_analyzer import analyze_gaps
from src.report_generator import generate_report


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _get_llm():
    """Instantiate the LLM based on environment variables.

    Supports two backends:
      - OpenAI (default): requires OPENAI_API_KEY
      - Ollama (free, local): set USE_OLLAMA=true
    """
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"

    if use_ollama:
        from langchain_community.llms import Ollama

        model_name = os.getenv("OLLAMA_MODEL", "llama3")
        print(f"🦙 Using Ollama model: {model_name}")
        return Ollama(model=model_name)
    else:
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key == "your-openai-api-key-here":
            print(
                "❌ OPENAI_API_KEY not set. Either:\n"
                "   1. Set it in your .env file, or\n"
                "   2. Set USE_OLLAMA=true to use a free local model."
            )
            sys.exit(1)
        print("🤖 Using OpenAI GPT model")
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def _get_embeddings():
    """Instantiate the embeddings model.

    We use a local HuggingFace model so there's no API cost for embeddings.
    sentence-transformers/all-MiniLM-L6-v2 is small, fast, and surprisingly
    good for academic text.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings

    print("📐 Loading embeddings model (this may take a moment the first time)...")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

PAPERS_DIR = os.path.join(os.path.dirname(__file__), "data", "papers")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "data", "index")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "data", "report.md")


def step_parse_papers(llm) -> list[dict]:
    """Parse all PDF papers in the data/papers/ directory."""
    print("\n📄 Step 1: Parsing papers...")
    pdf_files = sorted(
        f for f in os.listdir(PAPERS_DIR) if f.lower().endswith(".pdf")
    )

    if not pdf_files:
        print(
            f"⚠️  No PDF files found in {PAPERS_DIR}\n"
            "   Add some PDF papers and re-run."
        )
        return []

    papers_data = []
    for filename in pdf_files:
        file_path = os.path.join(PAPERS_DIR, filename)
        print(f"   Parsing: {filename}")
        paper = parse_paper(file_path, llm)
        papers_data.append(paper)
        print(f"   ✅ Extracted: {paper.get('title', filename)}")

    print(f"   Parsed {len(papers_data)} paper(s) total.\n")
    return papers_data


def step_index_papers(embeddings) -> "FAISS":
    """Build or load the FAISS vector index."""
    from langchain_community.vectorstores import FAISS

    print("📑 Step 2: Indexing papers...")
    # Try loading an existing index first
    vector_store = load_paper_index(INDEX_DIR, embeddings)
    if vector_store is not None:
        return vector_store

    # Build a new index
    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))

    vector_store = index_papers(
        PAPERS_DIR, embeddings,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    save_paper_index(vector_store, INDEX_DIR)
    return vector_store


def step_create_agent(papers_data, vector_store, llm):
    """Create the ReAct agent with all three tools."""
    print("🛠️  Step 3: Creating agent tools...")
    top_k = int(os.getenv("TOP_K", "5"))
    verbose = os.getenv("VERBOSE", "true").lower() == "true"

    tools = [
        create_search_tool(vector_store, top_k=top_k),
        create_summary_tool(papers_data, llm),
        create_compare_tool(papers_data, llm),
    ]
    agent = create_research_agent(tools, llm, verbose=verbose)
    print(f"   Agent ready with {len(tools)} tools.\n")
    return agent


def step_gap_analysis(papers_data, llm) -> dict:
    """Run gap analysis across all parsed papers."""
    print("🔬 Step 4: Running gap analysis...")
    results = analyze_gaps(papers_data, llm)
    print("   Gap analysis complete.\n")
    return results


def step_generate_report(analysis_results: dict) -> str:
    """Generate and save the Markdown report."""
    print("📝 Step 5: Generating report...")
    report = generate_report(analysis_results, REPORT_PATH)
    return report


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------


def interactive_loop(agent):
    """Let users ask follow-up questions via the agent."""
    print("\n" + "=" * 60)
    print("💬 Interactive Research Assistant")
    print("   Type your questions below. Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye! 👋")
            break

        run_agent(agent, query)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="AI Research Agent")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Skip full analysis and go straight to interactive Q&A",
    )
    args = parser.parse_args()

    # Load environment variables from .env file
    load_dotenv()

    print("🚀 AI Research Agent starting up...\n")

    # Initialise models
    llm = _get_llm()
    embeddings = _get_embeddings()

    # Parse papers
    papers_data = step_parse_papers(llm)

    # Index papers
    vector_store = step_index_papers(embeddings)

    # Create agent
    agent = step_create_agent(papers_data, vector_store, llm)

    if not args.interactive and papers_data:
        # Run full analysis pipeline
        analysis_results = step_gap_analysis(papers_data, llm)
        step_generate_report(analysis_results)
        print(f"\n✅ Full analysis complete! Report saved to {REPORT_PATH}")

    # Enter interactive mode
    interactive_loop(agent)


if __name__ == "__main__":
    main()
