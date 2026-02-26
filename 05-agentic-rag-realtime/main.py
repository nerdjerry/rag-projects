"""
main.py — Agentic RAG with Real-Time Tools

Entry point for the 05-agentic-rag-realtime project.  Assembles the full
pipeline: knowledge base indexing → tool registry → agent → interactive Q&A.

Usage examples:
    # Single query
    python main.py --query "What is AAPL's current stock price?"

    # Interactive multi-turn session
    python main.py --interactive

    # Use a specific knowledge base directory
    python main.py --kb-dir /path/to/docs --interactive

    # Disable conversation memory (stateless mode)
    python main.py --interactive --no-memory

    # Hide the agent's reasoning trace
    python main.py --query "Weather in Tokyo" --no-verbose
"""

import argparse
import os
import sys

from dotenv import load_dotenv

# Load .env before importing project modules (they may read env vars at import time).
load_dotenv()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_api_keys(config: dict) -> None:
    """
    Print a startup banner showing which tools are ready / missing API keys.
    This helps users quickly see what's available before running queries.
    """
    openai_key = config.get("openai_api_key")
    tavily_key = config.get("tavily_api_key")
    owm_key = config.get("openweathermap_api_key")

    rag_status    = "✅ RAG Tool ready"
    finance_status = "✅ Finance Tool ready (yfinance — no key needed)"
    wiki_status   = "✅ Wikipedia Tool ready (no key needed)"
    web_status    = "✅ Web Search ready" if tavily_key else "❌ Web Search (no TAVILY_API_KEY)"
    weather_status = "✅ Weather Tool ready" if owm_key else "⚠️  Weather Tool (mock mode — no OPENWEATHERMAP_API_KEY)"
    openai_status = "✅ OpenAI connected" if openai_key else "❌ OpenAI (no OPENAI_API_KEY — required)"

    print("\n" + "=" * 60)
    print("  Agentic RAG — Tool Availability")
    print("=" * 60)
    for status in [openai_status, rag_status, finance_status, wiki_status, web_status, weather_status]:
        print(f"  {status}")
    print("=" * 60)

    if not openai_key:
        print("\n[ERROR] OPENAI_API_KEY is required. Add it to your .env file.")
        sys.exit(1)


def _print_example_queries() -> None:
    """Print suggested example queries so new users know what to try."""
    print("\nExample queries:")
    print('  • "What is the current price of AAPL?"')
    print('  • "What\'s the weather in London today?"')
    print('  • "Search Wikipedia for transformer neural networks"')
    print('  • "What does our internal strategy document say about AI adoption?"')
    print('  • "What is AAPL price and how does it compare to our internal valuation?"')
    print('  • "What are latest AI news stories relevant to our strategy?"')
    print()


def _build_config() -> dict:
    """Read all configuration from environment variables and return as a dict."""
    return {
        "openai_api_key":          os.getenv("OPENAI_API_KEY", ""),
        "openai_model":            os.getenv("OPENAI_MODEL", "gpt-4"),
        "tavily_api_key":          os.getenv("TAVILY_API_KEY", ""),
        "openweathermap_api_key":  os.getenv("OPENWEATHERMAP_API_KEY", ""),
        "domain_description":      "internal company documents and knowledge base",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic RAG with real-time tools (finance, weather, web search, Wikipedia).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--kb-dir",
        default=os.getenv("KNOWLEDGE_BASE_DIR", "data/knowledge_base"),
        help="Directory containing .pdf and .txt files to index (default: data/knowledge_base)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="OpenAI model name to use (overrides OPENAI_MODEL env var, default: gpt-4)",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Run a single query and exit.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive multi-turn Q&A session.",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable conversation memory (each query is independent).",
    )
    verbose_group = parser.add_mutually_exclusive_group()
    verbose_group.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=True,
        help="Show the agent's reasoning trace (default: on).",
    )
    verbose_group.add_argument(
        "--no-verbose",
        dest="verbose",
        action="store_false",
        help="Hide the agent's reasoning trace.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    config = _build_config()

    # Allow --model to override the environment variable.
    if args.model:
        config["openai_model"] = args.model

    # --- Print startup banner ---
    _check_api_keys(config)

    # --- Step 1: Index / load the knowledge base ---
    print(f"\n[Setup] Indexing knowledge base from '{args.kb_dir}' …")
    from src.knowledge_indexer import index_knowledge_base  # noqa: PLC0415
    vector_store = index_knowledge_base(
        kb_dir=args.kb_dir,
        index_path=os.path.join(args.kb_dir, ".faiss_index"),
    )

    # --- Step 2: Build tool registry ---
    print("[Setup] Building tool registry …")
    from src.tool_registry import build_tool_registry, get_tool_descriptions  # noqa: PLC0415
    tools = build_tool_registry(vector_store, config)
    print(get_tool_descriptions(tools))

    # --- Step 3: Instantiate the LLM ---
    print(f"\n[Setup] Connecting to OpenAI model '{config['openai_model']}' …")
    from langchain_openai import ChatOpenAI  # noqa: PLC0415
    llm = ChatOpenAI(
        model=config["openai_model"],
        openai_api_key=config["openai_api_key"],
        temperature=0,  # deterministic tool selection
    )

    # --- Step 4: Create agent ---
    use_memory = not args.no_memory
    print(f"[Setup] Creating agent (memory={'on' if use_memory else 'off'}, verbose={args.verbose}) …")
    from src.agent import create_agent, run_agent_query  # noqa: PLC0415
    agent = create_agent(tools, llm, memory=use_memory, verbose=args.verbose)

    # --- Step 5: Run query/interactive loop ---
    from src.response_formatter import (  # noqa: PLC0415
        format_response,
        extract_tools_from_steps,
    )

    def _run_and_display(query: str) -> None:
        """Run a single query and print the formatted response."""
        print(f"\n[Query] {query}\n")
        try:
            result = agent.invoke({"input": query})
            answer = result.get("output", str(result))
            steps = result.get("intermediate_steps", [])
            tools_used = extract_tools_from_steps(steps)
        except Exception as exc:
            answer = f"Agent encountered an error: {exc}"
            tools_used = []

        print("\n" + format_response(answer, tools_used))

    if args.query:
        # Single-shot mode: run one query and exit.
        _run_and_display(args.query)

    elif args.interactive:
        # Interactive mode: loop until user types "quit" or "exit".
        _print_example_queries()
        print("Type 'quit' or 'exit' to end the session.\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                break

            _run_and_display(user_input)

    else:
        # No mode selected — show help and example queries.
        print("\nNo query mode selected. Use --query or --interactive.")
        _print_example_queries()
        print("Run with --help for all options.")


if __name__ == "__main__":
    main()
