"""
Agentic RAG with Real-Time Data — Main Entry Point
====================================================
This is the entry point for the Agentic RAG system.  It:
  1. Loads configuration from .env
  2. Indexes the local knowledge base (or loads an existing index)
  3. Registers all tools (RAG, web search, finance, weather, Wikipedia)
  4. Creates the agent
  5. Starts an interactive Q&A loop

Supports both OpenAI (cloud) and Ollama (free, local) as the LLM backend.

Usage:
  python main.py
"""

import os
import sys
from dotenv import load_dotenv

# Ensure the project root is on the Python path so `src.*` imports work
# when running as `python main.py` from the project directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.knowledge_indexer import index_knowledge_base, load_knowledge_base
from src.tool_registry import register_tools
from src.agent import create_agent, run_agent
from src.response_formatter import format_response


# ── Helpers ─────────────────────────────────────────────────────────────

def get_embeddings():
    """
    Create an embeddings model for the FAISS vector store.
    Uses HuggingFace sentence-transformers (free, runs locally).
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings

    print("  Loading embedding model (first run downloads ~90 MB)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
    )
    return embeddings


def get_llm(config: dict):
    """
    Create the LLM based on configuration.

    Supports:
      - OpenAI  (set OPENAI_API_KEY in .env)
      - Ollama  (set USE_OLLAMA=true in .env; free, runs locally)
    """
    use_ollama = config.get("USE_OLLAMA", "false").lower() == "true"

    if use_ollama:
        # --- Ollama (local, free) ---
        from langchain_community.llms import Ollama

        model_name = config.get("OLLAMA_MODEL", "llama3")
        print(f"  Using Ollama model: {model_name}")
        print("  Make sure Ollama is running: ollama serve")
        return Ollama(model=model_name)
    else:
        # --- OpenAI (cloud, requires API key) ---
        from langchain_openai import ChatOpenAI

        api_key = config.get("OPENAI_API_KEY", "")
        if not api_key or api_key == "your-openai-api-key-here":
            print("\n  ⚠  No valid OPENAI_API_KEY found.")
            print("  Either set it in .env or switch to Ollama (USE_OLLAMA=true).\n")
            sys.exit(1)

        print("  Using OpenAI GPT model.")
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=api_key,
        )


def print_welcome():
    """Print a welcome banner with example queries."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║           🤖 Agentic RAG with Real-Time Data 🌐            ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  This agent can answer questions using:                      ║
║    📄 Your local documents  (knowledge base)                 ║
║    🔍 Web search            (SerpAPI)                        ║
║    📈 Stock market data     (yfinance)                       ║
║    🌤  Weather data          (OpenWeatherMap)                 ║
║    📚 Wikipedia             (free, unlimited)                ║
║                                                              ║
║  Example queries:                                            ║
║    • "What does our policy say about remote work?"           ║
║    • "What's Apple's current stock price?"                   ║
║    • "What's the weather in Tokyo?"                          ║
║    • "Tell me about the history of quantum computing"        ║
║    • "What are the latest AI news?"                          ║
║                                                              ║
║  Type 'quit' or 'exit' to stop.                              ║
╚══════════════════════════════════════════════════════════════╝
""")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    # --- Step 1: Load configuration ---
    load_dotenv()

    config = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "USE_OLLAMA": os.getenv("USE_OLLAMA", "false"),
        "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL", "llama3"),
        "SERPAPI_API_KEY": os.getenv("SERPAPI_API_KEY", ""),
        "OPENWEATHERMAP_API_KEY": os.getenv("OPENWEATHERMAP_API_KEY", ""),
        "TOP_K": int(os.getenv("TOP_K", "3")),
        "VERBOSE": os.getenv("VERBOSE", "true").lower() == "true",
    }

    print("\n🔧 Configuration loaded.")

    # --- Step 2: Set up embeddings and knowledge base ---
    print("\n📄 Setting up knowledge base...")
    embeddings = get_embeddings()

    data_dir = os.path.join(os.path.dirname(__file__), "data", "knowledge_base")
    index_path = os.path.join(os.path.dirname(__file__), "faiss_index")

    # Try to load an existing index; if not found, build a new one.
    vector_store = load_knowledge_base(embeddings, index_path)
    if vector_store is None:
        print("  Building new index from documents...")
        vector_store = index_knowledge_base(data_dir, embeddings, index_path)

    # --- Step 3: Register tools ---
    print("🔧 Registering tools...")
    tools = register_tools(vector_store, config)

    # --- Step 4: Create the agent ---
    print("🧠 Creating agent...")
    llm = get_llm(config)
    agent = create_agent(tools, llm, verbose=config["VERBOSE"])

    # --- Step 5: Interactive Q&A loop ---
    print_welcome()

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("\n👋 Goodbye!")
            break

        # Run the query through the agent
        print("\n🔄 Thinking...\n")
        result = run_agent(agent, query)

        # Format and display the response
        formatted = format_response(result)
        print(formatted)


if __name__ == "__main__":
    main()
