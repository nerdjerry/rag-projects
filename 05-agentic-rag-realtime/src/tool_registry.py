"""
Tool Registry
=============
Registers all available tools in a list and returns them to the agent.

Why a registry?
  The agent's LLM reads the NAME and DESCRIPTION of every registered tool
  to decide which tool to call for a given query.  Think of it like a menu:

    Agent sees:
      1. knowledge_base  — "Search local PDFs and documents..."
      2. web_search      — "Search the internet for current info..."
      3. stock_market    — "Get current stock prices..."
      4. weather         — "Get current weather for a city..."
      5. wikipedia       — "Look up facts on Wikipedia..."

    User asks: "What's the weather in Paris?"
    Agent thinks: "This matches tool #4 (weather)" → calls weather("Paris")

  That's why writing clear, specific tool descriptions is so important!
"""

from src.tools.rag_tool import create_rag_tool
from src.tools.web_search_tool import create_web_search_tool
from src.tools.finance_tool import create_finance_tool
from src.tools.weather_tool import create_weather_tool
from src.tools.wiki_tool import create_wiki_tool


def register_tools(vector_store, config: dict) -> list:
    """
    Create and return a list of all tools the agent can use.

    Args:
        vector_store: The FAISS vector store for the RAG tool.
        config:       A dict with configuration values, typically from .env:
                      - SERPAPI_API_KEY: for web search
                      - OPENWEATHERMAP_API_KEY: for weather

    Returns:
        A list of LangChain Tool objects ready to be passed to the agent.
    """
    tools = []

    # --- 1. Knowledge Base (RAG) Tool ---
    # Always available — this is the core of the system.
    if vector_store is not None:
        rag_tool = create_rag_tool(vector_store)
        tools.append(rag_tool)
        print(f"  ✅ Registered: {rag_tool.name}")

    # --- 2. Web Search Tool ---
    # Requires a SerpAPI key; works gracefully without one.
    serpapi_key = config.get("SERPAPI_API_KEY", "")
    web_tool = create_web_search_tool(api_key=serpapi_key)
    tools.append(web_tool)
    if serpapi_key:
        print(f"  ✅ Registered: {web_tool.name}")
    else:
        print(f"  ⚠  Registered: {web_tool.name} (no API key — limited)")

    # --- 3. Finance / Stock Market Tool ---
    # No API key needed — uses free yfinance library.
    finance_tool = create_finance_tool()
    tools.append(finance_tool)
    print(f"  ✅ Registered: {finance_tool.name}")

    # --- 4. Weather Tool ---
    # Requires an OpenWeatherMap key; works gracefully without one.
    weather_key = config.get("OPENWEATHERMAP_API_KEY", "")
    weather_tool = create_weather_tool(api_key=weather_key)
    tools.append(weather_tool)
    if weather_key:
        print(f"  ✅ Registered: {weather_tool.name}")
    else:
        print(f"  ⚠  Registered: {weather_tool.name} (no API key — limited)")

    # --- 5. Wikipedia Tool ---
    # No API key needed — free and unlimited.
    wiki_tool = create_wiki_tool()
    tools.append(wiki_tool)
    print(f"  ✅ Registered: {wiki_tool.name}")

    # --- Summary ---
    print(f"\n  Total tools registered: {len(tools)}")
    tool_names = [t.name for t in tools]
    print(f"  Available: {', '.join(tool_names)}\n")

    return tools
