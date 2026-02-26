"""
src/tool_registry.py

Central registry that assembles all agent tools from their factory functions.

WHY A REGISTRY?
    The agent's LLM reads every tool's `name` and `description` when deciding
    what to call.  By building all tools in one place we can:
      • Conditionally include/exclude tools based on available API keys.
      • Replace real tools with mock/disabled stubs without touching agent.py.
      • Easily add new tools in one location.

GOOD TOOL DESCRIPTIONS:
    Think of tool descriptions like function docstrings aimed at another LLM.
    They should answer:
      1. WHAT the tool does.
      2. WHEN to use it (vs. other tools).
      3. WHAT INPUT FORMAT to provide.
    Vague descriptions → the agent picks the wrong tool.
    Overlapping descriptions → the agent gets confused about which to pick.

NOTE:
    The agent can ONLY use tools in this list — it cannot make up tools or call
    functions not registered here.  If a capability isn't listed, the agent will
    either say it can't help or try to approximate with another tool.
"""

from typing import List

from langchain.tools import Tool
from langchain_community.vectorstores import FAISS

from src.tools.rag_tool import create_rag_tool
from src.tools.web_search_tool import create_web_search_tool, create_mock_web_search_tool
from src.tools.finance_tool import create_finance_tool
from src.tools.weather_tool import create_weather_tool, create_mock_weather_tool
from src.tools.wiki_tool import create_wiki_tool


def build_tool_registry(
    vector_store: FAISS,
    config: dict,
) -> List[Tool]:
    """
    Instantiate and return the full list of tools available to the agent.

    Each tool factory either creates a real (API-backed) tool or a mock/disabled
    stub, depending on whether the relevant API key is present in `config`.

    Args:
        vector_store:  Loaded FAISS index for the RAG tool.
        config:        Dictionary with optional keys:
                         - tavily_api_key        (str | None)
                         - openweathermap_api_key (str | None)
                         - domain_description     (str) — what the KB contains

    Returns:
        List of LangChain Tool objects, ordered roughly by expected call frequency.
    """
    tools: List[Tool] = []

    # --- 1. RAG / Knowledge Base Tool ---
    # Always available — uses the local FAISS index, no external API needed.
    domain = config.get("domain_description", "internal company documents")
    rag = create_rag_tool(vector_store, domain_description=domain)
    tools.append(rag)

    # --- 2. Finance Tool ---
    # yfinance scrapes Yahoo Finance; no API key required.
    finance = create_finance_tool()
    tools.append(finance)

    # --- 3. Wikipedia Tool ---
    # Free, no API key required.  Useful for factual/encyclopaedic queries.
    wiki = create_wiki_tool()
    tools.append(wiki)

    # --- 4. Web Search Tool ---
    # Requires a Tavily API key.  Falls back to a disabled mock if not provided.
    tavily_key = config.get("tavily_api_key")
    if tavily_key:
        web = create_web_search_tool(tavily_key)
    else:
        web = create_mock_web_search_tool()
    tools.append(web)

    # --- 5. Weather Tool ---
    # Requires an OpenWeatherMap API key.  Falls back to mock data if not set.
    owm_key = config.get("openweathermap_api_key")
    if owm_key:
        weather = create_weather_tool(owm_key)
    else:
        weather = create_mock_weather_tool()
    tools.append(weather)

    return tools


def get_tool_descriptions(tools: List[Tool]) -> str:
    """
    Return a formatted string listing every tool's name and description.

    Useful for displaying the agent's capabilities at startup or for debugging
    which tools are available in the current session.

    Args:
        tools: List of LangChain Tool objects from build_tool_registry().

    Returns:
        Multi-line string, one tool per line.
    """
    lines = ["Available tools:"]
    for tool in tools:
        # Trim the description to one sentence for the summary display.
        first_sentence = tool.description.split(".")[0] + "."
        lines.append(f"  • {tool.name}: {first_sentence}")
    return "\n".join(lines)
