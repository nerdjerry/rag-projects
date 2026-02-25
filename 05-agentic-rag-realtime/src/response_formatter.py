"""
Response Formatter
==================
Formats the agent's raw output into a clean, user-friendly response that
includes the answer, which tools were used, and source attribution.

Why show sources?
  Showing sources builds user trust.  When a user sees:
    "Source: Wikipedia — Albert Einstein"
  they can verify the information themselves.  This is especially important
  for RAG systems where answers are generated from retrieved documents —
  users need to know WHERE the information came from.

Example output:
  ┌──────────────────────────────────────────────────┐
  │  🤖 Answer                                       │
  │  Apple's current stock price is $195.23.         │
  │  The 52-week high is $199.62 and the low is...   │
  │                                                  │
  │  🔧 Tools Used                                    │
  │  • stock_market                                  │
  │                                                  │
  │  📚 Sources                                       │
  │  • yfinance real-time data (AAPL)                │
  └──────────────────────────────────────────────────┘
"""


def format_response(result: dict, tools_used: list = None) -> str:
    """
    Format the agent's result into a structured, readable response.

    Args:
        result:      Dict from run_agent() with "output" and "error" keys.
        tools_used:  Optional list of tool names that were invoked.
                     (If not provided, we note that tool tracking wasn't available.)

    Returns:
        A formatted string ready to print to the console.
    """
    separator = "─" * 50

    # --- Handle errors ---
    if result.get("error"):
        return (
            f"\n{separator}\n"
            f"❌ Error\n"
            f"{result['error']}\n"
            f"{separator}\n"
        )

    answer = result.get("output", "No response generated.")

    # --- Build the formatted output ---
    parts = [
        f"\n{separator}",
        f"🤖 Answer",
        f"{answer}",
    ]

    # --- Tools Used section ---
    if tools_used:
        parts.append(f"\n🔧 Tools Used")
        for tool_name in tools_used:
            parts.append(f"  • {tool_name}")
    else:
        parts.append(f"\n🔧 Tools Used")
        parts.append(f"  (tool tracking not available — enable verbose mode to see)")

    # --- Sources section ---
    # Extract source hints from the answer text itself.
    sources = _extract_sources(answer)
    if sources:
        parts.append(f"\n📚 Sources")
        for source in sources:
            parts.append(f"  • {source}")

    parts.append(separator)

    return "\n".join(parts)


def _extract_sources(answer: str) -> list:
    """
    Try to extract source references from the answer text.

    This is a simple heuristic — it looks for common source patterns like
    URLs, "Source:", or file paths.  For more robust source tracking,
    you'd modify each tool to return structured metadata.

    Args:
        answer: The agent's answer text.

    Returns:
        A list of source strings found in the answer.
    """
    sources = []

    # Look for URLs
    import re
    urls = re.findall(r'https?://[^\s<>"\')\]]+', answer)
    for url in urls:
        if url not in sources:
            sources.append(url)

    # Look for "Source: ..." patterns
    source_patterns = re.findall(r'[Ss]ource[s]?:\s*(.+?)(?:\n|$)', answer)
    for pattern in source_patterns:
        cleaned = pattern.strip()
        if cleaned and cleaned not in sources:
            sources.append(cleaned)

    return sources
