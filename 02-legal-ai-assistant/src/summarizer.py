"""
summarizer.py — Contract Executive Summary Generator

Sends contract text to an LLM with a structured prompt and parses the JSON
response into a Python dict for downstream use and display.

Example transformation:
    Before (raw contract language):
        "This Agreement shall commence on the Effective Date and shall continue
         for a period of one (1) year unless sooner terminated..."

    After (plain-English summary field):
        "This is a one-year service agreement between Acme Corp and Beta LLC.
         Acme will provide software development services in exchange for monthly
         payments. Either party may terminate with 30 days written notice."

The structured JSON output (parties, contract_type, key_obligations, etc.)
makes it easy to build dashboards, comparison tools, or automated alerts on
top of this module without re-parsing free-form text.
"""

import json
import os
import re
from pathlib import Path

from langchain.schema import HumanMessage


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def _load_summary_prompt() -> str:
    """Load the summary prompt template from prompts/summary_prompt.txt."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "summary_prompt.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def generate_summary(contract_text: str, llm) -> dict:
    """
    Generate a structured executive summary of a contract.

    Parameters
    ----------
    contract_text : str  — full text (or a representative excerpt) of the contract
    llm           : LLM  — any LangChain-compatible chat model (e.g. ChatOpenAI)

    Returns
    -------
    dict with keys:
        parties          : list[str]
        contract_type    : str
        effective_date   : str
        duration         : str
        key_obligations  : list[str]
        summary          : str

    Falls back to {"raw_response": <text>} if JSON parsing fails, so callers
    always receive a dict even when the model returns malformed output.

    Note: We cap input at 8000 characters. Most consumer LLMs have a ~4k-token
    context window for GPT-3.5 or ~8k for GPT-4. 8000 chars ≈ 2000 tokens,
    leaving room for the prompt itself and the response.
    """
    prompt_template = _load_summary_prompt()

    # Truncate to avoid exceeding model context limits
    truncated_text = contract_text[:8000]
    if len(contract_text) > 8000:
        truncated_text += "\n\n[... document truncated for summary ...]"

    prompt = prompt_template.format(contract_text=truncated_text)

    # Invoke the LLM — works with both .invoke() (newer LangChain) and direct call
    response = llm.invoke([HumanMessage(content=prompt)])
    raw_content = response.content if hasattr(response, "content") else str(response)

    # --- Parse JSON response ---
    # The prompt instructs the model to return ONLY JSON, but it occasionally
    # wraps it in ```json ... ``` markdown fences — strip those first.
    clean = raw_content.strip()
    if clean.startswith("```"):
        # Remove opening fence (```json or ```)
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.MULTILINE)
        # Remove closing fence
        clean = re.sub(r"```\s*$", "", clean, flags=re.MULTILINE)
        clean = clean.strip()

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # Graceful fallback: return raw text so the caller can still display something
        return {"raw_response": raw_content}


# ---------------------------------------------------------------------------
# Display formatting
# ---------------------------------------------------------------------------

def format_summary_output(summary: dict) -> str:
    """
    Format a summary dict as a human-readable string for terminal output.

    Parameters
    ----------
    summary : dict — result from generate_summary()

    Returns
    -------
    str — multi-line formatted summary
    """
    if "raw_response" in summary:
        return f"[Raw LLM response — JSON parsing failed]\n\n{summary['raw_response']}"

    lines = []

    contract_type = summary.get("contract_type", "Unknown")
    lines.append(f"Contract Type : {contract_type}")

    parties = summary.get("parties", [])
    if parties:
        lines.append("Parties       :")
        for p in parties:
            lines.append(f"  • {p}")

    lines.append(f"Effective Date: {summary.get('effective_date', 'Not specified')}")
    lines.append(f"Duration      : {summary.get('duration', 'Not specified')}")

    obligations = summary.get("key_obligations", [])
    if obligations:
        lines.append("Key Obligations:")
        for ob in obligations:
            lines.append(f"  • {ob}")

    plain_summary = summary.get("summary", "")
    if plain_summary:
        lines.append(f"\nSummary:\n  {plain_summary}")

    return "\n".join(lines)

