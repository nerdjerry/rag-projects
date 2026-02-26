"""
clause_extractor.py — Legal Clause Extraction

Identifies and extracts specific, named clause types from a contract and
translates them into plain English. This is the "translation layer" between
dense legal prose and actionable information.

Clause type reference:
    indemnification     — Party A must pay for losses caused to Party B.
                          e.g. "Vendor shall indemnify Client against all claims
                          arising from Vendor's performance of the Services."
                          → "The vendor must cover any lawsuits or losses the
                             client suffers because of the vendor's work."

    limitation_of_liability — Maximum damages one party can claim.
                          e.g. "In no event shall either party's liability
                          exceed the fees paid in the prior 3 months."
                          → "Neither side can sue for more than 3 months of
                             contract payments."

    termination         — How and when the contract can be ended early.
                          e.g. "Either party may terminate with 30 days notice."
                          → "Either side can cancel with a month's warning."

    governing_law       — Which state/country's courts have jurisdiction.
                          e.g. "This Agreement shall be governed by the laws
                          of the State of New York."
                          → "Disputes go to New York courts."

    ip_ownership        — Who owns code, inventions, or designs created during
                          the contract.
                          e.g. "All work product created by Contractor shall
                          be deemed works made for hire owned by Company."
                          → "Everything you build belongs to the company."

    confidentiality     — What information must be kept secret and for how long.
                          e.g. "Each party agrees to keep Confidential
                          Information secret for 5 years after termination."
                          → "Both sides must keep secrets for 5 years after
                             the contract ends."
"""

import json
import re
from pathlib import Path

from langchain.schema import HumanMessage


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def _load_clause_prompt() -> str:
    """Load the clause extraction prompt from prompts/clause_prompt.txt."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "clause_prompt.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def extract_clauses(contract_text: str, llm) -> list[dict]:
    """
    Extract named clause types from a contract and provide plain-English
    translations.

    Parameters
    ----------
    contract_text : str — full or truncated contract text
    llm           : LLM — any LangChain-compatible chat model

    Returns
    -------
    list of dicts, each with:
        clause_type      : str — one of the six named types above
        original_text    : str — verbatim text from the contract
        plain_english    : str — plain-language explanation
        section_reference: str — section number or "Unknown"

    Returns an empty list if extraction fails or no clauses are found.
    """
    prompt_template = _load_clause_prompt()

    # Use up to 12 000 chars — clause extraction needs more context than summary
    # because clauses may appear anywhere across a long document.
    truncated_text = contract_text[:12000]
    if len(contract_text) > 12000:
        truncated_text += "\n\n[... document truncated ...]"

    prompt = prompt_template.format(contract_text=truncated_text)
    response = llm.invoke([HumanMessage(content=prompt)])
    raw_content = response.content if hasattr(response, "content") else str(response)

    # Strip markdown code fences if present
    clean = raw_content.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.MULTILINE)
        clean = re.sub(r"```\s*$", "", clean, flags=re.MULTILINE)
        clean = clean.strip()

    try:
        clauses = json.loads(clean)
        # Ensure we always return a list
        if isinstance(clauses, dict):
            clauses = [clauses]
        return clauses
    except json.JSONDecodeError:
        # Return empty list rather than crashing — downstream code handles this
        print(f"[ClauseExtractor] WARNING: Could not parse LLM response as JSON.\n{raw_content[:300]}")
        return []


# ---------------------------------------------------------------------------
# Display formatting
# ---------------------------------------------------------------------------

def format_clauses_output(clauses: list) -> str:
    """
    Format extracted clauses as a human-readable string for terminal output.

    Parameters
    ----------
    clauses : list — result from extract_clauses()

    Returns
    -------
    str — multi-line formatted clause list
    """
    if not clauses:
        return "No clauses extracted (or extraction failed)."

    lines = []
    for i, clause in enumerate(clauses, start=1):
        clause_type = clause.get("clause_type", "unknown").replace("_", " ").title()
        section = clause.get("section_reference", "Unknown")
        plain = clause.get("plain_english", "")
        original = clause.get("original_text", "")

        lines.append(f"[{i}] {clause_type}  (Section: {section})")
        lines.append(f"     Plain English: {plain}")
        # Truncate long original text for display purposes
        if len(original) > 200:
            original = original[:200] + "..."
        lines.append(f"     Original Text: {original}")
        lines.append("")  # blank line between clauses

    return "\n".join(lines)
