"""
conflict_detector.py — Contract Clause Conflict Detection

Compares extracted clauses against each other to surface internal
contradictions — places where one part of the contract conflicts with
another part. These inconsistencies are a common source of disputes.

⚠️  IMPORTANT DISCLAIMER ⚠️
────────────────────────────────────────────────────────────────────────────
LLM-based conflict detection is NOT 100% reliable. The model may:
  • Miss conflicts that require deep legal domain expertise to spot.
  • Flag "conflicts" that are actually intentional or legally complementary.
  • Fail on highly technical or jurisdiction-specific language.

This tool helps you KNOW WHAT TO LOOK FOR and directs your attention to
potentially problematic areas. It is NOT a substitute for a qualified
attorney's review. Always verify flagged conflicts with a licensed lawyer
before making any legal or business decisions.
────────────────────────────────────────────────────────────────────────────

Common conflict patterns this module targets:

  1. NOTICE PERIOD MISMATCH
     Termination clause: "30 days written notice required."
     Payment clause:     "Invoices are due 60 days after notice of termination."
     → You're legally required to pay for 60 days but can only terminate in 30.

  2. CONFIDENTIALITY vs DEFINITION CONFLICT
     Definition section: "Confidential Information means only written materials
                          marked CONFIDENTIAL."
     Confidentiality clause: "All information disclosed, including oral
                              communications, is confidential."
     → The definition is narrower than what the clause protects.

  3. TERMINATION vs AUTO-RENEWAL
     Termination clause: "Either party may terminate on 30 days notice."
     Auto-renewal clause: "This Agreement auto-renews annually unless notice
                           is given 90 days before expiry."
     → You need 90 days notice for auto-renewal but only 30 for termination —
       which governs if the contract expires and auto-renews in 35 days?

  4. IP OWNERSHIP vs CONFIDENTIALITY
     IP clause: "All work product is owned by Company and may be used freely."
     Confidentiality clause: "All work product is Confidential Information
                              and must not be disclosed."
     → Company claims ownership AND confidentiality — can they publish your work?
"""

import json
import re

from langchain.schema import HumanMessage


# ---------------------------------------------------------------------------
# LLM prompt (inline — short enough not to warrant a separate .txt file)
# ---------------------------------------------------------------------------

_CONFLICT_PROMPT = """
You are a legal contract analyst. Review the following extracted contract clauses
and identify any internal conflicts or contradictions between them.

Extracted clauses:
{clauses_json}

Look for conflicts such as:
- Different notice periods for the same event mentioned in two different clauses
- A definition that contradicts how a term is used elsewhere
- A termination clause that conflicts with an auto-renewal clause
- An IP ownership clause that contradicts a confidentiality clause
- Different liability caps stated in different sections
- Inconsistent governing law references

Respond with a JSON array. If no conflicts are found, return an empty array [].
[
  {{
    "conflict_type": "Short name for the type of conflict (e.g. notice_period_mismatch)",
    "clause_a": "Description or quote from the first clause",
    "clause_b": "Description or quote from the conflicting clause",
    "description": "Plain-English explanation of why these clauses conflict and the practical impact"
  }}
]
""".strip()


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def detect_conflicts(clauses: list[dict], llm) -> list[dict]:
    """
    Use an LLM to compare extracted clauses for internal contradictions.

    Parameters
    ----------
    clauses : list[dict] — output from clause_extractor.extract_clauses()
    llm     : LLM        — any LangChain-compatible chat model

    Returns
    -------
    list of dicts, each with:
        conflict_type : str — short category label
        clause_a      : str — description/quote from first clause
        clause_b      : str — description/quote from conflicting clause
        description   : str — plain-English explanation of the conflict

    Returns an empty list if no conflicts are found or detection fails.

    ⚠️  See module docstring for reliability limitations.
    """
    if not clauses:
        return []

    clauses_json = json.dumps(clauses, indent=2)
    prompt = _CONFLICT_PROMPT.format(clauses_json=clauses_json)

    response = llm.invoke([HumanMessage(content=prompt)])
    raw_content = response.content if hasattr(response, "content") else str(response)

    # Strip markdown code fences
    clean = raw_content.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.MULTILINE)
        clean = re.sub(r"```\s*$", "", clean, flags=re.MULTILINE)
        clean = clean.strip()

    try:
        conflicts = json.loads(clean)
        if isinstance(conflicts, dict):
            conflicts = [conflicts]
        return conflicts
    except json.JSONDecodeError:
        print(f"[ConflictDetector] WARNING: Could not parse LLM response as JSON.\n{raw_content[:300]}")
        return []


# ---------------------------------------------------------------------------
# Display formatting
# ---------------------------------------------------------------------------

def format_conflicts_output(conflicts: list) -> str:
    """
    Format detected conflicts as a human-readable terminal string.

    Parameters
    ----------
    conflicts : list — result from detect_conflicts()

    Returns
    -------
    str — formatted conflict report
    """
    if not conflicts:
        return "⚪ No internal conflicts detected."

    lines = [
        "⚠️  The following potential conflicts were detected.",
        "    Verify each finding with a qualified attorney before acting on it.",
        "",
    ]

    for i, conflict in enumerate(conflicts, start=1):
        conflict_type = conflict.get("conflict_type", "unknown").replace("_", " ").title()
        clause_a = conflict.get("clause_a", "")
        clause_b = conflict.get("clause_b", "")
        description = conflict.get("description", "")

        lines.append(f"⚡ [{i}] {conflict_type}")
        lines.append(f"     Clause A  : {clause_a}")
        lines.append(f"     Clause B  : {clause_b}")
        lines.append(f"     Impact    : {description}")
        lines.append("")

    return "\n".join(lines)
