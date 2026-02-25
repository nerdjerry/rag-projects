"""
conflict_detector.py — Detect contradictions between clauses in a legal document.

WHY THIS IS HARD (EVEN FOR LLMs):
    Conflict detection is one of the most challenging tasks in legal NLP. Contracts
    are written by multiple lawyers over multiple drafts, and contradictions creep in.

    Common contradiction patterns:
    1. DIRECT CONTRADICTION:
       Section 3: "This Agreement shall be governed by the laws of Delaware."
       Section 12: "All disputes shall be resolved under California law."
       → Two sections specify different governing law.

    2. DEFINITION vs. USAGE MISMATCH:
       Definitions: "'Confidential Information' means trade secrets only."
       Section 7: "All business information shared is Confidential Information."
       → The definition is narrow, but the usage is broad.

    3. TERM CONFLICTS:
       Section 2: "The term is 12 months, non-renewable."
       Section 14: "This Agreement auto-renews for successive 1-year periods."
       → One section says no renewal, another says auto-renewal.

    4. OBLIGATION vs. LIMITATION CONFLICTS:
       Section 5: "Provider shall deliver all reports within 5 business days."
       Section 9: "Provider shall not be liable for delays of any kind."
       → One section creates a deadline obligation, the other excuses delays.

    LLMs can catch obvious contradictions but may miss subtle ones. Always have a
    human lawyer review flagged conflicts — this tool is an assistant, not a replacement.
"""

import json
from typing import Any, Dict, List


# The prompt is built inline here (rather than from a file) because conflict
# detection requires a specialized structure comparing clause pairs.
CONFLICT_DETECTION_PROMPT = """You are a legal document reviewer specializing in
detecting internal contradictions and conflicts within contracts.

Review the following extracted clauses from a single contract. Look for:

1. **Direct Contradictions**: Two clauses that say opposite things.
2. **Definition Mismatches**: A term is defined one way but used differently elsewhere.
3. **Obligation Conflicts**: One clause creates an obligation that another clause
   undermines or negates.
4. **Scope Inconsistencies**: Different sections apply different scopes to the
   same concept (e.g., different time periods, different geographic limits).

For each conflict found, provide:
- The two conflicting clauses (by type or text)
- The nature of the conflict
- The potential legal impact
- A recommendation for resolution

--- CLAUSES ---
{clauses_text}
--- END CLAUSES ---

Return your response as a valid JSON array:
[
    {{
        "clause_a": "First conflicting clause type and key text...",
        "clause_b": "Second conflicting clause type and key text...",
        "conflict_type": "Direct Contradiction | Definition Mismatch | Obligation Conflict | Scope Inconsistency",
        "description": "Plain-English explanation of the conflict...",
        "impact": "What could go wrong because of this conflict...",
        "recommendation": "How to resolve the conflict..."
    }}
]

If no conflicts are found, return an empty array: []

Respond ONLY with the JSON array. Do not include any other text.
"""


def _format_clauses_for_prompt(clauses: List[Dict[str, str]]) -> str:
    """Format extracted clauses for the conflict detection prompt."""
    parts = []
    for i, clause in enumerate(clauses, start=1):
        clause_type = clause.get("clause_type", "Unknown")
        original = clause.get("original_text", "N/A")
        plain = clause.get("plain_english", "N/A")
        parts.append(
            f"--- Clause {i}: {clause_type} ---\n"
            f"Original: {original}\n"
            f"Plain English: {plain}\n"
        )
    return "\n".join(parts)


def _parse_conflicts_response(response_text: str) -> List[Dict[str, str]]:
    """Parse the LLM's JSON response, handling common formatting issues."""
    text = response_text.strip()

    # Strip markdown code fences if present.
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try to find JSON array boundaries.
    start = text.find("[")
    end = text.rfind("]") + 1
    if start != -1 and end > start:
        try:
            result = json.loads(text[start:end])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # If we can't parse but the response mentions "no conflicts", treat as empty.
    lower = response_text.lower()
    if "no conflict" in lower or "no contradiction" in lower:
        return []

    return [
        {
            "clause_a": "Unknown",
            "clause_b": "Unknown",
            "conflict_type": "Parse Error",
            "description": f"Could not parse LLM response: {response_text[:300]}",
            "impact": "Unable to determine.",
            "recommendation": "Re-run the conflict detection.",
        }
    ]


def detect_conflicts(clauses: List[Dict[str, str]], llm: Any) -> List[Dict[str, str]]:
    """Compare clauses against each other to find contradictions and conflicts.

    This is inherently imperfect — even human lawyers disagree on what constitutes
    a conflict. Use the results as starting points for human review, not as
    definitive legal analysis.

    Args:
        clauses: List of clause dicts from clause_extractor.extract_clauses().
        llm:     A LangChain LLM or ChatModel instance.

    Returns:
        A list of conflict dictionaries, each containing:
            - clause_a:        First conflicting clause
            - clause_b:        Second conflicting clause
            - conflict_type:   Category of conflict
            - description:     Plain-English explanation
            - impact:          Potential legal consequence
            - recommendation:  How to resolve it

        Returns an empty list if no conflicts are detected.
    """
    if len(clauses) < 2:
        print("  Need at least 2 clauses to check for conflicts — skipping.")
        return []

    print("  Formatting clauses for conflict detection...")
    clauses_text = _format_clauses_for_prompt(clauses)

    prompt = CONFLICT_DETECTION_PROMPT.replace("{clauses_text}", clauses_text)

    print("  Sending to LLM for conflict analysis (this may take a moment)...")
    response = llm.invoke(prompt)

    response_text = response.content if hasattr(response, "content") else str(response)

    print("  Parsing conflict detection results...")
    conflicts = _parse_conflicts_response(response_text)

    if conflicts:
        print(f"  ⚠ Found {len(conflicts)} potential conflict(s).")
    else:
        print("  ✓ No conflicts detected (note: LLMs may miss subtle contradictions).")

    return conflicts


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Conflict Detector — requires clauses from clause_extractor and an LLM.")
    print()
    print("Example contradicting clauses to test with:")
    print("  Clause 1 (Term): 'Non-renewable, 12-month term.'")
    print("  Clause 2 (Renewal): 'Auto-renews for successive 1-year periods.'")
    print()
    print("Usage:")
    print("  conflicts = detect_conflicts(clauses, llm)")
    print("  for c in conflicts:")
    print("      print(f\"CONFLICT: {c['description']}\")")
