"""
risk_analyzer.py — Flag risky patterns in contract clauses and rate their severity.

This module takes the clauses extracted by clause_extractor.py and evaluates each
one for common risk patterns. It assigns a severity rating and provides actionable
suggestions.

EXAMPLES OF HIGH-RISK vs. STANDARD CLAUSES:

    HIGH-RISK (Unlimited Liability):
        "Provider shall be liable for all direct, indirect, incidental, special,
         and consequential damages arising out of this Agreement."
        → No cap = unlimited financial exposure. Very dangerous.

    STANDARD (Capped Liability):
        "Provider's total aggregate liability shall not exceed the total fees
         paid by Client in the twelve (12) months preceding the claim."
        → Clear cap tied to contract value. Industry standard.

    HIGH-RISK (One-Sided Termination):
        "Company may terminate this Agreement at any time for any reason.
         Contractor may not terminate before the end of the Initial Term."
        → Only one party has the exit door. Unfair.

    STANDARD (Mutual Termination):
        "Either party may terminate this Agreement upon thirty (30) days
         prior written notice to the other party."
        → Both sides can walk away with reasonable notice. Fair.
"""

import json
import os
from typing import Any, Dict, List

from langchain.schema import Document


def _load_prompt_template() -> str:
    """Load the risk analysis prompt from the prompts/ directory."""
    prompt_path = os.path.join(
        os.path.dirname(__file__), "..", "prompts", "risk_prompt.txt"
    )
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def _format_clauses_for_prompt(clauses: List[Dict[str, str]]) -> str:
    """Format extracted clauses into a text block the LLM can analyze.

    We include both the original text and the clause type so the LLM has
    full context for its risk assessment.
    """
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


def _parse_risks_response(response_text: str) -> List[Dict[str, str]]:
    """Parse the LLM's JSON array response with fallback handling."""
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

    return [
        {
            "risk_category": "Parse Error",
            "severity": "UNKNOWN",
            "flagged_text": response_text[:500],
            "reason": "The LLM response could not be parsed as JSON.",
            "suggestion": "Try running the analysis again.",
        }
    ]


def analyze_risks(clauses: List[Dict[str, str]], llm: Any) -> List[Dict[str, str]]:
    """Analyze extracted clauses for risky patterns and rate their severity.

    Args:
        clauses: List of clause dicts from clause_extractor.extract_clauses().
        llm:     A LangChain LLM or ChatModel instance.

    Returns:
        A list of risk assessment dictionaries, each containing:
            - risk_category: e.g., "Unlimited Liability", "One-Sided Termination"
            - severity:      "HIGH", "MEDIUM", or "LOW"
            - flagged_text:  The specific clause text that triggered the flag
            - reason:        Why this is risky (plain English)
            - suggestion:    How to improve or negotiate this clause

    Example:
        [
            {
                "risk_category": "Unlimited Liability",
                "severity": "HIGH",
                "flagged_text": "Provider shall be liable for all damages...",
                "reason": "No cap on liability means unlimited financial exposure.",
                "suggestion": "Add a liability cap tied to fees paid (e.g., 12 months)."
            }
        ]
    """
    if not clauses:
        print("  No clauses to analyze — skipping risk analysis.")
        return []

    print("  Loading risk analysis prompt template...")
    template = _load_prompt_template()

    print("  Formatting clauses for analysis...")
    clauses_text = _format_clauses_for_prompt(clauses)

    prompt = template.replace("{clauses_text}", clauses_text)

    print("  Sending to LLM for risk analysis (this may take a moment)...")
    response = llm.invoke(prompt)

    response_text = response.content if hasattr(response, "content") else str(response)

    print("  Parsing risk assessments...")
    risks = _parse_risks_response(response_text)

    # Summarize findings.
    severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for risk in risks:
        sev = risk.get("severity", "UNKNOWN").upper()
        if sev in severity_counts:
            severity_counts[sev] += 1
    print(
        f"  Found {len(risks)} risks: "
        f"{severity_counts['HIGH']} HIGH, "
        f"{severity_counts['MEDIUM']} MEDIUM, "
        f"{severity_counts['LOW']} LOW"
    )

    return risks


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Risk Analyzer — requires clauses from clause_extractor and an LLM.")
    print("Example usage:")
    print("  risks = analyze_risks(clauses, llm)")
    print("  for r in risks:")
    print("      print(f\"[{r['severity']}] {r['risk_category']}: {r['reason']}\")")
