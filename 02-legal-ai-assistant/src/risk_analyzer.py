"""
risk_analyzer.py — Contract Risk Analysis

Scores extracted clauses for potential risks and explains WHY each clause
is risky and what a fair alternative would look like.

Example of HIGH-RISK vs STANDARD clause language:

    HIGH RISK (indemnification):
        "Employee agrees to indemnify, defend, and hold harmless Company and
         its officers, directors, and employees from any and all claims,
         losses, or damages, including those arising from Company's own
         negligence or intentional misconduct."
        → This is dangerous: the employee bears the cost of the company's
          OWN mistakes.

    STANDARD (indemnification):
        "Each party shall indemnify and hold harmless the other party for
         losses arising directly from that party's own negligence or
         willful misconduct."
        → Fair: each side is responsible only for their own actions.

    HIGH RISK (IP ownership):
        "Employee hereby assigns to Company all inventions, discoveries,
         and works of authorship conceived or reduced to practice at any
         time during employment, whether or not related to Company's business
         and whether or not made during working hours."
        → "At any time" + "whether or not related" = company owns your
          weekend side projects.

    STANDARD (IP ownership):
        "Employee assigns to Company inventions that relate to Company's
         business or are developed using Company resources or during
         working hours."
        → Scoped to actual work-related output.

Risk levels:
    HIGH   🔴 — Potential for significant financial or legal harm; seek
                attorney review before signing.
    MEDIUM 🟡 — Unusual or one-sided term; negotiate if possible.
    LOW    🟢 — Minor concern; worth noting but unlikely to cause harm.
"""

import json
import re
from pathlib import Path

from langchain.schema import HumanMessage


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def _load_risk_prompt() -> str:
    """Load the risk analysis prompt from prompts/risk_prompt.txt."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "risk_prompt.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def analyze_risks(clauses: list[dict], llm) -> list[dict]:
    """
    Analyze extracted clauses for legal and financial risks.

    Parameters
    ----------
    clauses : list[dict] — output from clause_extractor.extract_clauses()
    llm     : LLM        — any LangChain-compatible chat model

    Returns
    -------
    list of dicts, each with:
        clause_summary       : str — brief description of the risky clause
        risk_level           : str — "HIGH", "MEDIUM", or "LOW"
        risk_type            : str — category (e.g. "unlimited_liability")
        explanation          : str — why it's risky + what fair looks like
        original_text_excerpt: str — the specific concerning text

    Returns an empty list if analysis fails or no risks are found.
    """
    if not clauses:
        return []

    prompt_template = _load_risk_prompt()

    # Serialize clauses to a readable text block for the prompt
    clauses_text = json.dumps(clauses, indent=2)

    prompt = prompt_template.format(clauses_text=clauses_text)
    response = llm.invoke([HumanMessage(content=prompt)])
    raw_content = response.content if hasattr(response, "content") else str(response)

    # Strip markdown fences
    clean = raw_content.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.MULTILINE)
        clean = re.sub(r"```\s*$", "", clean, flags=re.MULTILINE)
        clean = clean.strip()

    try:
        risks = json.loads(clean)
        if isinstance(risks, dict):
            risks = [risks]
        return risks
    except json.JSONDecodeError:
        print(f"[RiskAnalyzer] WARNING: Could not parse LLM response as JSON.\n{raw_content[:300]}")
        return []


# ---------------------------------------------------------------------------
# Display formatting
# ---------------------------------------------------------------------------

# Emoji indicators for risk levels — visible at a glance in terminal output
_RISK_EMOJI = {
    "HIGH":   "🔴",
    "MEDIUM": "🟡",
    "LOW":    "🟢",
}


def format_risk_output(risks: list) -> str:
    """
    Format risk analysis results as a human-readable terminal string.

    Parameters
    ----------
    risks : list — result from analyze_risks()

    Returns
    -------
    str — multi-line formatted risk report with emoji indicators
    """
    if not risks:
        return "No risks identified (or risk analysis failed)."

    lines = []
    for i, risk in enumerate(risks, start=1):
        level = risk.get("risk_level", "UNKNOWN").upper()
        emoji = _RISK_EMOJI.get(level, "⚪")
        risk_type = risk.get("risk_type", "unknown").replace("_", " ").title()
        summary = risk.get("clause_summary", "")
        explanation = risk.get("explanation", "")
        excerpt = risk.get("original_text_excerpt", "")

        lines.append(f"{emoji} [{i}] {level} RISK — {risk_type}")
        lines.append(f"     Clause  : {summary}")
        lines.append(f"     Why     : {explanation}")
        if excerpt:
            # Truncate long excerpts for readability
            if len(excerpt) > 200:
                excerpt = excerpt[:200] + "..."
            lines.append(f"     Text    : \"{excerpt}\"")
        lines.append("")

    return "\n".join(lines)
