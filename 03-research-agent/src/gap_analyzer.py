"""
src/gap_analyzer.py
--------------------
Uses an LLM to synthesise research gaps across a collection of papers.

THIS IS PROMPTED REASONING, NOT DATABASE LOGIC
------------------------------------------------
Traditional literature review tools use citation graphs, keyword co-occurrence
matrices, or statistical topic models to find gaps.  We use a different
approach: we feed all paper summaries to an LLM and ask it to reason about
what is missing.

Advantages:
  - No need for structured metadata like citation counts or MeSH terms.
  - Can catch conceptual gaps ("nobody studied X in context Y") that keyword
    matching would miss.
  - Works on any research domain without domain-specific pre-processing.

Disadvantages (see LIMITATIONS below):
  - The LLM may hallucinate gaps or themes that are not actually present.
  - Subtle contradictions buried in technical detail may be missed.
  - The quality of the output is bounded by the quality of the summaries.

WHY SYNTHESIS REQUIRES READING ALL PAPERS, NOT JUST SEARCHING
---------------------------------------------------------------
Semantic search retrieves chunks relevant to a specific query.  Gap analysis
is a meta-level task: it needs to observe the DISTRIBUTION of topics across
the entire corpus.  If only 3 of 8 papers mention "dataset bias" we cannot
detect that gap by searching for "bias" — we need to compare absence vs
presence across all papers simultaneously.

LIMITATIONS
-----------
1. The LLM may invent plausible-sounding but false contradictions.
2. Gaps requiring deep domain expertise (e.g., specific biochemical pathways)
   may be missed or mischaracterised.
3. This prompt works best with 3–10 papers on the same topic.  With 20+
   papers the concatenated summaries may exceed the context window.
4. Results should always be reviewed by a domain expert before acting on them.
"""

import json


# ---------------------------------------------------------------------------
# Synthesis prompt
# ---------------------------------------------------------------------------

_GAP_ANALYSIS_PROMPT = """You are analyzing a collection of research papers on a topic.

Here are summaries of all the papers:
{all_summaries}

Based on these papers, provide a research gap analysis with:
1. common_themes: What topics/findings appear across multiple papers?
2. contradictions: Where do papers disagree or contradict each other?
3. missing_experiments: What experiments have NOT been done that would be valuable?
4. missing_populations: What groups or contexts haven't been studied?
5. methodological_gaps: What methodological approaches are missing?
6. suggested_next_steps: 3-5 specific research directions worth pursuing

Respond with JSON only."""


def analyze_gaps(paper_metadata_list: list, llm) -> dict:
    """Run a cross-paper synthesis prompt and return structured gap analysis.

    Parameters
    ----------
    paper_metadata_list : list[PaperMetadata]
        All parsed papers to analyse.
    llm :
        Any LangChain chat model.

    Returns
    -------
    dict with keys: common_themes, contradictions, missing_experiments,
                    missing_populations, methodological_gaps, suggested_next_steps
    """
    if not paper_metadata_list:
        return {
            "common_themes": [],
            "contradictions": [],
            "missing_experiments": [],
            "missing_populations": [],
            "methodological_gaps": [],
            "suggested_next_steps": [],
            "error": "No papers provided for gap analysis.",
        }

    # Build a human-readable summary block for each paper
    summary_blocks = []
    for pm in paper_metadata_list:
        authors_str = ", ".join(pm.authors) if pm.authors else "Unknown"
        findings_str = (
            "\n    ".join(f"• {f}" for f in pm.key_findings)
            if pm.key_findings
            else "(not extracted)"
        )
        limitations_str = (
            "\n    ".join(f"• {l}" for l in pm.limitations)
            if pm.limitations
            else "(not extracted)"
        )
        block = (
            f"--- Paper: {pm.title} ---\n"
            f"Authors    : {authors_str}\n"
            f"Year       : {pm.year or 'Unknown'}\n"
            f"Methodology: {pm.methodology or 'Not extracted'}\n"
            f"Key Findings:\n    {findings_str}\n"
            f"Limitations:\n    {limitations_str}"
        )
        summary_blocks.append(block)

    all_summaries = "\n\n".join(summary_blocks)
    prompt = _GAP_ANALYSIS_PROMPT.format(all_summaries=all_summaries)

    print("[gap_analyzer] Running synthesis prompt across all papers…")
    response = llm.invoke(prompt)
    raw = response.content if hasattr(response, "content") else str(response)

    # Strip markdown code fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()

    try:
        gaps = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"[gap_analyzer] Warning: could not parse JSON response: {exc}")
        # Return the raw text under a fallback key so nothing is lost
        gaps = {
            "common_themes": [],
            "contradictions": [],
            "missing_experiments": [],
            "missing_populations": [],
            "methodological_gaps": [],
            "suggested_next_steps": [],
            "raw_response": raw,
        }

    return gaps


def format_gap_analysis(gaps: dict) -> str:
    """Format a gap analysis dict as a human-readable string for console display.

    Parameters
    ----------
    gaps : dict
        Output of :func:`analyze_gaps`.

    Returns
    -------
    str
    """
    def _fmt_list(items) -> str:
        if not items:
            return "  (none identified)"
        if isinstance(items, list):
            return "\n".join(f"  • {item}" for item in items)
        return f"  {items}"

    sections = [
        ("Common Themes",        gaps.get("common_themes", [])),
        ("Contradictions",       gaps.get("contradictions", [])),
        ("Missing Experiments",  gaps.get("missing_experiments", [])),
        ("Missing Populations",  gaps.get("missing_populations", [])),
        ("Methodological Gaps",  gaps.get("methodological_gaps", [])),
        ("Suggested Next Steps", gaps.get("suggested_next_steps", [])),
    ]

    lines = ["=" * 60, "RESEARCH GAP ANALYSIS", "=" * 60]
    for heading, items in sections:
        lines.append(f"\n{heading}:")
        lines.append(_fmt_list(items))

    if "raw_response" in gaps:
        lines.append("\n[Raw LLM response — JSON parsing failed]")
        lines.append(gaps["raw_response"])

    return "\n".join(lines)
