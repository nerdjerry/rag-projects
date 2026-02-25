"""
src/report_generator.py
------------------------
Generates a structured Markdown report from parsed paper metadata and gap analysis.

HOW OUTPUT PARSERS CAN ENFORCE STRUCTURE
-----------------------------------------
Here we build the Markdown manually using Python f-strings.  An alternative
approach is to use LangChain's StructuredOutputParser or PydanticOutputParser:

  1. Define a Pydantic model with all report sections as fields.
  2. Attach the parser's format_instructions to the LLM prompt.
  3. The LLM fills the model; the parser deserialises it.

This is valuable when the *content* of each section needs to be LLM-generated
(e.g., "write a paragraph summarising the common themes").  For our report the
content comes from already-structured dicts (PaperMetadata, gap analysis dict),
so simple string formatting is cleaner and faster — no extra LLM call needed.

The general lesson: use output parsers when you need the LLM to produce
structured data; use string formatting when you already have structured data
and just need to render it.
"""

import os
from datetime import datetime
from pathlib import Path


def generate_report(
    paper_metadata_list: list,
    gap_analysis: dict,
    topic: str,
    output_path: str = None,
) -> str:
    """Generate a full Markdown research report and optionally save it to disk.

    Parameters
    ----------
    paper_metadata_list : list[PaperMetadata]
        All parsed papers.
    gap_analysis : dict
        Output of gap_analyzer.analyze_gaps().
    topic : str
        Human-readable topic label used in the report title.
    output_path : str or None
        If provided, the report is written to this path.
        If None, a timestamped filename is used automatically.

    Returns
    -------
    str
        The complete Markdown report as a string.
    """
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M")
    file_timestamp = now.strftime("%Y%m%d_%H%M%S")

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    def _list_section(items) -> str:
        """Render a list of strings as a Markdown bullet list."""
        if not items:
            return "_None identified._\n"
        return "\n".join(f"- {item}" for item in items) + "\n"

    def _numbered_section(items) -> str:
        """Render a list of strings as a Markdown numbered list."""
        if not items:
            return "_None identified._\n"
        return "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1)) + "\n"

    # ------------------------------------------------------------------
    # Section: Title & preamble
    # ------------------------------------------------------------------
    lines = [
        f"# Research Literature Analysis: {topic}",
        "",
        f"**Generated:** {timestamp}  ",
        f"**Papers analysed:** {len(paper_metadata_list)}",
        "",
    ]

    # ------------------------------------------------------------------
    # Section: Overview
    # ------------------------------------------------------------------
    lines += [
        "## Overview",
        "",
        f"This report analyses **{len(paper_metadata_list)}** research paper(s) on the topic of **{topic}**.",
        "",
        "### Papers in this collection",
        "",
    ]
    for pm in paper_metadata_list:
        authors_str = ", ".join(pm.authors[:3]) if pm.authors else "Unknown"
        if len(pm.authors) > 3:
            authors_str += " et al."
        year_str = f" ({pm.year})" if pm.year else ""
        lines.append(f"- **{pm.title}**{year_str} — {authors_str}")
    lines.append("")

    # ------------------------------------------------------------------
    # Section: Individual Paper Summaries
    # ------------------------------------------------------------------
    lines += ["## Individual Paper Summaries", ""]

    for pm in paper_metadata_list:
        authors_str = ", ".join(pm.authors) if pm.authors else "Unknown"
        lines += [
            f"### {pm.title}",
            "",
            f"**Authors:** {authors_str}  ",
            f"**Year:** {pm.year or 'Unknown'}  ",
            f"**File:** `{Path(pm.file_path).name}`",
            "",
        ]
        if pm.abstract:
            lines += ["**Abstract:**", "", pm.abstract, ""]
        if pm.methodology:
            lines += [f"**Methodology:** {pm.methodology}", ""]
        if pm.key_findings:
            lines += ["**Key Findings:**", ""]
            lines += [f"- {f}" for f in pm.key_findings]
            lines.append("")
        if pm.limitations:
            lines += ["**Limitations:**", ""]
            lines += [f"- {l}" for l in pm.limitations]
            lines.append("")
        lines.append("---")
        lines.append("")

    # ------------------------------------------------------------------
    # Section: Cross-Paper Analysis
    # ------------------------------------------------------------------
    lines += ["## Cross-Paper Analysis", ""]

    lines += ["### Common Themes", ""]
    lines.append(_list_section(gap_analysis.get("common_themes", [])))

    lines += ["### Contradictions Between Papers", ""]
    lines.append(_list_section(gap_analysis.get("contradictions", [])))

    # ------------------------------------------------------------------
    # Section: Research Gaps
    # ------------------------------------------------------------------
    lines += ["## Research Gaps", ""]

    lines += ["### Missing Experiments", ""]
    lines.append(_list_section(gap_analysis.get("missing_experiments", [])))

    lines += ["### Under-Studied Populations or Contexts", ""]
    lines.append(_list_section(gap_analysis.get("missing_populations", [])))

    lines += ["### Methodological Gaps", ""]
    lines.append(_list_section(gap_analysis.get("methodological_gaps", [])))

    # ------------------------------------------------------------------
    # Section: Suggested Next Steps
    # ------------------------------------------------------------------
    lines += ["## Suggested Next Steps", ""]
    lines.append(_numbered_section(gap_analysis.get("suggested_next_steps", [])))

    # ------------------------------------------------------------------
    # Section: Paper Index
    # ------------------------------------------------------------------
    lines += ["## Paper Index", ""]
    lines += ["| # | Title | File |", "|---|-------|------|"]
    for i, pm in enumerate(paper_metadata_list, 1):
        fname = Path(pm.file_path).name
        lines.append(f"| {i} | {pm.title} | `{fname}` |")
    lines.append("")

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    lines += [
        "---",
        "",
        "_Report generated by the 03-research-agent pipeline. "
        "Always verify findings against the original papers — "
        "LLMs can hallucinate citations and misrepresent content._",
        "",
    ]

    report = "\n".join(lines)

    # ------------------------------------------------------------------
    # Save to disk
    # ------------------------------------------------------------------
    if output_path is None:
        output_path = f"research_report_{file_timestamp}.md"

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"[report_generator] Report saved to '{output_path}'.")

    return report
