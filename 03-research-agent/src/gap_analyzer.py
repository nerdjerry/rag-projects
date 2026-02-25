"""
Gap Analyzer — Synthesises findings across multiple papers to identify
research gaps, contradictions, and opportunities.

THIS IS PROMPTED REASONING, NOT DATABASE LOGIC
===============================================
The gap analysis here is done entirely by the LLM — there's no rule-based
engine or SQL query behind it. We provide the LLM with a structured prompt
that lists all paper summaries and ask it to identify patterns:

  - What topics have been well-studied?
  - Where do papers contradict each other?
  - What questions remain unanswered?

This works because LLMs are excellent at synthesising information from
multiple sources into coherent narratives. However, it also means:

  ⚠️  The LLM might "hallucinate" gaps that don't exist, or miss real ones.
  ⚠️  Always verify the output against the actual papers.

Think of the LLM as a very fast research assistant who has read all the
papers — helpful, but not infallible.
"""

from langchain_core.prompts import ChatPromptTemplate


_GAP_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior research analyst. Given summaries of multiple "
            "research papers on a related topic, perform a gap analysis.\n\n"
            "Identify:\n"
            "1. **Well-studied areas** — topics covered by multiple papers\n"
            "2. **Key agreements** — findings that papers converge on\n"
            "3. **Contradictions** — areas where papers disagree\n"
            "4. **Research gaps** — questions that remain unanswered\n"
            "5. **Suggested next steps** — promising directions for future "
            "research\n\n"
            "Be specific: reference paper titles when citing evidence. "
            "If there are not enough papers to draw conclusions, say so.",
        ),
        (
            "human",
            "Here are the paper summaries to analyse:\n\n{paper_summaries}",
        ),
    ]
)


def _format_summaries(paper_summaries: list[dict]) -> str:
    """Format a list of paper metadata dicts into a readable block."""
    sections = []
    for i, paper in enumerate(paper_summaries, 1):
        title = paper.get("title", "Untitled")
        abstract = paper.get("abstract", "N/A")
        methodology = paper.get("methodology", "N/A")
        findings = paper.get("key_findings", [])
        limitations = paper.get("limitations", [])

        findings_str = "\n".join(f"  - {f}" for f in findings) or "  N/A"
        limitations_str = "\n".join(f"  - {l}" for l in limitations) or "  N/A"

        sections.append(
            f"### Paper {i}: {title}\n"
            f"**Abstract:** {abstract}\n"
            f"**Methodology:** {methodology}\n"
            f"**Key Findings:**\n{findings_str}\n"
            f"**Limitations:**\n{limitations_str}\n"
        )
    return "\n".join(sections)


def analyze_gaps(paper_summaries: list[dict], llm) -> dict:
    """Run a gap analysis across all paper summaries.

    Args:
        paper_summaries: List of parsed paper dicts (from paper_parser).
        llm: A LangChain LLM/ChatModel.

    Returns:
        A dict with keys:
          - raw_analysis: The full LLM response text
          - papers_analysed: Number of papers included
          - paper_titles: List of titles that were analysed
    """
    if not paper_summaries:
        return {
            "raw_analysis": "No papers provided for gap analysis.",
            "papers_analysed": 0,
            "paper_titles": [],
        }

    formatted = _format_summaries(paper_summaries)

    messages = _GAP_ANALYSIS_PROMPT.format_messages(paper_summaries=formatted)
    response = llm.invoke(messages)
    analysis_text = (
        response.content if hasattr(response, "content") else str(response)
    )

    return {
        "raw_analysis": analysis_text,
        "papers_analysed": len(paper_summaries),
        "paper_titles": [p.get("title", "Untitled") for p in paper_summaries],
    }
