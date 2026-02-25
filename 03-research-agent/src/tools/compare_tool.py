"""
Compare Tool — Compares two papers' methodologies and findings side-by-side.

MULTI-STEP REASONING ACROSS DOCUMENTS
Comparing two papers is inherently a *multi-step* task:
  1. Retrieve information about Paper A
  2. Retrieve information about Paper B
  3. Identify similarities and differences
  4. Synthesise a coherent comparison

An LLM can do all of this in a single prompt, but the quality improves when
the agent can first use other tools (search, summarise) to gather context
before calling this tool. This is why agent architectures are powerful:
the agent orchestrates multiple steps, each building on the last.

The compare tool itself receives pre-parsed metadata, so it focuses purely
on step 3-4. Steps 1-2 were handled earlier by the paper parser.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool


_COMPARE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a research analyst. Compare the two papers below. "
            "Provide a structured comparison covering:\n"
            "1. Research objectives\n"
            "2. Methodologies used\n"
            "3. Key findings — similarities and differences\n"
            "4. Relative strengths and weaknesses\n"
            "Be specific and cite details from each paper.",
        ),
        (
            "human",
            "=== PAPER A ===\n"
            "Title: {title_a}\n"
            "Methodology: {methodology_a}\n"
            "Key Findings: {findings_a}\n"
            "Limitations: {limitations_a}\n\n"
            "=== PAPER B ===\n"
            "Title: {title_b}\n"
            "Methodology: {methodology_b}\n"
            "Key Findings: {findings_b}\n"
            "Limitations: {limitations_b}\n",
        ),
    ]
)


def _find_paper(papers_data: list[dict], query: str) -> dict | None:
    """Find a paper by partial title match (case-insensitive)."""
    query_lower = query.lower().strip()
    for paper in papers_data:
        if query_lower in paper.get("title", "").lower():
            return paper
    return None


def _build_compare_fn(papers_data: list[dict], llm):
    """Return a closure that compares two papers by title.

    The agent provides a string like "Paper A vs Paper B" or
    "Paper A | Paper B". We split on common delimiters.
    """

    def compare_papers(input_str: str) -> str:
        """Compare two research papers. Input: two paper titles separated by
        'vs', 'versus', '|', or ','."""
        # Parse the two titles from the input string
        for delimiter in [" vs ", " versus ", " vs. ", "|", ","]:
            if delimiter in input_str.lower():
                # Use the original string (not lowered) for splitting
                idx = input_str.lower().index(delimiter)
                title_a = input_str[:idx].strip()
                title_b = input_str[idx + len(delimiter):].strip()
                break
        else:
            return (
                "Please provide two paper titles separated by 'vs', '|', or ','.\n"
                "Example: 'Paper A vs Paper B'"
            )

        # Look up both papers
        paper_a = _find_paper(papers_data, title_a)
        paper_b = _find_paper(papers_data, title_b)

        if paper_a is None or paper_b is None:
            available = [p.get("title", "Untitled") for p in papers_data]
            missing = []
            if paper_a is None:
                missing.append(title_a)
            if paper_b is None:
                missing.append(title_b)
            return (
                f"Could not find paper(s): {missing}. "
                f"Available papers: {available}"
            )

        # Build the comparison prompt
        messages = _COMPARE_PROMPT.format_messages(
            title_a=paper_a.get("title", "Unknown"),
            methodology_a=paper_a.get("methodology", "N/A"),
            findings_a="\n".join(
                f"- {f}" for f in paper_a.get("key_findings", [])
            )
            or "N/A",
            limitations_a="\n".join(
                f"- {l}" for l in paper_a.get("limitations", [])
            )
            or "N/A",
            title_b=paper_b.get("title", "Unknown"),
            methodology_b=paper_b.get("methodology", "N/A"),
            findings_b="\n".join(
                f"- {f}" for f in paper_b.get("key_findings", [])
            )
            or "N/A",
            limitations_b="\n".join(
                f"- {l}" for l in paper_b.get("limitations", [])
            )
            or "N/A",
        )
        response = llm.invoke(messages)
        return (
            response.content
            if hasattr(response, "content")
            else str(response)
        )

    return compare_papers


def create_compare_tool(papers_data: list[dict], llm) -> Tool:
    """Create a LangChain Tool for comparing two research papers.

    Args:
        papers_data: List of parsed paper metadata dicts.
        llm: A LangChain LLM/ChatModel.

    Returns:
        A LangChain Tool the agent can invoke.
    """
    return Tool(
        name="compare_papers",
        description=(
            "Compare two research papers side-by-side. Input should be two "
            "paper titles separated by 'vs', '|', or ','. Returns a "
            "structured comparison of methodologies, findings, and "
            "limitations."
        ),
        func=_build_compare_fn(papers_data, llm),
    )
