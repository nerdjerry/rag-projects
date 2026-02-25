"""
Summary Tool — Summarises a single paper when the agent provides its title.

SUMMARISATION IN AN AGENT CONTEXT
Unlike a standalone summariser that processes one document at a time, a tool
inside an agent is *called on demand*. The agent decides *when* a summary
would be useful — for example:

  User: "Compare the approaches in Paper A and Paper B."
  Agent thinks: "I should summarise each paper first, then compare."
        → calls summary_tool("Paper A")
        → calls summary_tool("Paper B")
        → synthesises the two summaries into a comparison

This "decide-then-act" pattern is what makes agents more flexible than
fixed pipelines. The summary tool doesn't need to know about comparisons —
it just needs to be good at summarising.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool


_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a research assistant. Provide a concise but thorough "
            "summary of the paper described below. Cover the objective, "
            "methodology, key findings, and limitations.",
        ),
        (
            "human",
            "Summarise the following paper:\n\n"
            "Title: {title}\n"
            "Authors: {authors}\n"
            "Abstract: {abstract}\n"
            "Methodology: {methodology}\n"
            "Key Findings: {key_findings}\n"
            "Limitations: {limitations}\n",
        ),
    ]
)


def _build_summary_fn(papers_data: list[dict], llm):
    """Return a closure that summarises a paper by title.

    Args:
        papers_data: List of parsed paper dicts (output of paper_parser).
        llm: A LangChain LLM/ChatModel instance.
    """

    def summarise_paper(title_query: str) -> str:
        """Summarise a research paper given its title (or partial title)."""
        # Find the best-matching paper by simple substring search.
        # A production system might use fuzzy matching or embeddings.
        title_lower = title_query.lower().strip()
        match = None
        for paper in papers_data:
            if title_lower in paper.get("title", "").lower():
                match = paper
                break

        if match is None:
            available = [p.get("title", "Untitled") for p in papers_data]
            return (
                f"Paper '{title_query}' not found. "
                f"Available papers: {available}"
            )

        # Build the prompt and call the LLM
        messages = _SUMMARY_PROMPT.format_messages(
            title=match.get("title", "Unknown"),
            authors=", ".join(match.get("authors", [])),
            abstract=match.get("abstract", "N/A"),
            methodology=match.get("methodology", "N/A"),
            key_findings="\n".join(
                f"- {f}" for f in match.get("key_findings", [])
            )
            or "N/A",
            limitations="\n".join(
                f"- {l}" for l in match.get("limitations", [])
            )
            or "N/A",
        )
        response = llm.invoke(messages)
        return (
            response.content
            if hasattr(response, "content")
            else str(response)
        )

    return summarise_paper


def create_summary_tool(papers_data: list[dict], llm) -> Tool:
    """Create a LangChain Tool for summarising individual papers.

    Args:
        papers_data: List of parsed paper metadata dicts.
        llm: A LangChain LLM/ChatModel.

    Returns:
        A LangChain Tool the agent can invoke by name.
    """
    return Tool(
        name="summarise_paper",
        description=(
            "Summarise a single research paper. Input should be the paper's "
            "title (or a distinctive part of it). Returns a structured "
            "summary covering objectives, methods, findings, and limitations."
        ),
        func=_build_summary_fn(papers_data, llm),
    )
