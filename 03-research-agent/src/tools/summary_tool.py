"""
src/tools/summary_tool.py
-------------------------
LangChain Tool that returns a structured summary of a single named paper.

TOOL PATTERN
------------
This tool is a good example of the lookup pattern:
  - Input  : a (possibly partial) paper title provided by the agent.
  - Process: find the matching PaperMetadata object, format its fields.
  - Output : a formatted string the agent can quote in its final answer.

The agent uses this tool when it already knows the paper's name and wants
detailed information about it, as opposed to the search_papers tool which
is used when the agent is still looking for relevant papers.

INPUT PARSING: FUZZY TITLE MATCHING
-------------------------------------
We do a case-insensitive substring match: a paper titled
"Attention Is All You Need" will match inputs like "attention", "all you need",
or the full title.  This is intentional — the agent's input may be an
approximation if it inferred the title from a previous search result.

If multiple papers match, we return the first one (alphabetical order from
the dict).  For a production system you might use fuzzy matching (e.g.,
rapidfuzz) or ask the agent to be more specific.
"""

from langchain.tools import Tool


def create_summary_tool(paper_metadata_dict: dict, llm) -> Tool:
    """Build a LangChain Tool that summarises a specific paper by title.

    Parameters
    ----------
    paper_metadata_dict : dict
        Maps paper title (str) → PaperMetadata object.
        Built in main.py as {pm.title: pm for pm in paper_metadata_list}.
    llm :
        Unused here but accepted for API consistency with other tool factories.

    Returns
    -------
    Tool
    """

    def _summarize(title_query: str) -> str:
        """Find a paper by (partial) title and return a formatted summary."""
        query_lower = title_query.strip().lower()

        # Fuzzy match: find the first paper whose title contains the query
        match = None
        for title, meta in paper_metadata_dict.items():
            if query_lower in title.lower():
                match = meta
                break

        if match is None:
            available = ", ".join(paper_metadata_dict.keys()) or "none"
            return (
                f"No paper found matching '{title_query}'. "
                f"Available papers: {available}"
            )

        # Format the metadata as a readable summary
        authors_str = ", ".join(match.authors) if match.authors else "Unknown"
        findings_str = (
            "\n".join(f"  • {f}" for f in match.key_findings)
            if match.key_findings
            else "  (not extracted)"
        )
        limitations_str = (
            "\n".join(f"  • {l}" for l in match.limitations)
            if match.limitations
            else "  (not extracted)"
        )

        return (
            f"Title      : {match.title}\n"
            f"Authors    : {authors_str}\n"
            f"Year       : {match.year or 'Unknown'}\n"
            f"Abstract   : {match.abstract or 'Not available'}\n\n"
            f"Methodology: {match.methodology or 'Not extracted'}\n\n"
            f"Key Findings:\n{findings_str}\n\n"
            f"Limitations:\n{limitations_str}"
        )

    return Tool(
        name="summarize_paper",
        description=(
            "Get a structured summary of a specific research paper. "
            "Input: the paper title (or a unique part of it)."
        ),
        func=_summarize,
    )
