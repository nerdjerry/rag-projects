"""
src/tools/compare_tool.py
--------------------------
LangChain Tool that uses an LLM to compare two papers' methodologies and findings.

WHERE AGENTS SHINE: MULTI-STEP REASONING ACROSS DOCUMENTS
----------------------------------------------------------
Simple RAG retrieves the nearest chunks to a query and returns them.  A
comparison task is inherently multi-step:
  1. Identify paper A and paper B by name.
  2. Retrieve full metadata for each.
  3. Synthesise similarities, differences, and contradictions.
  4. Produce a structured answer.

An agent can chain these steps autonomously.  Without the agent layer you
would need to hard-code this pipeline.  With an agent the user can simply ask
"Compare Paper A and Paper B" and the agent decides to call this tool.

HOW THE REACT AGENT USES THIS TOOL
------------------------------------
A typical agent trace might look like:

  Thought : The user wants to compare "Transformer" and "BERT".
            I should use the compare_papers tool.
  Action  : compare_papers
  Action Input: Transformer vs BERT
  Observation: [structured comparison returned by this tool]
  Thought : I now have the comparison. I can answer the user.
  Final Answer: …

The agent learns the input format ("A vs B") solely from the tool description —
no examples are needed.

INPUT FORMAT
------------
The tool expects: "<title of paper 1> vs <title of paper 2>"
The separator " vs " (with spaces) is chosen because it is unambiguous and
unlikely to appear in a paper title.
"""

from langchain.tools import Tool

# Comparison prompt template
_COMPARE_PROMPT = """Compare these two research papers:

Paper 1: {title1}
Methodology: {methodology1}
Key Findings: {findings1}

Paper 2: {title2}
Methodology: {methodology2}
Key Findings: {findings2}

Provide a structured comparison covering:
1. Methodological similarities and differences
2. Agreements in findings
3. Contradictions in findings
4. Which paper's approach is stronger and why"""


def create_compare_tool(paper_metadata_dict: dict, llm) -> Tool:
    """Build a LangChain Tool that compares two papers using an LLM.

    Parameters
    ----------
    paper_metadata_dict : dict
        Maps paper title (str) → PaperMetadata object.
    llm :
        Any LangChain chat model used to generate the comparison narrative.

    Returns
    -------
    Tool
    """

    def _find_paper(query: str):
        """Case-insensitive substring match against known paper titles."""
        q = query.strip().lower()
        for title, meta in paper_metadata_dict.items():
            if q in title.lower():
                return meta
        return None

    def _compare(input_str: str) -> str:
        """Parse 'Paper A vs Paper B', retrieve metadata, call LLM to compare."""
        # Parse the two titles from the 'X vs Y' format
        if " vs " not in input_str:
            return (
                "Invalid input format. Please use: 'Paper Title A vs Paper Title B'. "
                f"Got: '{input_str}'"
            )

        parts = input_str.split(" vs ", maxsplit=1)
        title_a, title_b = parts[0].strip(), parts[1].strip()

        paper_a = _find_paper(title_a)
        paper_b = _find_paper(title_b)

        # Report clearly which lookups failed so the agent can retry
        if paper_a is None and paper_b is None:
            return f"Could not find papers matching '{title_a}' or '{title_b}'."
        if paper_a is None:
            return f"Could not find a paper matching '{title_a}'."
        if paper_b is None:
            return f"Could not find a paper matching '{title_b}'."

        # Format findings lists as readable strings for the prompt
        def fmt_findings(meta) -> str:
            if not meta.key_findings:
                return "Not extracted"
            return "; ".join(meta.key_findings)

        prompt = _COMPARE_PROMPT.format(
            title1=paper_a.title,
            methodology1=paper_a.methodology or "Not extracted",
            findings1=fmt_findings(paper_a),
            title2=paper_b.title,
            methodology2=paper_b.methodology or "Not extracted",
            findings2=fmt_findings(paper_b),
        )

        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    return Tool(
        name="compare_papers",
        description=(
            "Compare two research papers' methodologies and findings. "
            "Input: two paper titles separated by ' vs ' "
            "(e.g., 'Paper A vs Paper B')"
        ),
        func=_compare,
    )
