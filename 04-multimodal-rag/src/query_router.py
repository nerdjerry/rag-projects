"""
query_router.py — Classify a user query and decide which indexes to search.

Not every question needs every index.  For example:

  "What does the architecture diagram show?"   → IMAGE index
  "What was Q3 revenue?"                       → TABLE index
  "Summarize the introduction"                 → TEXT  index
  "Explain the chart on page 5 and compare it
   with the numbers in Table 2"                → IMAGE + TABLE

By routing queries we:
  1. Reduce latency  – fewer indexes to search.
  2. Improve quality – avoid noisy results from irrelevant modalities.
  3. Save cost       – fewer embedding calls.

CLASSIFICATION PROMPT
─────────────────────
We ask the LLM to tag the query with one or more of: TEXT, IMAGE, TABLE.
The output is parsed with simple string matching (no fancy parser needed).
"""

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# The classification prompt.  We ask for a comma-separated list so parsing
# is trivial.  Few-shot examples make the model more reliable.
ROUTER_PROMPT = """\
You are a query classifier for a multimodal document search system.
The system has three indexes:
  TEXT  – paragraphs and prose from the document
  IMAGE – descriptions of charts, diagrams, photos, and figures
  TABLE – descriptions of data tables and spreadsheets

Given the user's question, reply with a comma-separated list of which
indexes should be searched.  Reply with ONLY the labels, nothing else.

Examples:
  Q: What does the flowchart on page 3 show?
  A: IMAGE

  Q: What was the revenue in Q2?
  A: TABLE

  Q: Summarize the executive summary.
  A: TEXT

  Q: Compare the bar chart with the numbers in Table 1.
  A: IMAGE, TABLE

  Q: Give me a full overview of the document.
  A: TEXT, IMAGE, TABLE

Now classify this query:
Q: {query}
A:"""

# Valid index types the router can return.
VALID_TYPES = {"TEXT", "IMAGE", "TABLE"}


def _parse_router_output(raw: str) -> list[str]:
    """
    Turn the LLM's comma-separated response into a clean list of index
    types.  Unknown labels are silently dropped.

    >>> _parse_router_output("IMAGE, TABLE")
    ['IMAGE', 'TABLE']
    >>> _parse_router_output("Hmm, probably TEXT and IMAGE")
    ['TEXT', 'IMAGE']
    """
    tokens = [t.strip().upper() for t in raw.replace(",", " ").split()]
    matched = [t for t in tokens if t in VALID_TYPES]
    return matched if matched else list(VALID_TYPES)  # default: search all


def route_query(query: str, llm=None) -> list[str]:
    """
    Classify a user query and return which index types to search.

    Parameters
    ----------
    query : str
        The user's natural-language question.
    llm : BaseLLM or BaseChatModel, optional
        LangChain LLM for classification.  If None, we default to
        searching all three indexes (safe but slower).

    Returns
    -------
    list[str]
        Subset of ["TEXT", "IMAGE", "TABLE"] indicating which indexes to
        search for this query.
    """
    if llm is None:
        logger.info("No LLM provided — defaulting to ALL indexes.")
        return list(VALID_TYPES)

    prompt = ROUTER_PROMPT.format(query=query)

    try:
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        result = _parse_router_output(raw)
        logger.info("Router: query=%r → %s  (raw=%r)", query[:60], result, raw.strip())
    except Exception as exc:
        logger.error("Router failed (%s) — searching all indexes.", exc)
        result = list(VALID_TYPES)

    return result
