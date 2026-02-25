"""
query_router.py
---------------
Classifies an incoming user query to determine which content modalities
(text, image, table) are most likely to contain the answer, then routes
retrieval to the appropriate FAISS indexes.

Why routing matters
--------------------
Without routing every query would hit all three indexes, which:
  * Wastes embedding / similarity-search compute.
  * Inflates cost when GPT-4V-captioned image indexes are large.
  * Dilutes the final context with irrelevant cross-modal results.

By classifying upfront we retrieve *only* from relevant indexes, reducing
latency and cost while keeping the context focused.

When to use ALL
----------------
Complex questions (e.g. "Summarise the findings from section 2") often span
all content types.  When the classifier is uncertain it returns ALL, which is
the safe default — it is better to over-search than to miss the answer.

Parsing the LLM output
-----------------------
We ask the LLM to respond with a JSON object `{"types": [...]}` to make
parsing deterministic.  If the response cannot be parsed as JSON we fall back
to ALL to maintain correctness at the cost of a broader search.
"""

import json
import re
from enum import Enum


class QueryType(Enum):
    TEXT = "TEXT"
    IMAGE = "IMAGE"
    TABLE = "TABLE"
    ALL = "ALL"


_CLASSIFICATION_PROMPT = """\
Classify this query to determine which type of document content would best answer it.

Query: {query}

Choose one or more from:
- TEXT: The answer is likely in text paragraphs
- IMAGE: The answer requires looking at a visual/diagram/photo
- TABLE: The answer requires numerical data from a table or chart
- ALL: Search all content types

Common patterns:
- "show me", "what does X look like", "diagram of" → IMAGE
- "how many", "revenue", "statistics", "percentage", "trend" → TABLE
- "explain", "describe", "what is", "how does" → TEXT
- Complex questions → ALL

Respond with JSON only: {{"types": ["TEXT", "TABLE"]}}
"""


def classify_query(query: str, llm) -> list[QueryType]:
    """
    Ask the LLM to classify a user query by relevant content modality.

    Parameters
    ----------
    query : The user's natural-language question.
    llm   : A LangChain LLM / chat model that supports .invoke() or .predict().

    Returns
    -------
    List of QueryType enum values indicating which indexes to search.
    Falls back to [QueryType.ALL] on any parsing error.
    """
    prompt = _CLASSIFICATION_PROMPT.format(query=query)

    try:
        if hasattr(llm, "invoke"):
            response = llm.invoke(prompt)
            raw = response.content if hasattr(response, "content") else str(response)
        else:
            raw = llm.predict(prompt)

        # Extract JSON from the response — the model may wrap it in markdown fences.
        json_match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON object found in LLM response.")

        parsed = json.loads(json_match.group())
        type_strings: list[str] = parsed.get("types", ["ALL"])

        query_types = []
        for t in type_strings:
            t_upper = t.upper()
            if t_upper == "ALL":
                # ALL expands to all three specific types.
                return [QueryType.TEXT, QueryType.IMAGE, QueryType.TABLE]
            try:
                query_types.append(QueryType(t_upper))
            except ValueError:
                pass  # Unknown type string — skip.

        if not query_types:
            raise ValueError("No valid QueryType values parsed.")

        return query_types

    except Exception as exc:
        # Fallback: search everything rather than potentially missing the answer.
        print(f"  [query_router] Classification failed ({exc}) — defaulting to ALL.")
        return [QueryType.TEXT, QueryType.IMAGE, QueryType.TABLE]
