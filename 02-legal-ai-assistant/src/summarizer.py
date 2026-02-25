"""
summarizer.py — Generate a plain-English executive summary of a legal document.

This module sends the full document text (or a representative sample) to an LLM
along with a structured prompt that asks for:
    - Parties involved
    - Contract type
    - Duration / key dates
    - Key obligations
    - A ≤300-word executive summary

BEFORE vs AFTER — why this matters:
    Raw clause:
        "The Receiving Party shall hold and maintain the Confidential Information
         of the Disclosing Party in strict confidence for the sole and exclusive
         benefit of the Disclosing Party for a period of five (5) years from the
         date of disclosure."

    Plain-English summary:
        "You must keep the other party's confidential information secret for 5 years."

    Business stakeholders don't have time to read 20-page contracts. A good summary
    lets them quickly decide whether to sign, negotiate, or walk away.
"""

import json
import os
from typing import Any, Dict, List

from langchain.schema import Document


def _load_prompt_template() -> str:
    """Load the summary prompt template from the prompts/ directory.

    We keep prompts in separate .txt files so they can be tweaked without
    touching Python code — a best practice for prompt engineering.
    """
    prompt_path = os.path.join(
        os.path.dirname(__file__), "..", "prompts", "summary_prompt.txt"
    )
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def _combine_document_text(documents: List[Document], max_chars: int = 15000) -> str:
    """Combine document chunks into a single text block for the LLM.

    We cap the text length because LLMs have context-window limits. For very
    long contracts, we take the first and last portions (the beginning usually
    contains parties/term, the end contains signatures/governing law).
    """
    full_text = "\n\n".join(doc.page_content for doc in documents)

    if len(full_text) <= max_chars:
        return full_text

    # Take the first ~60% and last ~40% to capture beginning and end sections.
    head_size = int(max_chars * 0.6)
    tail_size = max_chars - head_size
    return (
        full_text[:head_size]
        + "\n\n[... middle sections omitted for length ...]\n\n"
        + full_text[-tail_size:]
    )


def _parse_llm_response(response_text: str) -> Dict[str, Any]:
    """Try to parse the LLM's JSON response, with fallback handling.

    LLMs sometimes wrap JSON in markdown code fences or add commentary.
    We try to extract the JSON object even if there's extra text around it.
    """
    # Strip markdown code fences if present.
    text = response_text.strip()
    if text.startswith("```"):
        # Remove opening fence (possibly ```json)
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last resort: try to find JSON object boundaries.
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

        # If we still can't parse, return the raw text in our expected structure.
        return {
            "parties": [],
            "type": "Unknown",
            "duration": {"effective_date": "Unknown", "expiration_date": "Unknown",
                         "renewal_terms": "Unknown"},
            "key_obligations": [],
            "summary": response_text,
            "_parse_error": "Could not parse LLM response as JSON.",
        }


def generate_summary(documents: List[Document], llm: Any) -> Dict[str, Any]:
    """Generate a structured executive summary of the legal document.

    Args:
        documents: Parsed document chunks from document_parser.
        llm:       A LangChain LLM or ChatModel instance.

    Returns:
        A dictionary with keys: parties, type, duration, key_obligations, summary.

    Example return value:
        {
            "parties": [
                {"name": "Acme Corp", "role": "Service Provider"},
                {"name": "Widget Inc", "role": "Client"}
            ],
            "type": "SaaS Agreement",
            "duration": {
                "effective_date": "January 1, 2024",
                "expiration_date": "December 31, 2024",
                "renewal_terms": "Auto-renews annually"
            },
            "key_obligations": [
                {"party": "Acme Corp", "obligation": "Provide 99.9% uptime"},
                {"party": "Widget Inc", "obligation": "Pay $5,000/month"}
            ],
            "summary": "This is a one-year SaaS agreement between..."
        }
    """
    print("  Loading summary prompt template...")
    template = _load_prompt_template()

    print("  Combining document text...")
    document_text = _combine_document_text(documents)

    # Fill in the template placeholder.
    prompt = template.replace("{document_text}", document_text)

    print("  Sending to LLM for summarization (this may take a moment)...")
    response = llm.invoke(prompt)

    # LangChain chat models return an AIMessage; plain LLMs return a string.
    response_text = response.content if hasattr(response, "content") else str(response)

    print("  Parsing LLM response...")
    result = _parse_llm_response(response_text)

    return result


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Demonstrate with a mock — replace with a real LLM to test end-to-end.
    test_docs = [
        Document(
            page_content=(
                "SERVICE AGREEMENT\n\n"
                "This Service Agreement ('Agreement') is entered into as of January 1, 2024 "
                "('Effective Date') by and between Acme Corp, a Delaware corporation "
                "('Provider'), and Widget Inc, a California corporation ('Client').\n\n"
                "1. TERM. This Agreement shall remain in effect for twelve (12) months "
                "from the Effective Date.\n\n"
                "2. SERVICES. Provider shall deliver monthly analytics reports to Client."
            ),
            metadata={"source": "test_contract.pdf", "page": 1},
        )
    ]

    print("To test, provide a real LLM instance. Example:")
    print("  from langchain_openai import ChatOpenAI")
    print("  llm = ChatOpenAI(model='gpt-4o-mini')")
    print("  result = generate_summary(test_docs, llm)")
