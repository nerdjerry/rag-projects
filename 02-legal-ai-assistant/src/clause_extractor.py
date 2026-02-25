"""
clause_extractor.py — Extract and translate key legal clauses from a document.

This module identifies six important clause types commonly found in contracts:

1. INDEMNIFICATION — Who pays if something goes wrong?
   Example: "Party A shall indemnify Party B against all claims..."
   Why it matters: Determines financial exposure if there's a lawsuit.

2. LIMITATION OF LIABILITY — Is there a cap on damages?
   Example: "Total liability shall not exceed fees paid in the prior 12 months."
   Why it matters: Without this, you could owe unlimited damages.

3. TERMINATION — How can the contract end?
   Example: "Either party may terminate with 30 days written notice."
   Why it matters: Being locked into a bad contract is expensive.

4. GOVERNING LAW — Which state/country's laws apply?
   Example: "This Agreement shall be governed by the laws of Delaware."
   Why it matters: Different jurisdictions have very different rules.

5. INTELLECTUAL PROPERTY (IP) OWNERSHIP — Who owns the work product?
   Example: "All deliverables shall be considered work-for-hire."
   Why it matters: You might lose rights to your own creations.

6. CONFIDENTIALITY — What must be kept secret and for how long?
   Example: "Confidential Information shall be protected for 5 years."
   Why it matters: Breach of confidentiality can lead to lawsuits.
"""

import json
import os
from typing import Any, Dict, List

from langchain.schema import Document


def _load_prompt_template() -> str:
    """Load the clause extraction prompt from the prompts/ directory."""
    prompt_path = os.path.join(
        os.path.dirname(__file__), "..", "prompts", "clause_prompt.txt"
    )
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def _combine_document_text(documents: List[Document], max_chars: int = 15000) -> str:
    """Combine document chunks into a single text block for the LLM."""
    full_text = "\n\n".join(doc.page_content for doc in documents)
    if len(full_text) <= max_chars:
        return full_text
    # Prioritize beginning (definitions, parties) and end (signatures, governing law).
    head = int(max_chars * 0.6)
    tail = max_chars - head
    return (
        full_text[:head]
        + "\n\n[... middle sections omitted for length ...]\n\n"
        + full_text[-tail:]
    )


def _parse_clauses_response(response_text: str) -> List[Dict[str, str]]:
    """Parse the LLM's JSON array response with fallback handling."""
    text = response_text.strip()

    # Strip markdown code fences if present.
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try to find JSON array boundaries.
    start = text.find("[")
    end = text.rfind("]") + 1
    if start != -1 and end > start:
        try:
            result = json.loads(text[start:end])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Return a fallback if parsing completely fails.
    return [
        {
            "clause_type": "Parse Error",
            "original_text": response_text[:500],
            "plain_english": "The LLM response could not be parsed as JSON.",
        }
    ]


def extract_clauses(documents: List[Document], llm: Any) -> List[Dict[str, str]]:
    """Extract key clauses from the legal document with plain-English translations.

    Args:
        documents: Parsed document chunks from document_parser.
        llm:       A LangChain LLM or ChatModel instance.

    Returns:
        A list of dictionaries, each containing:
            - clause_type:    e.g., "Indemnification", "Termination"
            - original_text:  The exact text from the contract
            - plain_english:  A non-lawyer-friendly translation

    Example return value:
        [
            {
                "clause_type": "Termination",
                "original_text": "Either party may terminate this Agreement upon
                    thirty (30) days prior written notice to the other party.",
                "plain_english": "Either side can end the contract by giving the
                    other side 30 days' written notice."
            }
        ]
    """
    print("  Loading clause extraction prompt template...")
    template = _load_prompt_template()

    print("  Combining document text...")
    document_text = _combine_document_text(documents)

    # Fill in the template placeholder.
    prompt = template.replace("{document_text}", document_text)

    print("  Sending to LLM for clause extraction (this may take a moment)...")
    response = llm.invoke(prompt)

    response_text = response.content if hasattr(response, "content") else str(response)

    print("  Parsing extracted clauses...")
    clauses = _parse_clauses_response(response_text)

    # Log what we found.
    found_types = [c.get("clause_type", "Unknown") for c in clauses]
    print(f"  Found {len(clauses)} clauses: {', '.join(found_types)}")

    return clauses


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Clause Extractor — requires a real LLM instance to test.")
    print("Example usage:")
    print("  from langchain_openai import ChatOpenAI")
    print("  from src.document_parser import parse_legal_document")
    print("  docs = parse_legal_document('contract.pdf')")
    print("  llm = ChatOpenAI(model='gpt-4o-mini')")
    print("  clauses = extract_clauses(docs, llm)")
    print("  for c in clauses:")
    print("      print(c['clause_type'], '->', c['plain_english'])")
