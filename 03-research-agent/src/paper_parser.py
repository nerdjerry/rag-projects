"""
Paper Parser — Extracts structured fields from research papers using an LLM.

WHY STRUCTURED FIELDS?
Raw text from PDFs is messy — inconsistent formatting, broken lines, random
headers/footers. By asking the LLM to extract specific fields (title, authors,
abstract, etc.), we get clean, consistent data that downstream tools (search,
comparison, gap analysis) can reliably work with.

Think of it like filling out a form: instead of dumping the entire paper into
a single text blob, we ask the LLM to "fill in" each field separately. This
makes the data queryable, comparable, and much easier to reason about.
"""

import os
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from pypdf import PdfReader


# ---------------------------------------------------------------------------
# Pydantic model — defines the exact shape of the data we want back.
# Using Pydantic ensures the LLM output is validated at runtime, so we catch
# missing or malformed fields early rather than downstream.
# ---------------------------------------------------------------------------


class PaperMetadata(BaseModel):
    """Structured representation of a research paper's key information."""

    title: str = Field(description="Full title of the paper")
    authors: list[str] = Field(
        default_factory=list,
        description="List of author names",
    )
    year: Optional[str] = Field(
        default=None,
        description="Publication year (e.g. '2024')",
    )
    abstract: str = Field(
        default="",
        description="Paper abstract or brief summary",
    )
    methodology: str = Field(
        default="",
        description="Research methodology or approach used",
    )
    key_findings: list[str] = Field(
        default_factory=list,
        description="Main results and conclusions",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Stated limitations or caveats of the research",
    )


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------


def _extract_text_from_pdf(file_path: str) -> str:
    """Read a PDF and return its full text.

    PDFs can have all sorts of encoding quirks — embedded fonts, ligatures,
    column layouts. pypdf handles most cases, but the output may still be
    a bit rough. That's OK — the LLM is surprisingly good at making sense
    of messy input.
    """
    reader = PdfReader(file_path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)
    return "\n".join(pages_text)


# ---------------------------------------------------------------------------
# Main parsing function
# ---------------------------------------------------------------------------

# The prompt is deliberately explicit about the fields we want.  This
# "structured extraction" pattern is one of the most reliable ways to get
# consistently formatted output from an LLM.
_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a research paper analysis assistant. "
            "Extract the requested fields from the paper text below. "
            "If a field cannot be determined from the text, use an empty "
            "string or empty list as appropriate. Be concise but accurate.",
        ),
        (
            "human",
            "Extract the following fields from this paper text:\n\n"
            "1. title — Full title of the paper\n"
            "2. authors — List of author names\n"
            "3. year — Publication year\n"
            "4. abstract — The paper's abstract or a brief summary\n"
            "5. methodology — Research methodology or approach\n"
            "6. key_findings — Main results (as a list)\n"
            "7. limitations — Stated limitations (as a list)\n\n"
            "--- PAPER TEXT START ---\n{paper_text}\n--- PAPER TEXT END ---",
        ),
    ]
)


def parse_paper(file_path: str, llm) -> dict:
    """Parse a research paper PDF and return structured metadata.

    Args:
        file_path: Path to the PDF file.
        llm: A LangChain LLM/ChatModel instance used for extraction.

    Returns:
        A dict matching the PaperMetadata schema, plus a ``source_file`` key.

    How it works:
        1. Extract raw text from the PDF.
        2. Send the text (truncated if very long) to the LLM with a
           structured extraction prompt.
        3. Validate the response with Pydantic and return it as a dict.
    """
    # Step 1 — get raw text
    raw_text = _extract_text_from_pdf(file_path)

    if not raw_text.strip():
        # If the PDF is empty or unreadable, return a minimal stub so the
        # pipeline doesn't crash.
        return PaperMetadata(
            title=os.path.basename(file_path),
        ).model_dump() | {"source_file": file_path}

    # Step 2 — truncate to avoid exceeding context windows.
    # Most useful info (title, abstract, methodology) is near the top.
    max_chars = 15_000
    truncated_text = raw_text[:max_chars]

    # Step 3 — ask the LLM to extract fields
    # We use LangChain's `with_structured_output` when available (works with
    # OpenAI function-calling models). For models that don't support it we
    # fall back to a plain invoke + manual parsing.
    try:
        structured_llm = llm.with_structured_output(PaperMetadata)
        result: PaperMetadata = structured_llm.invoke(
            _EXTRACTION_PROMPT.format_messages(paper_text=truncated_text)
        )
    except (NotImplementedError, AttributeError):
        # Fallback: invoke without structured output and build a minimal
        # PaperMetadata from the raw response.
        raw_response = llm.invoke(
            _EXTRACTION_PROMPT.format_messages(paper_text=truncated_text)
        )
        response_text = (
            raw_response.content
            if hasattr(raw_response, "content")
            else str(raw_response)
        )
        result = PaperMetadata(title=response_text[:200])

    # Step 4 — return as dict with the source file path attached
    output = result.model_dump()
    output["source_file"] = file_path
    return output
