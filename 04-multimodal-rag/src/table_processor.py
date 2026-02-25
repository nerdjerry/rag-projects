"""
table_processor.py — Convert extracted CSV tables into natural-language descriptions.

THE CHALLENGE OF SEARCHING TABULAR DATA
────────────────────────────────────────
Tables are highly structured: rows, columns, headers, numeric values.
When you flatten a table into a single string it often looks like gibberish
to an embedding model:

    "Q1,Q2,Q3,Q4\n120,135,142,158"

An embedding model trained on prose will not understand that "Q3 revenue"
maps to the value 142.  So we ask an LLM to *describe* the table in plain
English:

    "This table shows quarterly revenue.  Q1 = $120M, Q2 = $135M, …"

That description embeds well and matches user queries like "What was Q3
revenue?".  We keep the raw CSV on disk so we can show exact numbers when
the table is retrieved.
"""

import os
import csv
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Prompt template for turning a CSV table into a natural-language summary.
TABLE_DESCRIPTION_PROMPT = (
    "You are a data analyst.  Below is a table extracted from a document.\n"
    "Describe its contents in 2-4 sentences of clear English.  Mention:\n"
    "  • What the table is about (topic / title if visible)\n"
    "  • Column names and what they represent\n"
    "  • Key numbers or trends\n"
    "  • Any notable patterns\n\n"
    "TABLE (CSV format):\n"
    "{table_csv}\n\n"
    "DESCRIPTION:"
)


def _read_csv_as_string(csv_path: str, max_rows: int = 50) -> str:
    """Read a CSV file and return its content as a single string (capped)."""
    rows: list[str] = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for i, row in enumerate(reader):
            if i >= max_rows:
                rows.append("... (truncated)")
                break
            rows.append(",".join(row))
    return "\n".join(rows)


def _describe_table_with_llm(table_csv: str, llm) -> str:
    """Send the CSV content to an LLM and get a natural-language summary."""
    prompt = TABLE_DESCRIPTION_PROMPT.format(table_csv=table_csv)
    response = llm.invoke(prompt)
    # LangChain LLMs return either a string or an AIMessage
    text = response.content if hasattr(response, "content") else str(response)
    return text.strip()


def _describe_table_heuristic(table_csv: str) -> str:
    """
    Fallback: build a basic description without an LLM.
    Useful for offline / no-API-key scenarios.
    """
    lines = table_csv.strip().splitlines()
    if not lines:
        return "Empty table."

    header = lines[0]
    num_rows = len(lines) - 1  # minus the header row
    return (
        f"Table with columns: {header}. "
        f"Contains {num_rows} data row(s). "
        f"First row: {lines[1] if num_rows > 0 else 'N/A'}."
    )


def process_tables(
    table_paths: list[str],
    llm=None,
) -> list[dict]:
    """
    Generate a natural-language description for every extracted table.

    Parameters
    ----------
    table_paths : list[str]
        Filesystem paths to CSV files produced by the parser.
    llm : BaseLLM or BaseChatModel, optional
        A LangChain LLM used to generate descriptions.  If None, a simple
        heuristic description is used instead.

    Returns
    -------
    list of dicts:  [{"path": str, "description": str}, ...]
    """
    results: list[dict] = []

    for idx, csv_path in enumerate(table_paths, start=1):
        logger.info("Describing table %d/%d: %s", idx, len(table_paths), os.path.basename(csv_path))

        table_csv = _read_csv_as_string(csv_path)

        if llm is not None:
            try:
                description = _describe_table_with_llm(table_csv, llm)
            except Exception as exc:
                logger.error("LLM failed for %s: %s — using heuristic", csv_path, exc)
                description = _describe_table_heuristic(table_csv)
        else:
            description = _describe_table_heuristic(table_csv)

        results.append({"path": csv_path, "description": description})
        logger.info("  → description length: %d chars", len(description))

    logger.info("Processed %d tables total.", len(results))
    return results
