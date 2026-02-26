"""
table_processor.py
------------------
Converts extracted tables into natural-language descriptions suitable for
semantic search, while also persisting the raw data as CSV files.

The challenge of searching tabular data semantically
-----------------------------------------------------
Tables are inherently 2-D structured objects.  A flat string representation
like "Q1 | 1000000 | Q2 | 1200000" is syntactically correct but semantically
opaque to a sentence-transformer trained on prose.

Why generate natural-language descriptions?
--------------------------------------------
Text embedding models (and LLMs used for generation) understand sentences
like "Q1 revenue was $1 M, representing 12 % growth quarter-over-quarter"
far better than a raw CSV row.  By asking an LLM to paraphrase a table, we
convert the structured 2-D data into a format that:
  1. Embeds meaningfully with all-MiniLM-L6-v2.
  2. Matches natural-language queries ("What were the Q1 sales figures?").
  3. Can be injected verbatim into a generation prompt for the final answer.

Why keep the raw CSV too?
--------------------------
Natural-language descriptions are lossy — they summarise, not enumerate.
For exact queries ("What was the exact revenue in row 7, column 3?") or for
programmatic downstream use (pandas, Excel), the CSV is the ground truth.
We store both and surface whichever is appropriate.
"""

import csv
import os
from pathlib import Path


def table_to_description(table: list[list], llm) -> str:
    """
    Use an LLM to convert a raw table (list of rows) into a prose description.

    Parameters
    ----------
    table : 2-D list where table[0] is typically the header row and
            subsequent rows are data rows.  Cell values are strings.
    llm   : A LangChain chat/LLM object that supports .invoke() or .predict().

    Returns
    -------
    A natural-language string describing the table's content and structure.
    """
    if not table:
        return "Empty table."

    # Format the table as a plain-text grid so the LLM can parse it easily.
    table_str = _format_table_as_text(table)

    prompt = (
        "Convert this table to a natural language description for search purposes. "
        "Describe what data the table contains, its structure, and key values.\n\n"
        f"Table:\n{table_str}"
    )

    try:
        # Support both .invoke() (LangChain ≥ 0.1) and .predict() (legacy).
        if hasattr(llm, "invoke"):
            response = llm.invoke(prompt)
            # .invoke() may return a string or an AIMessage depending on the model.
            description = response.content if hasattr(response, "content") else str(response)
        else:
            description = llm.predict(prompt)
        return description.strip()

    except Exception as exc:
        # Non-fatal fallback: return the raw text representation.
        print(f"  [table_processor] LLM unavailable for table description: {exc}")
        return f"Table data:\n{table_str}"


def save_table_as_csv(table: list[list], output_path: str) -> None:
    """
    Write a 2-D list to a CSV file.

    Parameters
    ----------
    table       : 2-D list of cell values.
    output_path : Full file path for the output CSV (directory must exist).
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(table)


def process_all_tables(
    tables: list[dict],
    llm,
    tables_dir: str = "data/extracted/tables",
) -> list[dict]:
    """
    Process every table extracted by the parser: save as CSV and generate a
    natural-language description via the LLM.

    Parameters
    ----------
    tables     : List of table dicts as returned by multimodal_parser —
                 each has keys "rows" (list[list]) and "page" (int).
    llm        : LangChain LLM / chat model for description generation.
    tables_dir : Directory where CSV files are written.

    Returns
    -------
    List of dicts:
      {
        "table_id"    : str        — unique identifier, e.g. "table_p2_0"
        "csv_path"    : str        — path to the saved CSV file
        "description" : str        — LLM-generated natural-language summary
        "raw_table"   : list[list] — original row data
        "page"        : int        — source page number
      }
    """
    Path(tables_dir).mkdir(parents=True, exist_ok=True)
    results = []

    for idx, table_meta in enumerate(tables):
        raw_rows = table_meta.get("rows", [])
        page = table_meta.get("page", 0)
        table_id = f"table_p{page}_{table_meta.get('table_index', idx)}"

        csv_filename = f"{table_id}.csv"
        csv_path = os.path.join(tables_dir, csv_filename)

        # Persist raw data.
        save_table_as_csv(raw_rows, csv_path)

        # Generate natural-language description.
        print(
            f"  [table_processor] Describing table {idx + 1}/{len(tables)} "
            f"(page {page}) …"
        )
        description = table_to_description(raw_rows, llm)

        results.append(
            {
                "table_id": table_id,
                "csv_path": csv_path,
                "description": description,
                "raw_table": raw_rows,
                "page": page,
            }
        )

    return results


# ── Private helpers ──────────────────────────────────────────────────────────


def _format_table_as_text(table: list[list]) -> str:
    """Render a 2-D list as a plain-text grid with | separators."""
    lines = []
    for row in table:
        lines.append(" | ".join(str(cell) for cell in row))
    return "\n".join(lines)
