"""
multimodal_parser.py
--------------------
Parses a PDF document and extracts three distinct modalities:
  1. Text blocks  — raw text per page, ready for embedding
  2. Images       — saved as PNG files; need vision model captioning before embedding
  3. Tables       — extracted as list-of-lists, converted to dict rows for downstream processing

Why separate modalities before indexing?
-----------------------------------------
Each content type requires a completely different processing pipeline:

  Text   → can be chunked and embedded directly with a sentence-transformer.

  Images → embedding raw pixel data is rarely useful for Q&A.  Instead we use a
           vision model (GPT-4V or LLaVA) to *describe* each image in plain English,
           then embed that description.  This bridges the "semantic gap" between a
           pixel array and a natural-language query.

  Tables → 2-D structured data doesn't embed well as a flat string of cell values.
           We convert each table into a short natural-language paragraph
           ("Q1 revenue was $1 M, up 12 % year-over-year …") that a sentence-
           transformer can compare against a user question.

Limitations
-----------
  * pdfplumber excels at text and table extraction from text-based PDFs.
  * Image extraction relies on the PDF's internal XObject stream; quality varies.
    Scanned PDFs with no embedded images will yield zero images here.
  * Large tables spanning multiple pages may be split; downstream code should
    handle partial tables gracefully.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
from PIL import Image


@dataclass
class ParsedDocument:
    """Container for all content extracted from a single PDF."""

    file_name: str
    # One entry per page; each entry is the full text of that page.
    text_blocks: list[str] = field(default_factory=list)
    # Absolute/relative paths to saved PNG files extracted from the PDF.
    image_paths: list[str] = field(default_factory=list)
    # Each table is a dict with keys "rows" (list[list]) and "page" (int).
    tables: list[dict] = field(default_factory=list)


def parse_document(
    file_path: str,
    images_dir: str = "data/extracted/images",
    tables_dir: str = "data/extracted/tables",
) -> ParsedDocument:
    """
    Open a PDF and extract text, images, and tables into a ParsedDocument.

    Parameters
    ----------
    file_path   : Path to the source PDF file.
    images_dir  : Directory where extracted PNG images are saved.
    tables_dir  : Directory where extracted tables are saved (CSV, handled downstream).

    Returns
    -------
    ParsedDocument with text_blocks, image_paths, and tables populated.
    """
    Path(images_dir).mkdir(parents=True, exist_ok=True)
    Path(tables_dir).mkdir(parents=True, exist_ok=True)

    file_name = Path(file_path).stem
    text_blocks: list[str] = []
    image_paths: list[str] = []
    tables: list[dict] = []

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):

            # ── 1. TEXT ──────────────────────────────────────────────────────────
            # extract_text() returns the full text of the page as a single string.
            # We keep one block per page; callers can chunk further if needed.
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_blocks.append(page_text.strip())

            # ── 2. TABLES ────────────────────────────────────────────────────────
            # extract_tables() returns a list of tables; each table is a list of
            # rows, and each row is a list of cell values (strings or None).
            for table_idx, raw_table in enumerate(page.extract_tables()):
                # Replace None cells with empty string to avoid downstream errors.
                clean_rows = [
                    [cell if cell is not None else "" for cell in row]
                    for row in raw_table
                ]
                tables.append(
                    {
                        "rows": clean_rows,       # list[list[str]]
                        "page": page_num,
                        "table_index": table_idx,
                    }
                )

            # ── 3. IMAGES ────────────────────────────────────────────────────────
            # pdfplumber exposes raw image XObjects via page.images.
            # Each entry is a dict with keys: "stream" (raw bytes), "x0", "y0",
            # "x1", "y1", "width", "height", etc.
            # We reconstruct a PIL Image from the raw stream and save as PNG.
            for img_idx, img_meta in enumerate(page.images):
                try:
                    raw_stream = img_meta.get("stream")
                    if raw_stream is None:
                        continue

                    # The stream is a pdfplumber PDFStream object; get its raw data.
                    raw_data = (
                        raw_stream.get_data()
                        if hasattr(raw_stream, "get_data")
                        else bytes(raw_stream)
                    )

                    # Attempt to open as a PIL Image (handles JPEG, PNG, etc.).
                    import io
                    try:
                        pil_img = Image.open(io.BytesIO(raw_data))
                        pil_img = pil_img.convert("RGB")  # normalise colour mode
                    except Exception:
                        # The raw bytes may be raw pixel data rather than an encoded
                        # image format. Fall back to using width/height from metadata.
                        width = int(img_meta.get("width", 100))
                        height = int(img_meta.get("height", 100))
                        pil_img = Image.frombytes("RGB", (width, height), raw_data)

                    img_filename = f"{file_name}_page{page_num}_img{img_idx}.png"
                    img_save_path = os.path.join(images_dir, img_filename)
                    pil_img.save(img_save_path, format="PNG")
                    image_paths.append(img_save_path)

                except Exception as exc:
                    # Non-fatal: log and continue — a single bad image shouldn't
                    # abort extraction of the rest of the document.
                    print(
                        f"  [parser] Could not extract image {img_idx} "
                        f"on page {page_num}: {exc}"
                    )

    print(
        f"[parser] '{file_name}': "
        f"{len(text_blocks)} text blocks, "
        f"{len(image_paths)} images, "
        f"{len(tables)} tables extracted."
    )

    return ParsedDocument(
        file_name=file_name,
        text_blocks=text_blocks,
        image_paths=image_paths,
        tables=tables,
    )
