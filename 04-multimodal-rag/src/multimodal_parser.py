"""
multimodal_parser.py — Extract text, images, and tables from PDF documents.

WHY SEPARATE MODALITIES BEFORE INDEXING?
─────────────────────────────────────────
A PDF is a bag of mixed content: paragraphs, diagrams, charts, and tables.
A plain text splitter would lose images entirely and mangle table structure
into meaningless strings.  By extracting each modality separately we can:

  1. TEXT   → chunk and embed with a text model (fast, cheap).
  2. IMAGES → describe with a vision model, then embed the caption.
  3. TABLES → convert to natural-language summaries, then embed.

This lets us search *across* modalities with a single vector query later.
"""

import os
import csv
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primary extractor – uses pdfplumber (pure-Python, always available)
# ---------------------------------------------------------------------------

def _extract_with_pdfplumber(file_path: str, output_dir: str):
    """
    Walk every page of a PDF and pull out:
      • text blocks  → returned as a list of strings (one per page)
      • images       → saved as PNG files under output_dir/images/
      • tables       → saved as CSV files under output_dir/tables/

    pdfplumber is a lightweight, pure-Python library that can extract text
    and tables reliably.  For images it gives bounding-box metadata; we
    crop the page image to get the actual picture.
    """
    import pdfplumber
    from PIL import Image

    text_chunks: list[str] = []
    image_paths: list[str] = []
    table_paths: list[str] = []

    images_dir = os.path.join(output_dir, "images")
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    pdf_stem = Path(file_path).stem  # e.g. "report"

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # --- TEXT -------------------------------------------------------
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text_chunks.append(page_text.strip())
                logger.info("Page %d: extracted %d chars of text", page_num, len(page_text))

            # --- TABLES -----------------------------------------------------
            # extract_tables() returns a list of tables; each table is a
            # list-of-lists (rows × columns).
            for table_idx, table in enumerate(page.extract_tables()):
                if not table:
                    continue
                csv_name = f"{pdf_stem}_p{page_num}_t{table_idx}.csv"
                csv_path = os.path.join(tables_dir, csv_name)
                with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    for row in table:
                        # Replace None cells with empty strings
                        writer.writerow([cell if cell is not None else "" for cell in row])
                table_paths.append(csv_path)
                logger.info("Page %d: saved table → %s", page_num, csv_name)

            # --- IMAGES -----------------------------------------------------
            # pdfplumber exposes page.images with bounding-box coordinates.
            # We render the page to a PIL Image, then crop each region.
            if page.images:
                page_image = page.to_image(resolution=200).original
                for img_idx, img_meta in enumerate(page.images):
                    try:
                        # Bounding box: (x0, top, x1, bottom)
                        bbox = (
                            img_meta["x0"],
                            img_meta["top"],
                            img_meta["x1"],
                            img_meta["bottom"],
                        )
                        cropped = page_image.crop(bbox)
                        # Skip tiny images (likely decorative lines / icons)
                        if cropped.width < 50 or cropped.height < 50:
                            continue
                        img_name = f"{pdf_stem}_p{page_num}_i{img_idx}.png"
                        img_path = os.path.join(images_dir, img_name)
                        cropped.save(img_path, "PNG")
                        image_paths.append(img_path)
                        logger.info("Page %d: saved image → %s", page_num, img_name)
                    except Exception as exc:
                        logger.warning("Page %d: could not crop image %d – %s", page_num, img_idx, exc)

    return text_chunks, image_paths, table_paths


# ---------------------------------------------------------------------------
# Optional: extraction via the `unstructured` library (richer heuristics)
# ---------------------------------------------------------------------------

def _extract_with_unstructured(file_path: str, output_dir: str):
    """
    The *unstructured* library applies layout-detection heuristics that can
    identify titles, narrative text, tables, and images.  It is heavier
    (needs extra system libs) but often more accurate on complex layouts.

    We fall back to pdfplumber if unstructured is not installed.
    """
    from unstructured.partition.pdf import partition_pdf

    images_dir = os.path.join(output_dir, "images")
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    pdf_stem = Path(file_path).stem

    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",              # full layout analysis
        extract_images_in_pdf=True,      # pull embedded images
        extract_image_block_output_dir=images_dir,
    )

    text_chunks: list[str] = []
    image_paths: list[str] = []
    table_paths: list[str] = []
    table_count = 0

    for elem in elements:
        category = elem.category  # "NarrativeText", "Table", "Image", …

        if category in ("NarrativeText", "Title", "ListItem", "UncategorizedText"):
            if str(elem).strip():
                text_chunks.append(str(elem).strip())

        elif category == "Table":
            # Unstructured may give us an HTML or text table; save as CSV.
            csv_name = f"{pdf_stem}_table_{table_count}.csv"
            csv_path = os.path.join(tables_dir, csv_name)
            with open(csv_path, "w", encoding="utf-8") as fh:
                fh.write(str(elem))
            table_paths.append(csv_path)
            table_count += 1

        elif category == "Image":
            # The image file was already written to images_dir by
            # partition_pdf; record its path if available.
            if hasattr(elem, "metadata") and hasattr(elem.metadata, "image_path"):
                image_paths.append(elem.metadata.image_path)

    # Also pick up any images that unstructured saved but didn't tag as
    # Image elements (belt-and-suspenders).
    for fname in os.listdir(images_dir):
        full = os.path.join(images_dir, fname)
        if full not in image_paths:
            image_paths.append(full)

    return text_chunks, image_paths, table_paths


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_multimodal_document(
    file_path: str,
    output_dir: str = "data/extracted",
    use_unstructured: bool = False,
) -> dict:
    """
    Parse a PDF and return separated modalities.

    Parameters
    ----------
    file_path : str
        Path to the PDF file.
    output_dir : str
        Base directory for extracted artefacts (images/, tables/).
    use_unstructured : bool
        If True, try the *unstructured* library first; fall back to
        pdfplumber if it is not installed.

    Returns
    -------
    dict with keys:
        text_chunks  – list[str]   : raw text passages (one per page/section)
        image_paths  – list[str]   : filesystem paths to extracted PNG images
        table_paths  – list[str]   : filesystem paths to extracted CSV tables
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    logger.info("Parsing document: %s", file_path)

    if use_unstructured:
        try:
            text_chunks, image_paths, table_paths = _extract_with_unstructured(file_path, output_dir)
            logger.info("Extraction via unstructured complete.")
        except ImportError:
            logger.warning("unstructured not available — falling back to pdfplumber.")
            text_chunks, image_paths, table_paths = _extract_with_pdfplumber(file_path, output_dir)
    else:
        text_chunks, image_paths, table_paths = _extract_with_pdfplumber(file_path, output_dir)

    logger.info(
        "Result: %d text chunks, %d images, %d tables",
        len(text_chunks), len(image_paths), len(table_paths),
    )

    return {
        "text_chunks": text_chunks,
        "image_paths": image_paths,
        "table_paths": table_paths,
    }


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python multimodal_parser.py <path-to-pdf>")
        sys.exit(1)

    result = parse_multimodal_document(sys.argv[1])
    print(f"\nText chunks : {len(result['text_chunks'])}")
    print(f"Images      : {len(result['image_paths'])}")
    print(f"Tables      : {len(result['table_paths'])}")
