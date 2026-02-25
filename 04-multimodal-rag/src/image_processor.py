"""
image_processor.py — Convert images into searchable text captions.

WHY DO WE CAPTION IMAGES?
──────────────────────────
Vector search works on *text* embeddings.  An image by itself cannot be
matched against a user's text query.  So we use a vision-capable LLM
(e.g. GPT-4o or LLaVA via Ollama) to describe every extracted image in
plain English.  The caption is then embedded just like any other text
chunk, and the original image path is kept as metadata so we can show it
to the user when it's relevant.

Flow:
  image.png  ──▶  Vision LLM  ──▶  "A bar chart showing Q3 revenue…"
                                         │
                                         ▼
                              embed caption → FAISS index
"""

import os
import base64
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# The prompt we send alongside each image.  It tells the model to produce
# a rich, factual description that will work well as a search target.
IMAGE_CAPTION_PROMPT = (
    "Describe this image in detail for a search system. "
    "Include all visible text, labels, numbers, colours, and the type of "
    "visual (photo, chart, diagram, screenshot, etc.). "
    "Be factual and thorough."
)


def _encode_image_base64(image_path: str) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _caption_with_openai(image_path: str, model: str = "gpt-4o") -> str:
    """
    Send an image to the OpenAI Chat Completions API (vision-capable model)
    and get back a textual description.

    The image is base64-encoded and passed inline so no public URL is needed.
    """
    from langchain_openai import ChatOpenAI
    from langchain.schema.messages import HumanMessage

    b64 = _encode_image_base64(image_path)
    # Determine MIME type from extension
    ext = os.path.splitext(image_path)[1].lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(ext, "image/png")

    llm = ChatOpenAI(model=model, max_tokens=512)
    message = HumanMessage(
        content=[
            {"type": "text", "text": IMAGE_CAPTION_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]
    )
    response = llm.invoke([message])
    return response.content.strip()


def _caption_with_ollama(image_path: str, model: str = "llava") -> str:
    """
    Use a local Ollama model (e.g. LLaVA) to caption an image.
    Ollama's Python client accepts raw image bytes.
    """
    from langchain_community.llms import Ollama

    b64 = _encode_image_base64(image_path)
    llm = Ollama(model=model)
    # Ollama vision models accept images via the `images` kwarg
    response = llm.invoke(
        IMAGE_CAPTION_PROMPT,
        images=[b64],
    )
    return response.strip()


def process_images(
    image_paths: list[str],
    llm=None,
    *,
    use_ollama: bool = False,
    vision_model: str = "gpt-4o",
) -> list[dict]:
    """
    Generate a text caption for every extracted image.

    Parameters
    ----------
    image_paths : list[str]
        Filesystem paths to PNG/JPEG images from the parser.
    llm : optional
        Not used directly — kept for API symmetry with other processors.
        Vision calls are made internally because they need multi-modal input.
    use_ollama : bool
        If True, use a local Ollama vision model instead of OpenAI.
    vision_model : str
        Model name (e.g. "gpt-4o" for OpenAI, "llava" for Ollama).

    Returns
    -------
    list of dicts:  [{"path": str, "caption": str}, ...]
    """
    results: list[dict] = []

    for idx, img_path in enumerate(image_paths, start=1):
        logger.info("Captioning image %d/%d: %s", idx, len(image_paths), os.path.basename(img_path))

        try:
            if use_ollama:
                caption = _caption_with_ollama(img_path, model=vision_model)
            else:
                caption = _caption_with_openai(img_path, model=vision_model)
        except Exception as exc:
            logger.error("Failed to caption %s: %s", img_path, exc)
            caption = f"[Image: {os.path.basename(img_path)}]"

        results.append({"path": img_path, "caption": caption})
        logger.info("  → caption length: %d chars", len(caption))

    logger.info("Processed %d images total.", len(results))
    return results
