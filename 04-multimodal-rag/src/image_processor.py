"""
image_processor.py
------------------
Converts image files into natural-language captions using a vision-capable
OpenAI model (e.g. gpt-4o-mini), making images semantically searchable.

Why convert images to text captions?
--------------------------------------
Semantic search engines (FAISS + sentence-transformers) operate in *text*
embedding space.  A raw PNG file cannot be compared to a natural-language
query like "architecture diagram of the data pipeline."

By asking a vision model to *describe* an image in detail, we produce a text
string that captures the visual content — labels, shapes, data, layout — in a
form that a sentence-transformer can embed and a user query can match against.

How vision input works
------------------------
Current OpenAI chat models (gpt-4o, gpt-4o-mini, etc.) accept *both* text and
images in the same prompt. Images are supplied as base64-encoded strings
inside a message with role "user". (The earlier gpt-4-vision-preview model
that pioneered this API has since been retired — any current chat model with
vision support uses the same message format.)

The base64 encoding pattern:
  1. Read the image file in binary mode.
  2. Encode with base64.b64encode(raw_bytes).decode("utf-8").
  3. Pass as {"type": "image_url", "image_url": {"url": "data:image/png;base64,<b64>"}}
     inside the messages list.

Cost consideration ⚠️
----------------------
Vision requests are more expensive than text-only calls:
  * A 1024×1024 image costs roughly 765 tokens at the "high" detail setting.
  * Caption all images once, then **cache** the results to avoid re-captioning
    on every run.  The main pipeline serialises captions to disk for this reason.

Alternative: LLaVA
-------------------
LLaVA (Large Language and Vision Assistant) is an open-source vision model
that runs locally with Ollama — zero API cost.  Swap `caption_image` to call
`ollama.chat(model="llava", ...)` for a cost-free local alternative, at the
expense of some caption quality.
"""

import base64
import io

from PIL import Image


def caption_image(
    image_path: str,
    openai_client,
    vision_model: str = "gpt-4o-mini",
) -> dict:
    """
    Generate a detailed text caption for a single image using a vision model.

    Parameters
    ----------
    image_path    : Path to the image file (PNG, JPEG, etc.).
    openai_client : An initialised openai.OpenAI() client instance.
    vision_model  : OpenAI vision-capable model identifier (must accept image
                    inputs — e.g. "gpt-4o" or "gpt-4o-mini").

    Returns
    -------
    dict with keys:
      "image_path" — the original path (used as a reference in search results)
      "caption"    — the generated natural-language description
      "image_type" — coarse type extracted from the caption (e.g. "chart")
    """
    # ── Step 1: read and base64-encode the image ─────────────────────────────
    with open(image_path, "rb") as f:
        raw_bytes = f.read()

    # Normalise to PNG via PIL to ensure a consistent MIME type.
    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    png_buffer = io.BytesIO()
    pil_img.save(png_buffer, format="PNG")
    b64_image = base64.b64encode(png_buffer.getvalue()).decode("utf-8")

    # ── Step 2: build the GPT-4V prompt ──────────────────────────────────────
    # The data-URI scheme embeds the image directly in the JSON payload.
    data_uri = f"data:image/png;base64,{b64_image}"

    prompt_text = (
        "Describe this image in detail for a document search system. "
        "Include: what the image shows, any text visible, any data or statistics shown, "
        "the type of visualization (chart, diagram, photo, etc.)."
    )

    try:
        response = openai_client.chat.completions.create(
            model=vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
            max_tokens=512,
        )
        caption = response.choices[0].message.content.strip()

        # Derive a coarse image_type by scanning the caption for keywords.
        image_type = _infer_image_type(caption)

    except Exception as exc:
        # Graceful degradation: if the vision model is unavailable (quota, model
        # access, or network issue) we return a placeholder so the pipeline keeps
        # running. The placeholder still gets indexed; it just won't match queries
        # well — process_all_images() surfaces a loud warning when this happens so
        # the failure isn't silently mistaken for a working image-search feature.
        print(f"  [image_processor] ⚠️  Vision model unavailable for '{image_path}': {exc}")
        caption = f"[Image caption unavailable — {image_path}]"
        image_type = "unknown"

    return {
        "image_path": image_path,
        "caption": caption,
        "image_type": image_type,
    }


def process_all_images(
    image_paths: list[str],
    openai_client,
    vision_model: str = "gpt-4o-mini",
) -> list[dict]:
    """
    Caption every image in the list and return combined results.

    Parameters
    ----------
    image_paths   : List of file paths returned by the multimodal parser.
    openai_client : An initialised openai.OpenAI() client instance.
    vision_model  : OpenAI vision-capable model identifier.

    Returns
    -------
    List of caption dicts (same structure as caption_image() return value).

    Note: captioning is done sequentially to stay within rate limits.
    For large document sets, consider batching with a short sleep between calls.
    """
    results = []
    for idx, path in enumerate(image_paths, start=1):
        print(f"  [image_processor] Captioning image {idx}/{len(image_paths)}: {path}")
        result = caption_image(path, openai_client, vision_model)
        results.append(result)

    failed = [r for r in results if r["image_type"] == "unknown"]
    if failed:
        print(
            f"\n  [image_processor] ⚠️  WARNING: {len(failed)}/{len(results)} image(s) "
            f"failed to caption and were indexed as placeholders — they will not "
            f"match search queries. Check the vision model name and API access above."
        )

    return results


# ── Private helpers ──────────────────────────────────────────────────────────


def _infer_image_type(caption: str) -> str:
    """Heuristically classify the image type from its caption text."""
    caption_lower = caption.lower()
    if any(w in caption_lower for w in ("chart", "bar", "pie", "line graph", "plot")):
        return "chart"
    if any(w in caption_lower for w in ("diagram", "flowchart", "architecture", "uml")):
        return "diagram"
    if any(w in caption_lower for w in ("table", "matrix", "grid")):
        return "table_image"
    if any(w in caption_lower for w in ("photo", "photograph", "picture", "image of")):
        return "photo"
    return "figure"
