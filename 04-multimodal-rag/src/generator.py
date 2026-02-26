"""
generator.py
------------
Builds a structured prompt from multimodal retrieved results and calls the
LLM to produce the final answer.

The prompt explicitly labels each piece of context by modality ([TEXT],
[IMAGE DESCRIPTIONS], [TABLE DATA]) so the model can reason about the
*source* of information — e.g. "the bar chart (image) shows Q4 was highest,
while the revenue table confirms $1.2M."  The model is instructed to
acknowledge which modality informed its answer, which improves transparency
and helps users verify the response against the source document.
"""


def generate_answer(
    query: str,
    retrieved_results: list[dict],
    llm,
    include_image_refs: bool = True,
) -> str:
    """
    Generate a natural-language answer from multimodal retrieved context.

    Parameters
    ----------
    query             : The user's original question.
    retrieved_results : Combined, ranked list from multi_retriever.merge_and_rank_results().
    llm               : LangChain LLM / chat model.
    include_image_refs: When True, append "See image: <path>" lines for any
                        image results so the user knows where to look.

    Returns
    -------
    Formatted answer string.
    """
    # ── Separate results by modality ─────────────────────────────────────────
    text_chunks: list[str] = []
    image_captions: list[str] = []
    table_descriptions: list[str] = []
    image_refs: list[str] = []

    for result in retrieved_results:
        modality = result.get("modality", "text")
        content = result.get("content", "").strip()

        if modality == "text":
            text_chunks.append(content)
        elif modality == "image":
            image_captions.append(content)
            if include_image_refs:
                img_path = result.get("metadata", {}).get("image_path", "")
                if img_path:
                    image_refs.append(img_path)
        elif modality == "table":
            table_descriptions.append(content)

    # ── Build context sections ────────────────────────────────────────────────
    text_section = "\n\n".join(text_chunks) if text_chunks else "No text context available."
    image_section = "\n\n".join(image_captions) if image_captions else "No image context available."
    table_section = "\n\n".join(table_descriptions) if table_descriptions else "No table context available."

    # ── Assemble prompt ───────────────────────────────────────────────────────
    prompt = f"""\
Answer the following question based on the provided context from a document.
The context includes text, image descriptions, and table data.

Context:
[TEXT]
{text_section}

[IMAGE DESCRIPTIONS]
{image_section}

[TABLE DATA]
{table_section}

Question: {query}

Answer (mention which type of content informed your answer — text/image/table):"""

    # ── Call the LLM ──────────────────────────────────────────────────────────
    try:
        if hasattr(llm, "invoke"):
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)
        else:
            answer = llm.predict(prompt)
        answer = answer.strip()
    except Exception as exc:
        answer = f"[generator] LLM call failed: {exc}"

    # ── Append image references if requested ─────────────────────────────────
    if include_image_refs and image_refs:
        refs_block = "\n".join(f"See image: {path}" for path in image_refs)
        answer = f"{answer}\n\n{refs_block}"

    return answer
