"""
multi_retriever.py — Retrieve and merge results from multiple FAISS indexes.

THE CHALLENGE OF CROSS-MODAL RANKING
─────────────────────────────────────
When we search three separate indexes (text, image-captions, table-
descriptions), each returns results with its own similarity scores.
These scores are **not directly comparable** across indexes because:

  • Different content lengths produce different score distributions.
  • A "0.82 match" in the text index is not the same quality as
    "0.82" in the image index.

Our strategy:
  1. Retrieve top-k from each relevant index independently.
  2. Normalize scores within each index to [0, 1].
  3. Tag each result with its modality (text / image / table).
  4. Merge, de-duplicate, and sort by normalized score.
  5. Return the top-k results overall.

This is a simple but effective approach.  Production systems may learn
cross-modal weights or use a re-ranker model.
"""

import logging
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """One piece of retrieved context, tagged with its modality."""
    content: str
    modality: str            # "TEXT", "IMAGE", or "TABLE"
    score: float = 0.0       # normalised similarity (higher = better)
    metadata: dict = field(default_factory=dict)


def _normalize_scores(docs_and_scores: list) -> list:
    """
    Min-max normalize FAISS distances into [0, 1] similarity scores.
    FAISS returns L2 distances (lower = better), so we invert them.
    """
    if not docs_and_scores:
        return []

    distances = [score for _, score in docs_and_scores]
    min_d = min(distances)
    max_d = max(distances)
    span = max_d - min_d if max_d != min_d else 1.0

    normalized = []
    for doc, dist in docs_and_scores:
        # Invert: small distance → high similarity
        sim = 1.0 - ((dist - min_d) / span)
        normalized.append((doc, sim))
    return normalized


def _search_index(index, query: str, modality: str, k: int) -> list[RetrievedChunk]:
    """Search a single FAISS index and return tagged, normalized results."""
    if index is None:
        return []

    # similarity_search_with_score returns (Document, float) tuples
    raw = index.similarity_search_with_score(query, k=k)
    normed = _normalize_scores(raw)

    chunks = []
    for doc, sim in normed:
        chunks.append(RetrievedChunk(
            content=doc.page_content,
            modality=modality,
            score=sim,
            metadata=doc.metadata,
        ))
    return chunks


def retrieve_multimodal(
    query: str,
    indexes: dict,
    router_result: list[str],
    k: int = 3,
) -> list[RetrievedChunk]:
    """
    Retrieve the top-k most relevant results across all routed indexes.

    Parameters
    ----------
    query : str
        The user's natural-language question.
    indexes : dict
        Mapping of modality name to FAISS vector store:
        {"TEXT": <FAISS>, "IMAGE": <FAISS>, "TABLE": <FAISS>}
        Any key may be None if that modality was not indexed.
    router_result : list[str]
        Which indexes to search, e.g. ["TEXT", "IMAGE"].
    k : int
        Number of results to return per index (before merge).

    Returns
    -------
    list[RetrievedChunk]
        Merged, de-duplicated, and scored results sorted best-first.
    """
    all_chunks: list[RetrievedChunk] = []

    for modality in router_result:
        index = indexes.get(modality)
        if index is None:
            logger.info("Skipping %s — no index available.", modality)
            continue

        hits = _search_index(index, query, modality, k)
        logger.info("Retrieved %d results from %s index.", len(hits), modality)
        all_chunks.extend(hits)

    # --- De-duplicate by content (exact match) ------------------------------
    seen: set[str] = set()
    unique: list[RetrievedChunk] = []
    for chunk in all_chunks:
        key = chunk.content.strip()
        if key not in seen:
            seen.add(key)
            unique.append(chunk)

    # --- Sort by normalized similarity score (descending) -------------------
    unique.sort(key=lambda c: c.score, reverse=True)

    # Keep only the overall top-k
    final = unique[:k]
    logger.info("Merged results: %d unique → returning top %d.", len(unique), len(final))

    return final


def format_context(chunks: list[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into a single context string for the generator.

    Each chunk is labelled with its modality so the LLM knows what kind of
    evidence it is working with (e.g. "[IMAGE] A bar chart showing …").
    """
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        label = f"[{chunk.modality}]"
        # Add source info when available
        source = ""
        if "source_image" in chunk.metadata:
            source = f" (source: {chunk.metadata['source_image']})"
        elif "source_table" in chunk.metadata:
            source = f" (source: {chunk.metadata['source_table']})"

        parts.append(f"{i}. {label}{source}\n{chunk.content}")

    return "\n\n".join(parts)
