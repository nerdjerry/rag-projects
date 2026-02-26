"""
multi_retriever.py
------------------
Queries one or more FAISS indexes in parallel based on the query types
returned by the router, then merges the results into a single ranked list.

The challenge of ranking across modalities
-------------------------------------------
Each FAISS index returns an L2 distance score in the embedding space of
all-MiniLM-L6-v2 (384 dimensions).  Because all three indexes use the *same*
embedding model, scores are theoretically comparable — but in practice:

  * The distribution of scores differs by modality (short captions tend to
    have lower variance than long text chunks).
  * A "0.3 score" for a text chunk may not be semantically equivalent to a
    "0.3 score" for an image caption.

Two ranking strategies are discussed here:

  Simple (implemented): interleave results — 1 text result, 1 image result,
  1 table result — so every modality is represented in the context, regardless
  of raw score magnitude.  Easy to implement, transparent to the user.

  Complex (alternative): normalise scores per-modality using min-max scaling,
  then sort globally.  More precise but can still suppress a modality entirely
  if its scores are consistently higher (worse) than others.

We use the simple interleaving approach and let the generator model weight
results contextually via its attention mechanism.

De-duplication
--------------
The same text snippet can theoretically appear in multiple indexes (e.g. a
table that was also mentioned verbatim in the text).  We de-duplicate on
content string to avoid feeding the same information twice to the generator.
"""

from langchain_community.vectorstores import FAISS

from .query_router import QueryType
from .text_indexer import search_text
from .image_indexer import search_images
from .table_indexer import search_tables


def retrieve_all(
    query: str,
    query_types: list[QueryType],
    text_index: FAISS | None,
    image_index: FAISS | None,
    table_index: FAISS | None,
    k: int = 3,
) -> list[dict]:
    """
    Retrieve top-k results from each relevant index and return a combined list.

    Parameters
    ----------
    query       : User's natural-language question.
    query_types : List of QueryType values from the router.
    text_index  : Loaded FAISS text index (or None if not built).
    image_index : Loaded FAISS image-caption index (or None if not built).
    table_index : Loaded FAISS table-description index (or None if not built).
    k           : Number of results to fetch from each relevant index.

    Returns
    -------
    List of result dicts:
      {
        "content"  : str  — the text content (chunk / caption / description)
        "modality" : str  — "text" | "image" | "table"
        "metadata" : dict — index-specific metadata (image_path, csv_path, etc.)
        "source"   : str  — human-readable source label
        "score"    : float
      }
    """
    results: list[dict] = []

    if QueryType.TEXT in query_types and text_index is not None:
        for doc, score in search_text(query, text_index, k=k):
            results.append(
                {
                    "content": doc.page_content,
                    "modality": "text",
                    "metadata": doc.metadata,
                    "source": f"text_chunk_{doc.metadata.get('chunk_id', '?')}",
                    "score": float(score),
                }
            )

    if QueryType.IMAGE in query_types and image_index is not None:
        for item in search_images(query, image_index, k=k):
            results.append(
                {
                    "content": item["caption"],
                    "modality": "image",
                    "metadata": {
                        "image_path": item["image_path"],
                        "image_type": item["image_type"],
                    },
                    "source": item["image_path"],
                    "score": item["score"],
                }
            )

    if QueryType.TABLE in query_types and table_index is not None:
        for item in search_tables(query, table_index, k=k):
            results.append(
                {
                    "content": item["description"],
                    "modality": "table",
                    "metadata": {
                        "table_id": item["table_id"],
                        "csv_path": item["csv_path"],
                        "page": item["page"],
                    },
                    "source": item["table_id"],
                    "score": item["score"],
                }
            )

    return results


def merge_and_rank_results(results: list[dict]) -> list[dict]:
    """
    De-duplicate and interleave results across modalities.

    De-duplication is done on content string (exact match).  Ranking uses a
    simple modality-interleaving strategy: we pick results round-robin from
    text → image → table buckets so every modality is represented early in
    the context window.

    Parameters
    ----------
    results : Combined list from retrieve_all().

    Returns
    -------
    De-duplicated, interleaved list of result dicts.
    """
    # De-duplicate on content string.
    seen_content: set[str] = set()
    unique: list[dict] = []
    for r in results:
        if r["content"] not in seen_content:
            seen_content.add(r["content"])
            unique.append(r)

    # Separate into modality buckets.
    buckets: dict[str, list[dict]] = {"text": [], "image": [], "table": []}
    for r in unique:
        bucket_key = r["modality"] if r["modality"] in buckets else "text"
        buckets[bucket_key].append(r)

    # Sort each bucket by ascending score (lower L2 = more similar).
    for bucket in buckets.values():
        bucket.sort(key=lambda x: x["score"])

    # Interleave: take one from each non-empty bucket in rotation.
    merged: list[dict] = []
    order = ["text", "image", "table"]
    max_len = max((len(b) for b in buckets.values()), default=0)
    for i in range(max_len):
        for modality in order:
            if i < len(buckets[modality]):
                merged.append(buckets[modality][i])

    return merged
