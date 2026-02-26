"""
image_indexer.py
----------------
Builds and queries a FAISS vector index over image *captions*.

Key insight: we are searching text (captions), but returning image references.
The caption is the searchable representation; the metadata carries the file
path so callers can retrieve or display the actual image.  This pattern —
"index the description, store the reference" — is the standard approach for
making non-text assets semantically searchable without specialised multimodal
embedding models.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document


_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=_EMBED_MODEL_NAME)


def index_image_captions(
    image_data: list[dict],
    index_path: str = "image_faiss_index",
) -> FAISS:
    """
    Embed image captions and save a FAISS index to disk.

    Parameters
    ----------
    image_data  : List of dicts with keys "image_path" and "caption"
                  (as returned by image_processor.process_all_images()).
    index_path  : Directory where FAISS index files are written.

    Returns
    -------
    A LangChain FAISS vector store whose documents are captions with
    image_path stored in metadata.
    """
    if not image_data:
        raise ValueError("image_data is empty — nothing to index.")

    # page_content is the caption text that will be embedded and searched.
    # metadata carries the image_path so we can return the file reference
    # when this document is retrieved.
    docs = [
        Document(
            page_content=item["caption"],
            metadata={
                "image_path": item["image_path"],
                "image_type": item.get("image_type", "figure"),
                "modality": "image",
            },
        )
        for item in image_data
    ]

    embeddings = _get_embeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(index_path)

    print(f"[image_indexer] Indexed {len(docs)} image captions → '{index_path}'")
    return vector_store


def load_image_index(index_path: str) -> FAISS:
    """
    Load a previously saved FAISS image-caption index from disk.

    Parameters
    ----------
    index_path : Directory path passed to index_image_captions().

    Returns
    -------
    A LangChain FAISS vector store.
    """
    embeddings = _get_embeddings()
    vector_store = FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )
    print(f"[image_indexer] Loaded image index from '{index_path}'")
    return vector_store


def search_images(
    query: str,
    vector_store: FAISS,
    k: int = 3,
) -> list[dict]:
    """
    Retrieve the top-k image captions most relevant to a query.

    Parameters
    ----------
    query        : Natural language question or search string.
    vector_store : A loaded or freshly-built FAISS image-caption index.
    k            : Number of results to return.

    Returns
    -------
    List of dicts:
      {
        "caption"    : str   — the generated image description
        "image_path" : str   — path to the original image file
        "image_type" : str   — coarse type (chart, diagram, photo, …)
        "score"      : float — FAISS L2 distance (lower = more similar)
      }
    """
    raw_results = vector_store.similarity_search_with_score(query, k=k)

    return [
        {
            "caption": doc.page_content,
            "image_path": doc.metadata.get("image_path", ""),
            "image_type": doc.metadata.get("image_type", "figure"),
            "score": float(score),
        }
        for doc, score in raw_results
    ]
