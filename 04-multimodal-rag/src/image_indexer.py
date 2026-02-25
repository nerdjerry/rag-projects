"""
image_indexer.py — Embed image captions and store them in a FAISS index.

Each document in this index contains:
  • page_content = the generated caption text
  • metadata     = {"source_image": "/path/to/image.png"}

This lets us retrieve relevant images by searching their captions, then
show the user the original image file.
"""

import os
import logging

from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def index_image_captions(
    image_data: list[dict],
    embeddings,
    index_path: str = "data/indexes/image_index",
) -> FAISS:
    """
    Build a FAISS vector store from image caption data.

    Parameters
    ----------
    image_data : list[dict]
        Each dict has {"path": str, "caption": str} — output of
        image_processor.process_images().
    embeddings : Embeddings
        LangChain embedding model (must match the one used for text).
    index_path : str
        Where to save the persisted index.

    Returns
    -------
    FAISS vector store, or None if there are no images to index.
    """
    if not image_data:
        logger.warning("No image captions to index.")
        return None

    # Wrap each caption in a LangChain Document with metadata pointing back
    # to the original image file.
    documents = [
        Document(
            page_content=item["caption"],
            metadata={"source_image": item["path"]},
        )
        for item in image_data
    ]

    vectorstore = FAISS.from_documents(documents, embeddings)

    os.makedirs(os.path.dirname(index_path) if os.path.dirname(index_path) else ".", exist_ok=True)
    vectorstore.save_local(index_path)
    logger.info("Image-caption index saved → %s  (%d entries)", index_path, len(documents))

    return vectorstore


def load_image_index(embeddings, index_path: str = "data/indexes/image_index") -> FAISS:
    """
    Reload a previously-saved image-caption FAISS index.

    Parameters
    ----------
    embeddings : Embeddings
        Must be the **same** model used at index-build time.
    index_path : str
        Directory containing the FAISS index files.

    Returns
    -------
    FAISS vector store, or None if the path does not exist.
    """
    if not os.path.isdir(index_path):
        logger.warning("Image index not found at %s", index_path)
        return None

    vectorstore = FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )
    logger.info("Loaded image index from %s", index_path)
    return vectorstore
