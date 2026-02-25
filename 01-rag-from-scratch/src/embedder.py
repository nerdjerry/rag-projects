"""
embedder.py — Turn text into numerical vectors (embeddings).

WHAT ARE EMBEDDINGS?
  An embedding is a list of numbers (a vector) that captures the *meaning*
  of a piece of text.  Texts with similar meanings produce vectors that
  are close together in this high-dimensional space.

  For example:
    "The cat sat on the mat"  →  [0.12, -0.03, 0.88, ...]
    "A kitten rested on a rug" →  [0.11, -0.04, 0.87, ...]   ← very similar!
    "Stock prices rose today"  →  [-0.55, 0.41, 0.02, ...]   ← very different

WHY COSINE SIMILARITY?
  We compare vectors using *cosine similarity* because it measures the
  angle between two vectors, ignoring their length.  This means a short
  sentence and a long paragraph about the same topic will still be
  recognised as similar, even though the raw numbers differ in magnitude.

MODEL CHOICE
  We use the lightweight "all-MiniLM-L6-v2" model from Sentence
  Transformers.  It runs locally (no API key needed), is fast, and
  produces 384-dimensional vectors — a great default for learning.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings


# Default model — small, fast, and free.  Feel free to swap in a larger
# model (e.g., "all-mpnet-base-v2") if you need higher accuracy.
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def get_embeddings(model_name: str = DEFAULT_MODEL_NAME) -> HuggingFaceEmbeddings:
    """
    Create and return a HuggingFace embedding model.

    The first call downloads the model weights (~80 MB).  Subsequent
    calls use the cached version.

    Args:
        model_name: Name of the Sentence Transformers model to use.

    Returns:
        A LangChain-compatible embedding object that can convert text
        into vectors.
    """
    print(f"Loading embedding model: {model_name} ...")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print("Embedding model ready.")
    return embeddings


def show_embedding_example(embeddings: HuggingFaceEmbeddings, text: str) -> None:
    """
    Embed a single piece of text and print its shape and first 5 values.

    This is a learning helper — call it to see what an embedding actually
    looks like under the hood.

    Args:
        embeddings: The embedding model returned by get_embeddings().
        text:       Any string you want to embed.
    """
    vector = embeddings.embed_query(text)
    print(f"\nExample embedding for: \"{text}\"")
    print(f"  Dimensions : {len(vector)}")
    print(f"  First 5 values: {vector[:5]}")


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    emb = get_embeddings()
    show_embedding_example(emb, "Retrieval-Augmented Generation is cool!")
