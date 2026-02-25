# src/embedder.py
#
# STEP 3 OF THE RAG PIPELINE: EMBEDDING TEXT INTO VECTORS
#
# WHAT ARE EMBEDDINGS?
# ----------------------
# An "embedding" is a way to represent text as a list of numbers (a vector).
# The key insight is that semantically similar text produces numerically similar vectors.
#
# For example, these two sentences will have very similar vectors:
#   "The cat sat on the mat."
#   "A feline rested on the rug."
#
# Even though they share no keywords, an embedding model understands they mean
# the same thing. This is the magic that makes semantic search work!
#
# WHY all-MiniLM-L6-v2?
# -----------------------
# We use the "all-MiniLM-L6-v2" model from HuggingFace for several reasons:
#
#   ✅ FREE — no API key required, runs entirely on your local machine
#   ✅ FAST — it's a small, distilled model (only ~80MB to download)
#   ✅ GOOD QUALITY — despite its size, it scores well on semantic benchmarks
#   ✅ 384 DIMENSIONS — each piece of text becomes a list of 384 numbers
#
# Alternative: OpenAI's text-embedding-ada-002 is more powerful but costs money
# and requires an API key. For learning, the free HuggingFace model is perfect.
#
# WHAT DOES "384 DIMENSIONS" MEAN?
# ----------------------------------
# Each text string gets converted to a list of 384 floating-point numbers.
# Think of it as a point in 384-dimensional space. Similar texts are "close"
# to each other in this space; unrelated texts are "far apart."
#
# WHY COSINE SIMILARITY?
# -----------------------
# To find which chunks are most relevant to a query, we compare their vectors.
# We use "cosine similarity" which measures the angle between two vectors:
#   - Score of 1.0 = identical direction = very similar meaning
#   - Score of 0.0 = perpendicular = unrelated
#   - Score of -1.0 = opposite direction = opposite meaning (rare in practice)
#
# Cosine similarity is preferred over Euclidean distance because it's insensitive
# to the magnitude of the vectors — only the direction matters.

from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    Load a HuggingFace sentence-transformer embedding model.

    The first time you call this, it will download the model (~80MB) from
    HuggingFace Hub and cache it locally. Subsequent calls use the cache.

    Args:
        model_name (str): HuggingFace model name. Default: "all-MiniLM-L6-v2"
                          Other options: "all-mpnet-base-v2" (higher quality, slower)

    Returns:
        HuggingFaceEmbeddings: A LangChain-compatible embedding model object.
                               Call model.embed_documents([...]) or model.embed_query("...")
    """

    print(f"\n🔢 Loading embedding model: '{model_name}'")
    print(f"   (First run will download ~80MB — subsequent runs use cache)")

    # model_kwargs: passed directly to the underlying sentence-transformers library
    # device="cpu" means we run on CPU — change to "cuda" if you have a GPU
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        # encode_kwargs: controls how the model encodes text into vectors
        # normalize_embeddings=True ensures vectors have length 1.0,
        # which makes cosine similarity equivalent to dot product (faster computation)
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"✅ Embedding model loaded successfully")
    return embedding_model


def embed_text(text: str, model) -> list:
    """
    Embed a single string and show what the resulting vector looks like.

    This is a teaching/demo function — it helps beginners see that "embedding"
    just means converting text into a list of numbers.

    Args:
        text (str): Any string to embed.
        model:      A loaded HuggingFaceEmbeddings (or compatible) model.

    Returns:
        list: A list of 384 floats representing the text's meaning as a vector.

    Example:
        model = get_embedding_model()
        vector = embed_text("Hello world", model)
        # Prints: Vector shape: 384 dimensions
        # Prints: First 5 values: [0.023, -0.041, 0.118, ...]
    """

    # embed_query() is the LangChain method for embedding a single string.
    # (embed_documents() is for embedding a list of strings all at once — more efficient.)
    vector = model.embed_query(text)

    # Show the learner what a vector actually looks like
    print(f"\n🔍 Embedding demo for: '{text[:60]}{'...' if len(text) > 60 else ''}'")
    print(f"   Vector shape:    {len(vector)} dimensions")
    print(f"   First 5 values:  {[round(v, 4) for v in vector[:5]]}")
    print(f"   (Each number encodes a tiny aspect of the text's meaning)")

    return vector
