# src/retriever.py
#
# STEP 5 OF THE RAG PIPELINE: RETRIEVING RELEVANT CHUNKS
#
# WHAT DOES "RETRIEVAL" DO IN THE RAG PIPELINE?
# -----------------------------------------------
# At this point we have:
#   - All our document chunks stored as vectors in FAISS
#   - A user's question
#
# The retriever's job is to:
#   1. Embed the question using the SAME embedding model we used for the chunks
#   2. Search FAISS for the k most similar chunk vectors to the question vector
#   3. Return those k chunks (as Document objects) so the LLM can read them
#
# The LLM never reads the whole document database — it only reads these k chunks.
# This is what makes RAG efficient and precise.
#
# WHAT IS k IN TOP-k RETRIEVAL?
# --------------------------------
# k is the number of chunks we retrieve. Think of it as:
#   "Give me the top 3 most relevant paragraphs from my documents."
#
#   k=1: Very focused. Only the single best match. May miss related info.
#   k=3: A good balance. Captures the main answer + nearby context. (default)
#   k=10: Comprehensive but may include loosely related chunks that confuse the LLM.
#
# Rule of thumb: Start with k=3 and increase if the LLM says "I don't know"
# on questions you KNOW are in your documents.
#
# WHY COSINE SIMILARITY BEATS KEYWORD SEARCH:
# --------------------------------------------
# Traditional search (like grep or SQL LIKE) requires exact keyword matches.
# If your document says "automobile" and you search for "car", you get nothing.
#
# Semantic (vector) search understands meaning:
#   "car" → very similar vector to "automobile", "vehicle", "sedan"
#
# This means you can ask questions in natural language and still find relevant
# chunks even when the exact words don't match. This is crucial for Q&A systems
# where users phrase questions differently than how documents are written.


def get_retriever(vector_store, k: int = 3):
    """
    Create a LangChain retriever from a FAISS vector store.

    A LangChain "retriever" is a standardized interface that wraps the vector store
    and exposes a simple .invoke(query) method. This makes it easy to plug into
    LangChain chains (like RetrievalQA in generator.py).

    Args:
        vector_store:   A FAISS vector store (from vector_store.py).
        k (int):        How many chunks to retrieve per query. Default: 3.
                        Increase if answers are missing info; decrease if too noisy.

    Returns:
        A LangChain VectorStoreRetriever object.

    Example:
        retriever = get_retriever(vector_store, k=3)
        docs = retriever.invoke("What is the refund policy?")
    """

    # as_retriever() wraps the FAISS store in a Retriever interface.
    # search_type="similarity" uses cosine similarity (since we normalized embeddings).
    # Other options: "mmr" (Maximal Marginal Relevance — reduces redundancy among results)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},  # retrieve top-k most similar chunks
    )

    print(f"\n🔎 Retriever configured (top-k={k}, search_type=similarity)")
    return retriever


def retrieve_chunks(question: str, retriever) -> list:
    """
    Retrieve the most relevant document chunks for a given question.

    Also prints the retrieved chunks so learners can inspect what gets passed
    to the LLM. This transparency is key to understanding and debugging RAG.

    Args:
        question (str):  The user's question in natural language.
        retriever:       A LangChain retriever (from get_retriever()).

    Returns:
        list: A list of LangChain Document objects — the most relevant chunks.
              Each has .page_content (the text) and .metadata (source file, page, etc.)

    Example:
        chunks = retrieve_chunks("What is the refund policy?", retriever)
        for chunk in chunks:
            print(chunk.page_content)
            print(chunk.metadata["source"])
    """

    print(f"\n🔍 Retrieving relevant chunks for: '{question}'")

    # .invoke() embeds the question and runs the similarity search
    relevant_chunks = retriever.invoke(question)

    print(f"\n📋 Top {len(relevant_chunks)} retrieved chunk(s):")
    print("-" * 60)

    for i, chunk in enumerate(relevant_chunks, 1):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "")
        page_info = f" (page {page})" if page != "" else ""

        print(f"\n[Chunk {i}] Source: {source}{page_info}")
        print(f"Content preview: {chunk.page_content[:200]}...")

    print("-" * 60)

    return relevant_chunks
