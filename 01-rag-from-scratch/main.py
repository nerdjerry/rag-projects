"""
main.py — Entry point for the RAG From Scratch pipeline.

This script ties every step together:
  1. Load documents   (document_loader.py)
  2. Chunk them       (chunker.py)
  3. Embed chunks     (embedder.py)
  4. Store in FAISS   (vector_store.py)
  5. Retrieve context (retriever.py)
  6. Generate answer  (generator.py)

Run:
    python main.py
"""

import os
import sys

from dotenv import load_dotenv

# Make sure Python can find the src/ package regardless of where the
# script is invoked from.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.document_loader import load_documents
from src.chunker import chunk_documents
from src.embedder import get_embeddings, show_embedding_example
from src.vector_store import create_vector_store, load_vector_store
from src.retriever import retrieve_relevant_chunks
from src.generator import create_qa_chain, ask_question


def main():
    # ------------------------------------------------------------------
    # 0. Load environment variables from .env
    # ------------------------------------------------------------------
    load_dotenv()

    chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
    top_k = int(os.getenv("TOP_K", "3"))
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3")

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    index_path = os.path.join(os.path.dirname(__file__), "faiss_index")

    print("=" * 60)
    print("  RAG From Scratch — Step-by-step Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load documents from the data/ folder
    # ------------------------------------------------------------------
    print("\n📄 Step 1: Loading documents ...")
    documents = load_documents(data_dir)

    if not documents:
        print(
            "\n⚠️  No documents found in the data/ folder.\n"
            "   Add .pdf, .txt, or .docx files to data/ (or data/sample_docs/) "
            "and try again."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Chunk documents
    # ------------------------------------------------------------------
    print("\n✂️  Step 2: Chunking documents ...")
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # ------------------------------------------------------------------
    # 3. Create embeddings
    # ------------------------------------------------------------------
    print("\n🔢 Step 3: Loading embedding model ...")
    embeddings = get_embeddings()
    # Show a quick example so learners can see what embeddings look like.
    show_embedding_example(embeddings, "What is Retrieval-Augmented Generation?")

    # ------------------------------------------------------------------
    # 4. Build (or load) the FAISS vector store
    # ------------------------------------------------------------------
    print("\n💾 Step 4: Building vector store ...")
    if os.path.exists(index_path):
        print(f"   Found existing index at '{index_path}' — loading from disk.")
        vector_store = load_vector_store(embeddings, index_path=index_path)
    else:
        vector_store = create_vector_store(chunks, embeddings, index_path=index_path)

    # ------------------------------------------------------------------
    # 5 & 6. Interactive Q&A loop
    # ------------------------------------------------------------------
    print("\n🤖 Step 5 & 6: Retrieve + Generate")
    print("-" * 60)

    # Build the QA chain once; reuse for every question.
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    qa_chain = create_qa_chain(
        retriever,
        use_ollama=use_ollama,
        model_name=ollama_model,
    )

    # Default sample question if the user just presses Enter.
    sample_question = "What are the main topics covered in these documents?"

    print(
        "\nAsk a question about your documents (or press Enter for a sample "
        "question). Type 'quit' to exit.\n"
    )

    while True:
        try:
            question = input("❓ Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not question:
            question = sample_question
            print(f"   Using sample question: \"{question}\"")

        # Step 5: Retrieve relevant chunks (printed inside the function).
        retrieve_relevant_chunks(vector_store, question, k=top_k)

        # Step 6: Generate an answer.
        print("💬 Generating answer ...\n")
        answer = ask_question(qa_chain, question, debug=True)

        print(f"\n✅ Answer:\n{answer}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()
