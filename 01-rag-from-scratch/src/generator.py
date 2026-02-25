"""
generator.py — Generate answers using an LLM grounded in retrieved context.

THE SYSTEM PROMPT PATTERN
  The key to preventing hallucination in RAG is the system prompt.
  We explicitly tell the LLM:
    "Answer the question based ONLY on the following context."
  This way the model is constrained to the information we retrieved
  from our documents rather than making things up from its training data.

SUPPORTED BACKENDS
  • OpenAI  — Requires an OPENAI_API_KEY in your .env file.
  • Ollama  — Runs locally, no API key needed.  Install from
               https://ollama.ai and pull a model (e.g., `ollama pull llama3`).
"""

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# -------------------------------------------------------------------
# Prompt template — this is the instruction sent to the LLM along
# with the retrieved context and the user's question.
# The "{context}" and "{question}" placeholders are filled in
# automatically by the RetrievalQA chain.
# -------------------------------------------------------------------
PROMPT_TEMPLATE = """Use the following pieces of context to answer the question.
If you don't know the answer based on the context, say "I don't have enough information to answer that."
Do NOT make up information that is not in the context.

Context:
{context}

Question: {question}

Answer:"""

QA_PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def _get_llm(use_ollama: bool = False, model_name: str = "llama3"):
    """Return a LangChain LLM instance for the chosen backend."""
    if use_ollama:
        # Ollama runs a local HTTP server (default http://localhost:11434).
        from langchain_community.llms import Ollama
        print(f"Using Ollama model: {model_name}")
        return Ollama(model=model_name)
    else:
        # OpenAI requires OPENAI_API_KEY to be set in the environment.
        from langchain_openai import ChatOpenAI
        print("Using OpenAI (gpt-3.5-turbo)")
        return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


def create_qa_chain(retriever, use_ollama: bool = False, model_name: str = "llama3"):
    """
    Build a RetrievalQA chain that ties together retrieval + generation.

    Args:
        retriever:   A LangChain retriever (e.g., vector_store.as_retriever()).
        use_ollama:  If True, use a local Ollama model instead of OpenAI.
        model_name:  Which Ollama model to use (ignored when use_ollama=False).

    Returns:
        A RetrievalQA chain you can call with a question.
    """
    llm = _get_llm(use_ollama=use_ollama, model_name=model_name)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" = concatenate all chunks into one prompt
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT},
    )
    return qa_chain


def ask_question(qa_chain, question: str, debug: bool = False) -> str:
    """
    Send a question through the QA chain and return the answer.

    Args:
        qa_chain:  The chain returned by create_qa_chain().
        question:  The user's natural-language question.
        debug:     If True, print the full prompt and source documents.

    Returns:
        The LLM's answer as a string.
    """
    result = qa_chain.invoke({"query": question})

    answer = result["result"]
    source_docs = result.get("source_documents", [])

    if debug:
        print("\n========== DEBUG: Source Documents ==========")
        for i, doc in enumerate(source_docs, start=1):
            print(f"\n--- Source {i} ({doc.metadata.get('source', 'unknown')}) ---")
            print(doc.page_content[:400])
        print("=============================================\n")

    return answer
