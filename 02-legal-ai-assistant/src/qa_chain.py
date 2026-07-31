"""
qa_chain.py — Retrieval-Augmented Q&A for Legal Documents

Builds a RAG Q&A chain that answers questions about a contract by:
  1. Retrieving the most relevant chunks from the FAISS index
  2. Sending those chunks + the user's question to the LLM
  3. Requiring the model to cite the specific section it's referencing

Why source citation is critical in legal Q&A:
    Unlike a general knowledge chatbot, a legal assistant's answers directly
    influence decisions with real financial and legal consequences. If a user
    asks "Can I terminate in 30 days?" and the model answers "Yes" without
    citing the source clause, the user cannot verify whether that answer is
    based on the actual contract or a hallucination. Forcing the model to
    cite sections:
      • Lets the user cross-check against the original document.
      • Makes hallucinations easier to spot (the cited section won't exist).
      • Builds appropriate trust — the user knows WHAT to verify, not just
        WHETHER to trust.

    This is fundamentally different from Q&A over, say, a technical manual,
    where a wrong answer is inconvenient. A wrong answer about a contract
    clause can result in a breach of contract, lawsuit, or financial loss.
"""

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# ---------------------------------------------------------------------------
# Custom legal Q&A prompt
# ---------------------------------------------------------------------------

# The prompt explicitly:
#   1. Restricts answers to provided context (reduces hallucination)
#   2. Requires section citations (enables user verification)
#   3. Provides a safe fallback for out-of-scope questions
#   4. Includes the "not legal advice" disclaimer in every answer
_LEGAL_QA_TEMPLATE = """You are a legal document assistant. Answer questions about the contract based ONLY on the provided context.
Always cite the specific section or clause you are referencing.
If the answer is not in the provided context, say "This information is not found in the provided contract."

Important: This is for informational purposes only, not legal advice.

Context from contract:
{context}

Question: {input}

Answer (include section references):"""

_QA_PROMPT = ChatPromptTemplate.from_template(_LEGAL_QA_TEMPLATE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_qa_chain(retriever, llm):
    """
    Construct a retrieval chain grounded in the indexed contract.

    Parameters
    ----------
    retriever : BaseRetriever — from indexer.get_retriever()
    llm       : LLM           — any LangChain-compatible chat model

    Returns
    -------
    A runnable retrieval chain — call .invoke({"input": question}) to get
    a dict with keys "input", "context", "answer".

    The chain uses a "stuff" document combination strategy — it concatenates
    retrieved chunks into a single context block. For very long contracts
    "map_reduce" or "refine" strategies may be preferable, but "stuff" is
    the most reliable for faithfully citing specific text.
    """
    combine_docs_chain = create_stuff_documents_chain(llm, _QA_PROMPT)
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return qa_chain


def ask_question(question: str, qa_chain) -> str:
    """
    Ask a natural-language question about the indexed contract.

    Parameters
    ----------
    question : str — the user's question (e.g. "What are my termination rights?")
    qa_chain       — retrieval chain built by build_qa_chain()

    Returns
    -------
    str — the model's answer with section citations

    The returned string always includes the LLM's answer. If source documents
    were returned they are appended as a "Sources" footer so users can quickly
    locate the referenced passage in the original document.
    """
    result = qa_chain.invoke({"input": question})

    answer = result.get("answer", "No answer returned.")

    # Append source chunk references if available — helps users locate the
    # exact passage that was used to generate the answer.
    source_docs = result.get("context", [])
    if source_docs:
        answer += "\n\n─── Sources (retrieved chunks) ───"
        for i, doc in enumerate(source_docs, start=1):
            # Show the first 150 chars of each source chunk as a reference hint
            snippet = doc.page_content[:150].replace("\n", " ").strip()
            answer += f"\n  [{i}] ...{snippet}..."

    return answer
