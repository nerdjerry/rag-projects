"""
qa_chain.py — RAG-based question-answering grounded in document chunks.

WHY SOURCE CITATION IS CRITICAL IN LEGAL CONTEXT:
    In everyday chatbots, a wrong answer is annoying. In legal contexts, a wrong
    answer can cost millions of dollars or land someone in court.

    By grounding every answer in specific document chunks and citing the source
    section, we:
    1. Let the user VERIFY the answer against the original contract.
    2. Make it clear WHICH section supports the answer.
    3. Reduce hallucination — the LLM can only reference text it was given.
    4. Build trust — lawyers won't use a tool that makes unsupported claims.

    Example:
        Question: "Can the client terminate early?"
        Bad answer: "Yes, the client can terminate at any time."
        Good answer: "Based on Section 8.2: 'Client may terminate this Agreement
            upon sixty (60) days written notice.' So yes, but 60 days' notice
            is required."
"""

import os
from typing import Any, Tuple

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS


# ---------------------------------------------------------------------------
# The QA prompt instructs the LLM to cite sources and stay grounded.
# ---------------------------------------------------------------------------
LEGAL_QA_PROMPT_TEMPLATE = """You are a legal document assistant. Answer the
question below based ONLY on the provided context from the contract.

RULES:
1. Only use information from the provided context. Do NOT make up information.
2. Cite the source for every claim using the format: "Based on Section X.Y: ..."
   or "According to [source document], page N: ..."
3. If the context does not contain enough information to answer, say:
   "I cannot find sufficient information in the document to answer this question."
4. Use plain English — explain legal terms if you reference them.
5. If the answer involves risk or obligation, highlight it clearly.

CONTEXT FROM THE CONTRACT:
{context}

QUESTION: {question}

ANSWER (with source citations):"""

LEGAL_QA_PROMPT = PromptTemplate(
    template=LEGAL_QA_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def create_legal_qa_chain(vector_store: FAISS, llm: Any) -> RetrievalQA:
    """Create a RAG question-answering chain grounded in the contract's text.

    The chain works in three steps:
    1. RETRIEVE: Find the top-k most relevant chunks from the FAISS index.
    2. AUGMENT:  Insert those chunks into the prompt as context.
    3. GENERATE: The LLM answers using only the provided context.

    Args:
        vector_store: A FAISS vector store containing the indexed document chunks.
        llm:          A LangChain LLM or ChatModel instance.

    Returns:
        A RetrievalQA chain ready to answer questions.
    """
    top_k = int(os.environ.get("TOP_K", 5))

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" = put all retrieved docs into one prompt.
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": LEGAL_QA_PROMPT},
    )

    return qa_chain


def ask_legal_question(chain: RetrievalQA, question: str) -> Tuple[str, list]:
    """Ask a question about the legal document and get a cited answer.

    Args:
        chain:    The QA chain from create_legal_qa_chain().
        question: A natural-language question about the contract.

    Returns:
        A tuple of (answer_text, source_documents).

    Example:
        answer, sources = ask_legal_question(chain, "What is the notice period?")
        print(answer)
        # "Based on Section 8.2, the notice period for termination is 60 days..."
        for src in sources:
            print(f"  Source: {src.metadata}")
    """
    result = chain.invoke({"query": question})

    answer = result.get("result", "No answer generated.")
    source_docs = result.get("source_documents", [])

    return answer, source_docs


def format_answer_with_sources(answer: str, source_docs: list) -> str:
    """Format the answer with source citations for display.

    This helper adds a "Sources" section below the answer so users can
    trace every claim back to the original document.
    """
    lines = [answer, "", "— Sources —"]

    for i, doc in enumerate(source_docs, start=1):
        meta = doc.metadata
        source = meta.get("source", "Unknown")
        page = meta.get("page", "")
        heading = meta.get("section_heading", "")
        label = meta.get("section_label", "")

        location = source
        if page:
            location += f", page {page}"
        if heading:
            location += f" ({heading})"
        elif label:
            location += f" [{label}]"

        preview = doc.page_content[:150].replace("\n", " ")
        lines.append(f"  [{i}] {location}")
        lines.append(f"      \"{preview}...\"")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Legal QA Chain — requires a FAISS vector store and an LLM.")
    print()
    print("Example usage:")
    print("  from src.indexer import load_index")
    print("  from langchain_openai import ChatOpenAI")
    print("  vs = load_index('faiss_index')")
    print("  llm = ChatOpenAI(model='gpt-4o-mini')")
    print("  chain = create_legal_qa_chain(vs, llm)")
    print("  answer, sources = ask_legal_question(chain, 'What is the term?')")
    print("  print(format_answer_with_sources(answer, sources))")
    print()
    print("Sample questions to try:")
    print("  - What is the notice period for termination?")
    print("  - Who owns the intellectual property created under this contract?")
    print("  - What happens if one party breaches the agreement?")
    print("  - Is there a limitation on liability? What is the cap?")
    print("  - Which state's laws govern this agreement?")
