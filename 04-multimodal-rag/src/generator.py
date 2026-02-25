"""
generator.py — Generate answers from multimodal context.

Supports both OpenAI (cloud) and Ollama (local, free) backends.
The chain receives retrieved context from text, image-captions, and
table-descriptions, and produces a grounded answer.
"""

import logging

from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# System prompt that tells the LLM how to use multimodal context.
SYSTEM_TEMPLATE = """\
You are a helpful assistant that answers questions based on the provided \
context.  The context may include three types of evidence:

  [TEXT]  — prose passages extracted from the document.
  [IMAGE] — descriptions of charts, diagrams, or photos.
  [TABLE] — summaries of data tables.

Rules:
1. Base your answer ONLY on the provided context.
2. When referencing a chart or image, say so explicitly.
3. When citing numbers from a table, mention the table source.
4. If the context does not contain enough information, say so honestly.
5. Be concise but thorough.
"""

HUMAN_TEMPLATE = """\
Context:
{context}

Question: {question}

Answer:"""


def get_llm(use_ollama: bool = False, model: str = "gpt-4o"):
    """
    Create a LangChain chat model.

    Parameters
    ----------
    use_ollama : bool
        If True, use a local Ollama model (free, no API key needed).
    model : str
        Model name — e.g. "gpt-4o" for OpenAI or "llama3" for Ollama.

    Returns
    -------
    A LangChain chat model instance.
    """
    if use_ollama:
        from langchain_community.chat_models import ChatOllama
        logger.info("Using Ollama model: %s", model)
        return ChatOllama(model=model)
    else:
        from langchain_openai import ChatOpenAI
        logger.info("Using OpenAI model: %s", model)
        return ChatOpenAI(model=model, temperature=0)


def create_multimodal_qa_chain(llm):
    """
    Build a LangChain chain that takes {context, question} and returns
    a string answer.

    The chain is intentionally simple:
      prompt → LLM → parse output as string

    Parameters
    ----------
    llm : BaseChatModel
        A LangChain chat model (OpenAI or Ollama).

    Returns
    -------
    A runnable chain with .invoke({"context": ..., "question": ...}).
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE),
        ("human", HUMAN_TEMPLATE),
    ])

    chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info("Multimodal QA chain created.")
    return chain


def generate_answer(chain, query: str, context: str) -> str:
    """
    Run the QA chain with the given query and multimodal context.

    Parameters
    ----------
    chain : Runnable
        The chain returned by create_multimodal_qa_chain().
    query : str
        The user's question.
    context : str
        Formatted context string from multi_retriever.format_context().

    Returns
    -------
    str — the generated answer.
    """
    logger.info("Generating answer for: %s", query[:80])

    answer = chain.invoke({"context": context, "question": query})
    return answer
