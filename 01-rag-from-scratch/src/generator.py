# src/generator.py
#
# STEP 6 OF THE RAG PIPELINE: GENERATING THE ANSWER WITH AN LLM
#
# WHAT DOES RetrievalQA DO?
# --------------------------
# RetrievalQA is a LangChain "chain" that combines two things:
#   1. A retriever (which fetches relevant chunks from FAISS)
#   2. An LLM (which reads those chunks and generates an answer)
#
# It handles the "stuffing" step: it takes the retrieved Document objects,
# extracts their page_content, concatenates them into a {context} block,
# and injects that into our prompt template before calling the LLM.
#
# WHY THE "ONLY USE CONTEXT" INSTRUCTION PREVENTS HALLUCINATION:
# ---------------------------------------------------------------
# LLMs are trained on massive datasets and have general knowledge baked in.
# Without explicit instructions, an LLM might answer from its training data
# instead of your documents — which defeats the entire purpose of RAG.
#
# The system instruction "answer ONLY based on the following context" tells
# the LLM to restrict itself to what we provide. The fallback phrase
# "I don't know based on the provided documents" prevents the LLM from
# making things up when the answer truly isn't in the documents.
#
# This is the most important prompt engineering technique in RAG systems.
#
# WHAT IS THE SYSTEM PROMPT PATTERN?
# ------------------------------------
# A "prompt template" is a string with placeholder variables (like {context}
# and {question}) that get filled in at runtime. This lets us:
#   - Set the LLM's behavior with clear instructions at the top
#   - Inject the retrieved context dynamically for each query
#   - Ask the user's question at the end
#
# The resulting filled-in prompt is what actually gets sent to the LLM API.
# With debug=True, you can print this full prompt to see exactly what the LLM receives.

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# The prompt template instructs the LLM to stay grounded in the provided context.
# {context} will be replaced by the retrieved chunks (as a single text block).
# {question} will be replaced by the user's question.
RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question based ONLY on the following context.
If the answer is not in the context, say "I don't know based on the provided documents."
Do not use your general knowledge.

Context:
{context}

Question: {question}

Answer:"""


def build_qa_chain(
    retriever,
    model_name: str = "gpt-3.5-turbo",
    debug: bool = False,
):
    """
    Build a RetrievalQA chain that combines document retrieval with LLM generation.

    This is the final assembly step of the RAG pipeline:
        User question
            → retriever fetches top-k relevant chunks from FAISS
            → chunks are injected into the prompt template as {context}
            → LLM reads context + question and generates a grounded answer

    Args:
        retriever:          A LangChain retriever (from retriever.py).
        model_name (str):   LLM to use. Options:
                              - "gpt-3.5-turbo"   (OpenAI, requires OPENAI_API_KEY)
                              - "gpt-4"            (OpenAI, more powerful, costs more)
                              - "ollama/llama3"    (local Ollama, no API key needed)
                              - "ollama/mistral"   (local Ollama, no API key needed)
        debug (bool):       If True, prints the full prompt sent to the LLM.
                            Useful for understanding what the LLM actually receives.

    Returns:
        RetrievalQA: A runnable chain. Call chain.invoke({"query": "your question"})
                     to get an answer dict with keys "query", "result", "source_documents".

    Example:
        chain = build_qa_chain(retriever, model_name="gpt-3.5-turbo", debug=True)
        result = chain.invoke({"query": "What is the refund policy?"})
        print(result["result"])
    """

    print(f"\n🤖 Building QA chain with model: '{model_name}'")

    # -------------------------------------------------------------------------
    # SELECT THE LLM BASED ON model_name
    # -------------------------------------------------------------------------
    if model_name.startswith("ollama/"):
        # Ollama runs LLMs locally on your machine — no API key, no cost.
        # Install Ollama from https://ollama.com and pull a model:
        #   ollama pull llama3
        #   ollama pull mistral
        #
        # The model_name format is "ollama/<model>" e.g. "ollama/llama3"
        import os
        from langchain_community.llms import Ollama

        # Extract the model tag after the "ollama/" prefix
        ollama_model = model_name.split("/", 1)[1]
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        print(f"   Using local Ollama model '{ollama_model}' at {base_url}")
        llm = Ollama(model=ollama_model, base_url=base_url)

    else:
        # OpenAI models (gpt-3.5-turbo, gpt-4, gpt-4o, etc.)
        # Requires OPENAI_API_KEY to be set in your .env file.
        #
        # temperature=0 means "deterministic" — the LLM always picks the highest
        # probability token. For Q&A this is ideal; you want consistent, factual
        # answers rather than creative variation.
        llm = ChatOpenAI(
            model_name=model_name,
            temperature=0,  # 0 = deterministic/factual, 1 = more creative/varied
        )
        print(f"   Using OpenAI model '{model_name}' (ensure OPENAI_API_KEY is set)")

    # -------------------------------------------------------------------------
    # BUILD THE PROMPT TEMPLATE
    # -------------------------------------------------------------------------
    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"],  # placeholders to fill at runtime
    )

    # If debug mode is on, show the template so learners can see the structure
    if debug:
        print("\n🐛 DEBUG: Prompt template being used:")
        print("-" * 60)
        print(RAG_PROMPT_TEMPLATE)
        print("-" * 60)

    # -------------------------------------------------------------------------
    # ASSEMBLE THE RetrievalQA CHAIN
    # -------------------------------------------------------------------------
    # chain_type="stuff" means: take all retrieved chunks, "stuff" them all into
    # the context at once. This works well for small k values (k=3 to k=5).
    #
    # Other chain_type options:
    #   "map_reduce"   — summarize each chunk separately, then combine (handles many chunks)
    #   "refine"       — iteratively refine the answer chunk by chunk (slower but thorough)
    #   "map_rerank"   — score each chunk separately and pick the best answer
    #
    # For most use cases with k<=5, "stuff" is the simplest and most effective.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,  # include source docs in the result dict
        chain_type_kwargs={
            "prompt": prompt,
            "verbose": debug,  # if debug=True, LangChain will print internal chain steps
        },
    )

    print(f"✅ QA chain ready")
    return qa_chain
