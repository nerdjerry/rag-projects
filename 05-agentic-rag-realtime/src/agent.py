"""
src/agent.py

Assembles the LangChain agent executor that ties together the LLM, all tools,
and optional conversation memory.

THE ReAct LOOP (Reason + Act):
    Every time the agent receives a question it goes through repeated cycles:
      1. REASON  — "What do I need to answer this? Which tool should I call?"
      2. ACT     — Calls a tool with a specific input string.
      3. OBSERVE — Reads the tool's output.
      4. REPEAT  — Reasons again with the new information; stops when confident.

    This is fundamentally different from standard RAG which does a single
    FAISS search every time regardless of the question type.

AGENT TYPES:
    • OPENAI_FUNCTIONS (default when using GPT-3.5 / GPT-4):
        Uses OpenAI's native function-calling API.  The LLM is trained to emit
        structured JSON for function calls, so tool invocation is very reliable.
        Requires an OpenAI model that supports function calling.

    • ZERO_SHOT_REACT_DESCRIPTION (fallback):
        Works with ANY LLM (Llama, Mistral, Claude, etc.).
        The LLM reasons in plain text using a "Thought/Action/Observation" format.
        Less reliable for tool selection but model-agnostic.

MEMORY:
    ConversationBufferWindowMemory(k=5) keeps the last 5 exchanges in context.
    k=5 is a pragmatic choice:
      • Enough to handle follow-up questions ("And what about MSFT?")
      • Small enough not to overflow the context window on long conversations
    Disable memory (--no-memory) for stateless single-query use cases.

VERBOSE MODE:
    verbose=True is essential for learning: you see every Thought → Action →
    Observation cycle printed to stdout.  In production set verbose=False.
"""

from typing import List, Optional

from langchain.agents import AgentExecutor, initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import SystemMessage
from langchain.tools import Tool


# System prompt injected before every conversation.
# Specific instructions improve tool selection accuracy significantly.
_SYSTEM_PROMPT = """You are a knowledgeable assistant with access to multiple tools.
You can search internal documents, look up live data, and search the web.

When answering:
1. First consider if you need real-time data (use web_search or get_stock_data)
2. Or if the question is about internal documents (use search_knowledge_base)
3. Or both (use multiple tools)

Always cite which tools you used and where information came from.
Think step by step before deciding which tools to use."""


def create_agent(
    tools: List[Tool],
    llm,
    memory: bool = True,
    verbose: bool = True,
) -> AgentExecutor:
    """
    Build and return a LangChain AgentExecutor wired to the provided tools.

    Args:
        tools:   List of LangChain Tool objects from tool_registry.
        llm:     An instantiated LangChain LLM (e.g. ChatOpenAI).
        memory:  If True, adds a sliding-window conversation memory (k=5).
        verbose: If True, prints the full reasoning trace to stdout.

    Returns:
        A configured AgentExecutor ready to accept queries.
    """
    # --- Memory ---
    # ConversationBufferWindowMemory keeps only the last k exchanges so the
    # context window doesn't grow unboundedly during long conversations.
    mem: Optional[ConversationBufferWindowMemory] = None
    if memory:
        mem = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            return_messages=True,
        )

    # --- Determine the best agent type ---
    # OPENAI_FUNCTIONS is more reliable for tool selection because it uses
    # OpenAI's native function-calling format instead of text-based reasoning.
    # We detect whether we're talking to an OpenAI chat model by checking the
    # class name — this avoids a hard dependency on langchain_openai at this level.
    llm_class = type(llm).__name__
    is_openai_chat = "ChatOpenAI" in llm_class or "AzureChatOpenAI" in llm_class

    if is_openai_chat:
        agent_type = AgentType.OPENAI_FUNCTIONS
        # Inject the system message through agent_kwargs for OPENAI_FUNCTIONS agents.
        agent_kwargs = {
            "system_message": SystemMessage(content=_SYSTEM_PROMPT),
        }
        if mem:
            agent_kwargs["extra_prompt_messages"] = []  # memory messages prepended automatically
    else:
        # ZERO_SHOT_REACT_DESCRIPTION works with any LLM via plain-text reasoning.
        agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION
        agent_kwargs = {}

    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=agent_type,
        memory=mem,
        agent_kwargs=agent_kwargs,
        verbose=verbose,
        # handle_parsing_errors=True prevents the agent from crashing when the
        # LLM produces a malformed tool call; it retries with an error message.
        handle_parsing_errors=True,
        # max_iterations caps runaway loops — agent stops after N tool calls.
        max_iterations=8,
    )

    return agent_executor


def run_agent_query(query: str, agent: AgentExecutor) -> str:
    """
    Submit a query to the agent and return the final answer string.

    Wraps the AgentExecutor.invoke() call with error handling so the main
    loop doesn't crash on unexpected LLM failures.

    Args:
        query: The user's natural-language question.
        agent: A configured AgentExecutor from create_agent().

    Returns:
        The agent's final answer as a plain string.
    """
    try:
        result = agent.invoke({"input": query})
        # AgentExecutor returns a dict; the final answer is under "output".
        return result.get("output", str(result))
    except Exception as exc:
        return f"Agent encountered an error: {exc}"
