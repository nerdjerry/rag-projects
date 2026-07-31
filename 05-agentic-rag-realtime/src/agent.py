"""
src/agent.py

Assembles the LangChain agent executor that ties together the LLM, all tools,
and optional conversation memory.

THE AGENT LOOP:
    Every time the agent receives a question it goes through repeated cycles:
      1. REASON  — the LLM decides what it needs and which tool (if any) to call,
                    using OpenAI's native function/tool-calling API.
      2. ACT     — the chosen tool runs with the structured arguments the LLM gave.
      3. OBSERVE — the tool's output is appended to the conversation.
      4. REPEAT  — the LLM reasons again with the new information; stops once it
                    replies without requesting another tool call.

    This is fundamentally different from standard RAG which does a single
    FAISS search every time regardless of the question type.

    This project builds the agent with create_tool_calling_agent, which relies
    on the model's native tool-calling support (OpenAI, Anthropic, and most
    current chat models implement this). It replaces the older text-based
    ReAct pattern (Thought/Action/Observation strings that LangChain had to
    parse), which was less reliable and is deprecated in current LangChain.

MEMORY:
    ConversationBufferWindowMemory(k=5) keeps the last 5 exchanges in context.
    k=5 is a pragmatic choice:
      • Enough to handle follow-up questions ("And what about MSFT?")
      • Small enough not to overflow the context window on long conversations
    Disable memory (--no-memory) for stateless single-query use cases.

VERBOSE MODE:
    verbose=True is essential for learning: you see every tool call and its
    result printed to stdout. In production set verbose=False.
"""

from typing import List, Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool


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
        llm:     An instantiated LangChain chat model that supports tool
                 calling (e.g. ChatOpenAI).
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
            # AgentExecutor's output dict has both "output" and (because of
            # return_intermediate_steps below) "intermediate_steps"; without
            # this, memory has to guess which key to store and warns about it.
            output_key="output",
        )

    # The prompt needs: a system message (persona + instructions), a slot for
    # prior turns (only used if memory is enabled), the user's input, and a
    # slot for the agent's own scratchpad (its tool calls + results this turn).
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    # create_tool_calling_agent binds the tools to the LLM via its native
    # tool-calling API — no text parsing of Thought/Action lines.
    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=mem,
        verbose=verbose,
        # handle_parsing_errors=True prevents the agent from crashing when the
        # LLM produces a malformed tool call; it retries with an error message.
        handle_parsing_errors=True,
        # max_iterations caps runaway loops — agent stops after N tool calls.
        max_iterations=8,
        # Without this, the executor's output dict never contains
        # "intermediate_steps", so callers can never see which tools were
        # actually used (see response_formatter.extract_tools_from_steps).
        return_intermediate_steps=True,
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
