"""
src/agent.py
------------
Wires together the tools and LLM into a LangChain tool-calling agent.

WHAT IS THE AGENT LOOP?
------------------------
This agent uses the model's native tool-calling ability (the same mechanism
behind OpenAI "function calling"): on each turn the LLM either emits one or
more tool calls or a final answer. The loop looks like:

  1. The LLM sees the conversation so far and the list of available tools.
  2. It either calls a tool (with structured, typed arguments) or answers.
  3. If it called a tool, AgentExecutor runs it and appends the result as a
     "tool" message, then loops back to step 1.
  4. Once the LLM responds without any tool calls, that's the final answer.

Example:
  LLM: [tool call] search_papers("transformer self-attention mechanism")
  Tool result: [Result 1] Paper: "Attention Is All You Need" ...
  LLM: [tool call] summarize_paper("Attention Is All You Need")
  Tool result: Title: Attention Is All You Need ...
  LLM: "The paper 'Attention Is All You Need' introduced ..." (final answer)

This is the modern replacement for the older text-based ReAct pattern
(Thought/Action/Observation), which relied on the LLM writing free-text that
LangChain then had to parse. Native tool calling is more reliable because the
model returns structured arguments directly instead of text that must be
parsed and can be malformed.

HOW THE AGENT SEES THE TOOLS
------------------------------
Each tool's name, description, and argument schema are sent to the LLM as
part of the tool-calling API request. The agent never sees function
signatures or source code — only what the tool object exposes. This is why
precise tool descriptions are critical: they are the agent's entire API docs.

THE INPUT/OUTPUT CONTRACT
--------------------------
  Input  - a plain string (the search query).
  Output - a plain string that the agent reads as a tool result.

LangChain enforces this contract: whatever your func returns is converted to
str and appended to the conversation as the tool's result message.

WHY TOOL DESCRIPTIONS MUST BE PRECISE
---------------------------------------
The agent is stateless — it has no memory of tool internals. If the
description says "search papers" without clarifying the expected input format,
the agent might pass a JSON object or a question instead of a keyword query,
producing poor results. Explicit examples in the description (like "Input: a
search query string") dramatically improve reliability.

THE DIFFERENCE BETWEEN AN AGENT AND A SIMPLE LLM CALL
--------------------------------------------------------
A simple LLM call is a single prompt → single response. The LLM cannot fetch
new information mid-response. An agent can:
  - Decide which tool to call based on intermediate results
  - Retry with a different query if the first search returns nothing
  - Chain multiple tool calls (search → summarize → compare)
  - Stop early if the first observation already answers the question
"""

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

from src.tools.compare_tool import create_compare_tool
from src.tools.search_tool import create_search_tool
from src.tools.summary_tool import create_summary_tool

# System prompt injected as the agent's persona and behavioural guidelines.
_AGENT_SYSTEM_PROMPT = """You are an AI research assistant. You have access to a collection of research papers.
Use the available tools to answer questions about the research literature.
Always cite your sources by mentioning which paper a piece of information comes from.
Think step by step about which tools to use."""


def create_research_agent(
    vector_store: FAISS,
    paper_metadata: list,
    llm,
) -> AgentExecutor:
    """Build and return a fully configured ReAct research agent.

    Parameters
    ----------
    vector_store : FAISS
        Populated FAISS index (from paper_indexer.index_papers).
    paper_metadata : list[PaperMetadata]
        List of parsed paper metadata objects.
    llm :
        Any LangChain chat model that supports tool calling (e.g., ChatOpenAI).

    Returns
    -------
    AgentExecutor
        The runnable agent. Call agent.invoke({"input": query}) to use it.
    """
    # Build a title → metadata dict for the summary and compare tools
    paper_metadata_dict = {pm.title: pm for pm in paper_metadata}

    # Instantiate each tool
    search_tool = create_search_tool(vector_store)
    summary_tool = create_summary_tool(paper_metadata_dict, llm)
    compare_tool = create_compare_tool(paper_metadata_dict, llm)

    tools = [search_tool, summary_tool, compare_tool]

    # The prompt needs three things: a system message (persona + instructions),
    # a slot for the user's input, and a slot for the agent's own scratchpad
    # (the running history of tool calls + results within this turn).
    prompt = ChatPromptTemplate.from_messages([
        ("system", _AGENT_SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # create_tool_calling_agent binds the tools to the LLM via its native
    # function/tool-calling API — no text parsing of Thought/Action lines.
    agent = create_tool_calling_agent(llm, tools, prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,                # print each tool call/result to stdout
        handle_parsing_errors=True,  # recover gracefully from malformed tool calls
        max_iterations=8,            # safety cap to prevent infinite loops
        return_intermediate_steps=True,  # expose which tools were called + their results
    )

    return executor


def run_agent(query: str, agent: AgentExecutor) -> str:
    """Run a single query through the research agent.

    Parameters
    ----------
    query : str
        The user's question or instruction.
    agent : AgentExecutor
        The agent built by :func:`create_research_agent`.

    Returns
    -------
    str
        The agent's final answer.
    """
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}\n")

    result = agent.invoke({"input": query})["output"]

    print(f"\n{'='*60}")
    print("Final Answer:")
    print(result)
    print(f"{'='*60}\n")

    return result
