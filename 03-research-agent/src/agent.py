"""
Agent — Defines a ReAct agent that orchestrates the research tools.

THE ReAct LOOP: Reason → Act → Observe → Repeat
=================================================
ReAct (Reasoning + Acting) is a prompting pattern where the LLM:

  1. **Reason** — Think about what it knows and what it still needs.
     ("I need to find the methodology used in Paper X.")

  2. **Act** — Choose a tool and call it with specific input.
     (Calls search_papers("Paper X methodology"))

  3. **Observe** — Read the tool's output.
     ("The paper used a transformer-based approach with...")

  4. **Repeat** — Decide if it has enough info to answer, or if it needs
     another tool call. If done, produce a final answer.

This loop lets the agent tackle complex, multi-step research questions
that no single tool could answer alone. For example:

  User: "What gaps exist between Paper A and Paper B?"
  Agent: Reason → summarise Paper A → summarise Paper B → compare them
         → identify gaps → produce final answer

LOGGING
We enable verbose mode so learners can see every step of the reasoning
chain in the console. This is invaluable for understanding *why* the
agent chose each tool and how it builds toward an answer.
"""

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool


# ---------------------------------------------------------------------------
# Agent prompt — this is what the LLM sees at every step of the loop.
# The prompt must include placeholders for tools, tool_names, agent_scratchpad
# (the running history of actions/observations), and the user input.
# ---------------------------------------------------------------------------

_REACT_PROMPT = PromptTemplate.from_template(
    """You are an AI Research Agent that helps users understand and analyse
academic papers. You have access to the following tools:

{tools}

Tool names: {tool_names}

Use the following format:

Question: the input question you must answer
Thought: think about what you need to do next
Action: the tool to use (must be one of [{tool_names}])
Action Input: the input to pass to the tool
Observation: the result of the tool call
... (this Thought/Action/Action Input/Observation cycle can repeat)
Thought: I now have enough information to answer
Final Answer: the final answer to the original question

Important guidelines:
- Always explain your reasoning before choosing a tool.
- If the question is about a specific paper, search or summarise it first.
- If comparing papers, gather info on each before comparing.
- Cite which paper your information comes from.
- If you cannot find relevant information, say so honestly.

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
)


def create_research_agent(
    tools: list[BaseTool],
    llm,
    verbose: bool = True,
) -> AgentExecutor:
    """Create a ReAct agent with the given tools and LLM.

    Args:
        tools:   List of LangChain Tool instances (search, summarise, compare).
        llm:     A LangChain LLM/ChatModel.
        verbose: If True, log every Thought/Action/Observation to stdout.

    Returns:
        An AgentExecutor that can be invoked with a query.
    """
    # create_react_agent builds the prompting logic; AgentExecutor adds
    # the execution loop (call tool → feed observation back → repeat).
    agent = create_react_agent(llm=llm, tools=tools, prompt=_REACT_PROMPT)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        # handle_parsing_errors lets the agent recover gracefully if the
        # LLM produces output that doesn't match the expected format.
        handle_parsing_errors=True,
        # max_iterations prevents infinite loops if the agent gets stuck.
        max_iterations=10,
    )
    return executor


def run_agent(agent: AgentExecutor, query: str) -> str:
    """Run the agent on a query and return the final answer.

    Args:
        agent: An AgentExecutor created by create_research_agent.
        query: A natural-language research question.

    Returns:
        The agent's final answer as a string.
    """
    print(f"\n{'='*60}")
    print(f"🔍 Agent Query: {query}")
    print(f"{'='*60}\n")

    result = agent.invoke({"input": query})

    # AgentExecutor returns a dict; the answer is under "output"
    answer = result.get("output", str(result))

    print(f"\n{'='*60}")
    print(f"✅ Agent Answer:\n{answer}")
    print(f"{'='*60}\n")

    return answer
