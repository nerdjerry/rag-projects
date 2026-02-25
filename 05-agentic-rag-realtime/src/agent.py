"""
Agent — The Decision-Making Brain
===================================
This module creates a LangChain agent that:
  1. Reads the user's question
  2. Decides which tool(s) to call  (based on tool descriptions)
  3. Calls the tool(s) and reads the results
  4. Synthesizes a final answer

Agent types explained:
  - OPENAI_FUNCTIONS:  Best with OpenAI models (gpt-3.5 / gpt-4).  Uses
    OpenAI's native function-calling feature for reliable tool selection.
  - ZERO_SHOT_REACT_DESCRIPTION:  Works with ANY LLM (including Ollama).
    Uses a ReAct-style prompt where the LLM reasons step-by-step:
      Thought → Action → Observation → ... → Final Answer

Verbose mode:
  When verbose=True, the agent prints its full thought process at each step.
  This is invaluable for debugging and understanding HOW the agent decides
  which tools to use.
"""

from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory


def create_agent(tools: list, llm, verbose: bool = True):
    """
    Create a LangChain agent with tools and conversational memory.

    Args:
        tools:    List of LangChain Tool objects from the tool registry.
        llm:      A LangChain LLM or ChatModel instance.
        verbose:  If True, print the agent's reasoning at every step.
                  Highly recommended during development!

    Returns:
        An initialized LangChain agent ready to answer queries.
    """
    # --- Conversational Memory ---
    # This lets the agent remember previous turns so users can ask
    # follow-up questions like "What about Microsoft?" after asking
    # about Apple's stock price.
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    # --- Choose agent type based on the LLM ---
    # OpenAI models support function calling natively → more reliable.
    # Other models (Ollama, Hugging Face) use the ReAct prompt pattern.
    llm_class_name = type(llm).__name__

    if "ChatOpenAI" in llm_class_name:
        agent_type = AgentType.OPENAI_FUNCTIONS
        print(f"  Using agent type: OPENAI_FUNCTIONS (optimized for OpenAI)")
    else:
        agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION
        print(f"  Using agent type: ZERO_SHOT_REACT_DESCRIPTION (universal)")

    # --- Initialize the Agent ---
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=agent_type,
        verbose=verbose,  # Print thought process at each step
        memory=memory,
        handle_parsing_errors=True,  # Gracefully handle malformed LLM output
        max_iterations=5,  # Prevent infinite loops
        early_stopping_method="generate",  # Let LLM generate a final answer if stuck
    )

    if verbose:
        print("  🧠 Agent created with verbose mode ON — you'll see its reasoning.\n")

    return agent


def run_agent(agent, query: str) -> dict:
    """
    Run a query through the agent and capture the result.

    This function wraps the agent invocation to provide consistent output
    and error handling.

    Args:
        agent:  The initialized LangChain agent.
        query:  The user's natural-language question.

    Returns:
        A dict with:
          - "output": The agent's final answer (string)
          - "error":  Error message if something went wrong, else None
    """
    try:
        # --- Run the agent ---
        # The agent will:
        #   1. Read the query
        #   2. Decide which tool(s) to use (printed if verbose=True)
        #   3. Call tool(s) and read results
        #   4. Generate a final answer
        result = agent.invoke({"input": query})

        return {
            "output": result.get("output", "No response generated."),
            "error": None,
        }

    except Exception as e:
        return {
            "output": None,
            "error": f"Agent error: {str(e)}",
        }
