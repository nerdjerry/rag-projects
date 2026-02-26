"""
src/response_formatter.py

Formats the agent's raw output into a structured, readable display.

WHY FORMAT RESPONSES?
    • Transparency: users should know whether an answer came from live data
      (could change in minutes) or stored documents (could be months old).
    • Trust: showing which tools were used lets users verify accuracy.
    • Debuggability: the reasoning trace (Thought → Action → Observation) is
      an audit trail that reveals HOW the agent reached its conclusion.

The box-drawing characters used here render well in any Unicode terminal.
"""

from typing import List, Optional


# Width of the output box in characters.
_BOX_WIDTH = 56


def format_response(
    answer: str,
    tools_used: List[str],
    agent_steps: Optional[list] = None,
) -> str:
    """
    Render the agent's answer inside a bordered box with a tools-used footer.

    Args:
        answer:      The final answer string from the agent.
        tools_used:  List of tool names that were called (e.g. ["get_stock_data"]).
        agent_steps: Optional raw intermediate steps from AgentExecutor for the
                     full reasoning trace footer.

    Returns:
        A multi-line formatted string ready to print to stdout.
    """
    lines: List[str] = []

    # ── Answer box ────────────────────────────────────────────────────────────
    lines.append("╔" + "═" * _BOX_WIDTH + "╗")
    lines.append("║  ANSWER" + " " * (_BOX_WIDTH - 7) + "║")
    lines.append("╚" + "═" * _BOX_WIDTH + "╝")
    lines.append(answer)
    lines.append("")

    # ── Tools / sources footer ─────────────────────────────────────────────
    tools_str = ", ".join(tools_used) if tools_used else "none"
    lines.append("┌" + "─" * _BOX_WIDTH + "┐")

    # Truncate tool list if it overflows the box width.
    tools_line = f" Tools Used: {tools_str}"
    if len(tools_line) > _BOX_WIDTH - 1:
        tools_line = tools_line[: _BOX_WIDTH - 4] + "…"
    lines.append("│" + tools_line.ljust(_BOX_WIDTH) + "│")
    lines.append("└" + "─" * _BOX_WIDTH + "┘")

    return "\n".join(lines)


def extract_tools_from_steps(agent_steps: list) -> List[str]:
    """
    Parse LangChain AgentExecutor intermediate steps to extract tool names.

    LangChain returns intermediate_steps as a list of (AgentAction, observation)
    tuples.  Each AgentAction has a `tool` attribute with the tool's name.

    Args:
        agent_steps: The `intermediate_steps` value from AgentExecutor output.

    Returns:
        Deduplicated list of tool names that were called, in call order.
    """
    seen = set()
    tools: List[str] = []

    for step in agent_steps or []:
        try:
            # Each step is a (AgentAction, str) tuple.
            action = step[0]
            tool_name = getattr(action, "tool", None)
            if tool_name and tool_name not in seen:
                tools.append(tool_name)
                seen.add(tool_name)
        except (IndexError, TypeError):
            # Malformed step — skip silently.
            continue

    return tools


def format_agent_trace(agent_steps: list) -> str:
    """
    Render the agent's full Thought → Action → Observation trace as text.

    This is the "reasoning audit trail": every decision the agent made is
    visible here.  Useful for debugging unexpected answers and for teaching
    users how the agent works.

    Args:
        agent_steps: The `intermediate_steps` list from AgentExecutor output.

    Returns:
        A formatted multi-line string showing each reasoning step.
    """
    if not agent_steps:
        return "(No intermediate steps recorded)"

    lines: List[str] = ["── Agent Reasoning Trace ──"]

    for i, step in enumerate(agent_steps, start=1):
        try:
            action, observation = step[0], step[1]
            tool_name = getattr(action, "tool", "unknown_tool")
            tool_input = getattr(action, "tool_input", "")
            log = getattr(action, "log", "").strip()

            lines.append(f"\nStep {i}:")

            # The `log` field contains the model's "Thought:" text for ReAct agents.
            if log:
                # Show only the Thought portion (first line) to keep it concise.
                thought_line = log.split("\n")[0]
                lines.append(f"  Thought   : {thought_line}")

            lines.append(f"  Action    : {tool_name}({tool_input!r})")
            # Truncate very long observations for readability.
            obs_str = str(observation)
            if len(obs_str) > 300:
                obs_str = obs_str[:300] + "…"
            lines.append(f"  Observation: {obs_str}")

        except Exception:
            lines.append(f"\nStep {i}: (could not parse step)")

    return "\n".join(lines)
