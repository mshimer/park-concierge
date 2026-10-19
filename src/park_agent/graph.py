"""The agent loop: a two-node LangGraph cycle between the model and its tools."""

import logging
from dataclasses import dataclass
from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph

# The real reducer: it merges by message id and coerces raw tuples to Message objects.
# (`from operator import add` is just list concatenation wearing the same name - it appends, but
# duplicates any message that gets re-sent and never coerces.)
from langgraph.graph.message import add_messages
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.server.mcpserver import MCPServer

from .mcp_tools import ToolTrace, load_mcp_tools

log = logging.getLogger(__name__)

# A tool-calling loop with no ceiling will happily spin until the context window bursts. Eight
# round trips is far more than any of these scenarios needs.
MAX_STEPS = 8


# keep the Agent in the bounds of it's purpose with concise instructions ;)

SYSTEM_PROMPT = """You are a National Park trip concierge. You help park-pass holders decide where
to go, using only the tools provided.

Rules you must follow:
1. If the question refers to a customer, call customer_lookup first. Never guess a customer's
   details, and if the lookup returns null, say plainly that no such customer exists.
2. Recommend only parks returned by the parks_* tools. Never name a park the tools did not return,
   and base every factual claim about a park - terrain, climate, elevation, activities, accessibility
   - on what a tool actually returned. Do not fill gaps from your own knowledge.
3. When asked where someone should go, offer two or three options rather than one, and say briefly
   why each fits. Give a single answer only when the question is about one specific park.
4. Respect the customer's stated accessibility needs and any max_elevation_ft limit. When the
   customer has an elevation limit, pass it as the max_elevation_ft argument to every parks_search
   and parks_similar_to call - that filters the results instead of merely hinting at it in the
   query text. Do not offer a park that fails a stated constraint; if you mention one anyway, say
   plainly why it does not fit.
5. Check parks_alerts for any park you recommend, and tell the visitor about anything that would
   affect their trip - closures, road status, hazards, reservation requirements.
6. If the question has nothing to do with national parks or a customer's trip, say that it is
   outside what you can help with. Do not answer it from general knowledge.

Be concise and concrete. Name parks with their state. Cite what the tools told you.
"""

# NOTE : Annotated is from Python's typing module. It attaches extra metadata to a type without changing the type itself.
# in this case it's saying add messages vs overwriting each time and thus losing context
# Annotated does this : Sequence[BaseMessage].__metadata__ = (add_messages,)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    steps: int


@dataclass
class AgentRun:
    """One complete turn: the answer, what it called, and what it saw."""

    answer: str
    trace: ToolTrace
    messages: list[BaseMessage]
    steps: int

    @property
    def hit_step_limit(self) -> bool:
        """True when the loop was cut off rather than finishing on its own.

        Read from the graph's own counter, not from len(trace.calls): the model gets MAX_STEPS
        turns, and the last one is spent answering rather than calling a tool, so the call count
        tops out one below the limit.
        """
        return self.steps >= MAX_STEPS


def build_graph(
    model: BaseChatModel, tools: list[BaseTool], system_prompt: str = SYSTEM_PROMPT
):
    """Compile the llm <-> tools cycle."""
    llm = model.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    async def call_llm(state: AgentState) -> dict[str, Any]:
        messages = [SystemMessage(content=system_prompt), *state["messages"]]
        return {"messages": [await llm.ainvoke(messages)], "steps": state["steps"] + 1}

    async def call_tools(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        results: list[BaseMessage] = []
        for call in getattr(last, "tool_calls", []):
            tool = tools_by_name.get(call["name"])
            if tool is None:
                content = f"No such tool {call['name']!r}. Choose from: {', '.join(tools_by_name)}"
            else:
                log.info("tool %s %s", call["name"], call["args"])
                content = await tool.ainvoke(call["args"])
            results.append(
                ToolMessage(
                    tool_call_id=call["id"], name=call["name"], content=str(content)
                )
            )
        return {"messages": results}

    def should_continue(state: AgentState) -> bool:
        last = state["messages"][-1]
        if state["steps"] >= MAX_STEPS:
            # Stop looping, but leave a trace of why - a silently truncated answer is worse than
            # one that admits it ran out of room.
            log.warning("hit MAX_STEPS=%s; stopping the loop", MAX_STEPS)
            return False
        return bool(getattr(last, "tool_calls", None))

    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("tools", call_tools)
    graph.add_conditional_edges("llm", should_continue, {True: "tools", False: END})
    graph.add_edge("tools", "llm")
    graph.set_entry_point("llm")
    return graph.compile()


@traceable(
    name="park_concierge_turn", run_type="chain", tags=["park-concierge"]
)  # LangSmith
async def run_agent(
    question: str,
    *,
    server: MCPServer | StdioServerParameters | str,
    model: BaseChatModel,
    system_prompt: str = SYSTEM_PROMPT,
) -> AgentRun:
    """Answer one question against an MCP server.

    `server` is explicit and has no default on purpose. An in-process MCPServer carries whatever
    ServerContext the caller built (which is how the evals reach their testcontainer); a
    StdioServerParameters spawns `uv run park-mcp`, which reads its own .env. If this defaulted to
    stdio, the eval suite would quietly test the developer's database and still pass.
    """

    trace = ToolTrace()

    # Everything below (graph, llm calls, tool calls) nests under this run in LangSmith. Metadata
    # makes runs filterable in the portal by model and by transport.

    if (
        rt := get_current_run_tree()
    ):  # LangSmith - adds some custom metadata for filtering
        rt.metadata.update(
            {
                "agent_model": getattr(model, "model_name", type(model).__name__),
                "transport": "inprocess" if isinstance(server, MCPServer) else "stdio",
            }
        )

    async with Client(
        server
    ) as client:  # Client is mcp client; __aexit__ runs to close the connection
        tools = await load_mcp_tools(client, trace)
        agent = build_graph(model, tools, system_prompt)
        state = await agent.ainvoke(
            {"messages": [HumanMessage(content=question)], "steps": 0}
        )

    final = state["messages"][-1]

    # NOTE: reasoning models return `content` as a list of blocks (an encrypted reasoning block
    # plus the text block), not a string. `.text` joins just the text blocks.
    # final.content could look like this:
    # [
    #   {"type": "reasoning", "encrypted_content": "gAAAA..."},
    #   {"type": "text", "text": "Yosemite (CA) has a rockfall alert..."},
    # ]
    answer = final.text if isinstance(final, AIMessage) else str(final.content)
    return AgentRun(
        answer=str(answer),
        trace=trace,
        messages=list(state["messages"]),
        steps=state["steps"],
    )
