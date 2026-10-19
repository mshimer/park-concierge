"""The agent loop, driven by a scripted model so it costs nothing and never flakes."""

import pytest
from langchain_core.messages import AIMessage
from support import ScriptedChatModel, tool_call

from park_agent.graph import MAX_STEPS, run_agent
from park_mcp.context import ServerContext
from park_mcp.server import build_server

pytestmark = pytest.mark.anyio

DONE = AIMessage(content="White Sands (NM) and Everglades (FL) both suit a low-elevation trip.")


async def test_loop_calls_a_tool_then_answers(ctx: ServerContext) -> None:
    model = ScriptedChatModel(script=[tool_call("customer_lookup", {"customer_id": 1042}), DONE])

    run = await run_agent("where for 1042?", server=build_server(None, ctx), model=model)

    assert run.trace.names == ["customer_lookup"]
    assert run.trace.calls[0][1] == {"customer_id": 1042}
    assert run.answer == DONE.content
    assert not run.hit_step_limit
    # Human, AI(tool_call), Tool, AI(final)
    assert len(run.messages) == 4


async def test_tool_results_reach_the_model(ctx: ServerContext) -> None:
    model = ScriptedChatModel(script=[tool_call("customer_lookup", {"customer_id": 1042}), DONE])

    run = await run_agent("x", server=build_server(None, ctx), model=model)

    tool_message = run.messages[2]
    assert tool_message.name == "customer_lookup"
    assert "Maria Delgado" in tool_message.content


async def test_retrieval_tools_populate_the_grounding_context(ctx: ServerContext) -> None:
    """parks_* results land in trace.retrieved; customer_* results deliberately do not."""
    model = ScriptedChatModel(
        script=[
            tool_call("parks_search", {"query": "low elevation warm", "k": 3}, "a"),
            tool_call("customer_lookup", {"customer_id": 1042}, "b"),
            DONE,
        ]
    )

    run = await run_agent("x", server=build_server(None, ctx), model=model)

    assert run.trace.names == ["parks_search", "customer_lookup"]
    assert len(run.trace.retrieved) == 3
    assert all("park_code" in item for item in run.trace.retrieved)


async def test_unwraps_the_mcp_envelope(ctx: ServerContext) -> None:
    """A list-returning tool must not reach the model as {"result": [...]}."""
    model = ScriptedChatModel(script=[tool_call("parks_search", {"query": "desert", "k": 2}), DONE])

    run = await run_agent("x", server=build_server(None, ctx), model=model)

    content = run.messages[2].content
    assert content.startswith("["), content[:80]
    assert '"result"' not in content


async def test_step_limit_stops_a_runaway_loop(ctx: ServerContext) -> None:
    """A model that only ever asks for tools must be capped, not allowed to spin."""
    # ScriptedChatModel repeats its last message once the script runs out.
    model = ScriptedChatModel(script=[tool_call("parks_search", {"query": "forest", "k": 1})])

    run = await run_agent("x", server=build_server(None, ctx), model=model)

    # The model gets MAX_STEPS turns; the last is spent answering, so MAX_STEPS - 1 tool rounds.
    assert run.steps == MAX_STEPS
    assert len(run.trace.calls) == MAX_STEPS - 1
    assert run.hit_step_limit


async def test_unknown_tool_is_reported_not_raised(ctx: ServerContext) -> None:
    model = ScriptedChatModel(script=[tool_call("does_not_exist", {}), DONE])

    run = await run_agent("x", server=build_server(None, ctx), model=model)

    assert "No such tool" in run.messages[2].content
    assert run.answer == DONE.content


async def test_tool_error_is_surfaced_to_the_model(ctx: ServerContext) -> None:
    """A bad argument should come back as a tool error the model can react to, not crash the loop."""
    model = ScriptedChatModel(script=[tool_call("customer_lookup", {"customer_id": "abc"}), DONE])

    run = await run_agent("x", server=build_server(None, ctx), model=model)

    assert "failed" in run.messages[2].content.lower()
    assert run.answer == DONE.content
