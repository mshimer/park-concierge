"""The Postgres half of the MCP server, against a real throwaway container."""

import pytest
from mcp import Client

from park_mcp.context import ServerContext
from park_mcp.server import build_server
from park_mcp.toolsets import TOOLSETS, unwrap

pytestmark = pytest.mark.anyio


async def call(client: Client, name: str, args: dict | None = None):
    result = await client.call_tool(name, args or {})
    assert not result.is_error, result.content
    return unwrap(result.structured_content)


async def test_toolset_selection(ctx: ServerContext) -> None:
    async with Client(build_server(["customers"], ctx)) as client:
        names = {t.name for t in (await client.list_tools()).tools}

    assert names == {"customer_lookup", "customer_find_by_email", "customer_visits"}
    with pytest.raises(ValueError, match="unknown toolsets"):
        build_server(["nope"], ctx)
    assert {"customers", "parks"} <= TOOLSETS.keys()


async def test_build_server_is_repeatable(ctx: ServerContext) -> None:
    # The lazy importlib registry caches the module, so a second build must still register tools.
    # (This is what a module-level @mcp.tool singleton would quietly break.)
    async with Client(build_server(["parks"], ctx)) as client:
        first = {t.name for t in (await client.list_tools()).tools}
    async with Client(build_server(["parks"], ctx)) as client:
        second = {t.name for t in (await client.list_tools()).tools}

    assert first == second != set()


async def test_customer_lookup(ctx: ServerContext) -> None:
    async with Client(build_server(["customers"], ctx)) as client:
        maria = await call(client, "customer_lookup", {"customer_id": 1042})
        assert maria["name"] == "Maria Delgado"
        assert maria["pass_tier"] == "access"
        assert maria["max_elevation_ft"] == 5000
        assert "wheelchair" in maria["accessibility_needs"]

        # A missing customer is null, not an error - the agent is told not to invent one.
        assert await call(client, "customer_lookup", {"customer_id": 99999}) is None


async def test_customer_find_by_email_is_case_insensitive(ctx: ServerContext) -> None:
    async with Client(build_server(["customers"], ctx)) as client:
        found = await call(client, "customer_find_by_email", {"email": "TOM.ISHIKAWA@EXAMPLE.COM"})
        assert found["id"] == 1043
        assert await call(client, "customer_find_by_email", {"email": "nobody@example.com"}) is None


async def test_customer_visits_are_newest_first(ctx: ServerContext) -> None:
    async with Client(build_server(["customers"], ctx)) as client:
        visits = await call(client, "customer_visits", {"customer_id": 1042})

    assert [v["park_code"] for v in visits] == ["whsa", "bibe", "sagu"]
    dates = [v["visited_on"] for v in visits]
    assert dates == sorted(dates, reverse=True)


async def test_visit_limit_is_capped(ctx: ServerContext) -> None:
    async with Client(build_server(["customers"], ctx)) as client:
        assert len(await call(client, "customer_visits", {"customer_id": 1042, "limit": 2})) == 2


async def test_tools_are_annotated_read_only(ctx: ServerContext) -> None:
    async with Client(build_server(None, ctx)) as client:
        tools = (await client.list_tools()).tools

    assert tools, "expected every toolset to register"
    for tool in tools:
        assert tool.annotations is not None, tool.name
        assert tool.annotations.read_only_hint is True, tool.name
