"""The vector-store half of the MCP server, against an offline-embedded collection.

HashEmbeddings is lexical, not semantic, so these tests assert on *wiring and shape* - dedupe,
filtering, envelope unwrapping, alert surfacing - and leave "is this a good recommendation?" to
the eval suite, where real embeddings and a judge are in play.
"""

import pytest
from mcp import Client

from park_mcp.context import ServerContext
from park_mcp.server import build_server
from park_mcp.toolsets import unwrap

pytestmark = pytest.mark.anyio


async def call(client: Client, name: str, args: dict | None = None):
    result = await client.call_tool(name, args or {})
    assert not result.is_error, result.content
    return unwrap(result.structured_content)


async def test_search_returns_distinct_parks(ctx: ServerContext) -> None:
    async with Client(build_server(["parks"], ctx)) as client:
        hits = await call(client, "parks_search", {"query": "desert canyon sandstone", "k": 5})

    assert len(hits) == 5
    codes = [h["park_code"] for h in hits]
    # A park split across several chunks must not come back more than once.
    assert len(codes) == len(set(codes))
    assert all({"park_code", "name", "states", "matched_text"} <= h.keys() for h in hits)


async def test_search_k_is_clamped(ctx: ServerContext) -> None:
    async with Client(build_server(["parks"], ctx)) as client:
        assert len(await call(client, "parks_search", {"query": "forest", "k": 500})) <= 10
        assert len(await call(client, "parks_search", {"query": "forest", "k": 0})) == 1


async def test_search_filters_by_state(ctx: ServerContext) -> None:
    async with Client(build_server(["parks"], ctx)) as client:
        hits = await call(client, "parks_search", {"query": "canyon", "k": 5, "state": "UT"})

    assert hits
    assert {h["states"] for h in hits} == {"UT"}


async def test_search_filters_by_max_elevation(ctx: ServerContext) -> None:
    """A hard ceiling must be enforced by the filter, not left to the embedding to infer."""
    async with Client(build_server(["parks"], ctx)) as client:
        unfiltered = await call(client, "parks_search", {"query": "alpine peaks glaciers", "k": 8})
        filtered = await call(
            client,
            "parks_search",
            {"query": "alpine peaks glaciers", "k": 8, "max_elevation_ft": 2000},
        )

    # Without the filter, high-elevation parks dominate an alpine query.
    assert any(h["elevation_ft_min"] > 2000 for h in unfiltered)
    assert filtered
    assert all(h["elevation_ft_min"] <= 2000 for h in filtered)


async def test_search_combines_state_and_elevation_filters(ctx: ServerContext) -> None:
    async with Client(build_server(["parks"], ctx)) as client:
        hits = await call(
            client,
            "parks_search",
            {"query": "scenery", "k": 5, "state": "CA", "max_elevation_ft": 1000},
        )

    assert hits
    assert all(h["states"] == "CA" and h["elevation_ft_min"] <= 1000 for h in hits)


async def test_get_returns_full_record_with_elevation(ctx: ServerContext) -> None:
    async with Client(build_server(["parks"], ctx)) as client:
        zion = await call(client, "parks_get", {"park_code": "ZION"})  # case-insensitive
        assert zion["name"] == "Zion National Park"
        assert zion["elevation_ft_min"] == 3666
        assert zion["elevation_ft_max"] == 8726
        assert zion["activities"]

        assert await call(client, "parks_get", {"park_code": "not-a-park"}) is None


async def test_alerts(ctx: ServerContext) -> None:
    async with Client(build_server(["parks"], ctx)) as client:
        alerts = await call(client, "parks_alerts", {"park_code": "bibe"})
        assert len(alerts) == 1
        assert "Chisos Basin" in alerts[0]["title"]
        assert alerts[0]["category"] == "Park Closure"

        # No alerts is an empty list, not null - the agent should not treat it as a failure.
        assert await call(client, "parks_alerts", {"park_code": "cony"}) == []
        assert await call(client, "parks_alerts", {"park_code": "whsa"}) == []


async def test_similar_to_excludes_itself(ctx: ServerContext) -> None:
    async with Client(build_server(["parks"], ctx)) as client:
        similar = await call(client, "parks_similar_to", {"park_code": "zion", "k": 4})

    codes = [p["park_code"] for p in similar]
    assert len(codes) == 4
    assert "zion" not in codes
    assert len(codes) == len(set(codes))


async def test_similar_to_honours_the_elevation_filter(ctx: ServerContext) -> None:
    """The constraint must hold on every retrieval path, not just parks_search.

    This is the regression the eval suite caught: the agent reached the knowledge base through
    parks_similar_to, which had no filter, and recommended a park above the visitor's ceiling.
    """
    async with Client(build_server(["parks"], ctx)) as client:
        unfiltered = await call(client, "parks_similar_to", {"park_code": "romo", "k": 8})
        filtered = await call(
            client, "parks_similar_to", {"park_code": "romo", "k": 8, "max_elevation_ft": 2000}
        )

    assert any(h["elevation_ft_min"] > 2000 for h in unfiltered)
    assert filtered
    assert all(h["elevation_ft_min"] <= 2000 for h in filtered)


async def test_similar_to_unknown_park_is_empty(ctx: ServerContext) -> None:
    async with Client(build_server(["parks"], ctx)) as client:
        assert await call(client, "parks_similar_to", {"park_code": "nope"}) == []


async def test_search_surfaces_alert_counts(ctx: ServerContext) -> None:
    """A search hit carries its alert count, so the agent knows when to call parks_alerts."""
    async with Client(build_server(["parks"], ctx)) as client:
        hits = await call(client, "parks_search", {"query": "Chihuahuan Desert Rio Grande", "k": 8})

    bibe = next((h for h in hits if h["park_code"] == "bibe"), None)
    assert bibe is not None, [h["park_code"] for h in hits]
    assert bibe["alert_count"] == 1
