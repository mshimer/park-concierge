"""One round trip over the real stdio transport.

Everything else talks to an in-process server, which mcp dispatches directly - no JSON-RPC
framing, no initialize handshake, no subprocess. That is fast, but it means the in-process tests
could never catch a tool returning something unserializable, or a stray print polluting stdout and
corrupting the protocol stream. This test exists to catch exactly that.

It deliberately touches only the tools that need no credentials, so the fast suite stays runnable
with no OPENAI_API_KEY and no database.
"""

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from park_core.config import PROJECT_ROOT
from park_mcp.toolsets import unwrap

pytestmark = pytest.mark.anyio

PARAMS = StdioServerParameters(
    command="uv", args=["run", "park-mcp"], cwd=str(PROJECT_ROOT)
)


async def test_stdio_round_trip() -> None:
    async with Client(PARAMS) as client:
        tools = {t.name for t in (await client.list_tools()).tools}
        assert {"customer_lookup", "parks_search", "parks_alerts", "parks_get"} <= tools

        # Serializes cleanly across the wire, including the nested alert objects.
        alerts = unwrap((await client.call_tool("parks_alerts", {"park_code": "bibe"})).structured_content)
        assert len(alerts) == 1
        assert "Chisos Basin" in alerts[0]["title"]

        park = unwrap((await client.call_tool("parks_get", {"park_code": "zion"})).structured_content)
        assert park["elevation_ft_max"] == 8726
