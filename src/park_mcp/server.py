from collections.abc import Iterable

from mcp.server.mcpserver import MCPServer

from .context import ServerContext
from .toolsets import register_toolsets

INSTRUCTIONS = """Tools for a National Park trip concierge.

`customer_*` tools read the park-pass customer database. `parks_*` tools read the park knowledge
base. Tool names are prefixed by the data source they touch. Everything is read-only.
"""


def build_server(
    toolsets: Iterable[str] | None = None, ctx: ServerContext | None = None
) -> MCPServer:
    """Build the server with the given toolsets (all when None)."""
    mcp = MCPServer(name="park-mcp", instructions=INSTRUCTIONS)

    # NOTE : ctx can be injected for testing or pulled from the settings for prod

    register_toolsets(mcp, ctx or ServerContext.from_settings(), toolsets)
    return mcp
