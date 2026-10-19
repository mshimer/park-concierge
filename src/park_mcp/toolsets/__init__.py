"""Toolset registry.

A toolset is a module with `register(mcp: MCPServer, ctx: ServerContext) -> None` that adds its
tools, each named `<source>_<action>`. Modules are imported lazily, so a toolset's extra deps only
matter when that toolset is enabled.

To add one: create `toolsets/<name>.py` with `register()`, then add it to TOOLSETS.
"""

import importlib
from collections.abc import Iterable
from typing import Any

from mcp.server.mcpserver import MCPServer

from ..context import ServerContext

TOOLSETS: dict[str, str] = {
    "customers": "park_mcp.toolsets.customers",
    "parks": "park_mcp.toolsets.parks",
    # "weather": "park_mcp.toolsets.weather",
    # "reservations": "park_mcp.toolsets.reservations",
}


def register_toolsets(
    mcp: MCPServer, ctx: ServerContext, names: Iterable[str] | None = None
) -> list[str]:
    """Register the named toolsets (all when None). Returns the names registered."""
    selected = list(TOOLSETS) if names is None else list(names)
    unknown = set(selected) - TOOLSETS.keys()
    if unknown:
        raise ValueError(f"unknown toolsets {sorted(unknown)}; known: {sorted(TOOLSETS)}")
    for name in selected:
        importlib.import_module(TOOLSETS[name]).register(mcp, ctx)
    return selected


def unwrap(structured_content: dict[str, Any] | None) -> Any:
    """Undo the MCP envelope.

    GOTCHA: a tool returning a non-dict (a list, a string) comes back as
    `{"result": <value>}` because the protocol's structured output must be an object. A tool
    returning a dict comes back as that dict. Callers that forget this end up handing
    `[{"result": [...]}]` to a model, or to an LLM judge, and wonder why grounding looks wrong.
    """
    if isinstance(structured_content, dict) and set(structured_content) == {"result"}:
        return structured_content["result"]
    return structured_content
