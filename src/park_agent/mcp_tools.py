"""Bridge: MCP tools -> LangChain tools, recording everything the agent does.

Hand-rolled rather than using langchain-mcp-adapters, for two reasons: it is about fifty lines
against the mcp 2.x API with no version-skew risk, and wrapping the call ourselves is what gives
the eval suite a tool trace to assert on.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from mcp import Client

log = logging.getLogger(__name__)

# Results from these tools are the agent's grounding - the eval judge scores the answer against
# them, so they are captured separately from the plain call log.
RETRIEVAL_TOOLS = {"parks_search", "parks_similar_to", "parks_get", "parks_alerts"}


@dataclass
class ToolTrace:
    """Ordered record of what the agent actually called.

    The evals assert on this *before* paying for a judge call: a broken tool wiring should fail
    fast and free, not show up as a mysteriously low rubric score.
    """

    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    retrieved: list[dict[str, Any]] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def called(self, name: str) -> bool:
        return name in self.names


def _unwrap(structured_content: dict[str, Any] | None) -> Any:
    """Undo the MCP structured-output envelope.

    GOTCHA: a tool returning a non-dict (a list, a string, null) comes back as `{"result": <value>}`
    because the protocol's structured output has to be an object; a tool returning a dict comes back
    as itself. Forget this and the model - and the judge scoring groundedness - sees
    `[{"result": [...]}]` instead of the parks.
    """
    if isinstance(structured_content, dict) and set(structured_content) == {"result"}:
        return structured_content["result"]
    return structured_content


async def load_mcp_tools(client: Client, trace: ToolTrace) -> list[BaseTool]:
    """Wrap every tool the server advertises as a LangChain tool, recording calls into `trace`."""
    tools: list[BaseTool] = []

    for spec in (await client.list_tools()).tools:
        name = spec.name

        def make_caller(tool_name: str):
            # StructuredTool calls coroutine(**args), so this takes **kwargs, not a dict.
            async def call(**kwargs: Any) -> str:
                trace.calls.append((tool_name, kwargs))
                result = await client.call_tool(tool_name, kwargs)
                if result.is_error:
                    log.warning("tool %s errored: %s", tool_name, result.content)
                    return f"Tool {tool_name} failed: {result.content}"
                payload = _unwrap(result.structured_content)
                if tool_name in RETRIEVAL_TOOLS and payload:
                    items = payload if isinstance(payload, list) else [payload]
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        # parks_alerts returns bare alert objects with no park_code of their own,
                        # which left the judge grading against context entries identified only as
                        # null. Stamp the code from the call arguments so every grounding item
                        # says which park it belongs to.
                        if "park_code" not in item and "park_code" in kwargs:
                            item = {"park_code": kwargs["park_code"], **item}
                        trace.retrieved.append(item)
                return json.dumps(payload, default=str)

            return call

        tools.append(
            StructuredTool(
                name=name,
                # Pass the description explicitly. CONFUSING-BIT WARNING: with a dict args_schema
                # LangChain only falls back to args_schema["description"], and an MCP input_schema
                # has no such key - omit this and you get a tool the model cannot understand.
                description=spec.description or name,
                # langchain-core accepts a raw JSON Schema dict here (ArgsSchema = BaseModel | dict),
                # so the MCP input_schema goes straight in with no pydantic conversion. The tradeoff:
                # a dict schema skips pydantic coercion, so a model emitting "5" for an int arg
                # reaches the server as a string and is rejected there rather than coerced. Fine for
                # a POC; swap in a generated BaseModel if that ever bites.
                args_schema=spec.input_schema,
                coroutine=make_caller(name),
            )
        )

    log.info("loaded %s MCP tools: %s", len(tools), ", ".join(t.name for t in tools))
    return tools
