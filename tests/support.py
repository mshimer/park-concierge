"""Test doubles shared across test modules.

Lives beside conftest rather than inside it: fixtures belong in conftest, plain helpers are
easier to find - and to import - here.
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedChatModel(BaseChatModel):
    """A chat model that replays a fixed list of AIMessages, in order.

    langchain-core's own fakes do not override `bind_tools`, so the base class raises
    NotImplementedError and they cannot drive a tool-calling graph. This one accepts the binding
    and ignores it, which is exactly what a scripted test needs.
    """

    script: list[AIMessage]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":  # type: ignore[override]
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Past the end of the script, keep returning the last message so a runaway loop is capped
        # by MAX_STEPS rather than by an IndexError.
        template = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1

        # CONFUSING-BIT WARNING: every turn needs a fresh message id and fresh tool_call ids.
        # `add_messages` merges by id, so replaying the same object would overwrite the previous
        # turn in place instead of appending - the graph would then see a ToolMessage as the last
        # message, decide there are no tool calls, and quietly stop after one round trip. A real
        # model never repeats an id; neither should a double that stands in for one.
        turn = self.calls
        message = template.model_copy(
            update={
                "id": f"scripted-{turn}",
                "tool_calls": [
                    {**tc, "id": f"{tc['id']}-{turn}"} for tc in (template.tool_calls or [])
                ],
            }
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def tool_call(name: str, args: dict[str, Any], call_id: str = "c1") -> AIMessage:
    """An AIMessage that requests one tool call."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )
