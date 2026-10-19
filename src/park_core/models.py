"""One factory for chat models, so every caller handles `temperature` the same way.

Newer OpenAI models (the gpt-5 / gpt-6 families and the o-series reasoning models) reject a
`temperature` parameter with a 400 "Unsupported parameter". langchain-openai only knows to strip it
for a hard-coded list of name prefixes, so a newer model name sails through and fails at the API.
Upgrading the library does not help until it learns the new name; omitting the parameter does.
"""

import os
from typing import Any

from langchain_openai import ChatOpenAI

# Model-name prefixes that do not accept a temperature. Extend this when a new family appears.
NO_TEMPERATURE_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")


def supports_temperature(model: str) -> bool:
    """LLM_TEMPERATURE=none forces omission, any number forces it; otherwise decide by name."""
    override = os.getenv("LLM_TEMPERATURE", "").strip().lower()
    if override:
        return override != "none"
    return not model.lower().startswith(NO_TEMPERATURE_PREFIXES)


def make_chat_model(model: str, **kwargs: Any) -> ChatOpenAI:
    """A ChatOpenAI that is deterministic (temperature 0) where the model allows it.

    Models without temperature control cannot be pinned, so evals against them are inherently a
    little noisier.
    """
    if supports_temperature(model):
        override = os.getenv("LLM_TEMPERATURE", "").strip().lower()
        kwargs.setdefault("temperature", float(override) if override else 0)
    return ChatOpenAI(model=model, **kwargs)
