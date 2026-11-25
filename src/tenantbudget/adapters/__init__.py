from __future__ import annotations

from typing import Any

from .anthropic import AnthropicMessages
from .base import Adapter, Prepared, StreamTracker, estimate_input_tokens
from .openai import OpenAIChat, OpenAIResponses

__all__ = ["Adapter", "Prepared", "StreamTracker", "AnthropicMessages", "OpenAIChat", "OpenAIResponses",
           "adapter_for", "estimate_input_tokens"]


def adapter_for(client: Any, api: str | None = None) -> Adapter:
    """Pick the adapter from the SDK client's package, without importing any SDK."""
    package = type(client).__module__.split(".")[0]
    if package == "anthropic":
        if api not in (None, "messages"):
            raise ValueError(f"unsupported Anthropic api {api!r}; expected 'messages'")
        return AnthropicMessages()
    if package == "openai":
        if api in (None, "chat"):
            return OpenAIChat()
        if api == "responses":
            return OpenAIResponses()
        raise ValueError(f"unsupported OpenAI api {api!r}; expected 'chat' or 'responses'")
    raise TypeError(
        f"cannot tell which provider {type(client).__name__} belongs to; "
        "pass adapter=... explicitly"
    )
