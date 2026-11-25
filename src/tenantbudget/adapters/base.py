"""What the guard needs to know about a provider API."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from ..pricing import Usage

# Request fields whose content is sent to the model as prompt tokens.
_PROMPT_FIELDS = ("messages", "system", "tools", "input", "instructions",
                  "response_format", "text", "output_config")


@dataclass(frozen=True)
class Prepared:
    request: dict[str, Any]
    max_output_tokens: int
    # The request asks the provider to write a prompt cache entry, which is
    # billed above the plain input rate.
    cache_write: bool = False


class Adapter(Protocol):
    provider: str

    def prepare(self, request: dict[str, Any], default_max_output_tokens: int) -> Prepared: ...

    def invoke(self, client: Any, request: dict[str, Any]) -> Any:
        """Make the call. Returns the response, or an awaitable for async clients."""

    def count_input_tokens(self, client: Any, request: dict[str, Any]) -> Any:
        """Exact prompt size from the provider, or None if it offers no counter."""

    def counted(self, result: Any) -> int:
        """Turn the (awaited) result of ``count_input_tokens`` into a number."""

    def usage(self, response: Any) -> Usage | None: ...

    def tool_calls(self, response: Any) -> list[tuple[str, str]]:
        """(tool name, canonical arguments) for each tool call the model requested."""

    def start_stream(self, request: dict[str, Any]) -> "tuple[dict[str, Any], StreamTracker]":
        """Turn a ``stream=True`` request into the one to send, plus the
        tracker that will read usage out of its events."""


class StreamTracker:
    """Watches the events of one streamed response.

    ``usage`` is whatever the provider has reported so far, ``complete`` says
    the final figures arrived, and ``output_bytes`` counts generated text so
    output can be estimated if the stream is cut short.
    """

    def __init__(self) -> None:
        self.usage: Usage | None = None
        self.model: str | None = None
        self.complete = False
        self.output_bytes = 0

    def observe(self, event: Any) -> bool:
        """Record one event. Returns False to hide it from the caller."""
        raise NotImplementedError

    def tool_calls(self) -> list[tuple[str, str]]:
        """Tool calls the model requested, as far as the stream has shown them."""
        return []

    def _count(self, *texts: Any) -> None:
        for text in texts:
            if isinstance(text, str):
                self.output_bytes += len(text.encode("utf-8"))


def estimate_input_tokens(request: dict[str, Any]) -> int:
    """Offline upper-leaning estimate of prompt tokens: one token per 3 bytes.

    English text runs near 4 bytes per token and CJK near 3, so this leans
    high, which is the safe direction for a reservation. It cannot see inside
    images or files passed by URL or ID; use exact counting for those.
    """
    payload = {k: request[k] for k in _PROMPT_FIELDS if request.get(k) is not None}
    text = json.dumps(payload, default=str, ensure_ascii=False)
    return len(text.encode("utf-8")) // 3 + 8


def mentions(value: Any, key: str) -> bool:
    """True if ``key`` appears as a mapping key anywhere inside ``value``."""
    if isinstance(value, dict):
        return key in value or any(mentions(v, key) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(mentions(v, key) for v in value)
    return False


def canonical_arguments(arguments: Any) -> str:
    """Tool arguments in one stable spelling, so equal calls compare equal."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return arguments
    return json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)


def as_int(value: Any) -> int:
    return int(value) if value else 0
