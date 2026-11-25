from __future__ import annotations

from typing import Any

from ..pricing import Usage
from .base import Prepared, StreamTracker, as_int, canonical_arguments, mentions

_COUNT_FIELDS = ("model", "messages", "system", "tools", "tool_choice", "thinking")


class AnthropicMessages:
    """Anthropic Messages API: ``client.messages.create``."""

    provider = "anthropic"

    def prepare(self, request: dict[str, Any], default_max_output_tokens: int) -> Prepared:
        request = dict(request)
        request.setdefault("max_tokens", default_max_output_tokens)
        return Prepared(request, int(request["max_tokens"]), mentions(request, "cache_control"))

    def invoke(self, client: Any, request: dict[str, Any]) -> Any:
        return client.messages.create(**request)

    def count_input_tokens(self, client: Any, request: dict[str, Any]) -> Any:
        params = {k: request[k] for k in _COUNT_FIELDS if k in request}
        return client.messages.count_tokens(**params)

    def counted(self, result: Any) -> int:
        return int(result.input_tokens)

    def start_stream(self, request: dict[str, Any]):
        return request, _MessagesTracker(self)

    def tool_calls(self, response: Any) -> list[tuple[str, str]]:
        return [(block.name, canonical_arguments(block.input))
                for block in getattr(response, "content", None) or []
                if getattr(block, "type", None) == "tool_use"]

    def usage(self, response: Any) -> Usage | None:
        u = getattr(response, "usage", None)
        if u is None:
            return None
        written = as_int(getattr(u, "cache_creation_input_tokens", 0))
        by_ttl = getattr(u, "cache_creation", None)
        if by_ttl is not None:
            write_5m = as_int(getattr(by_ttl, "ephemeral_5m_input_tokens", 0))
            write_1h = as_int(getattr(by_ttl, "ephemeral_1h_input_tokens", 0))
        else:
            write_5m, write_1h = written, 0
        return Usage(
            # Anthropic's input_tokens already excludes cached tokens.
            input_tokens=as_int(getattr(u, "input_tokens", 0)),
            output_tokens=as_int(getattr(u, "output_tokens", 0)),
            cache_read_tokens=as_int(getattr(u, "cache_read_input_tokens", 0)),
            cache_write_5m_tokens=write_5m,
            cache_write_1h_tokens=write_1h,
        )


class _Fields:
    """Usage fields gathered across stream events, shaped like a usage object."""

    cache_creation = None


class _MessagesTracker(StreamTracker):
    _FIELDS = ("input_tokens", "output_tokens", "cache_creation_input_tokens",
               "cache_read_input_tokens", "cache_creation")

    def __init__(self, adapter: AnthropicMessages):
        super().__init__()
        self._adapter = adapter
        self._seen = _Fields()
        self._tools: dict[int, list[str]] = {}  # block index -> [name, partial json]

    def tool_calls(self) -> list[tuple[str, str]]:
        return [(name, canonical_arguments(args or "{}"))
                for name, args in self._tools.values()]

    def observe(self, event: Any) -> bool:
        kind = getattr(event, "type", None)
        if kind == "message_start":
            self.model = getattr(event.message, "model", None)
            self._merge(getattr(event.message, "usage", None))
        elif kind == "content_block_start":
            block = event.content_block
            if getattr(block, "type", None) == "tool_use":
                self._tools[event.index] = [block.name, ""]
        elif kind == "content_block_delta":
            delta = event.delta
            if event.index in self._tools and isinstance(
                    getattr(delta, "partial_json", None), str):
                self._tools[event.index][1] += delta.partial_json
            self._count(getattr(delta, "text", None), getattr(delta, "partial_json", None),
                        getattr(delta, "thinking", None))
        elif kind == "message_delta":
            # Carries the cumulative output count, and sometimes revised input counts.
            self._merge(getattr(event, "usage", None))
        elif kind == "message_stop":
            self.complete = True
        return True

    def _merge(self, usage: Any) -> None:
        if usage is None:
            return
        for name in self._FIELDS:
            value = getattr(usage, name, None)
            if value is not None:
                setattr(self._seen, name, value)
        holder = type("_Response", (), {"usage": self._seen})
        self.usage = self._adapter.usage(holder)
