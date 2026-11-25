from __future__ import annotations

from typing import Any

from ..pricing import Usage
from .base import Prepared, StreamTracker, as_int, canonical_arguments


class OpenAIChat:
    """OpenAI Chat Completions: ``client.chat.completions.create``."""

    provider = "openai"

    def prepare(self, request: dict[str, Any], default_max_output_tokens: int) -> Prepared:
        request = dict(request)
        cap = request.get("max_completion_tokens") or request.get("max_tokens")
        if cap is None:
            cap = request["max_completion_tokens"] = default_max_output_tokens
        # Every one of the n choices can use the full cap.
        return Prepared(request, int(cap) * int(request.get("n") or 1))

    def invoke(self, client: Any, request: dict[str, Any]) -> Any:
        return client.chat.completions.create(**request)

    def start_stream(self, request: dict[str, Any]):
        # Chat Completions only reports usage on a stream when asked to, in an
        # extra final chunk with no choices. Ask for it, and hide that chunk
        # unless the caller requested it, so loops reading chunk.choices[0]
        # keep working.
        options = dict(request.get("stream_options") or {})
        asked = bool(options.get("include_usage"))
        options["include_usage"] = True
        return {**request, "stream_options": options}, _ChatTracker(self, hide_usage=not asked)

    def tool_calls(self, response: Any) -> list[tuple[str, str]]:
        calls = []
        for choice in getattr(response, "choices", None) or []:
            for call in getattr(choice.message, "tool_calls", None) or []:
                function = getattr(call, "function", None)
                if function is not None:
                    calls.append((function.name, canonical_arguments(function.arguments)))
        return calls

    def count_input_tokens(self, client: Any, request: dict[str, Any]) -> Any:
        return None  # no token-counting endpoint is used; the estimate applies

    def counted(self, result: Any) -> int:
        return int(result)

    def usage(self, response: Any) -> Usage | None:
        u = getattr(response, "usage", None)
        if u is None:
            return None
        cached = as_int(getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0))
        prompt = as_int(getattr(u, "prompt_tokens", 0))
        return Usage(
            # OpenAI's prompt_tokens includes the cached ones.
            input_tokens=max(prompt - cached, 0),
            output_tokens=as_int(getattr(u, "completion_tokens", 0)),
            cache_read_tokens=cached,
        )


class OpenAIResponses:
    """OpenAI Responses API: ``client.responses.create``."""

    provider = "openai"

    def prepare(self, request: dict[str, Any], default_max_output_tokens: int) -> Prepared:
        request = dict(request)
        request.setdefault("max_output_tokens", default_max_output_tokens)
        return Prepared(request, int(request["max_output_tokens"]))

    def invoke(self, client: Any, request: dict[str, Any]) -> Any:
        return client.responses.create(**request)

    def start_stream(self, request: dict[str, Any]):
        return request, _ResponsesTracker(self)

    def tool_calls(self, response: Any) -> list[tuple[str, str]]:
        return [(item.name, canonical_arguments(item.arguments))
                for item in getattr(response, "output", None) or []
                if getattr(item, "type", None) == "function_call"]

    def count_input_tokens(self, client: Any, request: dict[str, Any]) -> Any:
        return None  # no token-counting endpoint is used; the estimate applies

    def counted(self, result: Any) -> int:
        return int(result)

    def usage(self, response: Any) -> Usage | None:
        u = getattr(response, "usage", None)
        if u is None:
            return None
        cached = as_int(getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0))
        total_in = as_int(getattr(u, "input_tokens", 0))
        return Usage(
            input_tokens=max(total_in - cached, 0),
            output_tokens=as_int(getattr(u, "output_tokens", 0)),
            cache_read_tokens=cached,
        )


class _ChatTracker(StreamTracker):
    def __init__(self, adapter: OpenAIChat, hide_usage: bool):
        super().__init__()
        self._adapter = adapter
        self._hide_usage = hide_usage
        self._tools: dict[tuple[int, int], list[str]] = {}  # (choice, call) -> [name, args]

    def tool_calls(self) -> list[tuple[str, str]]:
        return [(name, canonical_arguments(args or "{}"))
                for name, args in self._tools.values() if name]

    def observe(self, chunk: Any) -> bool:
        self.model = getattr(chunk, "model", None) or self.model
        choices = getattr(chunk, "choices", None) or []
        for choice in choices:
            delta = getattr(choice, "delta", None)
            self._count(getattr(delta, "content", None))
            for call in getattr(delta, "tool_calls", None) or []:
                function = getattr(call, "function", None)
                self._count(getattr(function, "arguments", None))
                slot = self._tools.setdefault((choice.index, call.index), ["", ""])
                slot[0] = getattr(function, "name", None) or slot[0]
                slot[1] += getattr(function, "arguments", None) or ""
        if getattr(chunk, "usage", None) is not None:
            self.usage = self._adapter.usage(chunk)
            self.complete = True
            if self._hide_usage and not choices:
                return False
        return True


class _ResponsesTracker(StreamTracker):
    _FINAL = ("response.completed", "response.incomplete", "response.failed")

    def __init__(self, adapter: OpenAIResponses):
        super().__init__()
        self._adapter = adapter
        self._calls: list[tuple[str, str]] = []

    def tool_calls(self) -> list[tuple[str, str]]:
        return self._calls

    def observe(self, event: Any) -> bool:
        kind = getattr(event, "type", "") or ""
        if kind.endswith(".delta"):
            self._count(getattr(event, "delta", None))
        elif kind in self._FINAL:
            response = getattr(event, "response", None)
            self.model = getattr(response, "model", None) or self.model
            self._calls = self._adapter.tool_calls(response)
            usage = self._adapter.usage(response)
            if usage is not None:
                self.usage = usage
                self.complete = True
        return True
