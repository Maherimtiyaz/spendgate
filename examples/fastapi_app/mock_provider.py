"""A canned Anthropic API, so the example runs with no key and no network."""
from __future__ import annotations

import importlib
import json

from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient

# The HTTP library the installed SDK is built on (httpx or httpx2).
_http = importlib.import_module(DefaultAsyncHttpxClient.__mro__[1].__module__.split(".")[0])


def _handle(request):
    body = json.loads(request.content)
    prompt = json.dumps(body.get("messages", []))
    reply = "This is a canned reply from the mock provider."
    return _http.Response(200, json={
        "id": "msg_mock", "type": "message", "role": "assistant", "model": body["model"],
        "content": [{"type": "text", "text": reply}],
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": max(len(prompt) // 4, 1), "output_tokens": 400},
    })


def mock_client() -> AsyncAnthropic:
    transport = _http.MockTransport(_handle)
    return AsyncAnthropic(api_key="mock", max_retries=0,
                          http_client=_http.AsyncClient(transport=transport))
