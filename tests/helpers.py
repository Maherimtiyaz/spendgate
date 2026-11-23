"""Shared test plumbing: canned provider responses behind a mock HTTP transport,
so the real SDK clients run with no network and no API key."""
import json

import anthropic
import openai
import pytest

httpx = pytest.importorskip(type(openai.DefaultHttpxClient()).__mro__[1].__module__.split(".")[0])


def anthropic_body(model="claude-opus-5-5", **usage):
    usage = {"input_tokens": 100, "output_tokens": 50, **usage}
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn",
            "stop_sequence": None, "usage": usage}


def chat_body(model="gpt-5.4", **usage):
    usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, **usage}
    return {"id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "hi"}}],
            "usage": usage}


def responses_body(model="gpt-5.4", **usage):
    usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150,
             "input_tokens_details": {"cached_tokens": 0},
             "output_tokens_details": {"reasoning_tokens": 0}, **usage}
    return {"id": "resp_1", "object": "response", "created_at": 1, "model": model,
            "status": "completed", "output": [], "parallel_tool_calls": True,
            "tool_choice": "auto", "tools": [], "usage": usage}


class Server:
    """Records requests and answers each path with a canned body or status."""

    def __init__(self, routes):
        self.routes = routes
        self.requests = []

    def __call__(self, request):
        self.requests.append((request.url.path, json.loads(request.content or b"{}")))
        reply = self.routes[request.url.path]
        if isinstance(reply, str):  # a server-sent-events stream
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  content=reply.encode())
        if isinstance(reply, int):
            return httpx.Response(reply, json={"error": {"type": "api_error", "message": "boom"}})
        return httpx.Response(200, json=reply)


def clients(routes):
    server = Server(routes)
    sync, asyn = httpx.MockTransport(server), httpx.MockTransport(server)
    kw = {"api_key": "test", "max_retries": 0}
    return server, {
        "anthropic": anthropic.Anthropic(http_client=httpx.Client(transport=sync), **kw),
        "anthropic_async": anthropic.AsyncAnthropic(
            http_client=httpx.AsyncClient(transport=asyn), **kw),
        "openai": openai.OpenAI(http_client=httpx.Client(transport=sync), **kw),
        "openai_async": openai.AsyncOpenAI(http_client=httpx.AsyncClient(transport=asyn), **kw),
    }


MSG = [{"role": "user", "content": "hello"}]


def sse(*events, done=False):
    """Encode events as a server-sent-events body. Events with a "type" get an
    ``event:`` line, as Anthropic and the OpenAI Responses API send them."""
    out = []
    for event in events:
        if "type" in event:
            out.append(f"event: {event['type']}\n")
        out.append(f"data: {json.dumps(event)}\n\n")
    if done:
        out.append("data: [DONE]\n\n")
    return "".join(out)
