"""Streaming calls, through the real SDK clients' own stream parsers."""
import pytest

from helpers import MSG, clients, sse
from spendgate import BudgetExceeded, Guard, Usage


def anthropic_stream(text=("Hel", "lo"), output_tokens=42, finish=True):
    events = [
        {"type": "message_start", "message": {
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 100, "output_tokens": 1,
                      "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 0}}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        *[{"type": "content_block_delta", "index": 0,
           "delta": {"type": "text_delta", "text": t}} for t in text],
    ]
    if finish:
        events += [
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
             "usage": {"output_tokens": output_tokens}},
            {"type": "message_stop"},
        ]
    return sse(*events)


def chat_chunk(content=None, finish=None, usage=None, choices=True):
    chunk = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-5.4",
             "choices": [{"index": 0, "delta": {"content": content} if content else {},
                          "finish_reason": finish}] if choices else []}
    if usage:
        chunk["usage"] = usage
    return chunk


CHAT_USAGE = {"prompt_tokens": 1000, "completion_tokens": 50, "total_tokens": 1050,
              "prompt_tokens_details": {"cached_tokens": 600}}


def chat_stream(with_usage=True):
    chunks = [chat_chunk("Hel"), chat_chunk("lo"), chat_chunk(finish="stop")]
    if with_usage:
        chunks.append(chat_chunk(usage=CHAT_USAGE, choices=False))
    return sse(*chunks, done=True)


def responses_stream():
    response = {"id": "resp_1", "object": "response", "created_at": 1, "model": "gpt-5.4",
                "status": "completed", "output": [], "parallel_tool_calls": True,
                "tool_choice": "auto", "tools": [],
                "usage": {"input_tokens": 1000, "output_tokens": 50, "total_tokens": 1050,
                          "input_tokens_details": {"cached_tokens": 200},
                          "output_tokens_details": {"reasoning_tokens": 0}}}
    return sse(
        {"type": "response.output_text.delta", "item_id": "i", "output_index": 0,
         "content_index": 0, "delta": "Hello", "sequence_number": 1, "logprobs": []},
        {"type": "response.completed", "response": response, "sequence_number": 2},
    )


def month(guard):
    return next(s for s in guard.status("acme") if s.key.window == "month")


def test_anthropic_stream_passes_events_through_and_settles_at_the_end(guard, events):
    _, c = clients({"/v1/messages": anthropic_stream()})
    stream = guard.chat(c["anthropic"], tenant="acme", model="claude-opus-5-5",
                        max_tokens=500, messages=MSG, stream=True)
    assert month(guard).reserved_micros > 0 and events == []  # held while streaming
    text = "".join(e.delta.text for e in stream if e.type == "content_block_delta")

    assert text == "Hello"
    (event,) = events
    assert event.usage == Usage(input_tokens=100, output_tokens=42, cache_read_tokens=1000)
    assert not event.estimated and stream.usage_event is event
    assert (month(guard).reserved_micros, month(guard).spent_micros) == (0, event.cost_micros)


def test_openai_chat_stream_asks_for_usage_and_hides_the_extra_chunk(guard, events):
    server, c = clients({"/v1/chat/completions": chat_stream()})
    stream = guard.chat(c["openai"], tenant="acme", model="gpt-5.4", messages=MSG,
                        max_completion_tokens=200, stream=True)
    # A typical loop indexes choices[0]; the usage-only chunk would break it.
    text = "".join(chunk.choices[0].delta.content or "" for chunk in stream)

    assert text == "Hello"
    assert server.requests[0][1]["stream_options"] == {"include_usage": True}
    assert events[0].usage == Usage(input_tokens=400, output_tokens=50, cache_read_tokens=600)
    assert not events[0].estimated


def test_openai_usage_chunk_is_shown_to_callers_who_asked_for_it(guard, events):
    _, c = clients({"/v1/chat/completions": chat_stream()})
    stream = guard.chat(c["openai"], tenant="acme", model="gpt-5.4", messages=MSG,
                        stream=True, stream_options={"include_usage": True})
    chunks = list(stream)
    assert chunks[-1].choices == [] and chunks[-1].usage.completion_tokens == 50
    assert len(events) == 1


def test_openai_responses_stream(guard, events):
    _, c = clients({"/v1/responses": responses_stream()})
    stream = guard.chat(c["openai"], api="responses", tenant="acme", model="gpt-5.4",
                        input="hello", max_output_tokens=200, stream=True)
    kinds = [e.type for e in stream]
    assert kinds == ["response.output_text.delta", "response.completed"]
    assert events[0].usage == Usage(input_tokens=800, output_tokens=50, cache_read_tokens=200)


async def test_async_streams(guard, events):
    _, c = clients({"/v1/messages": anthropic_stream(), "/v1/chat/completions": chat_stream()})
    stream = await guard.achat(c["anthropic_async"], tenant="acme", model="claude-opus-5-5",
                               max_tokens=500, messages=MSG, stream=True)
    kinds = [e.type async for e in stream]
    assert kinds[0] == "message_start" and kinds[-1] == "message_stop"

    async with await guard.achat(c["openai_async"], tenant="acme", model="gpt-5.4",
                                 messages=MSG, stream=True) as stream:
        text = "".join([chunk.choices[0].delta.content or "" async for chunk in stream])
    assert text == "Hello"
    assert [(e.provider, e.estimated) for e in events] == [("anthropic", False),
                                                           ("openai", False)]
    assert month(guard).reserved_micros == 0


def test_closing_a_stream_early_books_an_estimate_and_frees_the_hold(guard, events):
    _, c = clients({"/v1/messages": anthropic_stream(text=["word " * 60] * 5)})
    with guard.chat(c["anthropic"], tenant="acme", model="claude-opus-5-5", max_tokens=2000,
                    messages=MSG, stream=True) as stream:
        for seen, event in enumerate(stream):
            if seen == 3:  # message_start, block start, two text deltas
                break
    (event,) = events
    assert event.estimated
    # Input figures came from message_start; output is estimated from the
    # 600 bytes of text received, never from the 1 token reported at start.
    assert event.usage.input_tokens == 100 and event.usage.cache_read_tokens == 1000
    assert event.usage.output_tokens == 600 // 3 + 1
    assert 0 < event.cost_micros <= event.reserved_micros
    assert month(guard).reserved_micros == 0


def test_stream_that_ends_without_final_usage_is_estimated(guard, events):
    _, c = clients({"/v1/chat/completions": chat_stream(with_usage=False),
                    "/v1/messages": anthropic_stream(finish=False)})
    list(guard.chat(c["openai"], tenant="acme", model="gpt-5.4", messages=MSG, stream=True))
    list(guard.chat(c["anthropic"], tenant="acme", model="claude-opus-5-5", max_tokens=100,
                    messages=MSG, stream=True))
    assert [e.estimated for e in events] == [True, True]
    # OpenAI reported nothing, so the prompt is billed at its reserved size.
    assert events[0].usage.input_tokens > 0 and events[0].usage.output_tokens == 5 // 3 + 1
    assert all(e.cost_micros <= e.reserved_micros for e in events)


def test_stream_settles_once_however_it_is_finished(guard, events):
    _, c = clients({"/v1/messages": anthropic_stream()})
    stream = guard.chat(c["anthropic"], tenant="acme", model="claude-opus-5-5",
                        max_tokens=500, messages=MSG, stream=True)
    list(stream)
    stream.close()
    del stream
    assert len(events) == 1


def test_dropped_stream_is_settled_when_collected(guard, events):
    _, c = clients({"/v1/messages": anthropic_stream()})
    stream = guard.chat(c["anthropic"], tenant="acme", model="claude-opus-5-5",
                        max_tokens=500, messages=MSG, stream=True)
    next(iter(stream))
    del stream
    assert len(events) == 1 and events[0].estimated
    assert month(guard).reserved_micros == 0


def test_stream_over_budget_is_refused_before_the_call(guard, events):
    server, c = clients({"/v1/messages": anthropic_stream()})
    with pytest.raises(BudgetExceeded):
        guard.chat(c["anthropic"], tenant="acme", model="claude-opus-5-5",
                   max_tokens=60_000, messages=MSG, stream=True)
    assert server.requests == []


def test_stream_wrapper_exposes_the_sdk_stream_attributes(guard):
    _, c = clients({"/v1/chat/completions": chat_stream()})
    stream = guard.chat(c["openai"], tenant="acme", model="gpt-5.4", messages=MSG, stream=True)
    assert stream.response.status_code == 200
    stream.close()


def test_long_running_stream_keeps_its_hold_past_the_ttl(clock):
    guard = Guard(clock=clock, reservation_ttl=60)
    guard.set_budget("acme", monthly_usd=1)
    _, c = clients({"/v1/messages": anthropic_stream(text=["a"] * 20)})
    stream = guard.chat(c["anthropic"], tenant="acme", model="claude-opus-5-5",
                        max_tokens=500, messages=MSG, stream=True)
    held = month(guard).reserved_micros
    for _ in stream:
        clock.now += 20  # 26 events, 20s apart: over eight minutes in all
        if not stream.usage_event:
            assert month(guard).reserved_micros == held  # never silently dropped
    assert month(guard).reserved_micros == 0 and not stream.usage_event.estimated
