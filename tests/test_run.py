"""Agent runs: cost and step ceilings, and the loop breaker."""
import pytest

from helpers import MSG, anthropic_body, chat_body, clients, sse
from spendgate import BudgetExceeded, Guard, LoopDetected, RunLimitExceeded, Usage

OPUS = {"model": "claude-opus-5-5", "max_tokens": 100, "messages": MSG}


def tool_reply(name="search", args=None, **usage):
    body = anthropic_body(**usage)
    body["content"] = [{"type": "tool_use", "id": "tu_1", "name": name,
                        "input": args if args is not None else {"q": "weather", "n": 1}}]
    body["stop_reason"] = "tool_use"
    return body


def chat_tool_reply(arguments='{"n": 1, "q": "weather"}'):
    body = chat_body()
    body["choices"][0]["message"] = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function",
         "function": {"name": "search", "arguments": arguments}}]}
    return body


def month(guard):
    return next(s for s in guard.status("acme") if s.key.window == "month")


def test_run_tracks_steps_and_cost_and_tags_events(guard, events):
    _, c = clients({"/v1/messages": anthropic_body()})
    with guard.run("acme", feature="agent", max_usd=1, max_steps=10) as run:
        run.chat(c["anthropic"], **OPUS)
        run.chat(c["anthropic"], **OPUS)
    assert (run.steps, run.spent_micros) == (2, 2 * 1400)
    assert {e.run_id for e in events} == {run.id}
    assert {e.feature for e in events} == {"agent"}


def test_step_ceiling_stops_the_run_before_the_next_call(guard):
    server, c = clients({"/v1/messages": anthropic_body()})
    run = guard.run("acme", max_steps=2)
    run.chat(c["anthropic"], **OPUS)
    run.chat(c["anthropic"], **OPUS)
    with pytest.raises(RunLimitExceeded) as exc:
        run.chat(c["anthropic"], **OPUS)
    assert exc.value.kind == "steps" and len(server.requests) == 2


def test_cost_ceiling_refuses_a_call_whose_worst_case_does_not_fit(guard):
    # Each call holds about 2030 micros and really costs 1400.
    server, c = clients({"/v1/messages": anthropic_body()})
    run = guard.run("acme", max_usd="0.004")
    run.chat(c["anthropic"], **OPUS)
    run.chat(c["anthropic"], **OPUS)  # 2800 spent
    with pytest.raises(RunLimitExceeded) as exc:
        run.chat(c["anthropic"], **OPUS)  # 2800 + ~2030 > 4000
    assert exc.value.kind == "cost" and exc.value.used == 2800
    assert len(server.requests) == 2 and run.spent_micros <= run.max_micros
    assert month(guard).reserved_micros == 0  # nothing left held on the tenant


def test_run_ceiling_is_separate_from_the_tenant_budget(guard):
    _, c = clients({"/v1/messages": anthropic_body()})
    tight = guard.run("acme", max_usd="0.001")
    with pytest.raises(RunLimitExceeded):
        tight.chat(c["anthropic"], **OPUS)
    guard.chat(c["anthropic"], tenant="acme", **OPUS)  # the tenant itself has room

    guard.set_budget("acme", monthly_usd="0.002")  # 1400 already spent
    roomy = guard.run("acme", max_usd=100)
    with pytest.raises(BudgetExceeded):
        roomy.chat(c["anthropic"], **OPUS)
    assert roomy.steps == 0  # a call the tenant budget refused is not a step


def test_failed_call_is_not_counted_as_a_step(guard):
    _, c = clients({"/v1/messages": 500})
    run = guard.run("acme", max_steps=1)
    with pytest.raises(Exception):
        run.chat(c["anthropic"], **OPUS)
    assert (run.steps, run.spent_micros) == (0, 0)


def test_loop_breaker_trips_on_the_third_identical_tool_call(guard, events):
    server, c = clients({"/v1/messages": tool_reply()})
    run = guard.run("acme")
    run.chat(c["anthropic"], **OPUS)
    run.chat(c["anthropic"], **OPUS)
    with pytest.raises(LoopDetected) as exc:
        run.chat(c["anthropic"], **OPUS)
    assert (exc.value.tool, exc.value.count) == ("search", 3)
    assert exc.value.response.content[0].name == "search"  # the response is not lost
    assert len(events) == 3  # the tripping call was made and is billed

    with pytest.raises(LoopDetected):  # the breaker stays open
        run.chat(c["anthropic"], **OPUS)
    assert len(server.requests) == 3


def test_argument_key_order_and_spacing_do_not_hide_a_loop(guard):
    replies = iter(['{"q": "weather", "n": 1}', '{"n":1,"q":"weather"}',
                    '{ "n": 1, "q": "weather" }'])
    server, c = clients({"/v1/chat/completions": chat_tool_reply()})
    original = server.routes
    server.routes = type("R", (dict,), {"__getitem__": lambda s, k: chat_tool_reply(next(replies))})(original)
    run = guard.run("acme")
    with pytest.raises(LoopDetected):
        for _ in range(3):
            run.chat(c["openai"], model="gpt-5.4", messages=MSG)


def test_different_arguments_are_not_a_loop(guard):
    args = iter([{"q": "a"}, {"q": "b"}, {"q": "c"}, {"q": "d"}])
    server, c = clients({})
    server.routes = type("R", (dict,), {"__getitem__": lambda s, k: tool_reply(args=next(args))})()
    run = guard.run("acme")
    for _ in range(4):
        run.chat(c["anthropic"], **OPUS)
    assert run.steps == 4


def test_old_repeats_fall_out_of_the_window(guard):
    names = iter(["a", "b", "c", "a", "d", "e", "a"])
    server, c = clients({})
    server.routes = type("R", (dict,), {"__getitem__": lambda s, k: tool_reply(name=next(names))})()
    run = guard.run("acme", loop_repeats=3, loop_window=4)
    for _ in range(7):  # "a" three times, but never three within any 4 calls
        run.chat(c["anthropic"], **OPUS)


def test_loop_detection_can_be_turned_off(guard):
    _, c = clients({"/v1/messages": tool_reply()})
    run = guard.run("acme", loop_repeats=None)
    for _ in range(5):
        run.chat(c["anthropic"], **OPUS)


def test_streamed_tool_calls_trip_the_breaker_on_the_next_call(guard):
    stream = sse(
        {"type": "message_start", "message": {
            "id": "m", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {
            "type": "tool_use", "id": "tu", "name": "search", "input": {}}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": '{"q": "wea'}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": 'ther"}'}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
         "usage": {"output_tokens": 20}},
        {"type": "message_stop"})
    server, c = clients({"/v1/messages": stream})
    run = guard.run("acme")
    for _ in range(3):
        list(run.chat(c["anthropic"], stream=True, **OPUS))
    with pytest.raises(LoopDetected) as exc:
        run.chat(c["anthropic"], stream=True, **OPUS)
    assert exc.value.arguments == '{"q":"weather"}' and len(server.requests) == 3


async def test_async_run(guard, events):
    _, c = clients({"/v1/messages": tool_reply()})
    async with guard.run("acme", max_steps=5) as run:
        await run.achat(c["anthropic_async"], **OPUS)
        await run.achat(c["anthropic_async"], **OPUS)
        with pytest.raises(LoopDetected):
            await run.achat(c["anthropic_async"], **OPUS)
    assert run.steps == 3 and len(events) == 3


def test_manual_reserve_and_tool_recording_in_a_run(guard):
    run = guard.run("acme", max_usd="0.02", max_steps=3)
    r = run.reserve("gpt-5.4", input_tokens=1000, max_output_tokens=500)  # holds 10_000
    with pytest.raises(RunLimitExceeded):
        run.reserve("gpt-5.4", input_tokens=1000, max_output_tokens=1000)  # needs 17_500
    guard.settle(r, Usage(input_tokens=1000, output_tokens=100))
    assert (run.steps, run.spent_micros) == (1, 4000)

    released = run.reserve("gpt-5.4", input_tokens=10, max_output_tokens=10)
    guard.release(released)
    assert run.steps == 1

    run.record_tool_call("fetch", {"url": "x"})
    run.record_tool_call("fetch", '{"url": "x"}')
    with pytest.raises(LoopDetected):
        run.record_tool_call("fetch", {"url": "x"})
