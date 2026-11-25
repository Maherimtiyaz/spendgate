"""End-to-end tests through the real OpenAI and Anthropic SDK clients."""
import anthropic
import pytest

from helpers import MSG, anthropic_body, chat_body, clients, responses_body
from tenantbudget import BudgetExceeded, Guard, UnknownModelError, Usage


def month(guard, tenant="acme"):
    return next(s for s in guard.status(tenant) if s.key.window == "month")


def test_anthropic_call_is_metered_with_cache_token_classes(guard, events):
    body = anthropic_body(cache_read_input_tokens=1000, cache_creation_input_tokens=300,
                          cache_creation={"ephemeral_5m_input_tokens": 200,
                                          "ephemeral_1h_input_tokens": 100})
    _, c = clients({"/v1/messages": body})
    response = guard.chat(c["anthropic"], tenant="acme", model="claude-opus-5-5",
                          max_tokens=1024, messages=MSG)

    assert response.content[0].text == "hi"  # the SDK response comes back untouched
    (event,) = events
    assert event.usage == Usage(input_tokens=100, output_tokens=50, cache_read_tokens=1000,
                                cache_write_5m_tokens=200, cache_write_1h_tokens=100)
    assert event.cost_micros == 100 * 4 + 50 * 20 + 200 + 200 * 5 + 100 * 8
    assert (event.tenant, event.provider, event.estimated) == ("acme", "anthropic", False)
    assert month(guard).spent_micros == event.cost_micros
    assert month(guard).reserved_micros == 0


def test_openai_chat_separates_cached_from_uncached_input(guard, events):
    body = chat_body(prompt_tokens=1000, prompt_tokens_details={"cached_tokens": 600})
    _, c = clients({"/v1/chat/completions": body})
    guard.chat(c["openai"], tenant="acme", model="gpt-5.4", messages=MSG,
               max_completion_tokens=200)
    assert events[0].usage == Usage(input_tokens=400, output_tokens=50, cache_read_tokens=600)
    assert events[0].cost_micros == 1000 + 150 + 750  # 400*2.5 + 600*0.25 + 50*15


def test_openai_responses_api(guard, events):
    body = responses_body(input_tokens=1000, input_tokens_details={"cached_tokens": 200})
    _, c = clients({"/v1/responses": body})
    guard.chat(c["openai"], api="responses", tenant="acme", model="gpt-5.4", input="hello",
               max_output_tokens=200)
    assert events[0].usage == Usage(input_tokens=800, output_tokens=50, cache_read_tokens=200)


async def test_async_clients(guard, events):
    _, c = clients({"/v1/messages": anthropic_body(), "/v1/chat/completions": chat_body()})
    await guard.achat(c["anthropic_async"], tenant="acme", model="claude-opus-5-5",
                      max_tokens=100, messages=MSG)
    await guard.achat(c["openai_async"], tenant="acme", model="gpt-5.4", messages=MSG)
    assert [e.provider for e in events] == ["anthropic", "openai"]
    assert month(guard).spent_micros == sum(e.cost_micros for e in events)


def test_call_over_budget_is_refused_before_reaching_the_provider(guard, events):
    server, c = clients({"/v1/messages": anthropic_body()})
    # 60k output tokens at $20/MTok is $1.20 worst case against a $1 budget.
    with pytest.raises(BudgetExceeded) as exc:
        guard.chat(c["anthropic"], tenant="acme", model="claude-opus-5-5",
                   max_tokens=60_000, messages=MSG)
    assert exc.value.tenant == "acme" and exc.value.window == "month"
    assert server.requests == [] and events == []
    assert month(guard).reserved_micros == 0


def test_spend_accumulates_until_the_budget_blocks(events):
    guard = Guard(on_usage=events.append)
    guard.set_budget("acme", monthly_usd="0.01")  # 10_000 micros
    _, c = clients({"/v1/messages": anthropic_body(input_tokens=500, output_tokens=200)})
    made = 0
    with pytest.raises(BudgetExceeded):
        for _ in range(20):
            guard.chat(c["anthropic"], tenant="acme", model="claude-opus-5-5",
                       max_tokens=200, messages=MSG)
            made += 1
    assert made == 1  # each call costs 6000 micros; a second cannot fit
    assert month(guard).spent_micros == 6000 <= month(guard).limit_micros


def test_provider_error_releases_the_reservation(guard, events):
    _, c = clients({"/v1/messages": 500})
    with pytest.raises(anthropic.APIStatusError):
        guard.chat(c["anthropic"], tenant="acme", model="claude-opus-5-5",
                   max_tokens=100, messages=MSG)
    assert events == []
    assert (month(guard).reserved_micros, month(guard).spent_micros) == (0, 0)


def test_missing_output_cap_is_added_so_the_reservation_is_a_real_bound(guard):
    server, c = clients({"/v1/chat/completions": chat_body(), "/v1/messages": anthropic_body()})
    guard.chat(c["openai"], tenant="acme", model="gpt-5.4", messages=MSG)
    assert server.requests[0][1]["max_completion_tokens"] == 4096
    guard.chat(c["openai"], tenant="acme", model="gpt-5.4", messages=MSG, max_tokens=10)
    assert "max_completion_tokens" not in server.requests[1][1]


def test_n_choices_multiply_the_reserved_output(guard, events):
    _, c = clients({"/v1/chat/completions": chat_body()})
    guard.chat(c["openai"], tenant="acme", model="gpt-5.4", messages=MSG,
               max_completion_tokens=1000, n=3)
    assert events[0].reserved_micros >= 3 * 1000 * 15


def test_cache_control_reserves_at_the_cache_write_rate(guard, events):
    _, c = clients({"/v1/messages": anthropic_body()})
    system = [{"type": "text", "text": "x" * 3000, "cache_control": {"type": "ephemeral"}}]
    guard.chat(c["anthropic"], tenant="acme", model="claude-opus-5-5", max_tokens=1,
               messages=MSG, system=system)
    guard.chat(c["anthropic"], tenant="acme", model="claude-opus-5-5", max_tokens=1,
               messages=MSG, system=[{"type": "text", "text": "x" * 3000}])
    assert events[0].reserved_micros > 1.9 * events[1].reserved_micros


def test_exact_token_counting_uses_the_provider_counter(events):
    guard = Guard(on_usage=events.append, count_tokens="api")
    server, c = clients({"/v1/messages/count_tokens": {"input_tokens": 7},
                         "/v1/messages": anthropic_body()})
    guard.chat(c["anthropic"], tenant="acme", model="claude-opus-5-5", max_tokens=10,
               messages=MSG)
    assert [path for path, _ in server.requests] == ["/v1/messages/count_tokens", "/v1/messages"]
    assert events[0].reserved_micros == 7 * 4 + 10 * 20


def test_served_model_is_billed_when_it_differs_from_the_requested_one(guard, events):
    _, c = clients({"/v1/messages": anthropic_body(model="claude-opus-4-8")})
    guard.chat(c["anthropic"], tenant="acme", model="claude-opus-5-5", max_tokens=100,
               messages=MSG)
    assert events[0].model == "claude-opus-4-8"
    assert events[0].cost_micros == 100 * 5 + 50 * 25


def test_unknown_model_is_refused_before_the_call(guard):
    server, c = clients({"/v1/chat/completions": chat_body()})
    with pytest.raises(UnknownModelError):
        guard.chat(c["openai"], tenant="acme", model="some-new-model", messages=MSG)
    assert server.requests == []


def test_tenant_without_budget_is_still_metered(guard, events):
    _, c = clients({"/v1/messages": anthropic_body()})
    guard.chat(c["anthropic"], tenant="free-tier", model="claude-opus-5-5", max_tokens=100,
               messages=MSG)
    assert events[0].tenant == "free-tier" and events[0].cost_micros == 1400


def test_feature_budget_blocks_only_that_feature(guard):
    guard.set_budget("acme", feature="summaries", monthly_usd="0.001")
    _, c = clients({"/v1/messages": anthropic_body()})
    with pytest.raises(BudgetExceeded) as exc:
        guard.chat(c["anthropic"], tenant="acme", feature="summaries",
                   model="claude-opus-5-5", max_tokens=1000, messages=MSG)
    assert exc.value.feature == "summaries"
    guard.chat(c["anthropic"], tenant="acme", feature="chat", model="claude-opus-5-5",
               max_tokens=1000, messages=MSG)


def test_soft_limit_callback_fires_once_and_listener_errors_are_contained(events):
    alerts = []

    def alert(status):
        alerts.append(status)
        raise RuntimeError("pager is down")

    guard = Guard(on_usage=events.append, on_soft_limit=alert)
    guard.set_budget("acme", monthly_usd="0.01", soft_pct=50)
    _, c = clients({"/v1/messages": anthropic_body(input_tokens=500, output_tokens=200)})
    guard.chat(c["anthropic"], tenant="acme", model="claude-opus-5-5", max_tokens=200,
               messages=MSG)  # 6000 of 10000 micros: crosses 50%
    assert len(alerts) == 1 and alerts[0].spent_micros == 6000


def test_manual_reserve_settle_for_calls_the_guard_does_not_wrap(guard, events):
    r = guard.reserve("acme", "gpt-5.4", input_tokens=1000, max_output_tokens=500)
    assert month(guard).reserved_micros == 2500 + 7500
    event = guard.settle(r, Usage(input_tokens=900, output_tokens=100))
    assert event.cost_micros == 2250 + 1500
    assert guard.settle(r, Usage(input_tokens=900, output_tokens=100)).id == event.id
    assert len(events) == 1  # the repeat was not reported or booked twice
    assert month(guard).spent_micros == 3750


def test_missing_usage_books_the_full_reservation(guard, events):
    r = guard.reserve("acme", "gpt-5.4", input_tokens=1000, max_output_tokens=500)
    event = guard.settle(r, None)
    assert event.estimated and event.cost_micros == r.amount_micros == 10_000
