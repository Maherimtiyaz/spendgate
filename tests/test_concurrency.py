"""The headline guarantee: many simultaneous calls cannot jointly overspend."""
import asyncio

from helpers import MSG, chat_body, clients
from spendgate import BudgetExceeded, Guard

CALLS = 1000


async def test_1000_parallel_calls_against_a_one_dollar_budget(make_store):
    events = []
    guard = Guard(make_store(), on_usage=events.append)
    guard.set_budget("acme", monthly_usd=1)
    # Each call really costs 20*2.5 + 500*15 = 7550 micros, so $1 covers 132.
    server, c = clients({"/v1/chat/completions": chat_body(prompt_tokens=20,
                                                           completion_tokens=500)})

    async def call():
        try:
            await guard.achat(c["openai_async"], tenant="acme", model="gpt-5.4",
                              messages=MSG, max_completion_tokens=500)
        except BudgetExceeded:
            return False
        return True

    results = await asyncio.gather(*(call() for _ in range(CALLS)))

    (status,) = guard.status("acme")
    made = sum(results)
    assert status.spent_micros <= status.limit_micros == 1_000_000
    assert status.spent_micros == made * 7550 == sum(e.cost_micros for e in events)
    assert status.reserved_micros == 0
    # Refused calls never reached the provider.
    assert len(server.requests) == made == len(events)
    # The budget was used, not just protected: no further call would have fit.
    assert status.remaining_micros < events[0].reserved_micros
