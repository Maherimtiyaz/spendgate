# spendgate

Per-customer LLM spend limits that actually hold, as a Python library. No
proxy, no gateway, no runtime dependencies.

If you sell a product built on OpenAI or Anthropic, each customer's usage
costs you money. spendgate lets you say "this customer may spend $200 a
month" and guarantees the calls stop there, even with a thousand requests in
flight, and gets every call's cost into your billing system.

```bash
pip install spendgate
```

```python
from anthropic import Anthropic
from spendgate import Guard, BudgetExceeded

guard = Guard(on_usage=lambda e: print(e.tenant, e.model, e.cost_usd))
guard.set_budget("acme", monthly_usd=200, soft_pct=80)
guard.set_budget("acme", feature="summaries", daily_usd=5)

client = Anthropic()
try:
    response = guard.chat(
        client, tenant="acme", feature="summaries",
        model="claude-opus-5-5", max_tokens=1024,
        messages=[{"role": "user", "content": "Summarise this ticket ..."}],
    )
except BudgetExceeded as exc:
    ...  # refused before the provider was called; exc.remaining_micros says what is left
```

What you get:

- **Hard budgets** per customer and per feature: daily, monthly or lifetime.
- **Your own SDK client and API keys.** Calls go straight to the provider;
  nothing sits in the network path.
- **Accurate cost** from the provider's reported usage, with cached and
  cache-write tokens priced separately.
- **Shared budgets across workers** with the PostgreSQL store.
- **Agent run ceilings** and a breaker for repeated tool calls.
- **Billing delivery** to Stripe meters or OpenMeter, at least once, with
  stable event IDs for deduplication.

Status: alpha. The API may change before 1.0.

`guard.chat` works with the synchronous OpenAI and Anthropic clients and
`await guard.achat` with the asynchronous ones. For OpenAI, pass
`api="responses"` to use the Responses API instead of Chat Completions. The
SDK response is returned unchanged.

## How a budget stays hard

Each call goes through **reserve → call → settle**:

1. **Reserve.** The worst-case cost (prompt size at the full input rate, plus
   `max_tokens` of output) is held against every budget that applies: the
   tenant's, and the feature's if one is named. If it does not fit in all of
   them, `BudgetExceeded` is raised and the provider is never called.
2. **Call.** The request goes to the provider through your own SDK client.
3. **Settle.** The real cost is computed from the response's usage (uncached,
   cache-read and cache-write tokens are priced separately) and booked; the
   rest of the hold is freed. If the provider call fails, the hold is released.

Because held amounts count against the limit, concurrent calls cannot jointly
overspend. A hold that is never settled (the process died mid-call) expires
after `reservation_ttl` seconds.

Two consequences worth knowing:

- A request with no output cap gets one (`default_max_output_tokens`, 4096),
  because an uncapped call has no cost bound to reserve.
- A model with no registered price is refused with `UnknownModelError`
  instead of being metered at $0. Add prices with
  `guard.prices.register("my-model", input="1.50", output="6.00")`.

## Wrapping calls yourself

For providers or call styles the guard does not wrap:

```python
r = guard.reserve("acme", "gpt-5.4", input_tokens=1200, max_output_tokens=500)
try:
    result = my_call()
except Exception:
    guard.release(r)
    raise
event = guard.settle(r, Usage(input_tokens=1180, output_tokens=212))
```

`UsageEvent.id` is unique per call and settling twice is a no-op, so the ID is
safe to use as an idempotency key when forwarding usage to a billing system.

## Streaming

Pass `stream=True` as you would to the SDK. You get the SDK's own events back,
and the call is settled when the stream ends:

```python
with guard.chat(client, tenant="acme", model="claude-opus-5-5", max_tokens=1024,
                messages=messages, stream=True) as stream:
    for event in stream:
        ...
print(stream.usage_event.cost_usd)
```

- With OpenAI Chat Completions the guard sets `stream_options.include_usage`,
  since that is the only way usage is reported on a stream. The extra
  usage-only chunk is hidden from you unless you asked for it yourself.
- If you stop reading early, close the stream (or use `with` / `async with`).
  The provider's final usage never arrives in that case, so the cost is
  estimated from the text received, marked `estimated=True` on the event, and
  capped at the reservation.

## Sharing budgets between workers: PostgreSQL

```bash
pip install "spendgate[postgres]"
```

```python
from spendgate import Guard, Sweeper
from spendgate.postgres import PostgresStore

store = PostgresStore("postgresql://user:pass@host/db").setup()  # creates 4 tables
guard = Guard(store)
Sweeper(store, interval=30).start()   # optional, see below
```

Every worker that points at the same database shares the same budgets, and
they survive restarts. The no-overspend guarantee is enforced by the database
(one conditional upsert per budget, in a transaction), not by a lock in your
process. With async clients, ledger queries run in a worker thread so the
event loop is not blocked.

A hold left behind by a crashed worker expires after `reservation_ttl`
seconds. Expired holds are reclaimed automatically the moment they would
otherwise cause a refusal, so `Sweeper` is optional: it only keeps the
`reserved` figures tidy for tenants that have gone quiet.

## Agent runs: stopping a runaway loop

A tenant budget limits the month; a run limits one agent task. Make the run's
calls through it:

```python
from spendgate import LoopDetected, RunLimitExceeded

with guard.run("acme", feature="research-agent", max_usd=2.00, max_steps=25) as run:
    while True:
        response = run.chat(client, model="claude-opus-5-5", max_tokens=2048,
                            messages=messages, tools=tools)
        ...
print(run.steps, run.cost_usd)
```

- `max_usd` and `max_steps` raise `RunLimitExceeded` before the call that
  would break them. Like budgets, the cost ceiling is checked against the
  call's worst case.
- The run raises `LoopDetected` when the model has requested the same tool
  with the same arguments 3 times within its last 10 tool calls
  (`loop_repeats`, `loop_window`; `loop_repeats=None` turns it off). Argument
  key order and whitespace are ignored. The exception carries the tripping
  `response`. For streamed calls the tool calls are only known when the stream
  ends, so the exception is raised by the next call instead.
- Once tripped, the run refuses every further call. Its usage events carry
  `run_id`.

Run ceilings are kept in the memory of the process driving the run.

## Getting usage into your billing system

`on_usage` callbacks are lost if the process dies between the call and the
callback. For billing, turn on the outbox: each usage event is then stored in
the same transaction that books its cost, and a dispatcher delivers it.

```python
from spendgate import CloudEventsSink, Dispatcher, Guard, StripeMeterSink
from spendgate.postgres import PostgresStore

store = PostgresStore(dsn).setup()
guard = Guard(store, outbox=True)

Dispatcher(store, [
    StripeMeterSink(stripe_key, "llm_spend", customer={"acme": "cus_123"}),
    CloudEventsSink("https://openmeter.example/api/v1/events",
                    headers={"Authorization": f"Bearer {openmeter_key}"}),
]).start()
```

- Delivery is at least once. An event stays queued until every sink has
  accepted it; a sink that already has it is not sent it again. Failures are
  retried with exponential backoff, and events that can never be delivered
  (a 4xx from the sink, or 12 failed attempts) are set aside as dead:
  `store.outbox_stats()` reports `pending` and `dead` counts.
- Every event has a stable `id`. `StripeMeterSink` sends it as Stripe's
  `identifier` and `CloudEventsSink` as the CloudEvents `id`, which is what
  those systems deduplicate on, so a redelivery does not double-bill. Stripe
  only guarantees this for 24 hours.
- `StripeMeterSink` meters cost in micro-dollars by default; pass `value=` to
  meter tokens or anything else. Tenants with no Stripe customer are skipped.
- Several dispatchers can run against one database without delivering an
  event twice between them.
- `store.dead_events()` lists dead events with their last error, and
  `store.retry_dead()` puts them back in the queue once the cause is fixed.
- Writing your own sink takes a `name` and a `send(event)` method.

## Example: a multi-tenant chat API

[`examples/fastapi_app`](examples/fastapi_app) is a small FastAPI service with
per-customer budgets, a `402` response when a customer runs out, and usage
delivered to billing. It runs without an API key against a canned model:

```bash
pip install "spendgate[dev]"
```

```bash
SPENDGATE_DEMO_MOCK=1 uvicorn app:app --factory --app-dir examples/fastapi_app
```

```bash
curl -X PUT localhost:8000/tenants/acme/budget -H 'content-type: application/json' -d '{"monthly_usd": 0.05}'
```

```bash
curl -X POST localhost:8000/chat -H 'content-type: application/json' -H 'X-Tenant-Id: acme' -d '{"message": "hi"}'
```

The third chat call returns `402` with what is left and what the call needed.
Set `SPENDGATE_DSN`, `STRIPE_API_KEY` or `USAGE_WEBHOOK_URL` to use PostgreSQL
and real billing sinks; the file's docstring lists the settings.

## Overhead

Measured with `benchmarks/overhead.py` on a 4-CPU cloud container, Python
3.13, with PostgreSQL 16 on the same machine (3,000 samples each):

| store | reserve + settle, p50 | p95 | p99 | added to a full `guard.chat` call, p50 |
|---|---|---|---|---|
| in-memory | 0.009 ms | 0.011 ms | 0.029 ms | 0.08 ms |
| PostgreSQL | 1.9 ms | 3.0 ms | 4.9 ms | 2.4 ms |

With 32 threads all drawing on a single tenant's budget, the stores handled
about 14,700 (in-memory) and 400 (PostgreSQL) reserve + settle pairs per
second. One tenant is the worst case for PostgreSQL, since every call
contends for the same row; different tenants do not block each other. A
remote database adds two network round trips per call to these figures.

## Status and limits of this version

- `Guard()` with no store keeps budgets in memory: fine for one process and
  for tests, but not shared between workers and lost on restart.
- Streaming covers `create(stream=True)` on the three supported APIs. The
  Anthropic `client.messages.stream()` helper is not wrapped.
- A running stream renews its hold while events keep arriving, so it can
  outlive `reservation_ttl`. A stream that goes silent for longer than the
  TTL loses its hold; its cost is still booked when it ends.
- Each metered call on PostgreSQL costs two transactions (reserve, settle).
  Settled reservations are kept in `spendgate_reservations` until you call
  `store.prune(older_than_seconds=...)`; schedule that yourself. Keep at least
  a day, since a kept record is what makes a repeated settle harmless.
- Prompt size is estimated offline at one token per 3 bytes, which leans high.
  `Guard(count_tokens="api")` uses Anthropic's token-counting endpoint for an
  exact figure at the price of one extra request; OpenAI calls always use the
  estimate. The estimate cannot see images or files passed by URL or ID.
- Prices in `prices.json` are standard-tier list prices retrieved on
  2026-11-18. Batch, flex, priority and fast-mode pricing, OpenAI long-context
  pricing, and per-use tool charges (such as web search) are not modelled.
- A call that times out client-side has its hold released, even though the
  provider may still bill it.

## Releases

Releases are built by a GitHub Actions workflow and published to PyPI through
trusted publishing, with no stored upload token. Each file carries a signed
provenance attestation; `SECURITY.md` shows how to verify one.

## Development

```bash
pip install -e ".[dev]"
```

```bash
pytest
```

The PostgreSQL tests are skipped unless `SPENDGATE_TEST_DSN` points at a
database they may create tables in and truncate:

```bash
SPENDGATE_TEST_DSN=postgresql://user:pass@localhost:5432/spendgate_test pytest
```
