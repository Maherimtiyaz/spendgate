"""How much latency does spendgate add to a model call?

    python benchmarks/overhead.py                      # in-memory store
    SPENDGATE_DSN=postgresql://... python benchmarks/overhead.py

Two measurements per store:

* ledger: one reserve + settle pair, which is everything spendgate does around
  a call that touches the store.
* guard.chat: a full metered call through the Anthropic SDK against an
  in-process mock transport, minus the same call without spendgate. This adds
  the request preparation, token estimate and usage parsing to the ledger time.

The Postgres numbers depend mostly on the round trip to your database.
"""
from __future__ import annotations

import importlib
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import anthropic

from spendgate import Guard, InMemoryStore

N = int(os.environ.get("BENCH_N", "3000"))
MODEL = "claude-opus-5-5"
MESSAGES = [{"role": "user", "content": "Summarise the attached ticket. " * 40}]

_http = importlib.import_module(
    anthropic.DefaultHttpxClient.__mro__[1].__module__.split(".")[0])
_BODY = {"id": "msg", "type": "message", "role": "assistant", "model": MODEL,
         "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
         "stop_sequence": None, "usage": {"input_tokens": 300, "output_tokens": 200}}


def mock_client() -> anthropic.Anthropic:
    transport = _http.MockTransport(lambda request: _http.Response(200, json=_BODY))
    return anthropic.Anthropic(api_key="bench", max_retries=0,
                               http_client=_http.Client(transport=transport))


def timed(fn, n=N, warmup=200):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return samples


def summary(samples):
    cuts = statistics.quantiles(samples, n=100)
    return statistics.median(samples), cuts[94], cuts[98]


def bench(name, store):
    guard = Guard(store)
    guard.set_budget("bench", total_usd=1_000_000)
    client = mock_client()
    request = dict(model=MODEL, max_tokens=1024, messages=MESSAGES)

    def ledger():
        r = store.reserve("bench", None, 5000, 600)
        store.settle(r.id, 3000)

    ledger_ms = summary(timed(ledger))
    bare = statistics.median(timed(lambda: client.messages.create(**request)))
    metered = summary(timed(lambda: guard.chat(client, tenant="bench", **request)))

    workers, per_worker = 32, max(N // 32, 20)
    start = time.perf_counter()
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda _: [ledger() for _ in range(per_worker)], range(workers)))
    throughput = workers * per_worker / (time.perf_counter() - start)

    return {"store": name,
            "ledger_p50": ledger_ms[0], "ledger_p95": ledger_ms[1], "ledger_p99": ledger_ms[2],
            "chat_added_p50": metered[0] - bare, "bare_sdk_p50": bare,
            "throughput": throughput}


def main():
    results = [bench("in-memory", InMemoryStore())]
    dsn = os.environ.get("SPENDGATE_DSN")
    if dsn:
        from spendgate.postgres import PostgresStore
        store = PostgresStore(dsn, max_connections=32).setup()
        results.append(bench("postgres", store))
        store.close()

    print(f"python {sys.version.split()[0]}, {os.cpu_count()} CPUs, {N} samples per measurement\n")
    print("| store | reserve+settle p50 | p95 | p99 | added to a chat call, p50 | "
          "reserve+settle pairs/s, 32 threads |")
    print("|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['store']} | {r['ledger_p50']:.3f} ms | {r['ledger_p95']:.3f} ms | "
              f"{r['ledger_p99']:.3f} ms | {r['chat_added_p50']:.3f} ms | "
              f"{r['throughput']:,.0f} |")
    print(f"\nbare SDK call against the mock transport, p50: {results[0]['bare_sdk_p50']:.3f} ms")
    if os.environ.get("BENCH_JSON"):
        print(json.dumps(results))


if __name__ == "__main__":
    main()
