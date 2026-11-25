"""The outbox, the dispatcher, and the sinks against a real local HTTP server."""
import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from helpers import MSG, anthropic_body, clients
from tenantbudget import (CloudEventsSink, Dispatcher, Guard, PermanentDeliveryError,
                       StripeMeterSink, Usage, UsageEvent)


class Receiver:
    """A local HTTP endpoint that records requests and answers with scripted statuses."""

    def __init__(self):
        self.requests = []
        self.statuses = []  # consumed one per request; 200 once empty
        self.body = "{}"
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                receiver.requests.append({"path": self.path, "headers": dict(self.headers),
                                          "body": raw.decode()})
                status = receiver.statuses.pop(0) if receiver.statuses else 200
                self.send_response(status)
                self.end_headers()
                self.wfile.write(receiver.body.encode())

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def receiver():
    r = Receiver()
    yield r
    r.close()


class ListSink:
    def __init__(self, name="list", fail=()):
        self.name = name
        self.got = []
        self.fail = list(fail)  # exceptions to raise, one per call, before succeeding

    def send(self, event):
        if self.fail:
            raise self.fail.pop(0)
        self.got.append(event)


@pytest.fixture
def store(make_store, clock):
    return make_store(clock)


@pytest.fixture
def guard(store, clock):
    g = Guard(store, outbox=True, clock=clock)
    g.set_budget("acme", monthly_usd=1)
    return g


def make_call(guard, tenant="acme"):
    r = guard.reserve(tenant, "gpt-5.4", input_tokens=1000, max_output_tokens=500)
    return guard.settle(r, Usage(input_tokens=900, output_tokens=100))


def test_settled_call_queues_exactly_one_event(guard, store):
    event = make_call(guard)
    r = guard.reserve("acme", "gpt-5.4", input_tokens=10, max_output_tokens=10)
    guard.release(r)  # a released call is not usage
    (item,) = store.claim_events(10, lease_seconds=60)
    assert UsageEvent.from_dict(item.payload) == event
    assert item.id == event.id and item.attempts == 1


def test_no_events_are_queued_unless_the_outbox_is_on(store, clock):
    guard = Guard(store, clock=clock)
    make_call(guard)
    assert store.outbox_stats() == {"pending": 0, "dead": 0}


def test_repeated_settle_does_not_queue_a_second_event(guard, store):
    r = guard.reserve("acme", "gpt-5.4", input_tokens=1000, max_output_tokens=500)
    guard.settle(r, Usage(input_tokens=900, output_tokens=100))
    guard.settle(r, Usage(input_tokens=900, output_tokens=100))
    assert store.outbox_stats()["pending"] == 1


def test_dispatcher_delivers_in_order_and_empties_the_outbox(guard, store, clock):
    sent = [make_call(guard) for _ in range(5)]
    sink = ListSink()
    assert Dispatcher(store, [sink], clock=clock).run_once() == 5
    assert sink.got == sent
    assert store.outbox_stats() == {"pending": 0, "dead": 0}


def test_failed_delivery_is_retried_after_a_backoff(guard, store, clock):
    event = make_call(guard)
    sink = ListSink(fail=[ConnectionError("down"), ConnectionError("down")])
    dispatcher = Dispatcher(store, [sink], clock=clock, backoff_base=5)

    assert dispatcher.run_once() == 0
    assert dispatcher.run_once() == 0 and sink.fail  # not due yet: no second attempt
    clock.now += 5
    assert dispatcher.run_once() == 0  # second failure, backoff doubles
    clock.now += 5
    assert dispatcher.run_once() == 0
    clock.now += 5
    assert dispatcher.run_once() == 1
    assert sink.got == [event]


def test_each_sink_gets_an_event_once_even_when_another_sink_fails(guard, store, clock):
    make_call(guard)
    good, flaky = ListSink("good"), ListSink("flaky", fail=[TimeoutError()])
    dispatcher = Dispatcher(store, [good, flaky], clock=clock)
    dispatcher.run_once()
    assert (len(good.got), len(flaky.got)) == (1, 0)
    clock.now += 60
    assert dispatcher.run_once() == 1
    assert (len(good.got), len(flaky.got)) == (1, 1)  # "good" was not sent it again


def test_permanent_failure_is_dead_lettered_without_retries(guard, store, clock):
    make_call(guard)
    sink = ListSink(fail=[PermanentDeliveryError("HTTP 400: no such meter")])
    dispatcher = Dispatcher(store, [sink], clock=clock)
    dispatcher.run_once()
    clock.now += 10_000
    assert dispatcher.run_once() == 0 and sink.got == []
    assert store.outbox_stats() == {"pending": 0, "dead": 1}


def test_event_is_dead_lettered_after_max_attempts(guard, store, clock):
    make_call(guard)
    sink = ListSink(fail=[ConnectionError()] * 10)
    dispatcher = Dispatcher(store, [sink], clock=clock, max_attempts=3)
    for _ in range(5):
        dispatcher.run_once()
        clock.now += 10_000
    assert len(sink.fail) == 7  # tried exactly three times
    assert store.outbox_stats() == {"pending": 0, "dead": 1}


def test_claimed_event_is_hidden_until_its_lease_runs_out(guard, store, clock):
    make_call(guard)
    assert len(store.claim_events(10, lease_seconds=60)) == 1  # this dispatcher then dies
    assert store.claim_events(10, lease_seconds=60) == []
    clock.now += 61
    (item,) = store.claim_events(10, lease_seconds=60)
    assert item.attempts == 2


def test_concurrent_dispatchers_do_not_deliver_an_event_twice(guard, store, clock):
    for _ in range(200):
        make_call(guard)
    sink = ListSink()
    start = threading.Barrier(4)

    def work():
        dispatcher = Dispatcher(store, [sink], clock=clock, batch_size=7)
        start.wait()
        while dispatcher.run_once():
            pass

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(sink.got) == 200 == len({e.id for e in sink.got})


def test_background_dispatcher_delivers_end_to_end(guard, store):
    sink = ListSink()
    delivered = threading.Event()
    sink.send = lambda event: (sink.got.append(event), delivered.set())
    _, c = clients({"/v1/messages": anthropic_body()})
    with Dispatcher(store, [sink], interval=0.01):
        guard.chat(c["anthropic"], tenant="acme", model="claude-opus-5-5", max_tokens=100,
                   messages=MSG)
        assert delivered.wait(5)
    assert sink.got[0].cost_micros == 1400 and sink.got[0].provider == "anthropic"


# -- sinks ------------------------------------------------------------------

EVENT = UsageEvent(
    id="e" * 32, tenant="acme", feature="chat", provider="openai", model="gpt-5.4",
    usage=Usage(input_tokens=900, output_tokens=100, cache_read_tokens=50),
    cost_micros=3763, reserved_micros=10_000, estimated=False,
    timestamp=1_800_000_000.25, run_id="run-1")


def test_cloudevents_sink_sends_a_valid_cloudevent(receiver):
    CloudEventsSink(receiver.url + "/api/v1/events", source="billing-svc",
                    headers={"Authorization": "Bearer om-key"}).send(EVENT)
    (request,) = receiver.requests
    doc = json.loads(request["body"])
    assert request["path"] == "/api/v1/events"
    assert request["headers"]["Content-Type"] == "application/cloudevents+json"
    assert request["headers"]["Authorization"] == "Bearer om-key"
    assert {k: doc[k] for k in ("specversion", "id", "source", "type", "subject", "time")} == {
        "specversion": "1.0", "id": "e" * 32, "source": "billing-svc",
        "type": "tenantbudget.usage", "subject": "acme", "time": "2027-01-15T08:00:00.250Z"}
    assert doc["data"]["cost_micros"] == 3763 and doc["data"]["cost_usd"] == "0.003763"
    assert doc["data"]["input_tokens"] == 900 and doc["data"]["run_id"] == "run-1"


def test_stripe_sink_sends_a_meter_event_with_the_event_id_as_identifier(receiver):
    StripeMeterSink("sk_test_123", "llm_spend", {"acme": "cus_ABC"},
                    base_url=receiver.url).send(EVENT)
    (request,) = receiver.requests
    assert request["path"] == "/v1/billing/meter_events"
    assert request["headers"]["Authorization"] == "Bearer sk_test_123"
    assert dict(urllib.parse.parse_qsl(request["body"])) == {
        "event_name": "llm_spend", "identifier": "e" * 32, "timestamp": "1800000000",
        "payload[stripe_customer_id]": "cus_ABC", "payload[value]": "3763"}


def test_stripe_sink_skips_tenants_with_no_customer_and_custom_values(receiver):
    sink = StripeMeterSink("sk", "tokens", lambda e: None if e.tenant == "acme" else "cus_X",
                           value=lambda e: e.usage.prompt_tokens + e.usage.output_tokens,
                           base_url=receiver.url)
    sink.send(EVENT)
    assert receiver.requests == []
    from dataclasses import replace
    sink.send(replace(EVENT, tenant="globex"))
    assert "payload%5Bvalue%5D=1050" in receiver.requests[0]["body"]


@pytest.mark.parametrize("status,permanent", [(500, False), (429, False), (400, True),
                                              (401, True)])
def test_http_errors_are_classified_as_retryable_or_permanent(receiver, status, permanent):
    receiver.statuses = [status]
    receiver.body = '{"error": {"message": "No meter found"}}'
    sink = StripeMeterSink("sk", "llm_spend", {"acme": "cus_ABC"}, base_url=receiver.url)
    with pytest.raises(Exception) as exc:
        sink.send(EVENT)
    assert isinstance(exc.value, PermanentDeliveryError) is permanent
    assert f"HTTP {status}" in str(exc.value)


def test_unreachable_sink_is_retryable():
    sink = CloudEventsSink("http://127.0.0.1:9/events", timeout=2)
    with pytest.raises(Exception) as exc:
        sink.send(EVENT)
    assert not isinstance(exc.value, PermanentDeliveryError)


def test_cloudevents_conflict_counts_as_already_delivered(receiver):
    receiver.statuses = [409]
    CloudEventsSink(receiver.url).send(EVENT)


def test_outbox_to_http_sink_end_to_end(guard, store, clock, receiver):
    event = make_call(guard)
    receiver.statuses = [503]  # the sink is down for the first attempt
    dispatcher = Dispatcher(store, [CloudEventsSink(receiver.url)], clock=clock)
    assert dispatcher.run_once() == 0
    clock.now += 5
    assert dispatcher.run_once() == 1
    ids = [json.loads(r["body"])["id"] for r in receiver.requests]
    assert ids == [event.id, event.id]  # same ID both times: the receiver can deduplicate


def test_dead_events_can_be_inspected_and_retried(guard, store, clock):
    first, second = make_call(guard), make_call(guard)
    broken = ListSink(fail=[PermanentDeliveryError("HTTP 400: no such meter")] * 2)
    Dispatcher(store, [broken], clock=clock).run_once()

    dead = store.dead_events()
    assert [item.id for item, _ in dead] == [first.id, second.id]
    assert "no such meter" in dead[0][1]

    assert store.retry_dead([first.id]) == 1  # the meter has been created since
    fixed = ListSink()
    assert Dispatcher(store, [fixed], clock=clock).run_once() == 1
    assert fixed.got == [first]
    assert store.outbox_stats() == {"pending": 0, "dead": 1}

    assert store.retry_dead() == 1
    Dispatcher(store, [fixed], clock=clock).run_once()
    assert fixed.got == [first, second] and store.dead_events() == []
