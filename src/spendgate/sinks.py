"""Billing sinks: where a Dispatcher sends usage events. Standard library only."""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from .errors import DeliveryError, PermanentDeliveryError
from .guard import UsageEvent

# Worth retrying: timeouts, throttling, and anything the server got wrong.
_RETRYABLE = {408, 425, 429}


def _post(url: str, body: bytes, headers: Mapping[str, str], timeout: float) -> None:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            return
    except urllib.error.HTTPError as exc:
        detail = exc.read(2000).decode("utf-8", "replace")
        message = f"HTTP {exc.code}: {detail}"
        if exc.code in _RETRYABLE or exc.code >= 500:
            raise DeliveryError(message) from None
        error = PermanentDeliveryError(message)
        error.status, error.body = exc.code, detail
        raise error from None
    except (urllib.error.URLError, OSError) as exc:  # includes timeouts
        raise DeliveryError(f"could not reach {url}: {exc}") from None


def event_data(event: UsageEvent) -> dict[str, Any]:
    """The flat set of properties sent as an event's data."""
    return {
        "tenant": event.tenant, "feature": event.feature, "run_id": event.run_id,
        "provider": event.provider, "model": event.model,
        "input_tokens": event.usage.input_tokens,
        "output_tokens": event.usage.output_tokens,
        "cache_read_tokens": event.usage.cache_read_tokens,
        "cache_write_5m_tokens": event.usage.cache_write_5m_tokens,
        "cache_write_1h_tokens": event.usage.cache_write_1h_tokens,
        "cost_micros": event.cost_micros,
        "cost_usd": f"{event.cost_micros / 1e6:.6f}",
        "estimated": event.estimated,
    }


class CloudEventsSink:
    """POSTs each event as a CloudEvents 1.0 JSON document.

    This is the format OpenMeter ingests, and a reasonable webhook format for
    anything else. The event ID is the CloudEvents ``id`` and the tenant is
    the ``subject``; receivers that deduplicate on ``source`` + ``id``, as
    OpenMeter does, make redelivery harmless. HTTP 409 is treated as "already
    have it".
    """

    def __init__(self, url: str, *, source: str = "spendgate", type: str = "spendgate.usage",
                 headers: Mapping[str, str] | None = None, name: str = "cloudevents",
                 timeout: float = 10.0):
        self.name = name
        self._url = url
        self._source = source
        self._type = type
        self._headers = {"Content-Type": "application/cloudevents+json", **(headers or {})}
        self._timeout = timeout

    def send(self, event: UsageEvent) -> None:
        moment = datetime.fromtimestamp(event.timestamp, tz=timezone.utc)
        document = {
            "specversion": "1.0", "id": event.id, "source": self._source, "type": self._type,
            "time": moment.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "subject": event.tenant, "datacontenttype": "application/json",
            "data": event_data(event),
        }
        try:
            _post(self._url, json.dumps(document).encode(), self._headers, self._timeout)
        except PermanentDeliveryError as exc:
            if getattr(exc, "status", None) != 409:
                raise


class StripeMeterSink:
    """Records each event as a Stripe billing meter event.

    ``customer`` maps a tenant to its Stripe customer ID: a dict, or a
    function of the event. Events for tenants with no customer are skipped.
    ``value`` is the number Stripe adds to the meter, by default the cost in
    micro-dollars (so a price of $0.000001 per unit bills cost at par; set
    your markup in the Stripe price). It must be a whole number.

    The event ID is sent as Stripe's ``identifier``, which Stripe uses to
    reject duplicates for at least 24 hours, so redelivery within that time
    does not double-bill.
    """

    def __init__(self, api_key: str, event_name: str,
                 customer: Mapping[str, str] | Callable[[UsageEvent], str | None], *,
                 value: Callable[[UsageEvent], int] = lambda event: event.cost_micros,
                 name: str = "stripe", base_url: str = "https://api.stripe.com",
                 timeout: float = 10.0):
        self.name = name
        self._event_name = event_name
        self._customer = customer if callable(customer) else (
            lambda event: customer.get(event.tenant))
        self._value = value
        self._url = base_url.rstrip("/") + "/v1/billing/meter_events"
        self._headers = {"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/x-www-form-urlencoded"}
        self._timeout = timeout

    def send(self, event: UsageEvent) -> None:
        customer = self._customer(event)
        value = int(self._value(event))
        if not customer or value <= 0:
            return  # nothing to bill
        form = urllib.parse.urlencode({
            "event_name": self._event_name,
            "identifier": event.id,
            "timestamp": int(event.timestamp),
            "payload[stripe_customer_id]": customer,
            "payload[value]": value,
        }).encode()
        try:
            _post(self._url, form, self._headers, self._timeout)
        except PermanentDeliveryError as exc:
            # A retry of an event Stripe already has is a success, not a failure.
            if not _is_duplicate(getattr(exc, "body", "")):
                raise


def _is_duplicate(body: str) -> bool:
    text = body.lower()
    return "identifier" in text and ("already" in text or "duplicate" in text)
