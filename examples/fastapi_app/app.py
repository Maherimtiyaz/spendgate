"""A small multi-tenant chat API with per-customer LLM budgets and billing.

Run it without any API key, against a canned model:

    TENANTBUDGET_DEMO_MOCK=1 uvicorn app:app --factory --app-dir examples/fastapi_app

Environment:

    ANTHROPIC_API_KEY     real model calls (not needed with TENANTBUDGET_DEMO_MOCK=1)
    TENANTBUDGET_DSN         PostgreSQL connection string; budgets are kept in
                          memory, per process, if unset
    USAGE_WEBHOOK_URL     POST each usage event there as a CloudEvent (OpenMeter)
    STRIPE_API_KEY        record each usage event as a Stripe meter event ...
    STRIPE_METER_EVENT    ... under this meter event name (default llm_spend)
    STRIPE_CUSTOMERS      ... for these tenants: JSON such as {"acme": "cus_123"}
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager

from anthropic import AsyncAnthropic
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from tenantbudget import (BudgetExceeded, CloudEventsSink, Dispatcher, Guard, InMemoryStore,
                       StripeMeterSink, UsageEvent)

MODEL = os.environ.get("TENANTBUDGET_DEMO_MODEL", "claude-opus-5-5")
MAX_TOKENS = int(os.environ.get("TENANTBUDGET_DEMO_MAX_TOKENS", "2048"))

log = logging.getLogger("uvicorn.error")  # visible in uvicorn's output


class LogSink:
    """Stands in for a billing system when none is configured."""

    name = "log"

    def send(self, event: UsageEvent) -> None:
        log.info("usage tenant=%s model=%s cost=$%.6f id=%s",
                 event.tenant, event.model, event.cost_usd, event.id)


def store_from_env():
    dsn = os.environ.get("TENANTBUDGET_DSN")
    if not dsn:
        return InMemoryStore()
    from tenantbudget.postgres import PostgresStore
    return PostgresStore(dsn).setup()


def sinks_from_env() -> list:
    sinks: list = []
    if os.environ.get("USAGE_WEBHOOK_URL"):
        sinks.append(CloudEventsSink(os.environ["USAGE_WEBHOOK_URL"]))
    if os.environ.get("STRIPE_API_KEY"):
        sinks.append(StripeMeterSink(
            os.environ["STRIPE_API_KEY"],
            os.environ.get("STRIPE_METER_EVENT", "llm_spend"),
            customer=json.loads(os.environ.get("STRIPE_CUSTOMERS", "{}"))))
    return sinks or [LogSink()]


def client_from_env() -> AsyncAnthropic:
    if os.environ.get("TENANTBUDGET_DEMO_MOCK"):
        from mock_provider import mock_client
        return mock_client()
    return AsyncAnthropic()


class BudgetIn(BaseModel):
    monthly_usd: float | None = None
    daily_usd: float | None = None
    soft_pct: float | None = None


class ChatIn(BaseModel):
    message: str
    feature: str | None = None


def create_app(client: AsyncAnthropic | None = None, store=None, sinks=None,
               dispatch_interval: float = 1.0) -> FastAPI:
    store = store if store is not None else store_from_env()
    client = client or client_from_env()
    # outbox=True: every call's usage event is stored with its booking, and
    # the dispatcher below delivers it to billing, retrying until it lands.
    guard = Guard(store, outbox=True)
    dispatcher = Dispatcher(store, sinks or sinks_from_env(), interval=dispatch_interval)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        dispatcher.start()
        yield
        dispatcher.stop()

    app = FastAPI(title="tenantbudget example", lifespan=lifespan)

    # In a real service the tenant comes from the caller's credentials, and
    # the two /tenants routes are admin-only. Here both are wide open.

    @app.put("/tenants/{tenant}/budget")
    def set_budget(tenant: str, budget: BudgetIn):
        guard.set_budget(tenant, monthly_usd=budget.monthly_usd, daily_usd=budget.daily_usd,
                         soft_pct=budget.soft_pct)
        return {"tenant": tenant, **budget.model_dump()}

    @app.get("/tenants/{tenant}/usage")
    def usage(tenant: str):
        return {"tenant": tenant, "budgets": [
            {"window": s.key.window, "period": s.period,
             "limit_usd": s.limit_micros / 1e6, "spent_usd": s.spent_micros / 1e6,
             "remaining_usd": s.remaining_micros / 1e6}
            for s in guard.status(tenant)]}

    @app.post("/chat")
    async def chat(body: ChatIn, x_tenant_id: str = Header()):
        try:
            response = await guard.achat(
                client, tenant=x_tenant_id, feature=body.feature, model=MODEL,
                max_tokens=MAX_TOKENS, messages=[{"role": "user", "content": body.message}])
        except BudgetExceeded as exc:
            # Refused before the model was called: nothing was spent.
            raise HTTPException(status_code=402, detail={
                "error": "budget_exceeded", "window": exc.window,
                "remaining_usd": exc.remaining_micros / 1e6,
                "needed_usd": exc.requested_micros / 1e6})
        event = guard.last_event
        return {
            "reply": "".join(b.text for b in response.content if b.type == "text"),
            "stop_reason": response.stop_reason,
            "cost_usd": event.cost_usd,
            "usage_event_id": event.id,
        }

    app.state.guard = guard
    return app


def app() -> FastAPI:  # for: uvicorn app:app --factory
    return create_app()
