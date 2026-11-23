"""The FastAPI example, driven over HTTP: budgets, refusals and billing events."""
import sys
import threading
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # needed by FastAPI's test client
from fastapi.testclient import TestClient  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "fastapi_app"))
from app import create_app  # noqa: E402
from mock_provider import mock_client  # noqa: E402


class BillingSink:
    name = "billing"

    def __init__(self):
        self.events = []
        self.arrived = threading.Semaphore(0)

    def send(self, event):
        self.events.append(event)
        self.arrived.release()


@pytest.fixture
def billing():
    return BillingSink()


@pytest.fixture
def api(make_store, billing):
    app = create_app(client=mock_client(), store=make_store(), sinks=[billing],
                     dispatch_interval=0.01)
    with TestClient(app) as client:
        yield client


def chat(api, tenant, message="hello"):
    return api.post("/chat", json={"message": message}, headers={"X-Tenant-Id": tenant})


def test_customer_is_served_until_the_budget_runs_out_then_gets_402(api, billing):
    # A mock call costs about $0.008; each call must also fit a $0.041 worst case.
    api.put("/tenants/acme/budget", json={"monthly_usd": 0.06})
    replies = [chat(api, "acme") for _ in range(6)]
    served = [r for r in replies if r.status_code == 200]
    refused = [r for r in replies if r.status_code == 402]

    assert len(served) == 3 and len(refused) == 3
    assert served[0].json()["reply"].startswith("This is a canned reply")
    detail = refused[0].json()["detail"]
    assert detail["error"] == "budget_exceeded" and detail["window"] == "month"
    assert detail["remaining_usd"] < detail["needed_usd"]

    usage = api.get("/tenants/acme/usage").json()["budgets"][0]
    assert usage["spent_usd"] == pytest.approx(sum(r.json()["cost_usd"] for r in served))
    assert usage["spent_usd"] <= usage["limit_usd"] == 0.06

    # Billing received one event per served call, and none for the refusals.
    for _ in served:
        assert billing.arrived.acquire(timeout=5)
    assert sorted(e.id for e in billing.events) == sorted(
        r.json()["usage_event_id"] for r in served)
    assert {e.tenant for e in billing.events} == {"acme"}


def test_one_customers_budget_does_not_affect_another(api):
    api.put("/tenants/acme/budget", json={"monthly_usd": 0.0001})
    api.put("/tenants/globex/budget", json={"monthly_usd": 5})
    assert chat(api, "acme").status_code == 402
    assert chat(api, "globex").status_code == 200
