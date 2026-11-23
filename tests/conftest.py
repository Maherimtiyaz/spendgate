import os

import pytest

from spendgate import Guard, InMemoryStore

# Postgres-backed tests run when this points at a database the tests may
# create tables in and truncate, e.g.
#   SPENDGATE_TEST_DSN=postgresql://user:pass@localhost:5432/spendgate_test
DSN = os.environ.get("SPENDGATE_TEST_DSN")


class Clock:
    def __init__(self, now=1_800_000_000.0):  # 2027-01-15 08:00 UTC
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture(params=["memory", "postgres"])
def make_store(request):
    """Factory for a fresh, empty ledger store of each kind."""
    if request.param == "memory":
        yield lambda clock=None: InMemoryStore(**({"clock": clock} if clock else {}))
        return
    if not DSN:
        pytest.skip("set SPENDGATE_TEST_DSN to run the Postgres tests")
    from spendgate.postgres import PostgresStore

    opened = []

    def factory(clock=None):
        store = PostgresStore(DSN, **({"clock": clock} if clock else {})).setup()
        opened.append(store)
        with store._pool.connection() as conn:
            conn.execute("TRUNCATE spendgate_budgets, spendgate_periods, "
                         "spendgate_reservations, spendgate_outbox")
        return store

    yield factory
    for store in opened:
        store.close()


@pytest.fixture
def events():
    return []


@pytest.fixture
def guard(events):
    g = Guard(on_usage=events.append)
    g.set_budget("acme", monthly_usd=1)
    return g
