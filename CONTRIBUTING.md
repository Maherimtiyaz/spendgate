# Contributing

```bash
pip install -e ".[dev]"
```

```bash
pytest
```

The PostgreSQL tests are skipped unless `TENANTBUDGET_TEST_DSN` points at a
database the tests may create tables in and truncate. Never point it at a
database you care about.

```bash
TENANTBUDGET_TEST_DSN=postgresql://user:pass@localhost:5432/tenantbudget_test pytest
```

Things worth knowing before you change code:

- **The core has no runtime dependencies** and must keep importing without
  the OpenAI, Anthropic or psycopg packages installed. Provider SDKs are
  recognised by module name, never imported.
- **Money is integer micro-dollars.** No floats in cost arithmetic.
- **Every store must pass `tests/test_ledger.py` and `tests/test_outbox.py`.**
  A new store is added to the `make_store` fixture in `tests/conftest.py`.
- **Tests go through the real SDK clients** with the HTTP layer mocked
  (`tests/helpers.py`), so usage parsing is tested against real response
  shapes. Please keep it that way for new providers.
- **Prices** live in `src/tenantbudget/prices.json`. A price change needs a link
  to the provider's pricing page in the pull request.
