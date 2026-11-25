# Changelog

## 0.1.1 (2026-11-25)

First published version. The project was renamed from `spendgate` to
`tenantbudget` before release; 0.1.0 was tagged under the old name and never
published.

- Per-tenant and per-feature budgets (daily, monthly, lifetime) enforced with
  a reserve → call → settle ledger; calls that do not fit are refused before
  they reach the provider.
- `Guard.chat` / `Guard.achat` for the Anthropic Messages API, OpenAI Chat
  Completions and the OpenAI Responses API, including streaming.
- Stores: in-memory and PostgreSQL.
- Agent runs with cost and step ceilings and a repeated-tool-call breaker.
- Durable usage-event outbox with a dispatcher, and sinks for Stripe billing
  meters and CloudEvents (OpenMeter).
- Bundled prices for current Claude and OpenAI models.
