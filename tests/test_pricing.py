import pytest

from tenantbudget import PriceRegistry, UnknownModelError, Usage, usd_to_micros


@pytest.fixture
def prices():
    return PriceRegistry.bundled()


def test_usd_to_micros_is_exact_for_decimal_strings_and_floats():
    assert usd_to_micros("0.10") == 100_000
    assert usd_to_micros(0.1) == 100_000
    assert usd_to_micros(200) == 200_000_000


def test_cost_uses_each_token_class_rate(prices):
    usage = Usage(input_tokens=1000, output_tokens=500, cache_read_tokens=10_000,
                  cache_write_5m_tokens=2000, cache_write_1h_tokens=100)
    # opus 5.5: in 4, out 20, read 0.20, write 5 / 8 (micros per token)
    assert prices.cost_micros("claude-opus-5-5", usage) == (
        1000 * 4 + 500 * 20 + 2000 + 2000 * 5 + 100 * 8)


def test_cost_rounds_up_to_a_whole_micro(prices):
    # gpt-5.4-mini cached input is 0.075 micros per token
    assert prices.cost_micros("gpt-5.4-mini", Usage(cache_read_tokens=1)) == 1


def test_missing_cache_price_falls_back_to_input_rate(prices):
    assert prices.cost_micros("o3-pro", Usage(cache_read_tokens=10)) == 200


def test_long_context_tier_applies_to_the_whole_request(prices):
    small = Usage(input_tokens=100_000, output_tokens=10)
    large = Usage(input_tokens=100_001, output_tokens=10)
    assert prices.cost_micros("claude-haiku-5-5", small) == 10_000 + 5
    assert prices.cost_micros("claude-haiku-5-5", large) == 50_001 + 25  # 50000.5 rounds up


def test_worst_case_assumes_no_cache_hits_and_full_output(prices):
    assert prices.worst_case_micros("claude-opus-5-5", 1000, 2000) == 1000 * 4 + 2000 * 20


def test_worst_case_prices_cache_writes_at_the_dearest_rate(prices):
    assert prices.worst_case_micros("claude-opus-5-5", 1000, 0, cache_write=True) == 8000


def test_worst_case_is_never_below_actual_cost(prices):
    usage = Usage(input_tokens=300, cache_read_tokens=500, cache_write_1h_tokens=200,
                  output_tokens=700)
    bound = prices.worst_case_micros("claude-sonnet-5-5", usage.prompt_tokens, 700,
                                     cache_write=True)
    assert bound >= prices.cost_micros("claude-sonnet-5-5", usage)


@pytest.mark.parametrize("alias", [
    "gpt-5.4-2026-03-05", "claude-opus-4-6-20260115", "claude-opus-4-6@20260115",
    "anthropic.claude-opus-4-6", "us.anthropic.claude-opus-4-6", "openai/gpt-5.4",
])
def test_dated_and_vendor_prefixed_ids_resolve(prices, alias):
    base = "gpt-5.4" if "gpt" in alias else "claude-opus-4-6"
    assert prices.get(alias) == prices.get(base)


def test_unknown_model_raises_rather_than_costing_nothing(prices):
    with pytest.raises(UnknownModelError):
        prices.get("gpt-5.4-turbo-preview")  # not a dated snapshot of gpt-5.4


def test_register_adds_and_overrides(prices):
    prices.register("my-model", input="1", output="2", cache_read="0.5")
    assert prices.cost_micros("my-model", Usage(10, 10, 10)) == 10 + 20 + 5
