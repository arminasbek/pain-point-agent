from types import SimpleNamespace

import pytest

from pain_point_agent.config import settings
from pain_point_agent.trace import Metered, Trace


def usage(input_tokens=0, output_tokens=0, cache_read=None, cache_write=None):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
    )


@pytest.fixture(autouse=True)
def prices(monkeypatch):
    monkeypatch.setattr(settings, "price_input_per_mtok", 1.0)
    monkeypatch.setattr(settings, "price_output_per_mtok", 10.0)


def test_cost_counts_loop_and_check_calls_with_cache_rates():
    trace = Trace()
    trace.round(1, usage(input_tokens=1_000_000, output_tokens=100_000, cache_read=1_000_000, cache_write=1_000_000))
    trace.check(usage(input_tokens=1_000_000, output_tokens=100_000))

    assert trace.summary() == {
        "rounds": 1,
        "loop_input_tokens": 1_000_000,
        "loop_cache_read_tokens": 1_000_000,
        "loop_cache_write_tokens": 1_000_000,
        "loop_output_tokens": 100_000,
        "check_calls": 1,
        "check_input_tokens": 1_000_000,
        "check_output_tokens": 100_000,
        "cost_usd": pytest.approx(1.0 + 0.1 + 1.25 + 1.0 + 1.0 + 1.0),
    }


def test_metered_client_forwards_the_call_and_records_usage():
    calls = []
    response = SimpleNamespace(usage=usage(input_tokens=300, output_tokens=20))
    inner = SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: calls.append(kwargs) or response))
    trace = Trace()

    result = Metered(inner, trace).messages.create(model="m", max_tokens=5)

    assert result is response
    assert calls == [{"model": "m", "max_tokens": 5}]
    assert trace.summary()["check_calls"] == 1
    assert trace.summary()["check_input_tokens"] == 300


def test_totals_print_checks_and_cost(capsys):
    trace = Trace(verbose=True)
    trace.round(1, usage(input_tokens=100, output_tokens=10))
    trace.check(usage(input_tokens=50, output_tokens=5))

    trace.totals()

    err = capsys.readouterr().err
    assert "── checks (judge, fit, theme): 1 calls · 50 input tokens · 5 output tokens" in err
    assert "── cost ≈ $0.0003" in err
