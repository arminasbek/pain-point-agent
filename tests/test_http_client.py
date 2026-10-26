import httpx
import pytest

from pain_point_agent.sources.models import SourceError
from pain_point_agent.sources.utils import http_client


def fake_responses(monkeypatch, responses):
    calls = iter(responses)
    sleeps = []
    request = httpx.Request("GET", "https://example.test")
    monkeypatch.setattr(http_client.httpx, "get", lambda *args, **kwargs: httpx.Response(**next(calls), request=request))
    monkeypatch.setattr(http_client.time, "sleep", sleeps.append)
    return sleeps


def test_get_retries_on_429_and_honours_retry_after(monkeypatch):
    sleeps = fake_responses(
        monkeypatch,
        [{"status_code": 429, "headers": {"Retry-After": "5"}}, {"status_code": 200, "json": {"ok": True}}],
    )

    assert http_client.get("https://example.test") == {"ok": True}
    assert sleeps == [5.0]


def test_get_backs_off_exponentially_then_gives_up(monkeypatch):
    sleeps = fake_responses(monkeypatch, [{"status_code": 503}] * 4)

    with pytest.raises(httpx.HTTPStatusError):
        http_client.get("https://example.test")
    assert sleeps == [1.0, 2.0, 4.0]


def test_get_does_not_retry_on_404(monkeypatch):
    sleeps = fake_responses(monkeypatch, [{"status_code": 404}])

    with pytest.raises(httpx.HTTPStatusError):
        http_client.get("https://example.test")
    assert sleeps == []


def test_get_adds_extra_headers_to_user_agent(monkeypatch):
    sent = {}
    request = httpx.Request("GET", "https://example.test")

    def fake_get(url, **kwargs):
        sent.update(kwargs["headers"])
        return httpx.Response(200, json={"ok": True}, request=request)

    monkeypatch.setattr(http_client.httpx, "get", fake_get)

    http_client.get("https://example.test", headers={"Authorization": "Bearer t"})

    assert sent == {"User-Agent": http_client.settings.user_agent, "Authorization": "Bearer t"}


def flaky_network(monkeypatch, failures, then=None):
    attempts = []
    sleeps = []
    request = httpx.Request("GET", "https://example.test")

    def get(*args, **kwargs):
        attempts.append(1)
        if len(attempts) <= failures:
            raise httpx.ConnectTimeout("timed out", request=request)
        return httpx.Response(200, json=then, request=request)

    monkeypatch.setattr(http_client.httpx, "get", get)
    monkeypatch.setattr(http_client.time, "sleep", sleeps.append)
    return sleeps


def test_get_retries_network_errors(monkeypatch):
    sleeps = flaky_network(monkeypatch, failures=2, then={"ok": True})

    assert http_client.get("https://example.test") == {"ok": True}
    assert sleeps == [1.0, 2.0]


def test_get_gives_up_on_network_errors_after_the_last_retry(monkeypatch):
    sleeps = flaky_network(monkeypatch, failures=10)

    with pytest.raises(httpx.ConnectTimeout):
        http_client.get("https://example.test")
    assert sleeps == [1.0, 2.0, 4.0]


def test_get_reports_a_response_that_is_not_json(monkeypatch):
    request = httpx.Request("GET", "https://example.test")
    monkeypatch.setattr(
        http_client.httpx, "get", lambda *args, **kwargs: httpx.Response(200, text="<html>maintenance</html>", request=request)
    )

    with pytest.raises(SourceError, match="did not return JSON"):
        http_client.get("https://example.test")
