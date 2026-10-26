from pain_point_agent.config import settings
from pain_point_agent.sources.utils.posts import trim_posts


def test_keeps_short_posts_as_they_are():
    posts, truncated = trim_posts([(1, "a", "first", "2026-10-01"), (2, "b", "second", "")])

    assert posts == [
        {"number": 1, "author": "a", "text": "first", "created_at": "2026-10-01"},
        {"number": 2, "author": "b", "text": "second", "created_at": ""},
    ]
    assert truncated is False


def test_cuts_a_long_post_and_keeps_going(monkeypatch):
    monkeypatch.setattr(settings, "max_post_chars", 5)

    posts, truncated = trim_posts([(1, "a", "0123456789", ""), (2, "b", "ok", "")])

    assert [post["text"] for post in posts] == ["01234", "ok"]
    assert truncated is True


def test_stops_when_the_thread_budget_runs_out(monkeypatch):
    monkeypatch.setattr(settings, "max_thread_chars", 8)

    posts, truncated = trim_posts([(1, "a", "12345", ""), (2, "b", "6789", ""), (3, "c", "x", "")])

    assert [post["number"] for post in posts] == [1]
    assert truncated is True
