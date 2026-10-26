import json
from pathlib import Path

import pytest
from pydantic import SecretStr

from pain_point_agent import sources
from pain_point_agent.config import settings
from pain_point_agent.sources import github
from pain_point_agent.sources.github import parse_search, parse_thread

API = "https://api.github.com"
FIXTURES = Path(__file__).parent / "fixtures"
ISSUE_URL = "https://api.github.com/repos/WhiskeySockets/Baileys/issues/2075"


def load(name: str):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture(autouse=True)
def fresh_issue_urls(monkeypatch):
    monkeypatch.setattr(github, "_issue_urls", {})


def test_parse_search_returns_trimmed_results():
    results = parse_search(load("github_search.json"))

    assert len(results) == 5
    first = results[0]
    assert set(first) == {"id", "title", "posts_count", "created_at", "url", "blurb"}
    assert first["id"] == 4880339745
    assert first["posts_count"] == 15
    assert first["url"] == "https://github.com/WhiskeySockets/Baileys/issues/2707"
    assert first["blurb"].startswith("We are experiencing a critical issue")
    assert len(first["blurb"]) <= settings.max_blurb_chars


def test_parse_search_handles_empty_response():
    assert parse_search({}) == []


def test_parse_thread_puts_issue_first_then_comments():
    thread = parse_thread(load("github_issue.json"), load("github_comments.json"))

    assert thread["id"] == 3633260896
    assert thread["title"] == "Repeated Number Bans on WhatsApp Service"
    assert thread["url"] == "https://github.com/WhiskeySockets/Baileys/issues/2075"
    assert thread["posts"][0]["author"] == "hosamalzagh"
    assert thread["posts"][0]["text"].startswith("I've been experiencing repeated bans")
    assert thread["posts"][1]["author"] == "echogit"
    assert thread["posts"][0]["created_at"] == "2025-11-17T13:05:30Z"
    assert thread["posts"][1]["created_at"] == "2025-11-17T19:09:55Z"
    assert [post["number"] for post in thread["posts"]] == list(range(1, 14))
    assert thread["posts_count"] == 13
    assert thread["truncated"] is False


def test_parse_thread_drops_html_comments_and_empty_comments():
    issue = load("github_issue.json")
    issue["body"] = "<!-- Describe the bug below -->\nIt crashes on start"
    comments = [{"id": 1, "user": {"login": "a"}, "body": ""}, {"id": 2, "user": {"login": "b"}, "body": None}]

    thread = parse_thread(issue, comments)

    assert [post["text"] for post in thread["posts"]] == ["It crashes on start"]


def test_parse_thread_marks_missing_comments_as_truncated():
    thread = parse_thread(load("github_issue.json"), load("github_comments.json")[:5])

    assert thread["truncated"] is True


def test_parse_thread_truncates_long_posts(monkeypatch):
    monkeypatch.setattr(settings, "max_post_chars", 20)

    thread = parse_thread(load("github_issue.json"), load("github_comments.json"))

    assert all(len(post["text"]) <= 20 for post in thread["posts"])
    assert thread["truncated"] is True


def fake_get(monkeypatch, *responses) -> list:
    calls = []
    replies = iter(responses)

    def get(url, params=None, headers=None):
        calls.append((url, params, headers))
        return next(replies)

    monkeypatch.setattr(github, "get", get)
    return calls


def test_search_asks_for_issues_with_discussion(monkeypatch):
    calls = fake_get(monkeypatch, load("github_search.json"))

    results = github.search(API, "whatsapp bot banned")

    assert calls == [
        (
            f"{API}/search/issues",
            {"q": "whatsapp bot banned is:issue comments:>1", "per_page": str(settings.search_limit)},
            {},
        )
    ]
    assert len(results) == 5


def test_fetch_uses_issue_url_from_search(monkeypatch):
    search = {"items": [dict(load("github_issue.json"))]}
    calls = fake_get(monkeypatch, search, load("github_issue.json"), load("github_comments.json"))

    github.search(API, "whatsapp bans")
    thread = github.fetch(API, 3633260896)

    assert calls[1:] == [(ISSUE_URL, None, {}), (f"{ISSUE_URL}/comments", {"per_page": "30"}, {})]
    assert thread["id"] == 3633260896


def test_fetch_issue_without_text_says_so(monkeypatch):
    issue = dict(load("github_issue.json"), body="", comments=0)
    fake_get(monkeypatch, {"items": [issue]}, issue, [])

    github.search(API, "anything")
    with pytest.raises(ValueError, match="have no text"):
        github.fetch(API, issue["id"])


def test_fetch_unknown_id_asks_to_search_first():
    with pytest.raises(ValueError, match="Search first"):
        github.fetch(API, 123)


def test_token_is_sent_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "github_token", SecretStr("secret"))
    calls = fake_get(monkeypatch, {"items": []})

    github.search(API, "anything")

    assert calls[0][2] == {"Authorization": "Bearer secret"}


def test_github_is_a_configured_source():
    assert settings.sources["github"].type == "github"
    assert settings.sources["github"].url == API
    assert sources.ADAPTERS["github"] is github
